import itertools
import random
import threading
from queue import Queue
from typing import Iterator

import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from vlm.data.processors import get_image_string


class BaseDataset(Dataset):
    def __init__(self, dataset, tokenizer, image_processor, mp_image_token_length):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length

        self.prefix_len = self._get_prefix_len()

    def __len__(self):
        return len(self.dataset)

    def _get_prefix_len(self):
        random_string_5_letters = "xzyvd"
        random_string_chat_templated = self.tokenizer.apply_chat_template(
            [{"role": "assistant", "content": random_string_5_letters}],
            tokenize=False,
            add_special_tokens=False,
        )
        random_string_location = random_string_chat_templated.find(random_string_5_letters)
        return len(self.tokenizer.encode(random_string_chat_templated[:random_string_location]))

    def _get_messages(self, item, splitted_image_counts):
        messages = []
        for text in item["texts"]:
            messages.append({"role": "user", "content": text["user"]})
            messages.append({"role": "assistant", "content": text["assistant"]})

        image_string = get_image_string(self.tokenizer, splitted_image_counts, self.mp_image_token_length)

        if len(splitted_image_counts) > 0:
            messages[0]["content"] = image_string + messages[0]["content"]

        return messages

    def _process_images(self, images):
        processed_images = []
        splitted_image_counts = []
        for image in images:
            if isinstance(image, Image.Image):
                if image.mode != "RGB":
                    image = image.convert("RGB")
                processed_image, splitted_image_count = self.image_processor(image)
                processed_images.append(processed_image)
                splitted_image_counts.append(splitted_image_count)
            else:
                raise ValueError("Error processing image")
        return processed_images, splitted_image_counts

    def _prepare_inputs_and_loss_mask(self, messages):
        conv_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            return_dict=True,
        )
        mask = [0] * len(conv_ids["input_ids"])

        # Locate each assistant turn and flip its mask to 1
        cursor = 0
        for msg in messages:
            segment_ids = self.tokenizer.apply_chat_template([msg], tokenize=True, add_special_tokens=False)
            seg_len = len(segment_ids)

            if msg["role"] == "assistant":
                start = cursor + self.prefix_len
                end = cursor + seg_len
                mask[start:end] = [1] * (end - start)  # attend to these tokens

            cursor += seg_len

        return (
            torch.tensor(conv_ids["input_ids"]),
            torch.tensor(mask).to(torch.bool),
            torch.tensor(conv_ids["attention_mask"]),
        )


class VQADataset(BaseDataset):  # Visual Question Answering Dataset
    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Handle images (should be a list)
        images_data = item["images"]
        if not isinstance(images_data, list):
            images_data = [images_data]

        # Now process the images
        processed_images, splitted_image_counts = self._process_images(images_data)

        messages = self._get_messages(item, splitted_image_counts)

        input_ids, mask, attention_mask = self._prepare_inputs_and_loss_mask(messages)
        labels = self._get_labels(input_ids, mask)

        return {
            "images": processed_images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _get_labels(self, input_ids, mask):
        labels = input_ids.clone().masked_fill(~mask, -100)
        labels = labels.roll(-1)  # Shift labels for causal LM
        labels[-1] = -100  # Last token has no target

        return labels


random.seed(42)  # Set the random seed to the meaning of life for good luck


class ConstantLengthDataset(IterableDataset):
    def __init__(
        self,
        dataset,
        infinite: bool = False,
        max_sample_length: int = 1024,
        seq_length: int = 1024,
        num_of_sequences: int = 1024,
        queue_size: int = 2048,
        max_images_per_example: int = 4,
        max_images_per_knapsack: int = 18,
    ):
        self.dataset = dataset
        self.max_sample_length = max_sample_length
        self.seq_length = seq_length
        self.max_length = seq_length * num_of_sequences
        self.epoch = 0  # only advanced when infinite=True
        self.infinite = infinite
        self.queue_size = queue_size
        self.max_images_per_example = max_images_per_example
        self.max_images_per_knapsack = max_images_per_knapsack
        self._sentinel = object()
        self._average_length_per_sample = (
            self.dataset.mp_image_token_length + 198
        )  # 198 is the average tokens for the cauldron dataset

    def __len__(self):
        return int(len(self.dataset) * self._average_length_per_sample / self.seq_length)

    def __iter__(self) -> Iterator[dict]:
        """
        Returns an iterator over the dataset that yields fixed-length sequences for training.

        The iterator uses a producer-consumer pattern with a background thread to efficiently
        pre-fetch and buffer samples. The producer thread continuously reads from the base
        dataset and fills a queue, while the main thread consumes from the queue.

        The dataset is automatically sharded across workers when using num_workers > 1.

        Returns:
            Iterator[dict]: An iterator that yields training samples with the following structure:
                - input_ids: Tensor of token ids of shape (seq_length,)
                - labels: Tensor of labels of shape (seq_length,)
                - attention_mask: Tensor of attention mask of shape (seq_length,)
                - images: List of processed image tensors
        """
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        def make_base_iterator():
            """Return a (sharded) iterator over the underlying dataset."""
            all_indices = range(len(self.dataset))

            # Shard the *indices* first, before any data is fetched.
            if num_workers > 1:
                worker_indices = itertools.islice(all_indices, worker_id, None, num_workers)
            else:
                worker_indices = all_indices

            # Create an iterator that only calls __getitem__ for the assigned indices.
            def sharded_item_iterator():
                for idx in worker_indices:
                    yield self.dataset[idx]

            return sharded_item_iterator()

        queue: Queue = Queue(maxsize=self.queue_size)

        producer = threading.Thread(target=self._producer, args=(make_base_iterator, queue), daemon=True)
        producer.start()

        while True:
            sample = queue.get()
            if sample is self._sentinel:
                break
            yield sample

    def _producer(
        self,
        make_iterator,  # a zero-arg lambda that returns a fresh (possibly sharded) iterator
        queue: Queue,
    ):
        """Runs in a separate daemon thread and keeps `queue` full."""
        iterator = make_iterator()
        more_examples = True

        while more_examples:
            # ------------- 1) pull raw samples until we have enough -------- #
            buffer, buffer_len = [], 0
            while buffer_len < self.max_length:
                try:
                    sample = next(iterator)
                except StopIteration:
                    if self.infinite:
                        iterator = make_iterator()
                        self.epoch += 1
                        print(f"Epoch {self.epoch} finished, restarting iterator")
                        continue
                    else:
                        more_examples = False
                        break

                if len(sample["input_ids"]) >= self.max_sample_length:
                    continue  # skip overly long samples
                if len(sample["images"]) > self.max_images_per_example:
                    continue  # skip samples that exceed the image constraint

                sample["input_ids"] = torch.cat(
                    [
                        sample["input_ids"],
                        torch.tensor([self.dataset.tokenizer.pad_token_id]),
                    ]
                )
                sample["attention_mask"] = torch.cat([sample["attention_mask"], torch.tensor([0])])
                sample["labels"] = torch.cat([sample["labels"], torch.tensor([-100])])

                buffer.append(sample)
                buffer_len += len(sample["input_ids"])

            if not buffer:
                break  # nothing left and not infinite

            # ------------- 2) run greedy knapsack & pack groups ------------ #
            groups = self._balanced_greedy_knapsack(
                buffer,
                self.seq_length,
                delta=5,
                max_images_per_knapsack=self.max_images_per_knapsack,
            )

            for g in groups:
                packed = self._pack_one_group(g, buffer, self.seq_length)

                # put blocks if queue is full.
                queue.put(
                    {
                        "input_ids": packed[0],
                        "labels": packed[1],
                        "attention_mask": packed[2],
                        "images": packed[3],
                    }
                )

        # finished → unblock consumer
        queue.put(self._sentinel)

    def _balanced_greedy_knapsack(self, buffer, L, delta=0, max_images_per_knapsack=None):
        # Extract lengths and image counts from buffer
        lengths = [len(x["input_ids"]) for x in buffer]
        image_counts = [len(x["images"]) for x in buffer]

        # keep the position while sorting
        items = sorted(enumerate(zip(lengths, image_counts)), key=lambda x: x[1][0], reverse=True)

        min_knapsacks = (sum(lengths) + L - 1) // L + delta
        knapsack_load = [0] * min_knapsacks
        knapsack_image_counts = [0] * min_knapsacks
        knapsack_groups = [[] for _ in range(min_knapsacks)]

        for idx, (item_len, item_image_count) in items:
            # Find a suitable knapsack that satisfies both length and image count constraints
            suitable_knapsack = None

            # First try to find a knapsack that can fit both constraints
            for ks_id in sorted(range(len(knapsack_load)), key=knapsack_load.__getitem__):
                length_fits = knapsack_load[ks_id] + item_len <= L
                image_fits = (
                    max_images_per_knapsack is None
                    or knapsack_image_counts[ks_id] + item_image_count <= max_images_per_knapsack
                )

                if length_fits and image_fits:
                    suitable_knapsack = ks_id
                    break

            # If no existing knapsack can fit, create a new one
            if suitable_knapsack is None:
                suitable_knapsack = len(knapsack_load)
                knapsack_load.append(0)
                knapsack_image_counts.append(0)
                knapsack_groups.append([])

            knapsack_groups[suitable_knapsack].append(idx)
            knapsack_load[suitable_knapsack] += item_len
            knapsack_image_counts[suitable_knapsack] += item_image_count

        # remove the completely empty bags that the +delta heuristic created
        random.shuffle(knapsack_groups)  # Knapsacks are semi-ordered after packing, thanks Luis for noticing!
        return [g for g in knapsack_groups if g]

    def _pack_one_group(self, group_indices, batch, max_len):
        ids, lbl, am, ims = [], [], [], []

        for i in group_indices:
            ids.extend(batch[i]["input_ids"])
            lbl.extend(batch[i]["labels"])
            am.extend(batch[i]["attention_mask"])
            ims.extend(batch[i]["images"])

        # safety: assert we never overflow
        if len(ids) > max_len:
            raise ValueError(f"Packed length {len(ids)} > max_len {max_len}")

        return torch.stack(ids), torch.stack(lbl), torch.stack(am), ims
