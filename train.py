import argparse
import contextlib
import math
import os
import time
from dataclasses import asdict
from statistics import mean

import PIL.PngImagePlugin
import torch
import torch.optim as optim
from datasets import concatenate_datasets, get_dataset_config_names, load_dataset
from dotenv import load_dotenv
from torch.utils.data import DataLoader, DistributedSampler

import wandb
from vlm.config import ModelConfig, TrainConfig
from vlm.data.collators import VQACollator
from vlm.data.datasets import ConstantLengthDataset, VQADataset
from vlm.data.processors import get_image_processor, get_tokenizer
from vlm.model.vision_language_model import VisionLanguageModel
from vlm.utils.dist import (
    destroy_dist,
    dist_gather,
    get_rank,
    get_world_size,
    init_dist,
    is_dist,
    is_master,
    synchronized_dataloader_step,
    wrap_model,
)
from vlm.utils.utils import get_device, get_run_name, seed_everything, seed_worker

load_dotenv()
seed_everything(42)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
PIL.PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024


def get_dataloader(train_config: TrainConfig, model_config: ModelConfig):
    image_processor = get_image_processor(model_config.max_img_size, model_config.vit_img_size)
    tokenizer = get_tokenizer(
        model_config.lm_tokenizer, model_config.vlm_extra_tokens, model_config.lm_chat_template
    )

    # Load and combine all training datasets
    combined_train_data = []
    dataset_names_to_load = train_config.train_dataset_name

    if "all" in dataset_names_to_load:
        dataset_names_to_load = get_dataset_config_names(train_config.train_dataset_path)

    for dataset_name in dataset_names_to_load:
        try:
            train_ds = load_dataset(train_config.train_dataset_path, dataset_name)
            combined_train_data.append(train_ds["train"])
        except Exception as e:
            if is_master():
                print(
                    f"Warning: Failed to load dataset config '{dataset_name}' from '{train_config.train_dataset_path}'. Error: {e}"
                )
            continue

    if not combined_train_data:
        raise ValueError("No valid datasets were loaded. Please check your dataset path and configurations.")

    train_ds = concatenate_datasets(combined_train_data)

    train_ds = train_ds.shuffle(
        seed=0
    )  # Shuffle the training dataset, so train and val get equal contributions from all concatenated datasets

    if is_dist():  # We need to shard the dataset in DDP since we are using an iterable dataset instead of the distributed sampler
        train_ds = train_ds.shard(num_shards=get_world_size(), index=get_rank())

    # Apply cutoff if specified
    if train_config.data_cutoff_idx is None:
        total_samples = len(train_ds)  # Use the entire dataset
    else:
        total_samples = min(len(train_ds), train_config.data_cutoff_idx)

    val_size = int(total_samples * train_config.val_ratio)
    train_size = total_samples - val_size

    train_dataset = VQADataset(
        train_ds.select(range(train_size)), tokenizer, image_processor, model_config.mp_image_token_length
    )

    train_dataset = ConstantLengthDataset(
        train_dataset,
        infinite=False,
        max_sample_length=train_config.max_sample_length,
        seq_length=model_config.lm_max_length,
        num_of_sequences=train_config.batch_size * 64,
        queue_size=train_config.batch_size * 64 * 2,
        max_images_per_example=train_config.max_images_per_example,
        max_images_per_knapsack=train_config.max_images_per_knapsack,
    )
    val_dataset = VQADataset(
        train_ds.select(range(train_size, total_samples)),
        tokenizer,
        image_processor,
        model_config.mp_image_token_length,
    )

    # Create collators
    vqa_collator = VQACollator(tokenizer, model_config.lm_max_length)

    g = torch.Generator()
    g.manual_seed(0)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,  # =per device BS in DDP
        collate_fn=vqa_collator,
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_sampler = DistributedSampler(
        val_dataset,
        rank=get_rank(),
        num_replicas=get_world_size(),
        shuffle=False,  # Usually False for validation
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.batch_size,
        sampler=val_sampler,
        collate_fn=vqa_collator,
        num_workers=8,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return train_loader, val_loader


def get_lr(it, max_lr, max_steps):
    min_lr = max_lr * 0.1
    warmup_steps = max_steps * 0.03
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


def train(train_config: TrainConfig, model_config: ModelConfig):
    train_loader, val_loader = get_dataloader(train_config, model_config)
    total_dataset_size = len(train_loader.dataset)

    # Start Wandb logging
    run_name = get_run_name(train_config, model_config)
    if train_config.log_wandb and is_master():
        if train_config.data_cutoff_idx is None:
            run_name = run_name.replace("full_ds", f"{total_dataset_size}samples")
    if train_config.log_wandb and is_master():
        run = wandb.init(
            entity=os.getenv("WANDB_ENTITY"),
            project="nanoVLM",
            config={"VLMConfig": asdict(model_config), "TrainConfig": asdict(train_config)},
            name=run_name,
        )

    # Initialize model or resume from checkpoint
    if train_config.resume_from_vlm_checkpoint:
        model = VisionLanguageModel.from_pretrained(model_config.vlm_checkpoint)
    else:
        model = VisionLanguageModel(
            model_config,
            load_backbone=model_config.vlm_load_backbone_weights,
        )

    # Start Training (ALL RANKS)
    if is_master():
        print(f"nanoVLM initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(
            f"Training summary{' (global)' if is_dist() else ''}: {len(train_loader.dataset)} samples, {int(len(train_loader) * get_world_size())} batches/epoch, batch size {int(train_config.batch_size * get_world_size() * train_config.gradient_accumulation_steps)}{', training on ' + str(get_world_size()) + ' GPUs' if is_dist() else ''}"
        )
        if is_dist():
            print(
                f"Training summary per GPU: {len(train_loader)} batches/epoch, batch size {train_loader.batch_size}"
            )
        print(
            f"Validation summary{' (global)' if is_dist() else ''}: {len(val_loader.dataset)} samples, {int(len(val_loader) * get_world_size())} batches/epoch, batch size {int(train_config.batch_size * get_world_size() * train_config.gradient_accumulation_steps)}{', training on ' + str(get_world_size()) + ' GPUs' if is_dist() else ''}"
        )
        if is_dist():
            print(
                f"Validation summary per GPU: {len(val_loader)} batches/epoch, batch size {val_loader.batch_size}"
            )

    # Get Optimizer (ALL RANKS)
    params_groups = [{"params": list(model.mp.parameters()), "lr": train_config.lr_mp}]
    if train_config.lr_backbones > 0:
        params_groups.append(
            {
                "params": list(model.decoder.parameters()) + list(model.vision_encoder.parameters()),
                "lr": train_config.lr_backbones,
            }
        )
    else:
        for p in list(model.decoder.parameters()) + list(model.vision_encoder.parameters()):
            p.requires_grad = False

    optimizer = optim.AdamW(params_groups)
    all_params = [p for group in optimizer.param_groups for p in group["params"]]

    device = get_device(get_rank())
    if device.type == "mps":
        torch.backends.mps.enable_fallback_to_cpu = True
        torch.mps.empty_cache()

    model.to(device)
    if train_config.compile:
        model = torch.compile(model)
    if is_dist():
        model = wrap_model(model)

    epoch_times = []
    best_val_loss = float("inf")
    global_step = 0
    epoch = 0

    accumulated_stats = {
        "tokens_per_second": [],
        "data_load_time": [],
        "fw_bw_time": [],
        "post_process_time": [],
        "images_per_sample": [],
    }

    while global_step < train_config.max_training_steps:
        epoch += 1
        epoch_start_time = time.time()
        model.train()

        total_train_loss = 0
        total_tokens_processed = 0
        optimizer.zero_grad()
        data_load_start = time.time()

        for i, batch in enumerate(synchronized_dataloader_step(train_loader, is_dist())):
            is_update_step = (i + 1) % train_config.gradient_accumulation_steps == 0 or i + 1 == len(
                train_loader
            )
            batch_start_time = time.time()

            # Get data
            images = batch["images"]
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            data_load_time = time.time() - data_load_start

            # Forward pass
            if is_dist() and train_config.gradient_accumulation_steps > 1 and not is_update_step:
                context = model.no_sync()
            else:
                context = contextlib.nullcontext()

            fw_bw_start = time.time()
            autocast_context = torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if device.type in ["cuda", "cpu"] else torch.float16,
            )
            with autocast_context, context:
                _, loss = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                    targets=labels,
                )

            loss.backward()
            fw_bw_time = time.time() - fw_bw_start

            post_process_start = time.time()
            if is_update_step:
                if train_config.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        all_params, max_norm=train_config.max_grad_norm
                    )

                adj_lr_mp = get_lr(global_step, train_config.lr_mp, train_config.max_training_steps)
                optimizer.param_groups[0]["lr"] = adj_lr_mp

                if train_config.lr_backbones > 0:
                    adj_lr_backbones = get_lr(
                        global_step, train_config.lr_backbones, train_config.max_training_steps
                    )
                    optimizer.param_groups[1]["lr"] = adj_lr_backbones

                optimizer.step()
                optimizer.zero_grad()

            batch_loss = loss.item()
            if train_config.gradient_accumulation_steps > 1:
                batch_loss = batch_loss * train_config.gradient_accumulation_steps

            total_train_loss += batch_loss

            num_tokens = torch.sum(attention_mask).item()
            total_tokens_processed += num_tokens
            post_process_time = time.time() - post_process_start

            images_per_sample = [len(image_pack) for image_pack in images]

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            tokens_per_second = (
                get_world_size() * num_tokens / batch_duration
            )  # Multiply by world size to get global tokens/s

            # Accumulate training stats
            accumulated_stats["tokens_per_second"].append(tokens_per_second)
            accumulated_stats["data_load_time"].append(data_load_time)
            accumulated_stats["fw_bw_time"].append(fw_bw_time)
            accumulated_stats["post_process_time"].append(post_process_time)
            accumulated_stats["images_per_sample"].extend(images_per_sample)

            if (
                train_config.eval_in_epochs
                and global_step % train_config.eval_interval == 0
                and is_update_step
                and global_step > 0
            ):
                model.eval()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                with torch.no_grad():
                    total_val_loss = 0
                    for batch in val_loader:
                        images = batch["images"]
                        input_ids = batch["input_ids"].to(device)
                        labels = batch["labels"].to(device)
                        attention_mask = batch["attention_mask"].to(device)

                        with autocast_context:
                            _, loss = model(input_ids, images, attention_mask=attention_mask, targets=labels)

                        total_val_loss += loss.item()
                    avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
                    avg_val_loss = mean(dist_gather(avg_val_loss)) if is_dist() else avg_val_loss
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        if is_master():
                            save_model = (
                                model.module if is_dist() else model
                            )  # unwrap the model for saving if DDP
                            save_model.save_pretrained(
                                save_directory=os.path.join(model_config.vlm_checkpoint_path, run_name)
                            )

                    lmms_results = {}
                    if train_config.use_lmms_eval:
                        from evaluation import cli_evaluate

                        eval_args = argparse.Namespace(
                            model=model.module if is_dist() else model,
                            tasks=train_config.lmms_eval_tasks,
                            limit=train_config.lmms_eval_limit,
                            batch_size=train_config.lmms_eval_batch_size,
                            process_with_media=True,
                            device=device,
                        )
                        # Evaluate using the CLI wrapper
                        eval_results = cli_evaluate(eval_args)

                        if is_master() and eval_results and "results" in eval_results[0]:
                            for task_name, task_results in eval_results[0]["results"].items():
                                for metric_name, metric_value in task_results.items():
                                    if isinstance(metric_value, (int, float)):
                                        lmms_results[f"{task_name}_{metric_name.split(',')[0]}"] = (
                                            metric_value
                                        )

                    if is_master():
                        print(
                            f"Step: {global_step}, Val Loss: {avg_val_loss:.4f}, Tokens/s: {tokens_per_second:.2f}"
                        )
                        if train_config.log_wandb:
                            run.log(
                                {
                                    "val_loss": avg_val_loss,
                                    **{f"lmms_eval/{key}": value for key, value in lmms_results.items()},
                                },
                                step=global_step,
                            )

                model.train()

            # Log training stats every N steps (ALL RANKS must participate in collective ops)
            if (
                global_step % train_config.stats_log_interval == 0
                and len(accumulated_stats["tokens_per_second"]) > 0
                and is_update_step
            ):
                # ALL RANKS: Perform collective operations for training stats
                stats = {}
                for key in [
                    "tokens_per_second",
                    "data_load_time",
                    "fw_bw_time",
                    "post_process_time",
                    "images_per_sample",
                ]:
                    if is_dist():
                        all_values = dist_gather(accumulated_stats[key])
                        all_values_flat = [
                            item for sublist in all_values for item in sublist
                        ]  # Flatten list of lists
                        stats[f"avg_{key}"] = mean(all_values_flat)
                    else:
                        stats[f"avg_{key}"] = mean(accumulated_stats[key])

                for key in ["data_load_time", "fw_bw_time", "post_process_time", "images_per_sample"]:
                    if is_dist():
                        all_values = dist_gather(accumulated_stats[key])
                        all_values_flat = [item for sublist in all_values for item in sublist]
                        stats[f"max_{key}"] = max(all_values_flat)
                    else:
                        stats[f"max_{key}"] = max(accumulated_stats[key])

                if is_dist():
                    all_images_values = dist_gather(accumulated_stats["images_per_sample"])
                    all_images_flat = [item for sublist in all_images_values for item in sublist]
                    stats["min_images_per_sample"] = min(all_images_flat)
                else:
                    stats["min_images_per_sample"] = min(accumulated_stats["images_per_sample"])

                # MASTER ONLY: Log to wandb
                if train_config.log_wandb and is_master():
                    run.log(
                        {
                            **{f"training_stats/{key}": value for key, value in stats.items()},
                        },
                        step=global_step,
                    )

                # ALL RANKS: Reset accumulators
                for key in accumulated_stats:
                    accumulated_stats[key] = []

            # Log batch loss
            if is_update_step:
                # ALL RANKS: gather loss from all ranks if DDP
                if is_dist():
                    batch_loss_gathered = mean(dist_gather(batch_loss))
                else:
                    batch_loss_gathered = batch_loss

                # MASTER ONLY: Log to wandb
                if train_config.log_wandb and is_master():
                    run.log(
                        {
                            "batch_loss": batch_loss_gathered,
                            **({"grad_norm": grad_norm} if train_config.max_grad_norm is not None else {}),
                        },
                        step=global_step,
                    )

            if is_update_step:
                global_step += 1
                if global_step >= train_config.max_training_steps:
                    break
            data_load_start = time.time()

        avg_train_loss = total_train_loss / len(train_loader)
        # gather average batch loss from all ranks if DDP
        avg_train_loss = mean(dist_gather(avg_train_loss)) if is_dist() else avg_train_loss

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        # gather and sum total_tokens_processed across all ranks if DDP
        total_tokens_processed = (
            sum(dist_gather(total_tokens_processed)) if is_dist() else total_tokens_processed
        )
        epoch_tokens_per_second = total_tokens_processed / epoch_duration

        if is_master():
            if train_config.log_wandb:
                run.log(
                    {
                        "epoch_loss": avg_train_loss,
                        "epoch_duration": epoch_duration,
                        "epoch_tokens_per_second": epoch_tokens_per_second,
                    }
                )

            print(
                f"Epoch: {epoch}, Step: {global_step}/{train_config.max_training_steps}, Train Loss: {avg_train_loss:.4f} | Time: {epoch_duration:.2f}s | T/s: {epoch_tokens_per_second:.2f}"
            )

    # Summary Statistics
    if is_master():
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        total_training_time = sum(epoch_times)
        batch_size = int(
            train_config.batch_size * get_world_size() * train_config.gradient_accumulation_steps
        )
        total_samples_processed = batch_size * global_step
        avg_time_per_sample = total_training_time / total_samples_processed
        print(f"Average time per epoch: {avg_epoch_time:.2f}s")
        print(f"Average time per sample: {avg_time_per_sample:.4f}s")

        if train_config.log_wandb:
            run.summary["avg_epoch_time"] = avg_epoch_time
            run.summary["avg_time_per_sample"] = avg_time_per_sample
            run.finish()


def get_args():
    parser = argparse.ArgumentParser(description="Train a Vision-Language Model")
    parser.add_argument("--lr_mp", type=float, help="Learning rate for the mapping network")
    parser.add_argument("--lr_backbones", type=float, help="Learning rate for the backbones")
    parser.add_argument(
        "--vlm_checkpoint_path", type=str, help="Path to the VLM checkpoint for loading or saving"
    )
    parser.add_argument("--compile", type=bool, help="Use torch.compile to optimize the model")
    parser.add_argument("--log_wandb", type=bool, help="Log to wandb")
    parser.add_argument(
        "--resume_from_vlm_checkpoint",
        type=bool,
        default=False,
        help="Resume training from VLM checkpoint specified by vlm_checkpoint_path (or default if not provided)",
    )
    parser.add_argument("--no_log_wandb", action="store_true", help="Do not log to wandb")
    parser.add_argument(
        "--dist_cpu",
        action="store_true",
        help="Force distributed training on CPU using Gloo (launch with torchrun)",
    )

    return parser.parse_args()


def merge_from_args(train_cfg, vlm_cfg, args):
    if args.lr_mp is not None:
        train_cfg.lr_mp = args.lr_mp
    if args.lr_backbones is not None:
        train_cfg.lr_backbones = args.lr_backbones
    if args.vlm_checkpoint_path is not None:
        vlm_cfg.vlm_checkpoint_path = args.vlm_checkpoint_path
    if args.compile is not None:
        train_cfg.compile = args.compile
    if args.no_log_wandb is True:
        train_cfg.log_wandb = False

    if args.resume_from_vlm_checkpoint and args.vlm_checkpoint_path is not None:
        train_cfg.resume_from_vlm_checkpoint = True
        # When resuming a full VLM, we don't need to load individual backbone weights from original sources
        vlm_cfg.vlm_load_backbone_weights = False

    return train_cfg, vlm_cfg


def main():
    args = get_args()
    model_config = ModelConfig()
    train_config = TrainConfig()
    train_config, model_config = merge_from_args(train_config, model_config, args)

    if train_config.log_wandb:
        # Access values
        api_key = os.getenv("WANDB_API_KEY")
        wandb.login(key=api_key)

    # Initialize distributed if launched with torchrun (envs provided by torchrun)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        init_dist()

    if is_master():
        print(os.environ.get("RANK", "0"))
        print(os.environ.get("WORLD_SIZE", "1"))
        print("--- VLM Config ---")
        print(model_config)
        print("--- Train Config ---")
        print(train_config)

    train(train_config, model_config)

    if is_dist():
        destroy_dist()


if __name__ == "__main__":
    main()
