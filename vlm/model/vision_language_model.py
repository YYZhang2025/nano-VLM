import json
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model

from vlm.config import ModelConfig
from vlm.data.processors import get_tokenizer
from vlm.model.language_model import LanguageModel
from vlm.model.modality_projector import ModalityProjector
from vlm.model.vision_model import ViT
from vlm.utils.sampling import top_k_top_p_filtering
from vlm.utils.utils import save_config


class VisionLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig, load_backbone=True):
        super().__init__()

        self.config = config
        if load_backbone:
            self.vision_encoder = ViT.from_pretrained(config)
            self.decoder = LanguageModel.from_pretrained(config)
        else:
            self.vision_encoder = ViT(config)
            self.decoder = LanguageModel(config)

        self.mp = ModalityProjector(config)

        self.load_backbone = load_backbone
        self.tokenizer = get_tokenizer(config.lm_tokenizer, config.vlm_extra_tokens, config.lm_chat_template)

    def _replace_img_tokens_with_embd(self, input_ids, token_embd, image_embd):
        updated_token_embd = token_embd.clone()

        mask = input_ids == self.tokenizer.image_token_id
        print(self.tokenizer.image_token_id)
        updated_token_embd[mask] = image_embd.view(-1, image_embd.size(-1)).to(updated_token_embd.dtype)

        return updated_token_embd

    def forward(self, input_ids, images, attention_mask=None, targets=None):
        device = input_ids.device
        if isinstance(images, list):
            if not images:
                images = torch.empty(
                    0, self.config.vit_channels, self.config.vit_img_size, self.config.vit_img_size
                ).to(device)
            else:
                if isinstance(images[0], list):
                    images = [img for sublist in images for img in sublist]
                images = torch.cat(images, dim=0).to(input_ids.device)

        # Encode Image
        image_embd = self.vision_encoder(images)
        image_embd = self.mp(image_embd)

        token_embd = self.decoder.token_embedding(input_ids)

        updated_token_embd = self._replace_img_tokens_with_embd(input_ids, token_embd, image_embd)

        # Decoder
        logits, _ = self.decoder(
            updated_token_embd,
            attention_mask=attention_mask,
        )

        loss = None
        if targets is not None:
            logits = self.decoder.head(logits)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.config.lm_ignore_index,
            )

        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        images,
        attention_mask=None,
        max_new_tokens=5,
        top_k=50,
        top_p=0.9,
        temperature=0.5,
        greedy=False,
    ):
        if isinstance(images, list):
            if not images:  # Handle cases with no images
                images = torch.empty(
                    0,
                    self.config.vit_channels,
                    self.config.vit_img_size,
                    self.config.vit_img_size,
                    device=input_ids.device,
                )
            else:
                if isinstance(images[0], list):
                    images = [img for sublist in images for img in sublist]
                images = torch.cat(images, dim=0).to(input_ids.device)

        image_embd = self.vision_encoder(images)  # [B, T_img_feat, D_model]
        image_embd = self.mp(image_embd)  # [B, mp_image_token_length, D_lm]

        # 2. Embed initial text prompt tokens
        prompt_token_embeds = self.decoder.token_embedding(input_ids)  # [B, T_prompt_text, D_lm]

        # 3. Combine image and text embeddings
        initial_combined_embeds = self._replace_img_tokens_with_embd(
            input_ids, prompt_token_embeds, image_embd
        )

        current_total_seq_len = initial_combined_embeds.size(1)
        batch_size = input_ids.size(0)

        prefill_output, kv_cache_list = self.decoder(
            initial_combined_embeds,
            attention_mask=attention_mask,  # Use the provided attention mask
            kv_cache=None,
            start_pos=0,
        )
        last_token_output_from_prefill = prefill_output[:, -1, :]

        if not self.decoder.lm_use_tokens:
            current_logits = self.decoder.head(last_token_output_from_prefill)
        else:
            current_logits = last_token_output_from_prefill

        newly_generated_ids_list = []
        for _ in range(max_new_tokens):
            if greedy:
                next_token_id = torch.argmax(current_logits, dim=-1, keepdim=True)
            else:
                filtered_logits = top_k_top_p_filtering(current_logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

            newly_generated_ids_list.append(next_token_id)

            # Embed the newly generated token
            next_token_embed = self.decoder.token_embedding(next_token_id)  # [B, 1, D_lm]

            # The start_pos for the new token is the current total sequence length *before* adding this new token
            current_token_start_pos = current_total_seq_len
            current_total_seq_len += 1

            # update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype),
                    ),
                    dim=1,
                )

            # With KV cache: only process the new token
            decode_step_output, kv_cache_list = self.decoder(
                next_token_embed,
                attention_mask=attention_mask,
                kv_cache=kv_cache_list,
                start_pos=current_token_start_pos,
            )

            last_token_output = decode_step_output[:, -1, :]

            # Apply head to get logits (if model is in embedding mode)
            if not self.decoder.lm_use_tokens:
                current_logits = self.decoder.head(last_token_output)
            else:
                current_logits = last_token_output

        if not newly_generated_ids_list:  # Handle case where max_new_tokens might be 0
            return torch.empty((batch_size, 0), dtype=torch.long, device=input_ids.device)

        generated_ids = torch.cat(newly_generated_ids_list, dim=1)

        # Post-process to handle EOS token.
        if (
            self.tokenizer.eos_token_id is not None and generated_ids.numel() > 0
        ):  # Ensure generated_ids is not empty
            seq_len = generated_ids.size(1)
            device = generated_ids.device

            eos_mask = generated_ids == self.tokenizer.eos_token_id  # Create a boolean mask for EOS tokens

            col_indices_for_min = torch.arange(
                seq_len, device=device
            )  # Create column indices [0, 1, ..., seq_len-1]

            # In eos_mask, mark positions with actual col_idx, others with a large number
            masked_col_indices = torch.where(
                eos_mask, col_indices_for_min.unsqueeze(0).expand_as(generated_ids), seq_len + 1
            )

            first_eos_indices_values = torch.min(masked_col_indices, dim=1).values

            # Clamp values to seq_len (if no EOS found, min will be seq_len + 1, clamp brings it to seq_len0. This means if no EOS, or EOS is the last token, no replacement will happen for that sample.
            actual_first_eos_indices = torch.clamp(first_eos_indices_values, max=seq_len)

            # Create column indices for comparison, shape [batch_size, seq_len]
            col_indices_for_comparison = (
                torch.arange(seq_len, device=device).unsqueeze(0).expand_as(generated_ids)
            )

            # Tokens are replaced if their column index is greater than the index of the first EOS token
            replace_mask = col_indices_for_comparison > actual_first_eos_indices.unsqueeze(1)

            generated_ids[replace_mask] = self.tokenizer.eos_token_id

        return generated_ids

    @classmethod
    def from_pretrained(
        cls, repo_id_or_path: str, *, revision: Optional[str] = None
    ) -> "VisionLanguageModel":
        """
        Load a VisionLanguageModel from a local directory or a repo on the Hugging Face Hub.

        Args:
            repo_id_or_path (str): The path to the local directory or the Hugging Face Hub repo ID.

        Returns:
            VisionLanguageModel: The loaded model.
        """
        # If local folder exists => load from there
        if os.path.exists(repo_id_or_path):
            config_path = os.path.join(repo_id_or_path, "config.json")
            weights_path = os.path.join(repo_id_or_path, "model.safetensors")

            if not os.path.exists(config_path):
                raise ValueError(f"Config file not found at {config_path}. Please provide a valid path.")
            if not os.path.exists(weights_path):
                raise ValueError(f"Weights file not found at {weights_path}. Please provide a valid path.")
        # Otherwise, assume it's a Hugging Face Hub repo
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(repo_id=repo_id_or_path, filename="config.json", revision=revision)
            weights_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="model.safetensors", revision=revision
            )

        # Load config
        with open(config_path, "r") as f:
            config = ModelConfig(**json.load(f))

        # Initialize model without loading the backbone
        model = cls(config, load_backbone=False)
        load_model(model, weights_path)

        return model

    def save_pretrained(self, save_directory: str) -> None:
        # Create directory if it doesn't exist
        os.makedirs(save_directory, exist_ok=True)

        # Save config
        save_config(self.config, os.path.join(save_directory, "config.json"))
        save_model(self, os.path.join(save_directory, "model.safetensors"))
