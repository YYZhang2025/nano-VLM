import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from vlm.config import ModelConfig


def merge_config(hf_config, config: ModelConfig):
    config.vit_dropout = hf_config.attention_dropout
    config.vit_hidden_dim = hf_config.hidden_size
    config.vit_img_size = hf_config.image_size
    config.vit_inter_dim = hf_config.intermediate_size
    config.vit_ln_eps = hf_config.layer_norm_eps
    config.vit_n_heads = hf_config.num_attention_heads
    config.vit_n_blocks = hf_config.num_hidden_layers
    config.vit_patch_size = hf_config.patch_size

    return config


class ViTPatchEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.img_size = config.vit_img_size
        self.patch_size = config.vit_patch_size

        assert self.img_size % self.patch_size == 0, "Image size must be divisible by patch size"
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.cls_flag = config.vit_cls_flag
        self.embd_dim = config.vit_hidden_dim

        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=self.embd_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        if self.cls_flag:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embd_dim))
            self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embd_dim))
        else:
            self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches, self.embd_dim))

    def forward(self, imgs):
        # (B, 3, H, W) -> (B, num_patches, embd_dim)
        x = self.conv(imgs).flatten(2).transpose(1, 2)

        if self.cls_flag:
            x = torch.cat((self.cls_token.expand(x.size(0), -1, -1), x), dim=1)

        x += self.position_embedding

        return x


class ViTMultiHeadAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.n_heads = config.vit_n_heads
        self.embd_dim = config.vit_hidden_dim
        assert self.embd_dim % self.n_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = self.embd_dim // self.n_heads
        self.dropout = config.vit_dropout

        self.qkv_proj = nn.Linear(self.embd_dim, self.embd_dim * 3)
        self.attn_drop = nn.Dropout(self.dropout)

        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim)
        self.proj_drop = nn.Dropout(self.dropout)

        self.sdap = hasattr(F, "scaled_dot_product_attention")
        if not self.sdap:
            print("Warning: scaled dot product attention not available. Using standard attention in ViT.")

    def forward(self, hidden_states):
        B, T, C = hidden_states.size()

        q, k, v = map(
            lambda x: x.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2),
            self.qkv_proj(hidden_states).chunk(3, dim=-1),
        )

        if self.sdap:
            attn_output = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            attn_output = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
            attn_output = F.softmax(attn_output, dim=-1)
            attn_output = self.attn_drop(attn_output)
            attn_output = attn_output @ v

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.proj_drop(self.out_proj(attn_output))
        return attn_output


class ViTMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.vit_hidden_dim, config.vit_hidden_dim * 4)
        self.fc2 = nn.Linear(config.vit_hidden_dim * 4, config.vit_hidden_dim)
        self.drop = nn.Dropout(config.vit_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VitBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.vit_hidden_dim)
        self.attn = ViTMultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.vit_hidden_dim)
        self.mlp = ViTMLP(config)

    def forward(self, hidden_states):
        x = hidden_states + self.attn(self.ln1(hidden_states))
        x = x + self.mlp(self.ln2(x))
        return x


class ViT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.patch_embedding = ViTPatchEmbedding(config)
        self.cls_flag = config.vit_cls_flag
        self.dropout = nn.Dropout(config.vit_dropout)
        self.blocks = nn.ModuleList([VitBlock(config) for _ in range(config.vit_n_blocks)])

        self.layer_norm = nn.LayerNorm(config.vit_hidden_dim)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)
        if self.cls_flag:
            x = x[:, 0]

        x = self.layer_norm(x)
        return x

    @classmethod
    def from_pretrained(cls, config: ModelConfig):
        import safetensors
        from huggingface_hub import hf_hub_download
        from transformers import SiglipVisionConfig

        hf_config = SiglipVisionConfig.from_pretrained(config.vit_model_type)
        config.merge_from_hf_vision(hf_config)

        model = cls(config)
        safetensors_file = hf_hub_download(repo_id=config.vit_model_type, filename="model.safetensors")

        sd = model.state_dict()

        mapping = {
            "vision_model.embeddings.patch_embedding.weight": "patch_embedding.conv.weight",
            "vision_model.embeddings.patch_embedding.bias": "patch_embedding.conv.bias",
            "vision_model.embeddings.position_embedding.weight": "patch_embedding.position_embedding",
            "vision_model.post_layernorm.weight": "layer_norm.weight",
            "vision_model.post_layernorm.bias": "layer_norm.bias",
        }

        for i in range(config.vit_n_blocks):
            # Layer norms
            mapping[f"vision_model.encoder.layers.{i}.layer_norm1.weight"] = f"blocks.{i}.ln1.weight"
            mapping[f"vision_model.encoder.layers.{i}.layer_norm1.bias"] = f"blocks.{i}.ln1.bias"
            mapping[f"vision_model.encoder.layers.{i}.layer_norm2.weight"] = f"blocks.{i}.ln2.weight"
            mapping[f"vision_model.encoder.layers.{i}.layer_norm2.bias"] = f"blocks.{i}.ln2.bias"

            # MLP
            mapping[f"vision_model.encoder.layers.{i}.mlp.fc1.weight"] = f"blocks.{i}.mlp.fc1.weight"
            mapping[f"vision_model.encoder.layers.{i}.mlp.fc1.bias"] = f"blocks.{i}.mlp.fc1.bias"
            mapping[f"vision_model.encoder.layers.{i}.mlp.fc2.weight"] = f"blocks.{i}.mlp.fc2.weight"
            mapping[f"vision_model.encoder.layers.{i}.mlp.fc2.bias"] = f"blocks.{i}.mlp.fc2.bias"

            # Output projection
            mapping[f"vision_model.encoder.layers.{i}.self_attn.out_proj.weight"] = (
                f"blocks.{i}.attn.out_proj.weight"
            )
            mapping[f"vision_model.encoder.layers.{i}.self_attn.out_proj.bias"] = (
                f"blocks.{i}.attn.out_proj.bias"
            )

        with safetensors.safe_open(filename=safetensors_file, framework="pt", device="cpu") as f:
            for hf_key, our_key in mapping.items():
                if hf_key in f.keys() and our_key in sd:
                    tensor = f.get_tensor(hf_key)
                    if tensor.shape == sd[our_key].shape:
                        sd[our_key].copy_(tensor)
                    else:
                        if "position_embedding" in hf_key:
                            sd[our_key].copy_(tensor.unsqueeze(0))
                        else:
                            print(
                                f"Shape mismatch for {hf_key} -> {our_key}: {tensor.shape} vs {sd[our_key].shape}"
                            )
                else:
                    if hf_key not in f.keys():
                        print(f"Warning: Key {hf_key} not found in safetensors file")
                    if our_key not in sd:
                        print(f"Warning: Key {our_key} not found in model state dict")

            # Manually handle QKV concatenation since our implementation combines Q, K, V into one
            for i in range(model.config.vit_n_blocks):
                q_weight = f.get_tensor(f"vision_model.encoder.layers.{i}.self_attn.q_proj.weight")
                k_weight = f.get_tensor(f"vision_model.encoder.layers.{i}.self_attn.k_proj.weight")
                v_weight = f.get_tensor(f"vision_model.encoder.layers.{i}.self_attn.v_proj.weight")

                qkv_weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
                sd[f"blocks.{i}.attn.qkv_proj.weight"].copy_(qkv_weight)

                q_bias = f.get_tensor(f"vision_model.encoder.layers.{i}.self_attn.q_proj.bias")
                k_bias = f.get_tensor(f"vision_model.encoder.layers.{i}.self_attn.k_proj.bias")
                v_bias = f.get_tensor(f"vision_model.encoder.layers.{i}.self_attn.v_proj.bias")

                qkv_bias = torch.cat((q_bias, k_bias, v_bias), dim=0)
                sd[f"blocks.{i}.attn.qkv_proj.bias"].copy_(qkv_bias)

        model.load_state_dict(sd)
        print(
            f"Successfully loaded {config.vit_model_type} weights from safetensors. Model has {sum(p.numel() for p in model.parameters()):,} parameters."
        )
        return model
