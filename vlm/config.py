from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Vision Model Configuration
    vit_model_type: str = "google/siglip2-base-patch16-512"  # Model type
    vit_img_size: int = 512
    vit_patch_size: int = 16
    vit_cls_flag: bool = False
    vit_hidden_dim: int = 768
    vit_inter_dim: int = 3072
    vit_n_heads: int = 12
    vit_dropout: float = 0.0
    vit_n_blocks: int = 12
    vit_ln_eps: float = 1e-6
    # Language Model Configuration

    lm_hidden_dim: int = 768

    # Multi-Modality Model Configuration
    mp_pixel_shuffle_factor: int = 2

    def merge_from_hf_vision(self, hf_config):
        """
        Merge fields from a Hugging Face vision config (e.g., SiglipVisionConfig)
        into this ModelConfig. Uses getattr with fallbacks so it is resilient to
        missing attributes across different HF configs.
        """
        self.vit_dropout = getattr(hf_config, "attention_dropout", self.vit_dropout)
        self.vit_hidden_dim = getattr(hf_config, "hidden_size", self.vit_hidden_dim)
        self.vit_img_size = getattr(hf_config, "image_size", self.vit_img_size)
        self.vit_inter_dim = getattr(hf_config, "intermediate_size", self.vit_inter_dim)
        self.vit_ln_eps = getattr(hf_config, "layer_norm_eps", self.vit_ln_eps)
        self.vit_n_heads = getattr(hf_config, "num_attention_heads", self.vit_n_heads)
        self.vit_n_blocks = getattr(hf_config, "num_hidden_layers", self.vit_n_blocks)
        self.vit_patch_size = getattr(hf_config, "patch_size", self.vit_patch_size)
        return self
