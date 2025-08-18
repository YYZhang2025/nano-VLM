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
    lm_vocab_size: int = 100

    lm_hidden_dim: int = 768
    lm_inter_dim: int = 3072
    lm_rms_eps: float = 1e-6
    lm_n_heads: int = 12
    lm_n_kv_heads: int = 2

    lm_re_base: float = 10000.0
    lm_max_position_embeddings: int = 1024
    lm_attn_scaling: float = 1.0
    lm_dropout: float = 0.0

    lm_n_blocks: int = 2
    lm_tie_weights: bool = True
    lm_use_tokens: bool = True

    lm_model_type: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    lm_tokenizer: str = "HuggingFaceTB/SmolLM2-360M-Instruct"

    # Multi-Modality Model Configuration
    mp_pixel_shuffle_factor: int = 4
    mp_image_token_length: int = 64

    max_img_size: int = 1024

    def __post_init__(self):
        # Perform any necessary post-initialization steps here
        assert self.lm_hidden_dim % self.lm_n_heads == 0, (
            "Language model hidden dimension must be divisible by number of heads."
        )
        assert self.lm_n_heads % self.lm_n_kv_heads == 0, (
            "Language model number of heads must be divisible by number of KV heads."
        )

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

    def merge_from_hf_language(self, hf_config):
        """
        Merge fields from a Hugging Face language config (e.g., GPT2Config)
        into this ModelConfig. Uses getattr with fallbacks so it is resilient to
        missing attributes across different HF configs.
        """
        self.lm_hidden_dim = getattr(hf_config, "hidden_size", self.lm_hidden_dim)
        self.lm_inter_dim = getattr(hf_config, "intermediate_size", self.lm_inter_dim)
        self.lm_rms_eps = getattr(hf_config, "rms_norm_eps", self.lm_rms_eps)
        self.lm_re_base = getattr(hf_config, "rope_theta", self.lm_re_base)
        self.lm_n_heads = getattr(hf_config, "num_attention_heads", self.lm_n_heads)
        self.lm_n_kv_heads = getattr(hf_config, "num_key_value_heads", self.lm_n_kv_heads)
        self.lm_max_position_embeddings = getattr(
            hf_config, "max_position_embeddings", self.lm_max_position_embeddings
        )
        self.lm_vocab_size = getattr(hf_config, "vocab_size", self.lm_vocab_size)
        self.lm_dropout = getattr(hf_config, "attention_dropout", self.lm_dropout)
        self.lm_n_blocks = getattr(hf_config, "num_hidden_layers", self.lm_n_blocks)
        return self
