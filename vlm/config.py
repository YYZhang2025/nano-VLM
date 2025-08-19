from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    # Vision Model Configuration
    vit_channels: int = 768
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
    lm_base_vocab_size: int = 49152
    lm_extra_token_amount: int = 17
    lm_vocab_size: int = lm_base_vocab_size + lm_extra_token_amount

    lm_hidden_dim: int = 768
    lm_inter_dim: int = 3072
    lm_rms_eps: float = 1e-6
    lm_n_heads: int = 12
    lm_n_kv_heads: int = 2

    lm_re_base: float = 10000.0
    lm_max_position_embeddings: int = 1024
    lm_attn_scaling: float = 1.0
    lm_dropout: float = 0.0
    lm_max_length: int = 1024
    lm_n_blocks: int = 2
    lm_tie_weights: bool = True
    lm_use_tokens: bool = False

    lm_ignore_index: int = -100

    lm_model_type: str = "HuggingFaceTB/SmolLM2-360M-Instruct"
    lm_tokenizer: str = "HuggingFaceTB/SmolLM2-360M-Instruct"

    lm_chat_template: str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    # Multi-Modality Model Configuration
    mp_pixel_shuffle_factor: int = 4
    mp_image_token_length: int = 64

    max_img_size: int = 1024

    vlm_extra_tokens: dict[str, str] = field(
        default_factory=lambda: {
            "image_token": "<|image|>",
            "r1c1": "<row_1_col_1>",
            "r1c2": "<row_1_col_2>",
            "r1c3": "<row_1_col_3>",
            "r1c4": "<row_1_col_4>",
            "r2c1": "<row_2_col_1>",
            "r2c2": "<row_2_col_2>",
            "r2c3": "<row_2_col_3>",
            "r2c4": "<row_2_col_4>",
            "r3c1": "<row_3_col_1>",
            "r3c2": "<row_3_col_2>",
            "r3c3": "<row_3_col_3>",
            "r3c4": "<row_3_col_4>",
            "r4c1": "<row_4_col_1>",
            "r4c2": "<row_4_col_2>",
            "r4c3": "<row_4_col_3>",
            "r4c4": "<row_4_col_4>",
        }
    )
    vlm_load_backbone_weights: bool = True
    vlm_checkpoint_path: str = "checkpoints"

    def __post_init__(self):
        # Perform any necessary post-initialization steps here
        assert self.lm_hidden_dim % self.lm_n_heads == 0, (
            "Language model hidden dimension must be divisible by number of heads."
        )
        assert self.lm_n_heads % self.lm_n_kv_heads == 0, (
            "Language model number of heads must be divisible by number of KV heads."
        )

        self.lm_extra_token_amount = len(self.vlm_extra_tokens) if self.vlm_extra_tokens is not None else 0

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
        self.lm_base_vocab_size = getattr(hf_config, "vocab_size", self.lm_base_vocab_size)
        self.lm_dropout = getattr(hf_config, "attention_dropout", self.lm_dropout)
        self.lm_n_blocks = getattr(hf_config, "num_hidden_layers", self.lm_n_blocks)

        self.lm_vocab_size = self.lm_base_vocab_size + self.lm_extra_token_amount
        return self


@dataclass
class TrainConfig:
    lr_mp: float = 0.00512
    lr_backbones: float = 5e-5
    data_cutoff_idx: int = None
    val_ratio: float = 0.025
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    eval_in_epochs: bool = True
    eval_interval: int = gradient_accumulation_steps * 100
    stats_log_interval: int = gradient_accumulation_steps * 25
    max_training_steps: int = 12
    max_images_per_example: int = 4
    max_images_per_knapsack: int = 18
    max_sample_length: int = 1024
    compile: bool = False
    resume_from_vlm_checkpoint: bool = False  # Indicate if the training should be resumed from a checkpoint of the whole VLM or you want to start from scratch

    # Dataset config
    train_dataset_path: str = "HuggingFaceM4/the_cauldron"
    train_dataset_name: tuple[str, ...] = ("intergps",)

    # Wandb Config
    log_wandb: bool = True

    # Evaluation Config
    use_lmms_eval: bool = True  # Use lmms-eval for evaluation
    lmms_eval_tasks: str = "mmstar,mmmu,ocrbench,textvqa"
    lmms_eval_limit: int = 2000
    lmms_eval_batch_size: int = 128
