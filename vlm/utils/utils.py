import json
import random
import time
from dataclasses import asdict

import numpy as np
import torch

from vlm.utils.dist import get_world_size


def seed_everything(seed: int = 42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(rank: int = 0) -> torch.device:
    """
    Get the device for the given rank.
    """
    if torch.cuda.is_available():
        print(f"Using GPU: cuda:{rank}")
        return torch.device(f"cuda:{rank}")
    elif torch.backends.mps.is_available():
        print("Using Apple Silicon: mps")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")


def save_config(config, filepath):
    """
    Save the model configuration to a JSON file.
    """
    with open(filepath, "w") as f:
        f.write(json.dumps(asdict(config), indent=4))


def get_run_name(train_cfg, vlm_cfg):
    dataset_size = "full_ds" if train_cfg.data_cutoff_idx is None else f"{train_cfg.data_cutoff_idx}samples"
    batch_size = f"bs{int(train_cfg.batch_size * get_world_size() * train_cfg.gradient_accumulation_steps)}"
    max_training_steps = f"{train_cfg.max_training_steps}"
    learning_rate = f"lr{train_cfg.lr_backbones}-{train_cfg.lr_mp}"
    num_gpus = f"{get_world_size()}xGPU"
    date = time.strftime("%m%d-%H%M%S")
    vit = f"{vlm_cfg.vit_model_type.split('/')[-1]}" + f"_{vlm_cfg.max_img_size}"
    mp = f"mp{vlm_cfg.mp_pixel_shuffle_factor}"
    llm = f"{vlm_cfg.lm_model_type.split('/')[-1]}"

    return f"nanoVLM_{vit}_{mp}_{llm}_{num_gpus}_{dataset_size}_{batch_size}_{max_training_steps}_{learning_rate}_{date}"
