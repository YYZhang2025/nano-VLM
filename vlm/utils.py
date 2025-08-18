import random

import numpy as np
import torch


def seed_everything(seed: int = 42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
