import torch
import torch.nn as nn

from vlm.config import ModelConfig


class ModalityProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.input_dim = config.vit_hidden_dim * (config.mp_pixel_shuffle_factor**2)
        self.output_dim = config.lm_hidden_dim

        self.proj = nn.Linear(self.input_dim, self.output_dim)

        self._init_weights()

        self.scale_factor = config.mp_pixel_shuffle_factor

    def _init_weights(self):
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def pixel_shuffle(self, x):
        B, S, D = x.size()
        seq_root = int(S**0.5)
        assert seq_root**2 == S  # Sequence length must be a perfect square for pixel shuffle
        assert seq_root % self.scale_factor == 0  # Sequence root must be divisible by scale factor

        height = width = seq_root
        x = x.view(B, height, width, D)
        h_out = height // self.scale_factor
        w_out = width // self.scale_factor

        x = x.reshape(B, h_out, self.scale_factor, w_out, self.scale_factor, D)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(B, h_out * w_out, D * self.scale_factor**2)

        return x

    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.proj(x)

        return x
