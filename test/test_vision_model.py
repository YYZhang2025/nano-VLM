import torch

from vlm.config import ModelConfig
from vlm.model.vision_model import ViT


def main():
    config = ModelConfig()
    config.vit_cls_flag = False

    model = ViT.from_pretrained(config=config)

    B = 8
    imgs = torch.randn(B, 3, config.vit_img_size, config.vit_img_size)
    out = model(imgs)
    assert out.shape == (B, (config.vit_img_size // config.vit_patch_size) ** 2, config.vit_hidden_dim), (
        f"Expected output shape {(B, config.vit_hidden_dim)}, got {out.shape}"
    )

    print("✅✅ Test passed!")


if __name__ == "__main__":
    main()
