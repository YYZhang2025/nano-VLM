import torch

from vlm.config import ModelConfig
from vlm.model.modality_projector import ModalityProjector
from vlm.model.vision_model import ViT


def main():
    config = ModelConfig()
    config.vit_cls_flag = False
    num_tokens = (config.vit_img_size // config.vit_patch_size) ** 2
    num_tokens //= config.mp_pixel_shuffle_factor**2
    hidden_dim = config.vit_hidden_dim

    model = ViT.from_pretrained(config=config)

    B = 8
    imgs = torch.randn(B, 3, config.vit_img_size, config.vit_img_size)
    projector = ModalityProjector(config)
    out = model(imgs)
    out = projector(out)

    assert out.shape == (B, num_tokens, hidden_dim), (
        f"Expected output shape {(B, num_tokens, hidden_dim)}, got {out.shape}"
    )

    print("✅✅ Test passed!")


if __name__ == "__main__":
    main()
