import torch
from transformers import AutoTokenizer

from vlm.config import ModelConfig
from vlm.model.language_model import LanguageModel


def test_language_model_from_pretrained():
    config = ModelConfig()
    config.lm_use_tokens = False
    model = LanguageModel.from_pretrained(config)
    model.eval()

    # Test output size
    input_tensor = torch.randn(8, 10, config.lm_hidden_dim)  # (batch_size, seq_len, hidden_dim)

    output, _ = model(input_tensor)
    assert output.size() == (8, 10, config.lm_hidden_dim), "Output size mismatch"

    print("✅✅ Test passed!")


def main():
    config = ModelConfig(
        lm_hidden_dim=64,
        lm_inter_dim=128,
        lm_rms_eps=1e-5,
        lm_re_base=10000.0,
        lm_max_position_embeddings=1024,
        lm_attn_scaling=1.0,
        lm_vocab_size=100,  # Small vocab for testing
        lm_n_heads=4,
        lm_n_kv_heads=2,
        lm_dropout=0.0,
        lm_n_blocks=2,
        lm_use_tokens=False,
        lm_tie_weights=True,
    )
    model = LanguageModel(config)
    model.eval()

    # Test output size
    input_tensor = torch.randn(8, 10, config.lm_hidden_dim)  # (batch_size, seq_len, hidden_dim)

    output, _ = model(input_tensor)
    assert output.size() == (8, 10, config.lm_hidden_dim), "Output size mismatch"

    print("✅✅ Test passed!")


if __name__ == "__main__":
    # main()
    test_language_model_from_pretrained()
