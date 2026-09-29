import pytest
import torch

from mini_gqa import ModelConfig, TinyModel, KVCachePool


@pytest.fixture
def tiny_configs():
    """MHA, MQA and a general GQA (2 KV heads, 4 Q heads)."""
    return {
        "mha": ModelConfig(vocab_size=64, d_model=32, n_layers=2,
                           n_q_heads=4, n_kv_heads=4),
        "gqa": ModelConfig(vocab_size=64, d_model=32, n_layers=2,
                           n_q_heads=4, n_kv_heads=2),
        "mqa": ModelConfig(vocab_size=64, d_model=32, n_layers=2,
                           n_q_heads=4, n_kv_heads=1),
    }


def make_model(cfg, seed=1234):
    return TinyModel(cfg, seed=seed).eval()


@pytest.fixture
def make_pool():
    def _make(cfg, capacity=512, block_size=4, seed=1234, dtype=torch.float32):
        model = make_model(cfg, seed=seed)
        return model, KVCachePool(model, capacity, block_size, dtype=dtype)
    return _make


@pytest.fixture
def deterministic_tokens():
    g = torch.Generator().manual_seed(2024)
    def _gen(shape, vocab=64):
        return torch.randint(0, vocab, shape, generator=g)
    return _gen
