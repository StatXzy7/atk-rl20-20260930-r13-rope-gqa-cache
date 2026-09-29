import torch
import pytest

from mini_decoder import CacheSpec, Engine, KVCacheManager, MiniDecoder, ModelConfig

ATOL = 1e-9  # float64 tolerance for cached-vs-reference logits


def make_model(n_heads=4, n_kv_heads=2, n_layers=2, head_dim=8, vocab=61, seed=0):
    cfg = ModelConfig(
        vocab_size=vocab,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
    )
    # init_std=0.3 keeps logit magnitudes O(1) so bug-induced drift is visible.
    return MiniDecoder(cfg, seed=seed, init_std=0.3).to(torch.float64)


def make_engine(model, num_blocks=64, block_size=4):
    cfg = model.cfg
    spec = CacheSpec(
        n_layers=cfg.n_layers,
        n_kv_heads=cfg.n_kv_heads,
        head_dim=cfg.head_dim,
        block_size=block_size,
        dtype=torch.float64,
    )
    mgr = KVCacheManager(spec, num_blocks=num_blocks, model_version=model.fingerprint())
    return Engine(model, mgr), mgr


@pytest.fixture(params=[(4, 4), (4, 2), (4, 1), (6, 2)], ids=["mha", "gqa-g2", "mqa", "gqa-g3"])
def model(request):
    return make_model(*request.param)


@pytest.fixture
def engine(model):
    eng, _ = make_engine(model)
    return eng
