"""Protocol errors, dtype byte accounting and release semantics."""

import pytest
import torch

from mini_gqa import (
    KVCachePool,
    KVCacheError,
    SessionReleasedError,
    decode,
    prefill,
)


def test_decode_before_prefill_rejected(make_pool, deterministic_tokens):
    model, pool = make_pool(__import__("mini_gqa").ModelConfig(
        vocab_size=64, d_model=32, n_q_heads=4, n_kv_heads=2))
    s = pool.create_session()
    with pytest.raises(KVCacheError):
        decode(pool, s, deterministic_tokens((1, 1)))


def test_prefill_into_nonempty_rejected(make_pool, deterministic_tokens):
    from mini_gqa import ModelConfig
    model, pool = make_pool(ModelConfig(n_q_heads=4, n_kv_heads=2,
                                        vocab_size=64, d_model=32))
    s = pool.create_session()
    prefill(pool, s, deterministic_tokens((1, 3)))
    with pytest.raises(KVCacheError):
        prefill(pool, s, deterministic_tokens((1, 2)))


def test_released_session_rejected(make_pool, deterministic_tokens):
    from mini_gqa import ModelConfig
    model, pool = make_pool(ModelConfig(n_q_heads=4, n_kv_heads=2,
                                        vocab_size=64, d_model=32))
    s = pool.create_session()
    prefill(pool, s, deterministic_tokens((1, 3)))
    pool.release_session(s)
    with pytest.raises(SessionReleasedError):
        decode(pool, s, deterministic_tokens((1, 1)))
    with pytest.raises(SessionReleasedError):
        prefill(pool, s, deterministic_tokens((1, 1)))
    # idempotent
    pool.release_session(s)


def test_duplicate_session_in_batch_rejected(make_pool, deterministic_tokens):
    from mini_gqa import ModelConfig
    model, pool = make_pool(ModelConfig(n_q_heads=4, n_kv_heads=2,
                                        vocab_size=64, d_model=32))
    s = pool.create_session()
    prefill(pool, s, deterministic_tokens((1, 2)))
    with pytest.raises(ValueError):
        pool.reserve_extend([s, s], [1, 1])


def test_backend_dtype_byte_size(make_pool, tiny_configs):
    cfg = tiny_configs["gqa"]
    model, pool16 = make_pool(cfg, capacity=32, block_size=4, dtype=torch.float16)
    assert pool16.elem_bytes == 2
    expected = 32 * cfg.n_layers * cfg.n_kv_heads * cfg.head_dim * 2 * 2
    assert pool16.capacity_bytes == expected
    assert pool16.k_backend.dtype == torch.float16
    # fp32 doubles the bytes
    model32, pool32 = make_pool(cfg, capacity=32, block_size=4, dtype=torch.float32)
    assert pool32.capacity_bytes == 2 * expected


def test_capacity_zero_or_negative_rejected(make_pool, tiny_configs):
    with pytest.raises(ValueError):
        make_pool(tiny_configs["gqa"], capacity=0)
