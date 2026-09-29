"""Prefix fork isolation, copy-on-write ownership, capacity and bytes."""

import pytest
import torch

from mini_gqa import (
    CacheCapacityError,
    KVCachePool,
    StaleCacheError,
    chunked_decode,
    decode,
    prefill,
)

ATOL = 1e-5


def tail_from(model, ids, base_pos=0):
    return model.forward_full_prefix(torch.tensor([ids]), base_pos=base_pos)[0, -1]


def test_fork_shared_prefix_then_isolated_writes(make_pool, deterministic_tokens):
    model, pool = make_pool(_gqa(), capacity=128, block_size=4)
    root = pool.create_session()
    prefix = deterministic_tokens((1, 6))[0].tolist()
    prefill(pool, root, torch.tensor([prefix]))

    a = pool.fork_session(root)
    b = pool.fork_session(root)
    assert a.length == b.length == root.length == 6
    # full prefix blocks are literally shared
    assert a.blocks == root.blocks
    assert all(pool._refcounts[bid] == 3 for bid in root.blocks)

    ta = deterministic_tokens((1, 3))[0].tolist()
    tb = deterministic_tokens((1, 5))[0].tolist()
    la, _, _ = chunked_decode(pool, a, torch.tensor([ta]), [3])
    lb, _, _ = chunked_decode(pool, b, torch.tensor([tb]), [5])

    # prefix 0..3 is a frozen full block, shared by all three sessions;
    # tokens 4..5 sat in a partial tail shared at fork time, so both branches
    # COW it into distinct tails and root keeps the original tail block.
    assert a.blocks[0] == b.blocks[0] == root.blocks[0]
    assert a.blocks[1] != b.blocks[1]
    assert a.blocks[1] != root.blocks[1]
    assert b.blocks[1] != root.blocks[1]
    # root never mutated by branch writes
    assert root.length == 6
    assert torch.allclose(
        tail_from(model, prefix),
        tail_from(model, root.stored_ids),
        atol=ATOL,
    )
    # branch outputs equal independent recomputations from the shared prefix
    assert torch.allclose(la[0, -1], tail_from(model, prefix + ta), atol=ATOL)
    assert torch.allclose(lb[0, -1], tail_from(model, prefix + tb), atol=ATOL)

    # physical content isolation: writing more on A must not change B's cache
    kb_before, vb_before = pool.gather_history(b, 0)
    ta2 = deterministic_tokens((1, 2))[0].tolist()
    chunked_decode(pool, a, torch.tensor([ta2]), [2])
    kb_after, vb_after = pool.gather_history(b, 0)
    assert torch.equal(kb_before, kb_after)
    assert torch.equal(vb_before, vb_after)


def test_release_one_branch_keeps_other_alive(make_pool, deterministic_tokens):
    model, pool = make_pool(_gqa(), capacity=128, block_size=4)
    root = pool.create_session()
    prefix = deterministic_tokens((1, 5))[0].tolist()
    prefill(pool, root, torch.tensor([prefix]))
    a = pool.fork_session(root)
    b = pool.fork_session(root)
    cont_b = deterministic_tokens((1, 4))[0].tolist()
    chunked_decode(pool, b, torch.tensor([cont_b]), [4])
    expected_b = tail_from(model, prefix + cont_b)
    shared_before = root.blocks[0]

    pool.release_session(a)
    assert not a.alive
    assert b.alive
    # b still computes correctly and its data survived
    kb, _ = pool.gather_history(b, 0)
    assert kb.shape[0] == 9
    assert torch.allclose(expected_b, tail_from(model, b.stored_ids), atol=ATOL)
    # shared full prefix block survived because root and b still own it
    assert pool._refcounts[shared_before] == 2

    pool.release_session(root)
    assert pool._refcounts[shared_before] == 1
    pool.release_session(b)
    assert pool._refcounts[shared_before] == 0
    # releasing twice is a no-op
    pool.release_session(b)
    # everything is reusable again
    assert len(pool._free_blocks) == pool.n_blocks


def test_cow_required_when_tail_is_partial_and_shared(make_pool, deterministic_tokens):
    model, pool = make_pool(_gqa(), capacity=128, block_size=8)
    root = pool.create_session()
    prefill(pool, root, deterministic_tokens((1, 3)))  # partial tail
    child = pool.fork_session(root)
    shared_tail = root.blocks[-1]
    assert pool._refcounts[shared_tail] == 2

    decode(pool, child, deterministic_tokens((1, 1)))
    # the append must have copied the partial tail, not written through
    assert child.blocks[-1] != shared_tail
    assert pool._refcounts[shared_tail] == 1  # only root left
    # child copied tail keeps the original three tokens byte-identical
    kc, _ = pool.gather_history(child, 0)
    kr, _ = pool.gather_history(root, 0)
    assert torch.equal(kc[:3], kr)
    assert child.length == 4 and root.length == 3


def test_capacity_full_prefix_and_atomic_failure(make_pool, deterministic_tokens):
    model, pool = make_pool(_gqa(), capacity=32, block_size=4)
    s = pool.create_session()
    # exact fit: 32 tokens = 8 blocks
    prefill(pool, s, deterministic_tokens((1, 32)))
    assert pool.allocated_blocks == 8
    with pytest.raises(CacheCapacityError):
        decode(pool, s, deterministic_tokens((1, 1)))
    assert s.length == 32  # no partial append
    assert pool.allocated_blocks == 8


def test_batch_capacity_failure_is_atomic(make_pool, deterministic_tokens):
    model, pool = make_pool(_gqa(), capacity=32, block_size=4)
    sa = pool.create_session()
    sb = pool.create_session()
    # prefill must succeed: 15 + 15 tokens = 30 -> 4 + 4 = 8 blocks (full pool)
    prefill(pool, [sa, sb],
            torch.cat([deterministic_tokens((1, 15)),
                       deterministic_tokens((1, 15))], 0),
            lens=[15, 15])
    assert (sa.length, sb.length) == (15, 15)
    assert len(pool._free_blocks) == 0
    used_before = pool.allocated_blocks
    blocks_before = (list(sa.blocks), list(sb.blocks))

    # one more token per session needs one fresh block each -> 2, but 0 free
    with pytest.raises(CacheCapacityError):
        decode(pool, [sa, sb],
               torch.cat([deterministic_tokens((1, 1)),
                          deterministic_tokens((1, 1))], 0))
    # neither session moved: no partial append, no COW leaked
    assert sa.length == 15 and sb.length == 15
    assert pool.allocated_blocks == used_before
    assert (list(sa.blocks), list(sb.blocks)) == blocks_before
    # and the pool still serves a request that fits into existing blocks
    # (extend within current tail capacity): nothing fits here, but releases
    # must reclaim exactly
    pool.release_session(sa)
    assert len(pool._free_blocks) == 4


def test_cow_failure_at_fork_boundary_is_atomic(make_pool, deterministic_tokens):
    model, pool = make_pool(_gqa(), capacity=32, block_size=4)
    root = pool.create_session()
    prefill(pool, root, deterministic_tokens((1, 29)))  # uses 8 blocks (32 slots)
    child = pool.fork_session(root)  # sharing costs no blocks
    assert len(pool._free_blocks) == 0
    # child appending 3 needs COW tail (1) + growth (1) but nothing is free
    with pytest.raises(CacheCapacityError):
        chunked_decode(pool, child, deterministic_tokens((1, 3)), [3])
    assert child.length == 29 and child.blocks == root.blocks
    # root untouched
    assert root.length == 29


def test_capacity_tokens_must_fit_block_grid(make_pool):
    with pytest.raises(ValueError):
        make_pool(_gqa(), capacity=30, block_size=4)


def test_cache_byte_accounting(make_pool, deterministic_tokens):
    cfg = _gqa()
    model, pool = make_pool(cfg, capacity=64, block_size=4)
    n_layers, n_kv, d = cfg.n_layers, cfg.n_kv_heads, cfg.head_dim
    elem = pool.elem_bytes
    per_slot = n_layers * n_kv * d * elem * 2
    assert pool.bytes_per_slot == per_slot
    assert pool.capacity_bytes == 64 * per_slot
    assert pool.k_backend.numel() * elem + pool.v_backend.numel() * elem == pool.capacity_bytes

    s = pool.create_session()
    prefill(pool, s, deterministic_tokens((1, 10)))
    # logical = exact token bytes; physical rounds up to whole blocks
    assert pool.session_logical_bytes(s) == 10 * per_slot
    assert pool.logical_used_bytes == 10 * per_slot
    assert pool.physical_used_bytes == 12 * per_slot  # 3 blocks * 4
    # after fork logical storage doubles (two owners of the same logical data)
    c = pool.fork_session(s)
    assert pool.logical_used_bytes == 20 * per_slot
    # but physical bytes do not grow: prefix is shared
    assert pool.physical_used_bytes == 12 * per_slot


def test_stale_cache_rejected_after_parameter_change(make_pool, deterministic_tokens):
    cfg = _gqa()
    model, pool = make_pool(cfg, capacity=64, block_size=4)
    s = pool.create_session()
    prefill(pool, s, deterministic_tokens((1, 4)))

    # mutate parameters: fingerprint changes, old K/V were rotated under old W
    with torch.no_grad():
        model.layers[0].qkv_proj.weight.add_(
            torch.randn_like(model.layers[0].qkv_proj.weight)
        )
    with pytest.raises(StaleCacheError):
        decode(pool, s, deterministic_tokens((1, 1)))
    # prefill into a fresh session is fine (new fingerprint)
    s2 = pool.create_session()
    prefill(pool, s2, deterministic_tokens((1, 4)))
    decode(pool, s2, deterministic_tokens((1, 1)))


def test_stale_cache_rejected_across_model_versions(tiny_configs):
    # structurally identical but separately versioned models cannot share:
    # build a real cache under m1, then point it at pool2 which owns m2.
    from mini_gqa import TinyModel
    cfg = _gqa()
    m1 = TinyModel(cfg, seed=1).eval()
    m2 = TinyModel(cfg, seed=2).eval()
    pool1 = KVCachePool(m1, capacity_tokens=64, block_size=4)
    pool2 = KVCachePool(m2, capacity_tokens=64, block_size=4)
    s1 = pool1.create_session()
    prefill(pool1, s1, torch.randint(0, 64, (1, 4)))
    # transplant session identity into pool2 (simulates cache reuse)
    s2 = pool2.create_session()
    s2.fingerprint = s1.fingerprint
    s2.length = s1.length
    s2.blocks = list(s1.blocks)
    with pytest.raises(StaleCacheError):
        decode(pool2, s2, torch.randint(0, 64, (1, 1)))


def _gqa():
    from mini_gqa import ModelConfig
    return ModelConfig(vocab_size=64, d_model=32, n_layers=2,
                       n_q_heads=4, n_kv_heads=2)
