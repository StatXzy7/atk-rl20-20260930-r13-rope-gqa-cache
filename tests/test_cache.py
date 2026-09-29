import torch
import pytest

from mini_decoder import CacheCapacityError, CacheSpec, KVCacheManager

SPEC = CacheSpec(n_layers=2, n_kv_heads=3, head_dim=8, block_size=4, dtype=torch.float64)


def make_mgr(num_blocks=8, version="v0"):
    return KVCacheManager(SPEC, num_blocks=num_blocks, model_version=version)


def rand_kv(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    ks = [torch.randn(n, SPEC.n_kv_heads, SPEC.head_dim, generator=g, dtype=torch.float64)
          for _ in range(SPEC.n_layers)]
    vs = [torch.randn(n, SPEC.n_kv_heads, SPEC.head_dim, generator=g, dtype=torch.float64)
          for _ in range(SPEC.n_layers)]
    return ks, vs


def test_pool_bytes_match_actual_tensor_storage():
    mgr = make_mgr(num_blocks=6)
    # block = n_layers * (K+V) * block_size * Hkv * D * itemsize
    assert mgr.block_bytes == 2 * 2 * 4 * 3 * 8 * 8
    assert mgr.nbytes == 6 * mgr.block_bytes
    actual = sum(t.numel() * t.element_size() for t in mgr.k + mgr.v)
    assert actual == mgr.nbytes
    assert mgr.allocated_bytes == 0


def test_sequence_bytes_track_appends():
    mgr = make_mgr()
    s = mgr.alloc_sequence()
    ks, vs = rand_kv(5)
    mgr.append(s, [k[:4] for k in ks], [v[:4] for v in vs], 4)
    assert mgr.sequence_bytes(s) == 1 * mgr.block_bytes  # 4 tokens = 1 block
    mgr.append(s, [k[4:] for k in ks], [v[4:] for v in vs], 1)
    assert mgr.sequence_bytes(s) == 2 * mgr.block_bytes  # 5 tokens spill over
    assert mgr.allocated_bytes == 2 * mgr.block_bytes


def test_gather_roundtrip_and_length():
    mgr = make_mgr()
    s = mgr.alloc_sequence()
    ks, vs = rand_kv(6, seed=3)
    mgr.append(s, ks, vs, 6)
    assert mgr.length(s) == 6
    for li in range(SPEC.n_layers):
        k, v = mgr.gather(s, li)
        assert k.shape == (6, SPEC.n_kv_heads, SPEC.head_dim)
        torch.testing.assert_close(k, ks[li])
        torch.testing.assert_close(v, vs[li])


def test_capacity_error_is_atomic():
    mgr = make_mgr(num_blocks=2)  # room for exactly 8 tokens
    s = mgr.alloc_sequence()
    ks, vs = rand_kv(4, seed=1)
    mgr.append(s, ks, vs, 4)
    assert mgr.num_free_blocks == 1
    # 5 more tokens would need 2 extra blocks; only 1 is free.
    ks5, vs5 = rand_kv(5, seed=2)
    with pytest.raises(CacheCapacityError):
        mgr.append(s, ks5, vs5, 5)
    # Nothing partially appended: length, blocks and content unchanged.
    assert mgr.length(s) == 4
    assert mgr.num_free_blocks == 1
    for li in range(SPEC.n_layers):
        k, v = mgr.gather(s, li)
        torch.testing.assert_close(k, ks[li])
        torch.testing.assert_close(v, vs[li])
    # A 4-token append fits exactly (pool now full).
    ks4, vs4 = rand_kv(4, seed=4)
    mgr.append(s, ks4, vs4, 4)
    assert mgr.length(s) == 8
    assert mgr.num_free_blocks == 0
    with pytest.raises(CacheCapacityError):
        mgr.append(s, *rand_kv(1, seed=5), 1)


def test_fork_shares_prefix_and_cow_isolates_writes():
    mgr = make_mgr()
    a = mgr.alloc_sequence()
    ks, vs = rand_kv(6, seed=7)  # 6 tokens: block0 full, block1 half full
    mgr.append(a, ks, vs, 6)
    b = mgr.fork(a)
    assert mgr.allocated_bytes == 2 * mgr.block_bytes  # fork copies no data
    assert all(mgr.refcount[blk] == 2 for blk in mgr.block_table(a))

    # Appending to A must copy the shared, half-filled tail block (COW).
    ksa, vsa = rand_kv(1, seed=8)
    mgr.append(a, ksa, vsa, 1)
    assert mgr.allocated_bytes == 3 * mgr.block_bytes
    ta, tb = mgr.block_table(a), mgr.block_table(b)
    assert ta[0] == tb[0]  # full prefix block still shared
    assert ta[1] != tb[1]  # tail block was copied on write

    # B's view of the shared prefix is byte-identical to the original.
    for li in range(SPEC.n_layers):
        kb, _ = mgr.gather(b, li)
        torch.testing.assert_close(kb, ks[li])
        ka, _ = mgr.gather(a, li)
        torch.testing.assert_close(ka[:6], ks[li])  # A kept its prefix too
        torch.testing.assert_close(ka[6:], ksa[li])

    # B now owns its tail block exclusively and writes in place (no new block).
    ksb, vsb = rand_kv(2, seed=9)
    mgr.append(b, ksb, vsb, 2)
    assert mgr.allocated_bytes == 3 * mgr.block_bytes
    for li in range(SPEC.n_layers):
        ka, _ = mgr.gather(a, li)
        assert ka.shape[0] == 7  # A untouched by B's writes


def test_free_one_branch_keeps_the_other_intact():
    mgr = make_mgr()
    a = mgr.alloc_sequence()
    ks, vs = rand_kv(5, seed=11)
    mgr.append(a, ks, vs, 5)
    b = mgr.fork(a)
    mgr.free(a)
    assert mgr.allocated_bytes == 2 * mgr.block_bytes  # shared prefix survives
    for li in range(SPEC.n_layers):
        k, v = mgr.gather(b, li)
        torch.testing.assert_close(k, ks[li])
        torch.testing.assert_close(v, vs[li])
    # The survivor can keep appending (COW on the still-shared tail block).
    ks2, vs2 = rand_kv(3, seed=12)
    mgr.append(b, ks2, vs2, 3)
    assert mgr.length(b) == 8
    mgr.free(b)
    assert mgr.allocated_bytes == 0
    with pytest.raises(KeyError):
        mgr.length(b)


def test_cow_copy_counts_against_capacity():
    # Pool with exactly 2 blocks: prefix uses 1 (half full), fork shares it,
    # branch A's append needs a COW copy -> 1 block. Branch B's append then
    # writes in place (sole owner) -> 0 blocks.
    mgr = make_mgr(num_blocks=2)
    a = mgr.alloc_sequence()
    ks, vs = rand_kv(3, seed=13)
    mgr.append(a, ks, vs, 3)
    b = mgr.fork(a)
    assert mgr.required_blocks(a, 1) == 1  # COW copy
    mgr.append(a, *rand_kv(1, seed=14), 1)
    assert mgr.num_free_blocks == 0
    assert mgr.required_blocks(b, 1) == 0  # B is now sole owner of the tail
    mgr.append(b, *rand_kv(1, seed=15), 1)
    assert mgr.length(b) == 4
