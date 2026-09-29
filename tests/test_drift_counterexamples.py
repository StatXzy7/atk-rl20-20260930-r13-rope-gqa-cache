"""Counterexamples: classic KV-cache bugs that make cached logits drift.

Each case runs identical tokens through a correct cached path (which matches
the full-prefix recomputation) and through the engine with one specific,
realistic fault injected, then proves the faulty logits diverge materially.

The injected faults are test-only underscore keyword arguments; production
callers never set them.
"""

import torch

from mini_gqa import (
    KVCachePool,
    ModelConfig,
    TinyModel,
    chunked_decode,
    decode,
    prefill,
    reference_logits,
)

ATOL = 1e-5
DRIFT_MIN = 0.1


def _setup(prefix_len=20, seed=7, init_std=0.3):
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2,
                      n_q_heads=4, n_kv_heads=2)
    model = TinyModel(cfg, seed=seed, init_std=init_std).eval()
    pool = KVCachePool(model, capacity_tokens=512, block_size=4)
    g = torch.Generator().manual_seed(seed + 1000)
    root = pool.create_session()
    prefix = torch.randint(0, 64, (1, prefix_len), generator=g)
    prefill(pool, root, prefix)
    return model, pool, root, g


def test_counterexample_decode_restarts_position_at_zero():
    """Decode numbers every new request from position 0 (classic porting bug).

    History K stays at its true absolute angles while the new Q/K restart at
    0, so every q-k relative angle is wrong. Correct path matches recompute;
    the buggy one drifts strongly, and the drift grows with prefix length.
    """
    model, pool, root, g = _setup(prefix_len=20)
    nxt = torch.randint(0, 64, (1, 1), generator=g)
    ref = reference_logits(model, root, nxt)[-1]

    good = pool.fork_session(root)
    bad = pool.fork_session(root)
    lg_good, _, _ = decode(pool, good, nxt)
    lg_bad, _, _ = decode(pool, bad, nxt, _restart_position=True)

    assert torch.allclose(lg_good[0, -1], ref, atol=ATOL)
    drift_short = (lg_bad[0, -1] - ref).abs().max().item()
    assert drift_short > DRIFT_MIN

    # the same bug after a longer prefix drifts even more (more wrong angles)
    model2, pool2, root2, g2 = _setup(prefix_len=40)
    nxt2 = torch.randint(0, 64, (1, 1), generator=g2)
    ref2 = reference_logits(model2, root2, nxt2)[-1]
    bad2 = pool2.fork_session(root2)
    lg2, _, _ = decode(pool2, bad2, nxt2, _restart_position=True)
    drift_long = (lg2[0, -1] - ref2).abs().max().item()
    assert drift_long > drift_short


def test_counterexample_rope_off_by_one_rotates_at_previous_position():
    """Off-by-one: decode rotates new tokens at length-1 rather than length.

    On a realistically scaled model this is a smaller, but unambiguous, drift;
    the correct sibling stays at the reference.
    """
    model, pool, root, g = _setup(prefix_len=20)
    nxt = torch.randint(0, 64, (1, 1), generator=g)
    ref = reference_logits(model, root, nxt)[-1]
    good = pool.fork_session(root)
    bad = pool.fork_session(root)
    lg_good, _, _ = decode(pool, good, nxt)
    lg_bad, _, _ = decode(pool, bad, nxt, _position_bug=True)
    assert torch.allclose(lg_good[0, -1], ref, atol=ATOL)
    assert (lg_bad[0, -1] - ref).abs().max().item() > 100 * ATOL


def test_counterexample_chunk_sees_its_own_future():
    """Chunked decode drops the within-chunk causal mask.

    Token j then attends to j+1.. within the same chunk; rows before the last
    drift (row 0 leaks three future tokens), while the final row -- which has
    no future token -- is unaffected.
    """
    model, pool, root, g = _setup(prefix_len=20, seed=31)
    chunk = torch.randint(0, 64, (1, 4), generator=g)

    per = pool.fork_session(root)
    leak = pool.fork_session(root)
    per_rows = []
    for j in range(4):
        lg, _, _ = decode(pool, per, chunk[:, j:j + 1])
        per_rows.append(lg[0, -1])
    lg_leak, _, _ = chunked_decode(pool, leak, chunk, [4], _leak_future=True)

    drifts = [(per_rows[j] - lg_leak[0, j]).abs().max().item() for j in range(4)]
    assert drifts[0] > DRIFT_MIN and drifts[1] > DRIFT_MIN
    assert drifts[2] > 100 * ATOL
    assert drifts[3] < 0.1  # last token has no within-chunk future (fp noise only)
    # the correct per-token path matches full-prefix recomputation
    prefix_ids = torch.tensor([root.stored_ids])
    for j in range(4):
        full = torch.cat([prefix_ids, chunk[:, :j + 1]], 1)
        assert torch.allclose(
            per_rows[j], model.forward_full_prefix(full)[0, -1], atol=ATOL
        )


def test_counterexample_wrong_gqa_head_grouping():
    """KV heads are assigned to Q heads by the wrong repeat mapping.

    The correct mapping is consecutive blocks; the faulty engine cyclically
    rotates kv-head ids, sending each Q head to a different K/V stream. The
    correct path matches recompute; the rotated mapping drifts, and only for
    true GQA/MQA-with-multiple-kv configs (MHA hkv==hq would be unaffected).
    """
    model, pool, root, g = _setup(prefix_len=20)
    nxt = torch.randint(0, 64, (1, 1), generator=g)
    ref = reference_logits(model, root, nxt)[-1]
    good = pool.fork_session(root)
    bad = pool.fork_session(root)
    lg_good, _, _ = decode(pool, good, nxt)
    lg_bad, _, _ = decode(pool, bad, nxt, _group_mode="rotated")
    assert torch.allclose(lg_good[0, -1], ref, atol=ATOL)
    assert (lg_bad[0, -1] - ref).abs().max().item() > DRIFT_MIN


def test_counterexample_cached_k_not_rotated():
    """New K is stored/used without RoPE while Q (and all history K) are rotated.

    Dot products at the decode step then combine rotated Q with unrotated K,
    corrupting the attention distribution and logits.
    """
    model, pool, root, g = _setup(prefix_len=20)
    nxt = torch.randint(0, 64, (1, 1), generator=g)
    ref = reference_logits(model, root, nxt)[-1]
    good = pool.fork_session(root)
    bad = pool.fork_session(root)
    lg_good, _, _ = decode(pool, good, nxt)
    lg_bad, _, _ = decode(pool, bad, nxt, _rotate_new_k=False)
    assert torch.allclose(lg_good[0, -1], ref, atol=ATOL)
    assert (lg_bad[0, -1] - ref).abs().max().item() > DRIFT_MIN
