"""Counterexamples: classic KV-cache bugs that make logits drift.

Each test runs (a) a deliberately buggy variant of the cached step, which
must NOT match the independent whole-prefix reference, and (b) the correct
engine, which must match -- demonstrating the failure mode each design rule
protects against.  Buggy variants run first: they never write to the cache,
so the correct decode afterwards still sees the pristine prefix.
"""
import torch

from mini_decoder import reference_logits
from conftest import ATOL, make_engine

DRIFT_MIN = 1e-3  # drift must be clearly visible, far above float64 noise


def toks(seed, n, vocab=61):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (n,), generator=g).tolist()


def manual_step(model, mgr, s, chunk, positions, allowed):
    """Single-sequence cached decode step with caller-supplied positions/mask.

    Reads the cache but never appends to it.
    """
    past = mgr.length(s)
    ids = torch.tensor(chunk)[None, :]
    x = model.embed(ids)
    for li, layer in enumerate(model.layers):
        if past > 0:
            kb, vb = mgr.gather(s, li)  # already-rotated cached K
            past_kv = (kb.transpose(0, 1)[None], vb.transpose(0, 1)[None])
        else:
            past_kv = None
        h = layer.ln1(x)
        q, k_new, v_new = layer.attn.project_qkv(h, positions)
        if past_kv is not None:
            k_all = torch.cat([past_kv[0], k_new], dim=2)
            v_all = torch.cat([past_kv[1], v_new], dim=2)
        else:
            k_all, v_all = k_new, v_new
        x = x + layer.attn.attend(q, k_all, v_all, allowed)
        x = x + layer.mlp(layer.ln2(x))
    return model.logits(x)[0]


def test_counterexample_position_reset_on_decode(model):
    """Bug: decode rotates Q/K with chunk-relative positions (0,1,...) instead
    of absolute ones (past_len, past_len+1, ...)."""
    eng, mgr = make_engine(model)
    prefix, chunk = toks(90, 5), toks(91, 2)
    s = eng.create_sequence()
    eng.prefill([s], [prefix])
    past, S = mgr.length(s), len(chunk)

    # Buggy: positions restart at 0 for the new tokens.
    bad_pos = torch.arange(S)[None, :]  # should be past + arange(S)
    allowed = torch.tril(torch.ones(S, past + S, dtype=torch.bool), diagonal=past)[None]
    buggy = manual_step(model, mgr, s, chunk, bad_pos, allowed)

    ref = reference_logits(model, prefix + chunk)[-S:]
    drift = (buggy - ref).abs().max().item()
    assert drift > DRIFT_MIN, f"expected visible drift, got {drift}"

    correct = eng.decode([s], [chunk])[0]
    torch.testing.assert_close(correct, ref, atol=ATOL, rtol=ATOL)


def test_counterexample_causal_off_by_one_in_chunk(model):
    """Bug: chunked decode lets query i also see chunk token i+1 (one future
    token leaks into the past)."""
    eng, mgr = make_engine(model)
    prefix, chunk = toks(92, 4), toks(93, 3)
    s = eng.create_sequence()
    eng.prefill([s], [prefix])
    past, S = mgr.length(s), len(chunk)

    pos = (past + torch.arange(S))[None, :]  # positions are right...
    # ...but the mask allows j <= past + i + 1: one future token visible.
    bad_mask = torch.tril(torch.ones(S, past + S, dtype=torch.bool), diagonal=past + 1)[None]
    buggy = manual_step(model, mgr, s, chunk, pos, bad_mask)

    ref = reference_logits(model, prefix + chunk)[-S:]
    drift = (buggy - ref).abs().max().item()
    assert drift > DRIFT_MIN, f"expected visible drift, got {drift}"

    correct = eng.decode([s], [chunk])[0]
    torch.testing.assert_close(correct, ref, atol=ATOL, rtol=ATOL)


def test_counterexample_padding_leaks_into_attention(model):
    """Bug: zero-padded cache slots of shorter sequences are not masked out
    of the softmax when batching unequal lengths."""
    eng, mgr = make_engine(model)
    batch = [toks(94, 3), toks(95, 6)]  # different prefix lengths
    seqs = [eng.create_sequence() for _ in batch]
    eng.prefill(seqs, batch)
    chunk = toks(96, 1)

    # Buggy batched step: past slots padded to Pmax are ALL marked valid.
    pasts = [mgr.length(s) for s in seqs]
    Pmax = max(pasts)
    cfg = model.cfg
    ids = torch.tensor([chunk, chunk])
    q_pos = torch.tensor(pasts)[:, None] + torch.arange(1)[None, :]
    x = model.embed(ids)
    for li, layer in enumerate(model.layers):
        K = torch.zeros(2, cfg.n_kv_heads, Pmax, cfg.head_dim, dtype=torch.float64)
        V = torch.zeros_like(K)
        for b, s in enumerate(seqs):
            kb, vb = mgr.gather(s, li)
            K[b, :, : pasts[b]] = kb.transpose(0, 1)
            V[b, :, : pasts[b]] = vb.transpose(0, 1)
        h = layer.ln1(x)
        q, k_new, v_new = layer.attn.project_qkv(h, q_pos)
        k_all = torch.cat([K, k_new], dim=2)
        v_all = torch.cat([V, v_new], dim=2)
        past_valid = torch.ones(2, Pmax, dtype=torch.bool)  # BUG: ignores real lengths
        new_valid = torch.ones(2, 1, dtype=torch.bool)
        key_valid = torch.cat([past_valid, new_valid], dim=1)
        key_pos = torch.cat([torch.arange(Pmax)[None, :].expand(2, -1), q_pos], dim=1)
        allowed = key_valid[:, None, :] & (key_pos[:, None, :] <= q_pos[:, :, None])
        x = x + layer.attn.attend(q, k_all, v_all, allowed)
        x = x + layer.mlp(layer.ln2(x))
    buggy = model.logits(x)[0, -1]  # shorter sequence is the victim

    ref = reference_logits(model, batch[0] + chunk)[-1]
    drift = (buggy - ref).abs().max().item()
    assert drift > DRIFT_MIN, f"expected visible drift, got {drift}"

    correct = eng.decode(seqs, [chunk, chunk])
    for b in range(2):
        ref_b = reference_logits(model, batch[b] + chunk)[-1]
        torch.testing.assert_close(correct[b][-1], ref_b, atol=ATOL, rtol=ATOL)
