"""Prefill / per-token decode / chunked decode vs full-prefix recomputation."""

import itertools

import pytest
import torch

from mini_gqa import (
    chunked_decode,
    decode,
    prefill,
    reference_logits,
)

ATOL = 1e-5


def full_tail(model, stored, cont, base_pos=0):
    """Ground-truth last-token logits over stored prefix + continuation."""
    ids = torch.tensor([stored + list(cont)], dtype=torch.long)
    return model.forward_full_prefix(ids, base_pos=base_pos)[0, -1]


def full_rows(model, stored, cont, base_pos=0):
    ids = torch.tensor([stored + list(cont)], dtype=torch.long)
    return model.forward_full_prefix(ids, base_pos=base_pos)[0, len(stored):]


def _prefill_one(pool, session, tokens):
    t = torch.tensor([tokens], dtype=torch.long)
    logits, valid, _ = prefill(pool, session, t)
    return logits[0, len(tokens) - 1]


@pytest.mark.parametrize("config_name", ["mha", "mqa", "gqa"])
def test_prefill_matches_stateless_causal(tiny_configs, make_pool,
                                          deterministic_tokens, config_name):
    cfg = tiny_configs[config_name]
    model, pool = make_pool(cfg)
    sess = pool.create_session()
    tokens = deterministic_tokens((1, 7))[0].tolist()

    logits, valid, lens = prefill(
        pool, sess, torch.tensor([tokens]), lens=[len(tokens)]
    )
    ref = model.forward_full_prefix(torch.tensor([tokens]))
    # every prefill row must match, not only the last one
    assert torch.allclose(logits[0], ref[0], atol=ATOL)
    assert lens == [7] and valid.all()


@pytest.mark.parametrize("config_name", ["mha", "mqa", "gqa"])
def test_per_token_decode_matches_full_recompute(tiny_configs, make_pool,
                                                 deterministic_tokens, config_name):
    cfg = tiny_configs[config_name]
    model, pool = make_pool(cfg)
    sess = pool.create_session()
    prefix = deterministic_tokens((1, 6))[0].tolist()
    cont = deterministic_tokens((1, 8))[0].tolist()
    prefill(pool, sess, torch.tensor([prefix]))

    max_drift = 0.0
    for step, tok in enumerate(cont):
        nxt = torch.tensor([[tok]])
        # reference computed against the pre-commit session (independent recompute)
        ref = reference_logits(model, sess, nxt)[-1]
        logits, _, _ = decode(pool, sess, nxt)
        got = logits[0, -1]
        drift = (got - ref).abs().max().item()
        max_drift = max(max_drift, drift)
        assert drift < ATOL
        assert sess.length == 6 + step + 1
        # committed session must reproduce the same via fresh recompute too
        assert torch.allclose(
            got, full_tail(model, sess.stored_ids[:-1], [tok]), atol=ATOL
        )
    assert max_drift < 1e-5


@pytest.mark.parametrize("config_name", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("chunk_widths", [(2, 2, 2), (3, 2), (1, 4, 1), (7, 1), (8,)])
def test_chunked_decode_matches_per_token_and_reference(
    tiny_configs, make_pool, deterministic_tokens, config_name, chunk_widths
):
    cfg = tiny_configs[config_name]
    model, pool = make_pool(cfg)
    root = pool.create_session()
    prefix = deterministic_tokens((1, 5))[0].tolist()
    cont = deterministic_tokens((1, sum(chunk_widths)))[0].tolist()
    prefill(pool, root, torch.tensor([prefix]))

    # independent per-token branch
    single = pool.fork_session(root)
    per_token_rows = []
    for tok in cont:
        lg, _, _ = decode(pool, single, torch.tensor([[tok]]))
        per_token_rows.append(lg[0, -1])

    # chunked branch
    chunked_rows = []
    cursor = 0
    for w in chunk_widths:
        block = torch.tensor([cont[cursor:cursor + w]])
        lg, _, _ = chunked_decode(pool, root, block, [w])
        chunked_rows.extend(lg[0, :w])
        cursor += w

    for j, (a, b) in enumerate(zip(chunked_rows, per_token_rows)):
        assert torch.allclose(a, b, atol=ATOL), f"chunk row {j} drifts"
        ref = full_tail(model, prefix, cont[:j + 1])
        assert torch.allclose(a, ref, atol=ATOL), f"row {j} vs reference"


@pytest.mark.parametrize("config_name", ["mha", "mqa", "gqa"])
def test_ragged_batch_validity_padding_positions_and_causal(
    tiny_configs, make_pool, deterministic_tokens, config_name
):
    cfg = tiny_configs[config_name]
    model, pool = make_pool(cfg)
    la, lb = 3, 6
    ta = deterministic_tokens((1, la))
    tb = deterministic_tokens((1, lb))
    padded = torch.cat([torch.nn.functional.pad(ta, (0, lb - la)), tb], dim=0)

    sa = pool.create_session()
    sb = pool.create_session()
    logits, valid, lens = prefill(pool, [sa, sb], padded, lens=[la, lb])

    # validity mask: only the real tokens
    assert valid[0, :la].all() and not valid[0, la:].any()
    assert valid[1].all()
    # padding rows must produce zero (never attend)
    assert torch.count_nonzero(logits[0, la:]) == 0
    # last valid rows match independent single-sequence prefill
    assert torch.allclose(
        logits[0, la - 1], model.forward_full_prefix(ta)[0, -1], atol=ATOL
    )
    assert torch.allclose(
        logits[1, lb - 1], model.forward_full_prefix(tb)[0, -1], atol=ATOL
    )
    assert sa.length == la and sb.length == lb

    # batched per-token decode of two now-different-length sessions
    na = deterministic_tokens((1, 1))
    nb = deterministic_tokens((1, 1))
    dl, _, _ = decode(pool, [sa, sb], torch.cat([na, nb], 0))
    assert torch.allclose(
        dl[0, -1], full_tail(model, ta[0].tolist(), na[0].tolist()), atol=ATOL
    )
    assert torch.allclose(
        dl[1, -1], full_tail(model, tb[0].tolist(), nb[0].tolist()), atol=ATOL
    )

    # ragged chunked decode: widths differ inside the same batch
    ca = deterministic_tokens((1, 2))
    cb = deterministic_tokens((1, 3))
    width = 3
    batch = torch.cat([torch.nn.functional.pad(ca, (0, 1)), cb], dim=0)
    cl, _, _ = chunked_decode(pool, [sa, sb], batch, [2, 3])
    ref_a_rows = full_rows(model, ta[0].tolist() + na[0].tolist(), ca[0].tolist())
    ref_b_rows = full_rows(model, tb[0].tolist() + nb[0].tolist(), cb[0].tolist())
    assert torch.allclose(cl[0, :2], ref_a_rows, atol=ATOL)
    assert torch.allclose(cl[1, :3], ref_b_rows, atol=ATOL)
    assert torch.count_nonzero(cl[0, 2]) == 0  # padded column zero


def test_position_offset_baked_into_cached_k(tiny_configs, make_pool,
                                                 deterministic_tokens):
    # RoPE attention is translation invariant in logits (scores depend on
    # relative positions), so absolute numbering is verified directly on the
    # cached, already-rotated K: entry i must be R(base_pos + i) applied to
    # the raw projected K.
    cfg = tiny_configs["gqa"]
    model, pool = make_pool(cfg)
    base_pos = 37
    sess = pool.create_session(base_pos=base_pos)
    tokens = deterministic_tokens((1, 6))
    prefill(pool, sess, tokens)

    layer = model.layers[0]
    with torch.no_grad():
        raw = layer.qkv_proj(layer.attn_norm(model.tok_emb(tokens)))
    q_len = cfg.n_q_heads * cfg.head_dim
    kv_len = cfg.n_kv_heads * cfg.head_dim
    raw_k = raw[..., q_len:q_len + kv_len].view(1, 6, cfg.n_kv_heads, cfg.head_dim)
    abs_pos = torch.arange(base_pos, base_pos + 6).view(1, 6)
    expected_k = model.rope.rotate(raw_k, abs_pos)[0]
    cached_k, _ = pool.gather_history(sess, 0)
    assert torch.allclose(cached_k, expected_k, atol=ATOL)

    # same tokens numbered from 0 must produce *different* cached K angles,
    # proving the offset is actually consumed (even though final logits of a
    # pure-RoPE model are translation invariant).
    zero = pool.create_session(base_pos=0)
    prefill(pool, zero, tokens)
    cached_k0, _ = pool.gather_history(zero, 0)
    assert not torch.allclose(cached_k, cached_k0, atol=1e-3)

    # decode after offset keeps numbering contiguous: new K at base_pos+6
    nxt = deterministic_tokens((1, 1))
    lg, _, _ = decode(pool, sess, nxt)
    with torch.no_grad():
        raw_all = layer.qkv_proj(layer.attn_norm(model.tok_emb(torch.cat([tokens, nxt], 1))))
    raw_new = raw_all[:, 6:7, q_len:q_len + kv_len].view(1, 1, cfg.n_kv_heads, cfg.head_dim)
    expected_new = model.rope.rotate(raw_new, torch.tensor([[base_pos + 6]]))[0]
    cached_new, _ = pool.gather_history(sess, 0)
    assert torch.allclose(cached_new[6:7], expected_new, atol=ATOL)


def test_decode_continues_position_where_prefill_stopped(make_pool, tiny_configs,
                                                         deterministic_tokens):
    # decode cannot restart RoPE at 0: cached path matches absolute numbering
    cfg = tiny_configs["gqa"]
    model, pool = make_pool(cfg)
    sess = pool.create_session()
    prefix = deterministic_tokens((1, 4))[0].tolist()
    cont = deterministic_tokens((1, 3))[0].tolist()
    prefill(pool, sess, torch.tensor([prefix]))
    for j, tok in enumerate(cont):
        nxt = torch.tensor([[tok]])
        ref = reference_logits(model, sess, nxt)
        lg, _, _ = decode(pool, sess, nxt)
        assert torch.allclose(lg[0, -1], ref[-1], atol=ATOL)


@pytest.mark.parametrize("block_size,pref_len,chunk_widths", [
    (4, 7, (2, 3, 4)),   # chunks straddle 8-token and 12-token block edges
    (8, 1, (7, 1, 8)),   # prefill shorter than one block, chunks hit edges
    (2, 3, (1, 2, 3)),   # tiny blocks force many COW/growth decisions
])
def test_chunked_decode_across_block_boundaries(
    tiny_configs, make_pool, deterministic_tokens, block_size, pref_len, chunk_widths
):
    cfg = tiny_configs["gqa"]
    model, pool = make_pool(cfg, capacity=512, block_size=block_size)
    sess = pool.create_session()
    prefix = deterministic_tokens((1, pref_len))[0].tolist()
    cont = deterministic_tokens((1, sum(chunk_widths)))[0].tolist()
    prefill(pool, sess, torch.tensor([prefix]))

    rows = []
    cursor = 0
    for w in chunk_widths:
        block = torch.tensor([cont[cursor:cursor + w]])
        lg, _, _ = chunked_decode(pool, sess, block, [w])
        rows.extend(lg[0, :w])
        cursor += w

    for j in range(len(cont)):
        full = torch.tensor([prefix + cont[:j + 1]])
        assert torch.allclose(rows[j], model.forward_full_prefix(full)[0, -1], atol=ATOL)


def test_cached_v_matches_projected_v_all_layers(make_pool, deterministic_tokens):
    # V carries no rotation but must be the exact projected V for every layer;
    # a wrong slice/index order would corrupt deep layers.
    from mini_gqa import ModelConfig, TinyModel, KVCachePool
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=3,
                      n_q_heads=4, n_kv_heads=2)
    model = TinyModel(cfg, seed=42).eval()
    pool = KVCachePool(model, capacity_tokens=128, block_size=4)
    sess = pool.create_session()
    tokens = deterministic_tokens((1, 10))
    prefill(pool, sess, tokens)
    x = model.tok_emb(tokens)
    for lidx, layer in enumerate(model.layers):
        h = layer.attn_norm(x)
        _, k, v = layer.project_qkv(h)
        _, cached_v = pool.gather_history(sess, lidx)
        assert torch.allclose(cached_v, v[0], atol=ATOL)
        # advance the stateless hidden state through the layer
        q, k, v = layer.project_qkv(h)
        q = model.rope.rotate(q, torch.arange(10).view(1, 10))
        k = model.rope.rotate(k, torch.arange(10).view(1, 10))
        from mini_gqa.model import masked_attention
        allow = torch.tril(torch.ones(10, 10, dtype=torch.bool)).unsqueeze(0)
        attn = masked_attention(q, k, v, n_kv_heads=cfg.n_kv_heads,
                                scale=model.scale, allow=allow)
        x = x + layer.attention_out(attn)
        x = x + layer.ffn(layer.ffn_norm(x))
