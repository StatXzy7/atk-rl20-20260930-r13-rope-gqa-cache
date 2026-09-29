import torch
import pytest

from mini_decoder import (
    CacheCapacityError,
    CacheInvalidError,
    Engine,
    KVCacheManager,
    reference_logits,
)
from conftest import ATOL, make_engine, make_model


def toks(seed, n, vocab=61):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (n,), generator=g).tolist()


def assert_close(a, b):
    torch.testing.assert_close(a, b, atol=ATOL, rtol=ATOL)


# --------------------------------------------------------------------- prefill
def test_prefill_unequal_batch_matches_reference(model):
    eng, _ = make_engine(model)
    batch = [toks(1, 7), toks(2, 3), toks(3, 5)]
    seqs = [eng.create_sequence() for _ in batch]
    out = eng.prefill(seqs, batch)
    for tokens, got in zip(batch, out):
        assert_close(got, reference_logits(model, tokens))


def test_prefill_logits_match_across_batch_composition(model):
    # The same sequence must give identical logits alone or padded in a batch.
    eng, _ = make_engine(model)
    solo, other = toks(4, 4), toks(5, 9)
    s1 = eng.create_sequence()
    solo_logits = eng.prefill([s1], [solo])[0]
    s2, s3 = eng.create_sequence(), eng.create_sequence()
    batched = eng.prefill([s2, s3], [solo, other])
    assert_close(batched[0], solo_logits)
    assert_close(batched[1], reference_logits(model, other))


# ---------------------------------------------------------------------- decode
def test_token_by_token_decode_matches_reference(model):
    eng, _ = make_engine(model)
    batch = [toks(10, 5), toks(12, 8)]  # unequal lengths in the same batch
    seqs = [eng.create_sequence() for _ in batch]
    eng.prefill(seqs, batch)
    conts = [toks(11, 6), toks(13, 6)]
    full = [list(p) for p in batch]
    for step in range(6):
        out = eng.decode(seqs, [[conts[b][step]] for b in range(2)])
        for b in range(2):
            full[b].append(conts[b][step])
            assert_close(out[b][-1], reference_logits(model, full[b])[-1])


@pytest.mark.parametrize("chunking", [[1, 1, 1, 1, 1, 1], [2, 2, 2], [3, 3], [6], [1, 3, 2]])
def test_chunked_decode_independent_of_chunking(model, chunking):
    eng, _ = make_engine(model)
    prefix, cont = toks(20, 5), toks(21, 6)
    s = eng.create_sequence()
    eng.prefill([s], [prefix])
    full = list(prefix)
    pos = 0
    for size in chunking:
        chunk = cont[pos : pos + size]
        pos += size
        out = eng.decode([s], [chunk])[0]
        full.extend(chunk)
        ref = reference_logits(model, full)
        assert out.shape[0] == size
        assert_close(out, ref[-size:])  # every position of the chunk, not just the last


def test_ragged_chunks_in_one_decode_call(model):
    eng, _ = make_engine(model)
    batch = [toks(30, 4), toks(31, 6)]
    seqs = [eng.create_sequence() for _ in batch]
    eng.prefill(seqs, batch)
    full = [list(p) for p in batch]
    for c0, c1 in [(toks(32, 2), toks(33, 3)), (toks(34, 1), toks(35, 4))]:
        out = eng.decode(seqs, [c0, c1])
        for b, chunk in enumerate((c0, c1)):
            full[b].extend(chunk)
            assert_close(out[b], reference_logits(model, full[b])[-len(chunk):])


def test_position_offset_prefill_and_decode(model):
    eng, _ = make_engine(model)
    offs = [0, 7]
    batch = [toks(40, 5), toks(41, 3)]
    seqs = [eng.create_sequence(pos_offset=o) for o in offs]
    out = eng.prefill(seqs, batch)
    for tokens, off, got in zip(batch, offs, out):
        assert_close(got, reference_logits(model, tokens, pos_offset=off))
    conts = [toks(42, 2), toks(43, 2)]
    out = eng.decode(seqs, conts)
    for b in range(2):
        full = batch[b] + conts[b]
        assert_close(out[b], reference_logits(model, full, pos_offset=offs[b])[-2:])


def test_cache_stores_rotated_k(model):
    # The cached K must equal the RoPE-rotated K of the reference computation.
    eng, mgr = make_engine(model)
    tokens = toks(50, 6)
    s = eng.create_sequence()
    eng.prefill([s], [tokens])
    ids = torch.tensor(tokens)[None, :]
    positions = torch.arange(len(tokens))[None, :]
    allowed = torch.tril(torch.ones(len(tokens), len(tokens), dtype=torch.bool))[None]
    x = model.embed(ids)
    for li, layer in enumerate(model.layers):
        h = layer.ln1(x)
        q, k_ref, v_ref = layer.attn.project_qkv(h, positions)  # [1, Hkv, L, D]
        x = x + layer.attn.attend(q, k_ref, v_ref, allowed)
        x = x + layer.mlp(layer.ln2(x))
        k_cached, v_cached = mgr.gather(s, li)
        assert_close(k_cached, k_ref[0].transpose(0, 1))
        assert_close(v_cached, v_ref[0].transpose(0, 1))


# ------------------------------------------------------- forks through engine
def test_fork_two_continuations_do_not_pollute_each_other(model):
    eng, mgr = make_engine(model)
    prefix = toks(60, 6)  # 6 tokens: one full block + half-filled block (bs=4)
    a = eng.create_sequence()
    eng.prefill([a], [prefix])
    b = eng.fork(a)
    shared_block = mgr.block_table(b)[0]
    shared_before = [mgr.k[li][shared_block].clone() for li in range(model.cfg.n_layers)]

    cont_a, cont_b = toks(61, 5), toks(62, 5)
    fa, fb = list(prefix), list(prefix)
    for i in range(5):
        (out,) = eng.decode([a], [[cont_a[i]]])
        fa.append(cont_a[i])
        assert_close(out[-1], reference_logits(model, fa)[-1])
        (out,) = eng.decode([b], [[cont_b[i]]])
        fb.append(cont_b[i])
        assert_close(out[-1], reference_logits(model, fb)[-1])

    # The shared prefix block was never written by either branch.
    for li in range(model.cfg.n_layers):
        assert_close(mgr.k[li][shared_block], shared_before[li])

    # Freeing one branch must not break the other.
    eng.free(a)
    tail = toks(63, 3)
    (out,) = eng.decode([b], [tail])
    fb.extend(tail)
    assert_close(out, reference_logits(model, fb)[-3:])


# ------------------------------------------------------------- invalidation
def test_parameter_change_invalidates_cache(model):
    eng, mgr = make_engine(model)
    s = eng.create_sequence()
    eng.prefill([s], [toks(70, 4)])
    with torch.no_grad():
        model.layers[0].attn.q_proj.weight.add_(1e-3)
    with pytest.raises(CacheInvalidError):
        eng.decode([s], [[1]])
    with pytest.raises(CacheInvalidError):
        Engine(model, mgr)  # old cache must not be reusable by a changed model
    # A fresh cache stamped with the new fingerprint works again.
    mgr2 = KVCacheManager(mgr.spec, num_blocks=8, model_version=model.fingerprint())
    eng2 = Engine(model, mgr2)
    s2 = eng2.create_sequence()
    tokens = toks(71, 3)
    out = eng2.prefill([s2], [tokens])
    assert_close(out[0], reference_logits(model, tokens))


def test_config_change_means_different_version():
    m1 = make_model(4, 2, seed=0)
    m2 = make_model(4, 1, seed=0)  # different head layout
    assert m1.fingerprint() != m2.fingerprint()


# --------------------------------------------------------------- capacity
def test_engine_step_capacity_is_atomic_across_batch(model):
    eng, mgr = make_engine(model, num_blocks=3, block_size=4)
    t0, t1 = toks(80, 4), toks(81, 4)
    s0, s1 = eng.create_sequence(), eng.create_sequence()
    eng.prefill([s0, s1], [t0, t1])  # 2 blocks used, 1 free
    with pytest.raises(CacheCapacityError):
        eng.decode([s0, s1], [toks(82, 4), toks(83, 4)])  # needs 2 blocks
    assert [mgr.length(s0), mgr.length(s1)] == [4, 4]  # neither sequence grew
    assert mgr.num_free_blocks == 1
    c = toks(84, 2)
    (out,) = eng.decode([s0], [c])  # 1 block, fits
    assert [mgr.length(s0), mgr.length(s1)] == [6, 4]
    assert_close(out, reference_logits(model, t0 + c)[-2:])
