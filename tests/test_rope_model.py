"""RoPE definition, config invariants and stateless causal-model checks."""

import math

import pytest
import torch

from mini_gqa import ModelConfig, RoPE, TinyModel


def test_head_multiple_invariant():
    # n_q must be an integer multiple of n_kv
    with pytest.raises(ValueError):
        ModelConfig(n_q_heads=5, n_kv_heads=2)
    with pytest.raises(ValueError):
        ModelConfig(n_q_heads=4, n_kv_heads=3)
    with pytest.raises(ValueError):
        ModelConfig(n_q_heads=4, n_kv_heads=0)
    # MHA / MQA / GQA all legal
    assert ModelConfig(n_q_heads=4, n_kv_heads=4).n_kv_heads == 4
    assert ModelConfig(n_q_heads=4, n_kv_heads=1).n_kv_heads == 1
    assert ModelConfig(n_q_heads=4, n_kv_heads=2).n_kv_heads == 2


def test_rope_even_dimension_required():
    RoPE(head_dim=8)
    with pytest.raises(ValueError):
        RoPE(head_dim=7)
    with pytest.raises(ValueError):
        RoPE(head_dim=0)


def test_rope_frequency_formula():
    # theta_k = base ** (-2k / d_head), explicitly verified against the table
    d, base = 8, 10000.0
    rope = RoPE(d, base=base)
    for k in range(d // 2):
        expected = base ** (-2.0 * k / d)
        assert torch.allclose(rope.inv_freq[k], torch.tensor(expected, dtype=rope.inv_freq.dtype))


def test_rope_rotation_is_planar_and_position0_identity():
    d = 8
    rope = RoPE(d)
    x = torch.randn(1, 1, 2, d)
    pos0 = torch.zeros(1, 1, dtype=torch.long)
    out = rope.rotate(x, pos0)
    assert torch.allclose(out, x, atol=1e-6)

    # 90-degree rotation at phi = pi/2 on pair k=0 (theta_0 = 1)
    rope_unit = RoPE(8, base=10000.0)
    pos = torch.tensor([[int(round(math.pi / 2))]])
    # exact angle through the formula, not via integer rounding:
    angle = torch.tensor([[[[math.pi / 2]]]])
    # build a synthetic check using rotate with position p where p*theta0=pi/2
    # theta0 = 1, so choose p analytically instead: verify algebra on cos/sin.
    cos, sin = rope_unit._table(torch.tensor([math.pi / 2]), torch.device("cpu"), torch.float32)
    assert torch.allclose(cos[0, 0], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(sin[0, 0], torch.tensor(1.0), atol=1e-6)


def test_rope_interleaved_pair_convention():
    # (x0,x1) pair rotates together with angle p (theta_0=1), so at p=1
    # [x0,x1] -> [cos1*x0 - sin1*x1, sin1*x0 + cos1*x1].
    rope = RoPE(4)
    x = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])  # [B=1,T=1,H=1,D=4]
    out = rope.rotate(x, torch.tensor([[1]]))
    c, s = math.cos(1.0), math.sin(1.0)
    # second pair theta_1 = 10000**(-1/2) != 1; check the first pair only
    assert torch.allclose(out[0, 0, 0, :2], torch.tensor([c, s]), atol=1e-5)
    # and the second pair rotates by its own smaller angle
    theta1 = 10000.0 ** (-2.0 * 1 / 4)
    assert torch.allclose(
        out[0, 0, 0, 2:],
        torch.tensor([math.cos(theta1), math.sin(theta1)]),
        atol=1e-5,
    )


def test_rope_absolute_positions_and_offset():
    # Rotating at absolute position p must equal rotating at p0 then (p-p0):
    # R(p1)R(p0) x == R(p0+p1) x because the same pair rotates by (p0+p1)*theta.
    rope = RoPE(8)
    x = torch.randn(1, 1, 1, 8)
    p3 = rope.rotate(x, torch.tensor([[3]]))
    p1a = rope.rotate(x, torch.tensor([[1]]))
    p1b = rope.rotate(p1a, torch.tensor([[2]]))
    assert torch.allclose(p3, p1b, atol=1e-5)


def test_causal_model_future_independence(tiny_configs):
    # logits at row t cannot depend on tokens > t: perturb a future token.
    cfg = tiny_configs["gqa"]
    model = TinyModel(cfg, seed=5).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    base = model.forward_full_prefix(ids)
    perturbed = ids.clone()
    perturbed[0, 5] = (perturbed[0, 5] + 7) % cfg.vocab_size
    after = model.forward_full_prefix(perturbed)
    # rows 0..4 must be unchanged; rows >=5 may change
    assert torch.allclose(base[0, :5], after[0, :5], atol=1e-6)


def test_param_change_changes_fingerprint(tiny_configs):
    cfg = tiny_configs["gqa"]
    m1 = TinyModel(cfg, seed=1)
    fp1 = m1.fingerprint()
    m2 = TinyModel(cfg, seed=2)
    assert fp1 != m2.fingerprint()
    with torch.no_grad():
        p = next(m1.parameters())
        p.add_(torch.randn_like(p) * 0.01)
    assert m1.fingerprint() != fp1
    # same seed + same config -> identical fingerprint
    assert TinyModel(cfg, seed=1).fingerprint() == fp1
