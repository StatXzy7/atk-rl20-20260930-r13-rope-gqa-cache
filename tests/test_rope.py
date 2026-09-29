import torch
import pytest

from mini_decoder import ModelConfig, rope


def test_rotate_matches_complex_definition():
    torch.manual_seed(0)
    d, base = 8, 10000.0
    x = torch.randn(3, 5, d, dtype=torch.float64)
    positions = torch.arange(5)[None, :].expand(3, 5)
    got = rope.rotate(x, positions, base)

    # Independent restatement: treat pairs as complex numbers z_i and
    # multiply by exp(j * m * theta_i).
    theta = base ** (-2.0 * torch.arange(d // 2, dtype=torch.float64) / d)
    angles = positions.to(torch.float64)[..., None] * theta  # [B, S, d/2]
    z = torch.complex(x[..., 0::2], x[..., 1::2])
    rot = z * torch.exp(torch.complex(torch.zeros_like(angles), angles))
    want = torch.stack([rot.real, rot.imag], dim=-1).flatten(-2)
    torch.testing.assert_close(got, want, atol=1e-12, rtol=1e-12)


def test_frequency_formula():
    d, base = 8, 10000.0
    theta = rope.inv_freq(d, base, torch.float64, torch.device("cpu"))
    want = torch.tensor([base ** (-2.0 * i / d) for i in range(d // 2)], dtype=torch.float64)
    torch.testing.assert_close(theta, want)
    assert theta[0].item() == 1.0  # i=0 -> base**0


def test_rotation_preserves_norm():
    x = torch.randn(2, 7, 8, dtype=torch.float64)
    positions = torch.arange(7)[None, :].expand(2, 7)
    y = rope.rotate(x, positions, 10000.0)
    torch.testing.assert_close(y.norm(dim=-1), x.norm(dim=-1), atol=1e-12, rtol=1e-12)


def test_relative_position_property():
    # <R_m q, R_n k> == <R_{m-n} q, k>: attention scores depend on distance.
    torch.manual_seed(1)
    q = torch.randn(8, dtype=torch.float64)
    k = torch.randn(8, dtype=torch.float64)
    for m, n in [(0, 0), (5, 2), (11, 7), (100, 3)]:
        lhs = rope.rotate(q, torch.tensor(m), 10000.0) @ rope.rotate(k, torch.tensor(n), 10000.0)
        rhs = rope.rotate(q, torch.tensor(m - n), 10000.0) @ k
        torch.testing.assert_close(lhs, rhs, atol=1e-10, rtol=1e-10)


def test_odd_head_dim_rejected():
    with pytest.raises(ValueError, match="even"):
        rope.rotate(torch.randn(4, 5), torch.tensor(0), 10000.0)
    with pytest.raises(ValueError, match="even"):
        ModelConfig(head_dim=7).validate()


def test_head_config_validation():
    ModelConfig(n_heads=4, n_kv_heads=4).validate()  # MHA
    ModelConfig(n_heads=4, n_kv_heads=2).validate()  # GQA
    ModelConfig(n_heads=4, n_kv_heads=1).validate()  # MQA
    with pytest.raises(ValueError, match="integer multiple"):
        ModelConfig(n_heads=5, n_kv_heads=2).validate()
    with pytest.raises(ValueError, match="integer multiple"):
        ModelConfig(n_heads=4, n_kv_heads=3).validate()
