"""Rotary Position Embedding (RoPE) -- the single source of truth for positions.

Definition (used identically by prefill, token decode and chunked decode):

* The per-head dimension ``d`` (``head_dim``) must be **even**.  Dimensions are
  grouped into ``d/2`` pairs ``(2i, 2i+1)`` for ``i = 0 .. d/2 - 1``.
* Frequencies (inverse wavelengths)::

      theta_i = rope_base ** (-2i / d),        i = 0 .. d/2 - 1

* Absolute position ``m``: the 0-based index of a token within its own
  sequence's stream of *valid* tokens, plus an optional per-sequence
  ``pos_offset``.  Padding slots never receive a position and never enter the
  cache.  Prefill, single-token decode and chunked decode all number positions
  with this same rule, so a token's rotation depends only on ``m``, never on
  how the sequence was split into prefill/decode chunks.
* Rotation angle for pair ``i``:  ``phi_i = m * theta_i``.
* The rotation applied to a vector ``x`` (per head) is::

      x'_{2i}   = x_{2i} * cos(phi_i) - x_{2i+1} * sin(phi_i)
      x'_{2i+1} = x_{2i} * sin(phi_i) + x_{2i+1} * cos(phi_i)

  i.e. the complex multiplication ``(x_{2i} + j*x_{2i+1}) * exp(j*phi_i)``.

The KV cache stores K **after** this rotation (and V untouched), so decode
never re-rotates old keys.
"""
from __future__ import annotations

import torch


def inv_freq(head_dim: int, base: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return ``theta_i = base ** (-2i / d)`` for ``i = 0 .. d/2 - 1``."""
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")
    i = torch.arange(0, head_dim // 2, dtype=dtype, device=device)
    return base ** (-2.0 * i / head_dim)


def rotate(x: torch.Tensor, positions: torch.Tensor, base: float) -> torch.Tensor:
    """Apply RoPE to ``x``.

    ``x``: ``[..., d]`` with even ``d``.  ``positions``: integer tensor
    broadcastable to ``x.shape[:-1]`` holding absolute positions ``m``.
    """
    d = x.shape[-1]
    theta = inv_freq(d, base, x.dtype, x.device)
    angles = positions.to(theta.dtype).unsqueeze(-1) * theta  # [..., d/2]
    cos, sin = angles.cos(), angles.sin()
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
    return out.flatten(-2)
