"""Rotary position embedding (RoPE).

Definition (used identically by prefill, per-token decode and chunked decode):

* The per-head feature dimension ``d_head`` is split into ``d_head / 2``
  rotation pairs ``(x_0, x_1), (x_2, x_3), ...`` using the "interleaved
  half-split": index 2*k is paired with index 2*k + 1.
* Every rotation pair has an *even* dimension: it rotates exactly two
  coordinates, so the head dimension must be even.
* Frequencies are indexed by pair index ``k = 0 .. d_head/2 - 1``::

      theta_k = base ** (-2 * k / d_head)
  (equivalently ``1 / (base ** (2*k / d_head))``; the common Neox-style
  rewrite ``inv_freq = base ** (-2k / d_head)`` is exact, not approximate).
* Absolute position numbering: the token that is ``t`` tokens after the
  (optional) position offset gets angle ``phi_{t,k} = (t + base_pos) *
  theta_k``. Positions are *absolute* positions in the logical sequence,
  not indices inside the current request, so decode continues exactly where
  prefill stopped.

* The rotation is::

      [x_{2k}  ] -> [ cos(phi) * x_{2k} - sin(phi) * x_{2k+1} ]
      [x_{2k+1}]    [ sin(phi) * x_{2k} + cos(phi) * x_{2k+1} ]

Only the sign convention (this one) must stay fixed; the *cache path* applies
the identical function ``rotate(q/k, position)`` to both paths.
"""

from __future__ import annotations

import torch


class RoPE:
    """Stateless RoPE table with ``d_head`` even and absolute positions."""

    def __init__(self, head_dim: int, base: float = 10000.0, device=None, dtype=torch.float32):
        if head_dim % 2 != 0 or head_dim < 2:
            raise ValueError(f"head_dim must be a positive even number, got {head_dim}")
        self.head_dim = head_dim
        self.base = float(base)
        k = torch.arange(0, head_dim // 2, device=device, dtype=torch.float64)
        inv_freq = self.base ** (-2.0 * k / head_dim)
        self.inv_freq = inv_freq.to(dtype)

    def _table(self, positions: torch.Tensor, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) of shape [len(positions), head_dim/2]."""
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)
        pos = positions.to(device=device, dtype=torch.float32)
        angle = pos[:, None] * inv_freq[None, :]
        return angle.cos().to(dtype), angle.sin().to(dtype)

    def rotate(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rotate ``x`` in place-free functional style.

        Args:
            x: tensor of shape ``[B, T, H, d_head]``.
            positions: integer tensor broadcastable to ``[B, T]``. Entries at
                padded positions are ignored when ``valid`` is given.
            valid: optional bool tensor ``[B, T]``; rows that are False are
                copied through untouched (they never reach the scores).

        Returns:
            Rotated tensor with the same shape.
        """
        b, t, h, d = x.shape
        if d != self.head_dim:
            raise ValueError(f"last dim {d} != head_dim {self.head_dim}")
        positions = positions.to(x.device)
        if positions.shape != (b, t):
            positions = positions.expand(b, t).contiguous()
        pos_flat = positions.reshape(-1)
        cos, sin = self._table(pos_flat, x.device, x.dtype)
        half = d // 2
        x2 = x.reshape(b, t, h, half, 2)
        x0 = x2[..., 0]
        x1 = x2[..., 1]
        rot0 = cos.view(b, t, 1, half) * x0 - sin.view(b, t, 1, half) * x1
        rot1 = sin.view(b, t, 1, half) * x0 + cos.view(b, t, 1, half) * x1
        out = torch.stack((rot0, rot1), dim=-1).reshape(b, t, h, d)
        if valid is not None:
            keep = ~valid.unsqueeze(-1).unsqueeze(-1)
            out = torch.where(keep, x, out)
        return out
