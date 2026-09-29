"""Independent no-cache reference: recompute the whole valid prefix.

This is the ground truth every cached step is checked against.  It never
touches the KV cache: each call runs the full prefix through the model with a
plain causal mask and absolute RoPE positions.
"""
from __future__ import annotations

import torch

from .model import MiniDecoder


@torch.no_grad()
def reference_logits(model: MiniDecoder, tokens: list[int], pos_offset: int = 0) -> torch.Tensor:
    """Logits of every position of ``tokens`` (shape [L, vocab])."""
    ids = torch.tensor(list(tokens), dtype=torch.long).unsqueeze(0)
    L = ids.shape[1]
    positions = (torch.arange(L) + pos_offset).unsqueeze(0)
    allowed = torch.tril(torch.ones(L, L, dtype=torch.bool)).unsqueeze(0)
    x = model.embed(ids)
    for layer in model.layers:
        x, _ = layer(x, positions, None, allowed)
    return model.logits(x)[0]
