"""A tiny autoregressive decoder with RoPE and grouped-query attention (GQA).

Head layout: ``n_heads`` query heads, ``n_kv_heads`` key/value heads, with
``n_heads % n_kv_heads == 0``.  Query head ``h`` is served by KV head
``h // (n_heads // n_kv_heads)`` (contiguous grouping).  MHA is the special
case ``n_kv_heads == n_heads``, MQA is ``n_kv_heads == 1``.

The model only computes projections / norms / MLPs.  All cache bookkeeping
lives in :mod:`mini_decoder.cache` and :mod:`mini_decoder.engine`.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
from torch import nn

from . import rope


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 61
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 2
    head_dim: int = 8
    rope_base: float = 10000.0
    mlp_ratio: int = 2

    @property
    def d_model(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def n_groups(self) -> int:
        return self.n_heads // self.n_kv_heads

    def validate(self) -> None:
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {self.head_dim}")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be an integer multiple of "
                f"n_kv_heads ({self.n_kv_heads})"
            )
        if self.n_layers < 1 or self.n_heads < 1 or self.n_kv_heads < 1:
            raise ValueError("n_layers / n_heads / n_kv_heads must be positive")


class GQAAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def project_qkv(self, x: torch.Tensor, positions: torch.Tensor):
        """Project and RoPE-rotate.  Returns q [B,H,S,D], k/v [B,Hkv,S,D].

        The returned K is already rotated with the absolute ``positions`` and
        is exactly what gets written into the KV cache.
        """
        cfg = self.cfg
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        pos = positions[:, None, :]  # [B, 1, S] broadcasts over heads
        q = rope.rotate(q, pos, cfg.rope_base)
        k = rope.rotate(k, pos, cfg.rope_base)
        return q, k, v

    def attend(self, q, k_all, v_all, allowed):
        """Scaled-dot-product attention over ``k_all``/``v_all``.

        ``k_all``/``v_all``: [B, Hkv, L, D] ordered by ascending absolute
        position (cached prefix first, then the current chunk).
        ``allowed``: bool [B, S, L]; ``allowed[b, s, j]`` means query ``s`` of
        sequence ``b`` may attend key ``j``.
        """
        cfg = self.cfg
        B, H, S, D = q.shape
        k = k_all.repeat_interleave(cfg.n_groups, dim=1)  # [B, H, L, D]
        v = v_all.repeat_interleave(cfg.n_groups, dim=1)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(D)  # [B, H, S, L]
        scores = scores.masked_fill(~allowed[:, None, :, :], float("-inf"))
        probs = scores.softmax(dim=-1)
        out = (probs @ v).transpose(1, 2).reshape(B, S, H * D)
        return self.o_proj(out)


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = GQAAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        hidden = cfg.mlp_ratio * cfg.d_model
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, cfg.d_model),
        )

    def forward(self, x, positions, past_kv, allowed):
        """Run one layer.

        ``past_kv``: optional ``(k, v)`` of already-rotated cached tensors
        [B, Hkv, Lpast, D].  Returns ``(x, (k_new, v_new))`` where ``k_new``
        is rotated and ready to be appended to the cache.
        """
        h = self.ln1(x)
        q, k_new, v_new = self.attn.project_qkv(h, positions)
        if past_kv is not None:
            k_all = torch.cat([past_kv[0], k_new], dim=2)
            v_all = torch.cat([past_kv[1], v_new], dim=2)
        else:
            k_all, v_all = k_new, v_new
        x = x + self.attn.attend(q, k_all, v_all, allowed)
        x = x + self.mlp(self.ln2(x))
        return x, (k_new, v_new)


class MiniDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig, seed: int = 0, init_std: float = 0.05):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(DecoderLayer(cfg) for _ in range(cfg.n_layers))
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            for p in self.parameters():
                nn.init.normal_(p, std=init_std)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.final_norm(x))

    def fingerprint(self) -> str:
        """Content hash of the config and every parameter tensor.

        Any parameter mutation changes the fingerprint, which is how stale
        KV caches are detected (see :class:`mini_decoder.engine.Engine`).
        """
        h = hashlib.sha256()
        h.update(repr(self.cfg).encode())
        for name, p in sorted(self.named_parameters()):
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
