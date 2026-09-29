"""A tiny decoder-only transformer with grouped-query attention.

Layer convention (pre-norm residual)::

    h <- h + Attention(RMSNorm(h))
    h <- h + FFN(RMSNorm(h))
    logits = h @ W_lm_head^T

The attention projection is a fused QKV projection; ``n_q_heads`` must be an
integer multiple of ``n_kv_heads``:

* n_q == n_kv -> multi-head attention (MHA)
* n_kv == 1   -> multi-query attention (MQA)
* 1 < n_kv < n_q (and divides n_q) -> grouped-query attention (GQA)

KV heads are repeated by *consecutive* groups::

    q_head i attends to kv_head i // (n_q // n_kv)

All attention math lives in :func:`masked_attention`; it scores exactly the
positions allowed by the caller-supplied boolean visibility matrix, so the
prefill causal mask, the chunked-decode "history + earlier-in-chunk" mask and
the stateless full-prefix reference all run through one implementation.
"""

from __future__ import annotations

import hashlib
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RoPE


class ModelConfig:
    """Structural + size configuration of the tiny decoder."""

    def __init__(
        self,
        vocab_size: int = 257,
        d_model: int = 64,
        n_layers: int = 2,
        n_q_heads: int = 4,
        n_kv_heads: int = 2,
        ffn_mult: int = 2,
        rope_base: float = 10000.0,
        max_position: int = 4096,
    ):
        if d_model % n_q_heads != 0:
            raise ValueError("d_model must be divisible by n_q_heads")
        if n_kv_heads < 1 or n_q_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_q_heads ({n_q_heads}) must be a positive integer multiple "
                f"of n_kv_heads ({n_kv_heads})"
            )
        self.vocab_size = int(vocab_size)
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_q_heads = int(n_q_heads)
        self.n_kv_heads = int(n_kv_heads)
        self.head_dim = d_model // n_q_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim (d_model // n_q_heads) must be even for RoPE")
        self.ffn_hidden = int(ffn_mult * d_model)
        self.rope_base = float(rope_base)
        self.max_position = int(max_position)

    def signature(self) -> dict:
        """Structural fingerprint fields (cache is incompatible if changed)."""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_q_heads": self.n_q_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "ffn_hidden": self.ffn_hidden,
            "rope_base": self.rope_base,
        }


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        qkv_out = (cfg.n_q_heads + 2 * cfg.n_kv_heads) * cfg.head_dim
        self.n_q_heads = cfg.n_q_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.qkv_proj = nn.Linear(cfg.d_model, qkv_out, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_norm = RMSNorm(cfg.d_model)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def project_qkv(self, x: torch.Tensor):
        b, t, _ = x.shape
        q_len = self.n_q_heads * self.head_dim
        kv_len = self.n_kv_heads * self.head_dim
        qkv = self.qkv_proj(x)
        q = qkv[..., :q_len].view(b, t, self.n_q_heads, self.head_dim)
        k = qkv[..., q_len:q_len + kv_len].view(b, t, self.n_kv_heads, self.head_dim)
        v = qkv[..., q_len + kv_len:].view(b, t, self.n_kv_heads, self.head_dim)
        return q, k, v

    def attention_out(self, attn: torch.Tensor) -> torch.Tensor:
        b, t, _ = attn.shape
        return self.o_proj(attn)

    def ffn(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    n_kv_heads: int,
    scale: float,
    allow: torch.Tensor,
    group_mode: str = "consecutive",
) -> torch.Tensor:
    """Scaled dot-product GQA with an explicit boolean visibility matrix.

    Args:
        q: [B, Tq, Hq, D] (already RoPE-rotated).
        k: [B, Tk, Hkv, D] (already RoPE-rotated, same rule as q).
        v: [B, Tk, Hkv, D].
        allow: bool [B, Tq, Tk]; True means query may see that key. Rows that
            are all False (padded queries) produce zero output.

    Returns:
        [B, Tq, Hq * D], zero on all-False query rows.
    """
    b, tq, hq, d = q.shape
    hkv = n_kv_heads
    group_size = hq // hkv
    if group_mode == "consecutive":
        rep = group_size
        k_r = k.repeat_interleave(rep, dim=2)
        v_r = v.repeat_interleave(rep, dim=2)
    elif group_mode == "rotated":
        # Classic wrong grouping: instead of consecutive blocks
        # (q heads 0..g-1 -> kv 0, g..2g-1 -> kv 1, ...) the kv heads are
        # cyclically shifted, so each Q head reads a different K/V stream.
        # Kept only for the drift counterexamples.
        idx = (torch.arange(hq, device=q.device) + 1) % hkv
        k_r = k[:, :, idx, :]
        v_r = v[:, :, idx, :]
    else:
        raise ValueError(group_mode)

    qh = q.permute(0, 2, 1, 3)
    kh = k_r.permute(0, 2, 1, 3)
    vh = v_r.permute(0, 2, 1, 3)
    scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale
    mask = allow.unsqueeze(1)
    scores = scores.masked_fill(~mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    out = torch.matmul(probs, vh)
    out = out.permute(0, 2, 1, 3).reshape(b, tq, hq * d)
    return out


class TinyModel(nn.Module):
    """The full decoder; owns RoPE, embeddings, layers and the LM head."""

    def __init__(self, cfg: ModelConfig, seed: int = 0, init_std: float = 0.02):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.rope = RoPE(cfg.head_dim, base=cfg.rope_base)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.reset_parameters(seed, init_std)

    def reset_parameters(self, seed: int = 0, init_std: float = 0.02) -> None:
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for p in self.parameters():
                if p.dim() >= 2:
                    p.normal_(0.0, init_std, generator=g)
                else:
                    p.zero_()
            for layer in self.layers:
                layer.attn_norm.weight.fill_(1.0)
                layer.ffn_norm.weight.fill_(1.0)
            self.norm.weight.fill_(1.0)

    @property
    def scale(self) -> float:
        return self.cfg.head_dim ** -0.5

    @torch.no_grad()
    def fingerprint(self) -> str:
        """Version id: structural config + exact parameter byte contents.

        Any parameter change (re-init, optimizer step, manual edit) yields a
        different fingerprint; the KV pool refuses to serve old caches.
        """
        h = hashlib.sha256()
        h.update(json.dumps(self.cfg.signature(), sort_keys=True).encode())
        for name, p in sorted(self.named_parameters(), key=lambda kv: kv[0]):
            h.update(name.encode())
            h.update(p.detach().to(torch.float32).cpu().numpy().tobytes())
        return h.hexdigest()

    def forward_full_prefix(self, token_ids: torch.Tensor, base_pos: int = 0) -> torch.Tensor:
        """Stateless causal forward over one full prefix [B, T] -> logits.

        This is the independent reference implementation: it recomputes the
        whole valid prefix from scratch, with no cache at all.
        """
        b, t = token_ids.shape
        x = self.tok_emb(token_ids)
        positions = (torch.arange(t, device=token_ids.device) + base_pos).unsqueeze(0).expand(b, t)
        allow = torch.tril(torch.ones(t, t, dtype=torch.bool, device=token_ids.device))
        allow = allow.unsqueeze(0).expand(b, t, t)
        for layer in self.layers:
            residual = x
            h = layer.attn_norm(x)
            q, k, v = layer.project_qkv(h)
            q = self.rope.rotate(q, positions)
            k = self.rope.rotate(k, positions)
            attn = masked_attention(
                q, k, v,
                n_kv_heads=self.cfg.n_kv_heads,
                scale=self.scale,
                allow=allow,
            )
            x = residual + layer.attention_out(attn)
            residual = x
            x = residual + layer.ffn(layer.ffn_norm(x))
        return self.lm_head(self.norm(x))
