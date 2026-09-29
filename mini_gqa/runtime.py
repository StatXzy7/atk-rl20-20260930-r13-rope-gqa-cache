"""Runtime: prefill, per-token decode, chunked decode, and the reference.

All three cache paths share one extend engine and therefore one RoPE rule:

* Each stored token keeps its **absolute** position
  ``base_pos + session.length + offset``. Decode continues numbering exactly
  where prefill stopped; ``base_pos`` allows sequences that logically start
  at a non-zero position.
* Cached K (and V) are written *already rotated* for that absolute position
  and never rotated again. New Q and new K in a request are rotated with the
  identical :class:`RoPE` call used by the stateless reference.
* Visibility:

  - prefill:        token j sees new tokens 0..j (strict causal), no history;
  - decode (1 tok): sees all history + itself;
  - chunked decode: sees all history + new tokens 0..j
                    (never later tokens inside the same chunk).

Padding rows never attend and their outputs are discarded, so ragged batches
(sequences of different lengths in one request) cannot bleed into each other.
"""

from __future__ import annotations

import torch

from .cache import CacheCapacityError, KVCacheError, StaleCacheError
from .model import masked_attention

__all__ = [
    "prefill",
    "decode",
    "chunked_decode",
    "reference_logits",
    "CacheCapacityError",
    "StaleCacheError",
    "KVCacheError",
]


def _as_list(sessions):
    if not isinstance(sessions, (list, tuple)):
        return [sessions]
    return list(sessions)


def _extend(
    pool,
    sessions,
    token_ids,
    lens=None,
    *,
    mode,
    _position_bug=False,
    _restart_position=False,
    _leak_future=False,
    _group_mode="consecutive",
    _rotate_new_k=True,
):
    """Batched extend engine. Underscore kwargs inject the classic mistakes
    used by the drift counterexample tests; production callers never set them."""
    model = pool.model
    cfg = model.cfg
    device = pool.device
    sessions = _as_list(sessions)
    token_ids = torch.as_tensor(token_ids, device=device)
    if token_ids.dim() == 1:
        token_ids = token_ids.unsqueeze(0)
    b, width = token_ids.shape
    if b != len(sessions):
        raise ValueError("batch size mismatch between sessions and token_ids")
    if lens is None:
        lens = [width] * b if mode == "prefill" else [1] * b
    lens = [int(n) for n in lens]
    if any(n <= 0 or n > width for n in lens):
        raise ValueError("each lens value must be in [1, token width]")

    # 1) Atomic reservation, then COW + block growth (raises before mutation).
    pool.reserve_extend(sessions, lens)
    pool.commit_extend(sessions, lens)

    # 2) Absolute positions and validity matrix for this request.
    positions = torch.zeros((b, width), dtype=torch.long, device=device)
    valid = torch.zeros((b, width), dtype=torch.bool, device=device)
    hist_lens = [s.length for s in sessions]
    for i, (s, nnew) in enumerate(zip(sessions, lens)):
        valid[i, :nnew] = True
        abs_pos = torch.arange(s.length, s.length + nnew, device=device) + s.base_pos
        if _restart_position and mode != "prefill":
            abs_pos = torch.arange(nnew, device=device)  # restart numbering at 0
        elif _position_bug and mode != "prefill":
            abs_pos = abs_pos - 1  # classic off-by-one drift
        positions[i, :nnew] = abs_pos

    safe_tokens = torch.where(valid, token_ids, torch.zeros_like(token_ids))
    x = model.tok_emb(safe_tokens)

    total_k = [h + n for h, n in zip(hist_lens, lens)]
    tmax = max(total_k)
    for lidx, layer in enumerate(model.layers):
        h_norm = layer.attn_norm(x)
        q_new, k_new, v_new = layer.project_qkv(h_norm)
        q_new = model.rope.rotate(q_new, positions, valid=valid)
        k_new_rot = model.rope.rotate(k_new, positions, valid=valid)
        if not _rotate_new_k:
            k_new_rot = torch.where(
                valid[:, :, None, None], k_new, k_new_rot
            )

        # 3) Persist already-rotated K (and V), then gather each history.
        for i, s in enumerate(sessions):
            nnew = lens[i]
            pool.write_new(s, lidx, 0, k_new_rot[i, :nnew], v_new[i, :nnew])
            if lidx == 0:
                s.stored_ids.extend(token_ids[i, :nnew].tolist())

        k_batch = torch.zeros((b, tmax, cfg.n_kv_heads, cfg.head_dim),
                              dtype=q_new.dtype, device=device)
        v_batch = torch.zeros_like(k_batch)
        for i, s in enumerate(sessions):
            khist, vhist = pool.gather_history(s, lidx, length=hist_lens[i] + lens[i])
            k_batch[i, :khist.shape[0]] = khist
            v_batch[i, :vhist.shape[0]] = vhist

        # 4) Visibility: full history + strict-causal within the new block.
        allow = torch.zeros((b, width, tmax), dtype=torch.bool, device=device)
        for i, nnew in enumerate(lens):
            hist = hist_lens[i]
            allow[i, :nnew, :hist] = True
            sub = torch.ones((nnew, nnew), dtype=torch.bool, device=device)
            if not _leak_future:
                sub = torch.tril(sub)
            allow[i, :nnew, hist:hist + nnew] = sub

        attn = masked_attention(
            q_new, k_batch, v_batch,
            n_kv_heads=cfg.n_kv_heads,
            scale=model.scale,
            allow=allow,
            group_mode=_group_mode,
        )
        attn = torch.where(valid.unsqueeze(-1), attn, torch.zeros_like(attn))
        x = x + layer.attention_out(attn)
        residual = x
        x = residual + layer.ffn(layer.ffn_norm(x))

    x = torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x))
    logits = model.lm_head(model.norm(x))
    pool.finish_extend(sessions, lens)
    return logits, valid, lens


def prefill(pool, sessions, token_ids, lens=None, **faults):
    """Causal prefill into empty sessions; returns (logits, valid, lens)."""
    sessions = _as_list(sessions)
    for s in sessions:
        pool._require_alive(s)
        if s.length != 0:
            raise KVCacheError(f"prefill requires an empty session, got length {s.length}")
    return _extend(pool, sessions, token_ids, lens, mode="prefill", **faults)


def decode(pool, sessions, token_ids, lens=None, **faults):
    """Extend non-empty sessions. ``lens=None`` means one new token per row;
    ``lens`` widths > 1 is chunked decode (history + earlier chunk tokens)."""
    sessions = _as_list(sessions)
    for s in sessions:
        pool._require_alive(s)
        if s.length == 0:
            raise KVCacheError("decode requires a non-empty session; call prefill first")
    return _extend(pool, sessions, token_ids, lens, mode="decode", **faults)


def chunked_decode(pool, sessions, token_ids, lens, **faults):
    """Explicit chunked decode with per-row chunk widths ``lens``."""
    return decode(pool, sessions, token_ids, lens=lens, **faults)


@torch.no_grad()
def reference_logits(model, session, cont_ids, last_only: bool = True):
    """Independent reference: recompute the entire valid prefix from scratch.

    Concatenates the stored prefix ids with ``cont_ids``, numbers every token
    by its absolute position ``base_pos + i``, and runs a full stateless
    causal forward (no cache). Returns the continuation rows (or just the last
    row). This is the ground truth every cached step must match.
    """
    if not session.alive:
        raise KVCacheError("reference_logits on released session")
    if session.fingerprint != model.fingerprint():
        raise StaleCacheError("reference_logits: model parameters changed vs session")
    device = next(model.parameters()).device
    cont_ids = torch.as_tensor(cont_ids, device=device).reshape(-1)
    prefix = torch.as_tensor(session.stored_ids, device=device, dtype=torch.long)
    full = torch.cat([prefix, cont_ids]) if prefix.numel() else cont_ids
    out = model.forward_full_prefix(full.unsqueeze(0), base_pos=session.base_pos)[0]
    if prefix.numel():
        out = out[prefix.numel():]
    return out[-1:] if last_only else out
