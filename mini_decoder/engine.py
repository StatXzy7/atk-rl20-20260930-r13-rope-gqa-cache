"""Batch engine driving prefill, token decode and chunked decode over the KV cache.

All three paths share one code path (:meth:`Engine._step`): prefill is a step
with an empty cache, token decode is a chunk of size 1.  Positions always
follow the absolute numbering defined in :mod:`mini_decoder.rope`, so cached
logits must equal an independent whole-prefix recomputation.
"""
from __future__ import annotations

import torch

from .cache import CacheCapacityError, CacheInvalidError, KVCacheManager
from .model import MiniDecoder


class Engine:
    def __init__(self, model: MiniDecoder, manager: KVCacheManager):
        self.model = model
        self.manager = manager
        self.assert_fresh()

    # --------------------------------------------------------- cache validity
    def assert_fresh(self) -> None:
        """Refuse to run if the model no longer matches the cache's version."""
        fp = self.model.fingerprint()
        if fp != self.manager.model_version:
            raise CacheInvalidError(
                "model parameters/config changed since this KV cache was "
                "created; the old cache is invalid and must not be reused"
            )

    # ------------------------------------------------------------- sequences
    def create_sequence(self, pos_offset: int = 0) -> int:
        self.assert_fresh()
        return self.manager.alloc_sequence(pos_offset=pos_offset)

    def fork(self, seq_id: int) -> int:
        self.assert_fresh()
        return self.manager.fork(seq_id)

    def free(self, seq_id: int) -> None:
        self.manager.free(seq_id)

    # ------------------------------------------------------------------ steps
    @torch.no_grad()
    def prefill(self, seq_ids: list[int], tokens: list[list[int]]) -> list[torch.Tensor]:
        """Run the initial prompt(s).  Returns per-sequence logits [len_b, V]."""
        for s in seq_ids:
            if self.manager.length(s) != 0:
                raise ValueError("prefill requires a freshly created sequence")
        if any(len(t) == 0 for t in tokens):
            raise ValueError("prefill needs at least one token per sequence")
        return self._step(seq_ids, tokens)

    @torch.no_grad()
    def decode(self, seq_ids: list[int], chunks: list[list[int]]) -> list[torch.Tensor]:
        """Append a chunk (size >= 1) per sequence.  Returns logits [len_b, V].

        Within a chunk, query ``i`` attends to the whole cached prefix plus
        chunk tokens ``0..i`` -- never to future tokens of its own chunk.
        """
        if any(len(c) == 0 for c in chunks):
            raise ValueError("decode chunks must be non-empty")
        return self._step(seq_ids, chunks)

    # ------------------------------------------------------------------ core
    def _step(self, seq_ids: list[int], batch_tokens: list[list[int]]) -> list[torch.Tensor]:
        self.assert_fresh()
        model, mgr = self.model, self.manager
        if len(seq_ids) != len(batch_tokens):
            raise ValueError("seq_ids and batch_tokens must have the same length")
        B = len(seq_ids)
        lens = [len(t) for t in batch_tokens]
        past = [mgr.length(s) for s in seq_ids]
        offsets = [mgr.pos_offset(s) for s in seq_ids]
        Smax = max(lens)
        Pmax = max(past) if past else 0
        param = next(model.parameters())
        device, dtype = param.device, param.dtype

        ids = torch.zeros(B, Smax, dtype=torch.long, device=device)
        for b, toks in enumerate(batch_tokens):
            ids[b, : lens[b]] = torch.tensor(toks, dtype=torch.long, device=device)

        arange_s = torch.arange(Smax)
        past_t = torch.tensor(past)
        off_t = torch.tensor(offsets)
        q_pos = off_t[:, None] + past_t[:, None] + arange_s[None, :]  # [B, S]

        # Key layout: [cached prefix (Pmax) | current chunk (Smax)].
        if Pmax > 0:
            past_pos = off_t[:, None] + torch.arange(Pmax)[None, :]
            past_valid = torch.arange(Pmax)[None, :] < past_t[:, None]
            key_pos = torch.cat([past_pos, q_pos], dim=1)
        else:
            past_valid = torch.zeros(B, 0, dtype=torch.bool)
            key_pos = q_pos
        new_valid = arange_s[None, :] < torch.tensor(lens)[:, None]
        key_valid = torch.cat([past_valid, new_valid], dim=1)  # [B, L]

        # Causal visibility in absolute-position space + padding exclusion.
        allowed = key_valid[:, None, :] & (key_pos[:, None, :] <= q_pos[:, :, None])
        pad_row = ~new_valid
        if pad_row.any():  # keep padding query rows finite; they are discarded
            rows = allowed[pad_row]
            rows[:] = False
            rows[:, 0] = True
            allowed[pad_row] = rows
        allowed = allowed.to(device)

        # Gather the cached prefix per layer, zero-padded to Pmax.
        past_kv = None
        if Pmax > 0:
            cfg = model.cfg
            past_kv = []
            for li in range(cfg.n_layers):
                K = torch.zeros(B, cfg.n_kv_heads, Pmax, cfg.head_dim, dtype=dtype, device=device)
                V = torch.zeros_like(K)
                for b, s in enumerate(seq_ids):
                    if past[b] > 0:
                        kb, vb = mgr.gather(s, li)
                        K[b, :, : past[b]] = kb.transpose(0, 1).to(device=device, dtype=dtype)
                        V[b, :, : past[b]] = vb.transpose(0, 1).to(device=device, dtype=dtype)
                past_kv.append((K, V))

        x = model.embed(ids)
        new_k: list[torch.Tensor] = []
        new_v: list[torch.Tensor] = []
        for li, layer in enumerate(model.layers):
            pkv = past_kv[li] if past_kv is not None else None
            x, (k, v) = layer(x, q_pos.to(device), pkv, allowed)
            new_k.append(k)
            new_v.append(v)
        logits = model.logits(x)  # [B, Smax, V]

        # All-or-nothing across the whole batch: preflight capacity first.
        need = sum(mgr.required_blocks(s, n) for s, n in zip(seq_ids, lens))
        if need > mgr.num_free_blocks:
            raise CacheCapacityError(
                f"step needs {need} block(s), only {mgr.num_free_blocks} free; "
                "no sequence was appended"
            )
        for b, s in enumerate(seq_ids):
            n = lens[b]
            k_layers = [new_k[li][b, :, :n].transpose(0, 1).contiguous() for li in range(len(new_k))]
            v_layers = [new_v[li][b, :, :n].transpose(0, 1).contiguous() for li in range(len(new_v))]
            mgr.append(s, k_layers, v_layers, n)

        return [logits[b, : lens[b]].clone() for b in range(B)]
