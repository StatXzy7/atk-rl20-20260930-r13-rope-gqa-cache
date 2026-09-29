"""Paged KV cache with reference-counted sharing and copy-on-write fork.

Storage layout
--------------
The cache is a pool of ``num_blocks`` blocks.  Each block holds
``block_size`` token slots of K **and** V for **all** layers, i.e. block id
``b`` addresses slot ``b`` in every per-layer tensor.  A block's footprint is::

    block_bytes = n_layers * 2 (K,V) * block_size * n_kv_heads * head_dim * itemsize

A sequence is a *block table* (ordered list of block ids) plus a ``length``
(number of valid cached tokens).  Token ``t`` of a sequence lives in block
``table[t // block_size]`` at offset ``t % block_size``, for every layer.
Only K that has already been RoPE-rotated with its absolute position is ever
written, together with its V.

Ownership / copy-on-write
-------------------------
Blocks are reference counted.  ``fork`` copies the block *table* and bumps
refcounts -- O(#blocks), no data copy.  A sequence may only mutate a block it
solely owns: when an append would write into a partially-filled last block
that is still shared (refcount > 1), that block is first copied into a
private block (copy-on-write) and the shared one is released.  Full blocks of
a shared prefix are therefore immutable in practice and freeing one branch
only decrements refcounts; the shared prefix survives as long as any branch
references it.

Capacity
--------
``required_blocks`` computes exactly how many fresh blocks an append needs
(including a possible COW copy).  ``append`` checks the free-list *before*
mutating anything and raises :class:`CacheCapacityError` on shortage, so a
failed append never partially writes tokens.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


class CacheCapacityError(RuntimeError):
    """Raised when an append would exceed the pool; nothing is modified."""


class CacheInvalidError(RuntimeError):
    """Raised when a cache is used with a model version it was not built for."""


@dataclass(frozen=True)
class CacheSpec:
    n_layers: int
    n_kv_heads: int
    head_dim: int
    block_size: int
    dtype: torch.dtype

    @property
    def token_bytes(self) -> int:
        """Bytes of one token slot (K+V, all layers)."""
        return self.n_layers * 2 * self.n_kv_heads * self.head_dim * self.dtype.itemsize


@dataclass
class _SeqState:
    table: list[int]
    length: int
    pos_offset: int


class KVCacheManager:
    def __init__(self, spec: CacheSpec, num_blocks: int, model_version: str):
        if num_blocks < 1:
            raise ValueError("num_blocks must be >= 1")
        self.spec = spec
        self.num_blocks = num_blocks
        self.model_version = model_version
        s = spec
        self.k = [torch.zeros(num_blocks, s.block_size, s.n_kv_heads, s.head_dim, dtype=s.dtype)
                  for _ in range(s.n_layers)]
        self.v = [torch.zeros(num_blocks, s.block_size, s.n_kv_heads, s.head_dim, dtype=s.dtype)
                  for _ in range(s.n_layers)]
        self.refcount = [0] * num_blocks
        self._free_ids = list(range(num_blocks))
        self._seqs: dict[int, _SeqState] = {}
        self._next_id = 0

    # ------------------------------------------------------------------ bytes
    @property
    def block_bytes(self) -> int:
        s = self.spec
        return s.n_layers * 2 * s.block_size * s.n_kv_heads * s.head_dim * self.k[0].element_size()

    @property
    def nbytes(self) -> int:
        """Total bytes of the pool (allocated + free)."""
        return self.num_blocks * self.block_bytes

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_ids)

    @property
    def allocated_bytes(self) -> int:
        return (self.num_blocks - len(self._free_ids)) * self.block_bytes

    def sequence_bytes(self, seq_id: int) -> int:
        """Logical bytes of one sequence (shared blocks counted per holder)."""
        return len(self._seq(seq_id).table) * self.block_bytes

    # -------------------------------------------------------------- sequences
    def _seq(self, seq_id: int) -> _SeqState:
        try:
            return self._seqs[seq_id]
        except KeyError:
            raise KeyError(f"unknown or freed sequence id {seq_id}") from None

    def alloc_sequence(self, pos_offset: int = 0) -> int:
        if pos_offset < 0:
            raise ValueError("pos_offset must be >= 0")
        seq_id = self._next_id
        self._next_id += 1
        self._seqs[seq_id] = _SeqState(table=[], length=0, pos_offset=pos_offset)
        return seq_id

    def fork(self, seq_id: int) -> int:
        """Share the prefix with a new branch (refcount bump, no data copy)."""
        src = self._seq(seq_id)
        new_id = self._next_id
        self._next_id += 1
        self._seqs[new_id] = _SeqState(table=list(src.table), length=src.length,
                                       pos_offset=src.pos_offset)
        for b in src.table:
            self.refcount[b] += 1
        return new_id

    def free(self, seq_id: int) -> None:
        st = self._seq(seq_id)
        for b in st.table:
            self._release(b)
        del self._seqs[seq_id]

    def length(self, seq_id: int) -> int:
        return self._seq(seq_id).length

    def pos_offset(self, seq_id: int) -> int:
        return self._seq(seq_id).pos_offset

    def block_table(self, seq_id: int) -> list[int]:
        return list(self._seq(seq_id).table)

    # ------------------------------------------------------------------ blocks
    def _alloc_block(self) -> int:
        b = self._free_ids.pop()
        self.refcount[b] = 1
        return b

    def _release(self, b: int) -> None:
        self.refcount[b] -= 1
        if self.refcount[b] == 0:
            self._free_ids.append(b)

    def required_blocks(self, seq_id: int, n_tokens: int) -> int:
        """Fresh blocks an append of ``n_tokens`` would need (incl. COW copy)."""
        st = self._seq(seq_id)
        if n_tokens < 0:
            raise ValueError("n_tokens must be >= 0")
        if n_tokens == 0:
            return 0
        total = st.length + n_tokens
        need_total = math.ceil(total / self.spec.block_size)
        extra = max(0, need_total - len(st.table))
        cow = 1 if (st.table and st.length % self.spec.block_size != 0
                    and self.refcount[st.table[-1]] > 1) else 0
        return extra + cow

    # ------------------------------------------------------------------- IO
    def append(self, seq_id: int, k_layers: list[torch.Tensor],
               v_layers: list[torch.Tensor], n_tokens: int) -> None:
        """Append ``n_tokens`` rotated K / V pairs (all-or-nothing).

        ``k_layers[li]`` / ``v_layers[li]``: ``[n_tokens, n_kv_heads, head_dim]``.
        Raises :class:`CacheCapacityError` before any mutation if the pool
        cannot hold the append.
        """
        st = self._seq(seq_id)
        if len(k_layers) != self.spec.n_layers or len(v_layers) != self.spec.n_layers:
            raise ValueError("expected one K/V tensor per layer")
        need = self.required_blocks(seq_id, n_tokens)
        if need > len(self._free_ids):
            raise CacheCapacityError(
                f"append of {n_tokens} token(s) to seq {seq_id} needs {need} "
                f"block(s), only {len(self._free_ids)} free; nothing was written"
            )
        if n_tokens == 0:
            return
        bs = self.spec.block_size
        # Copy-on-write: never write into a shared partially-filled block.
        if st.table and st.length % bs != 0 and self.refcount[st.table[-1]] > 1:
            old = st.table[-1]
            new = self._alloc_block()
            for li in range(self.spec.n_layers):
                self.k[li][new] = self.k[li][old]
                self.v[li][new] = self.v[li][old]
            self._release(old)
            st.table[-1] = new
        # Allocate any further blocks needed to hold `length + n_tokens`.
        while math.ceil((st.length + n_tokens) / bs) > len(st.table):
            st.table.append(self._alloc_block())
        # Scatter the tokens into their slots.
        pos = torch.arange(st.length, st.length + n_tokens)
        blocks = torch.tensor(st.table, dtype=torch.long)[pos // bs]
        offs = pos % bs
        for li in range(self.spec.n_layers):
            self.k[li][blocks, offs] = k_layers[li].to(self.k[li].dtype)
            self.v[li][blocks, offs] = v_layers[li].to(self.v[li].dtype)
        st.length += n_tokens

    def gather(self, seq_id: int, layer: int):
        """Return ``(K, V)`` of the valid prefix, each ``[length, Hkv, D]``.

        K comes back exactly as stored, i.e. already RoPE-rotated.
        """
        st = self._seq(seq_id)
        if st.length == 0:
            s = self.spec
            empty = torch.zeros(0, s.n_kv_heads, s.head_dim, dtype=s.dtype)
            return empty, empty
        idx = torch.tensor(st.table, dtype=torch.long)
        k = self.k[layer][idx].reshape(-1, self.spec.n_kv_heads, self.spec.head_dim)[: st.length]
        v = self.v[layer][idx].reshape(-1, self.spec.n_kv_heads, self.spec.head_dim)[: st.length]
        return k, v
