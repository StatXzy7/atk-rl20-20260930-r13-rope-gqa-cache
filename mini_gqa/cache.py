"""Paged KV cache with reference counting and tail copy-on-write.

Storage model
-------------
* K and V are each stored as one contiguous tensor shaped
  ``[num_blocks, block_size, n_layers, n_kv_heads, head_dim]``.
* A sequence's cache is an ordered list of block ids. Full blocks are
  **frozen**: once a block is full it is never mutated, so it can be shared
  by every branch that owns that prefix.
* Only the *tail* block (partially filled) is mutable. Two branches sharing
  a tail diverge the moment either one writes; that writer copies the tail
  into a fresh block first (copy-on-write) and decrements the old block's
  reference count. The shared immutable prefix stays shared forever.
* Capacity reservation is batch-atomic: a request that would not fit raises
  ``CacheCapacityError`` before *any* allocation or copy happens, so a failed
  multi-sequence append never partially mutates the pool.
* Every session is stamped with its model fingerprint; requests against a
  pool whose parameters changed raise ``StaleCacheError`` instead of reusing
  stale rotated K/V.

Bytes per KV slot (one token, one layer)::

    bytes_per_slot = n_layers * n_kv_heads * head_dim * elem_bytes * 2   # K and V

Total physical bytes = ``num_blocks * block_size * bytes_per_slot``.
"""

from __future__ import annotations

import math

import torch


class KVCacheError(RuntimeError):
    """Base class for cache protocol errors."""


class CacheCapacityError(KVCacheError):
    """Raised when the pool cannot reserve the requested blocks/slots."""


class StaleCacheError(KVCacheError):
    """Raised when a session's cache was built under different parameters."""


class SessionReleasedError(KVCacheError):
    """Raised when an operation targets a released session."""


class Session:
    def __init__(self, sid: int, fingerprint: str, base_pos: int):
        self.sid = sid
        self.fingerprint = fingerprint
        self.base_pos = int(base_pos)
        self.length = 0
        self.blocks: list[int] = []
        self.alive = True
        self.stored_ids: list[int] = []

    def __repr__(self) -> str:
        return (
            f"Session(id={self.sid}, length={self.length}, "
            f"blocks={self.blocks}, alive={self.alive})"
        )


class KVCachePool:
    def __init__(
        self,
        model,
        capacity_tokens: int,
        block_size: int = 8,
        device=None,
        dtype=torch.float32,
    ):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if capacity_tokens <= 0 or capacity_tokens % block_size != 0:
            raise ValueError("capacity_tokens must be positive and divisible by block_size")
        cfg = model.cfg
        self.model = model
        self.cfg = cfg
        self.block_size = int(block_size)
        self.n_blocks = capacity_tokens // block_size
        self.device = device or next(model.parameters()).device
        self.dtype = dtype
        shape = (
            self.n_blocks,
            block_size,
            cfg.n_layers,
            cfg.n_kv_heads,
            cfg.head_dim,
        )
        self.k_backend = torch.empty(shape, dtype=dtype, device=self.device)
        self.v_backend = torch.empty(shape, dtype=dtype, device=self.device)
        self._refcounts = [0] * self.n_blocks
        self._free_blocks = list(range(self.n_blocks))
        self._sessions: dict[int, Session] = {}
        self._next_sid = 0

    # ------------------------------------------------------------------ bytes
    @property
    def elem_bytes(self) -> int:
        return self.k_backend.element_size()

    @property
    def bytes_per_slot(self) -> int:
        """Bytes for K+V of one token across all layers."""
        cfg = self.cfg
        return cfg.n_layers * cfg.n_kv_heads * cfg.head_dim * self.elem_bytes * 2

    @property
    def capacity_bytes(self) -> int:
        """Bytes of the fully allocated backend tensors (K and V)."""
        return self.n_blocks * self.block_size * self.bytes_per_slot

    @property
    def allocated_blocks(self) -> int:
        return self.n_blocks - len(self._free_blocks)

    @property
    def physical_used_bytes(self) -> int:
        """Bytes held by blocks currently referenced (full blocks counted)."""
        return self.allocated_blocks * self.block_size * self.bytes_per_slot

    @property
    def logical_used_bytes(self) -> int:
        """Sum of per-session stored token bytes (shared prefixes counted
        once per owner)."""
        return sum(
            s.length * self.bytes_per_slot for s in self._sessions.values() if s.alive
        )

    def session_logical_bytes(self, session: Session) -> int:
        return session.length * self.bytes_per_slot

    # -------------------------------------------------------------- sessions
    def create_session(self, base_pos: int = 0) -> Session:
        if base_pos < 0:
            raise ValueError("base_pos must be non-negative")
        sid = self._next_sid
        self._next_sid += 1
        sess = Session(sid, self.model.fingerprint(), base_pos)
        self._sessions[sid] = sess
        return sess

    def fork_session(self, session: Session) -> Session:
        """Branch from ``session`` sharing its whole current prefix.

        The fork shares block ids (refcounts bumped); future diverging writes
        are isolated by tail copy-on-write. Releasing one branch leaves the
        shared full blocks alive for the other.
        """
        self._require_alive(session)
        child = self.create_session(session.base_pos)
        child.length = session.length
        child.blocks = list(session.blocks)
        child.stored_ids = list(session.stored_ids)
        for bid in child.blocks:
            self._refcounts[bid] += 1
        return child

    def release_session(self, session: Session) -> None:
        """Idempotent release; shared blocks survive until the last owner."""
        if not session.alive:
            return
        for bid in session.blocks:
            self._refcounts[bid] -= 1
            if self._refcounts[bid] == 0:
                self._free_blocks.append(bid)
        session.blocks = []
        session.length = 0
        session.stored_ids = []
        session.alive = False

    def _require_alive(self, session: Session) -> None:
        if not session.alive:
            raise SessionReleasedError(f"session {session.sid} was released")

    def _check_fingerprint(self, session: Session) -> None:
        current = self.model.fingerprint()
        if session.fingerprint != current:
            raise StaleCacheError(
                f"session {session.sid} cache fingerprint "
                f"{session.fingerprint[:12]}... does not match model "
                f"{current[:12]}...; old cache is invalid"
            )

    # -------------------------------------------------------------- planning
    def _tail_needs_cow(self, session: Session) -> bool:
        if session.length == 0 or session.length % self.block_size == 0:
            return False
        return self._refcounts[session.blocks[-1]] > 1

    @staticmethod
    def _blocks_for_tokens(length: int, block_size: int) -> int:
        return math.ceil(length / block_size) if length else 0

    def reserve_extend(self, sessions: list[Session], lens: list[int]) -> None:
        """Validate and atomically reserve blocks for a batched extend.

        Raises before mutating anything when capacity is insufficient.
        """
        if len(sessions) != len(lens):
            raise ValueError("sessions and lens must align")
        if len(set(s.sid for s in sessions)) != len(sessions):
            raise ValueError("a session cannot appear twice in one batch")
        for s, nnew in zip(sessions, lens):
            self._require_alive(s)
            self._check_fingerprint(s)
            if nnew <= 0:
                raise ValueError("each sequence must extend by at least one token")
            if s.base_pos + s.length + nnew > self.cfg.max_position:
                raise KVCacheError("extend exceeds model max_position")

        needed = 0
        for s, nnew in zip(sessions, lens):
            # blocks the session currently owns vs will own after appending;
            # owned count uses floor (a full tail already owns the next slot),
            # while the ceil helper sizes the final post-extend block list.
            owned = s.length // self.block_size
            total = self._blocks_for_tokens(s.length + nnew, self.block_size)
            needed += total - owned + (1 if self._tail_needs_cow(s) else 0)
        if needed > len(self._free_blocks):
            raise CacheCapacityError(
                f"need {needed} free blocks, only {len(self._free_blocks)} "
                f"available; no state was mutated"
            )

    def commit_extend(self, sessions: list[Session], lens: list[int]) -> None:
        """Apply the previously validated reservation (COW + block growth)."""
        for s, nnew in zip(sessions, lens):
            if self._tail_needs_cow(s):
                tail = s.blocks[-1]
                used = s.length % self.block_size
                fresh = self._alloc_block()
                self.k_backend[fresh, :used] = self.k_backend[tail, :used]
                self.v_backend[fresh, :used] = self.v_backend[tail, :used]
                self._refcounts[tail] -= 1
                if self._refcounts[tail] == 0:
                    self._free_blocks.append(tail)
                s.blocks[-1] = fresh
            new_total = s.length + nnew
            while len(s.blocks) * self.block_size < new_total:
                s.blocks.append(self._alloc_block())

    def _alloc_block(self) -> int:
        try:
            bid = self._free_blocks.pop()
        except IndexError as exc:  # pragma: no cover - reserve_extend guards
            raise CacheCapacityError("no free blocks") from exc
        assert self._refcounts[bid] == 0
        self._refcounts[bid] = 1
        return bid

    def finish_extend(self, sessions: list[Session], lens: list[int]) -> None:
        for s, nnew in zip(sessions, lens):
            s.length += nnew

    # --------------------------------------------------------------- access
    def gather_history(self, session: Session, layer: int, length: int | None = None):
        """Return (k, v) for ``session`` cache, each [length, Hkv, D].

        ``length`` defaults to the committed session length; during an extend
        the engine passes the post-write length (history + new tokens), whose
        blocks are already allocated and filled.
        """
        cfg = self.cfg
        if length is None:
            length = session.length
        k = torch.empty(
            (length, cfg.n_kv_heads, cfg.head_dim),
            dtype=self.dtype, device=self.device,
        )
        v = torch.empty_like(k)
        copied = 0
        remaining = length
        for bid in session.blocks:
            take = min(self.block_size, remaining)
            k[copied:copied + take] = self.k_backend[bid, :take, layer]
            v[copied:copied + take] = self.v_backend[bid, :take, layer]
            copied += take
            remaining -= take
        return k, v

    def write_new(self, session: Session, layer: int, offset: int,
                  k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Write ``k_new``/``v_new`` [nnew, Hkv, D] at logical offset."""
        nnew = k_new.shape[0]
        start = session.length + offset
        logical_end = start + nnew
        first_block = start // self.block_size
        last_block = (logical_end - 1) // self.block_size
        cursor = start
        written = 0
        for bidx in range(first_block, last_block + 1):
            block_start = bidx * self.block_size
            within_start = max(start, block_start) - block_start
            within_end = min(logical_end, block_start + self.block_size) - block_start
            take = within_end - within_start
            bid = session.blocks[bidx]
            self.k_backend[bid, within_start:within_end, layer] = k_new[written:written + take]
            self.v_backend[bid, within_start:within_end, layer] = v_new[written:written + take]
            written += take
