"""Tiny autoregressive decoder with RoPE + grouped-query attention.

Public API:
    ModelConfig, TinyModel
    RoPE
    KVCachePool, Session
    CacheCapacityError, StaleCacheError, KVCacheError
    prefill, decode, chunked_decode, reference_logits
"""

from .rope import RoPE
from .model import ModelConfig, TinyModel
from .cache import (
    KVCachePool,
    Session,
    KVCacheError,
    CacheCapacityError,
    StaleCacheError,
    SessionReleasedError,
)
from .runtime import prefill, decode, chunked_decode, reference_logits

__all__ = [
    "RoPE",
    "ModelConfig",
    "TinyModel",
    "KVCachePool",
    "Session",
    "KVCacheError",
    "CacheCapacityError",
    "StaleCacheError",
    "prefill",
    "decode",
    "chunked_decode",
    "reference_logits",
]
