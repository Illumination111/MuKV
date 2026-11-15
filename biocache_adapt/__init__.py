"""MuKV BioCache streaming evaluation adapter."""

from .adapter import MuKVBioCacheAdapter, MuKVConfig
from .common import TimestampedFrame, iter_benchmark

__all__ = ["MuKVBioCacheAdapter", "MuKVConfig", "TimestampedFrame", "iter_benchmark"]


def __getattr__(name):
    if name in {"QwenStreamAdapter", "StreamConfig"}:
        from .qwen_stream import QwenStreamAdapter, StreamConfig
        return {"QwenStreamAdapter": QwenStreamAdapter, "StreamConfig": StreamConfig}[name]
    raise AttributeError(name)


__all__ += ["QwenStreamAdapter", "StreamConfig"]
