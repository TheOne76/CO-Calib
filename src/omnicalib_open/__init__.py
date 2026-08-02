"""Public NN-Detector, multi-camera Datawash, and Kalibr integration."""

from __future__ import annotations

from typing import Any


__all__ = ["NNDetector", "read_cache", "run_datawash", "write_cache"]


def __getattr__(name: str) -> Any:
    if name == "NNDetector":
        from .nn_detector import NNDetector

        return NNDetector
    if name in {"read_cache", "write_cache"}:
        from .cache import read_cache, write_cache

        return {"read_cache": read_cache, "write_cache": write_cache}[name]
    if name == "run_datawash":
        from .pipeline import run_datawash

        return run_datawash
    raise AttributeError(name)
