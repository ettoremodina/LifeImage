"""Summary: Gradient generation utilities for environmental modulation (food spawn & move cost)."""
from __future__ import annotations
import numpy as np


def radial_gradient(size: int) -> np.ndarray:
    c = (size - 1) / 2.0
    y, x = np.ogrid[:size, :size]
    dist = np.sqrt((x - c) ** 2 + (y - c) ** 2)
    maxd = dist.max() or 1.0
    g = 1.0 - (dist / maxd)
    g /= g.mean()
    return g.astype(np.float32)


def build_gradient(kind: str, size: int):
    if kind == 'radial':
        return radial_gradient(size)
    if kind == 'none':
        return None
    return None
