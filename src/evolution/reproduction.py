"""Summary: Reproduction via uniform crossover between two parents producing child brain."""
from __future__ import annotations
import numpy as np


def uniform_crossover(parent_a, parent_b, rng):
    child = parent_a.clone()
    for cw, aw, bw in zip(child.weights, parent_a.weights, parent_b.weights):
        mask = rng.np.integers(0, 2, size=cw.shape, endpoint=False)
        cw[:] = np.where(mask == 0, aw, bw)
    return child
