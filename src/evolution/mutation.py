"""Summary: Mutation utilities with linear decay schedule."""
from __future__ import annotations
import numpy as np


class MutationSchedule:
    def __init__(self, initial: float, minimum: float, decay_ticks: int):
        self.initial = initial
        self.minimum = minimum
        self.decay_ticks = max(1, decay_ticks)

    def current_sigma(self, tick: int) -> float:
        if tick >= self.decay_ticks:
            return self.minimum
        frac = 1 - (tick / self.decay_ticks)
        return self.minimum + (self.initial - self.minimum) * frac


def mutate_uniform(mlp, sigma: float, rng):
    for w in mlp.weights:
        noise = rng.np.normal(0.0, sigma, w.shape).astype(w.dtype)
        w += noise
    return mlp
