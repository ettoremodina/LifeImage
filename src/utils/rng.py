"""Summary: RNG utilities for deterministic seeding and reproducibility."""
from __future__ import annotations
import random
import numpy as np


class RNG:
    def __init__(self, seed: int):
        self.master_seed = seed
        self.random = random.Random(seed)
        self.np = np.random.default_rng(seed)

    def randint(self, a: int, b: int) -> int:
        return self.random.randint(a, b)

    def choice(self, seq):
        return self.random.choice(seq)

    def uniform(self, a: float, b: float) -> float:
        return self.random.uniform(a, b)
