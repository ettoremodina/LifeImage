"""Summary: Simple global image fitness comparing occupancy map to brightness threshold of target image."""
from __future__ import annotations
import numpy as np


class GlobalImageFitness:
    def __init__(self, target_image: np.ndarray):
        self.target = target_image
        self.last_score = 0.0

    def evaluate(self, occupancy: np.ndarray) -> float:
        bright = self.target.mean(axis=2)
        desired = (bright > 0.5).astype(np.float32)
        occupied = (occupancy != -1).astype(np.float32)
        diff = (desired - occupied) ** 2
        self.last_score = 1.0 - diff.mean()
        return self.last_score
