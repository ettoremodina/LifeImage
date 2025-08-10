"""Summary: Organism entity with energy, brain, sensing, and action selection."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Organism:
    id: int
    x: int
    y: int
    energy: float
    brain: any

    def alive(self) -> bool:
        return self.energy > 0

    def sense(self, grid, sense_radius: int) -> np.ndarray:
        size = grid.size
        feats = []
        for dy in range(-sense_radius, sense_radius+1):
            for dx in range(-sense_radius, sense_radius+1):
                sx, sy = self.x+dx, self.y+dy
                if 0 <= sx < size and 0 <= sy < size:
                    occ = grid.occupancy[sy, sx] != -1
                    food = grid.food[sy, sx] > 0
                else:
                    occ = True
                    food = False
                feats.append(1.0 if occ else 0.0)
                feats.append(-1.0 if food else 0.0)
        feats.append(self.energy)
        return np.asarray(feats, dtype=np.float32)

    def act(self, logits: np.ndarray, stochastic: bool = False):
        """Select action: argmax (default) or sample from softmax if stochastic."""
        if not stochastic:
            return int(np.argmax(logits))
        # softmax sampling
        exps = np.exp(logits - np.max(logits))
        probs = exps / np.sum(exps)
        return int(np.random.choice(len(logits), p=probs))
