"""Summary: Simple feedforward MLP for organism decision making."""
from __future__ import annotations
import numpy as np
from typing import List


class MLP:
    def __init__(self, input_size: int, hidden_layers: List[int], output_size: int, rng):
        self.shapes = []
        prev = input_size
        self.weights = []
        for h in hidden_layers:
            self.shapes.append((prev, h))
            self.weights.append(rng.np.normal(0, 1/np.sqrt(prev), (prev, h)).astype(np.float32))
            prev = h
        self.shapes.append((prev, output_size))
        self.weights.append(rng.np.normal(0, 1/np.sqrt(prev), (prev, output_size)).astype(np.float32))

    def forward(self, x: np.ndarray) -> np.ndarray:
        a = x
        for w in self.weights[:-1]:
            a = np.tanh(a @ w)
        return a @ self.weights[-1]

    def forward_batch(self, x: np.ndarray) -> np.ndarray:
        """Vectorized forward for a batch (batch, input)."""
        a = x
        for w in self.weights[:-1]:
            a = np.tanh(a @ w)
        return a @ self.weights[-1]

    def clone(self):
        c = object.__new__(MLP)
        c.shapes = list(self.shapes)
        c.weights = [w.copy() for w in self.weights]
        return c

    def parameters(self):
        for w in self.weights:
            yield w
