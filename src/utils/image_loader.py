"""Summary: Loading target image, fallback generation if missing."""
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image


def load_or_create_target(path: str, size: int, rng) -> np.ndarray:
    p = Path(path)
    if p.exists():
        img = Image.open(p).convert("RGB").resize((size, size))
        return np.asarray(img, dtype=np.float32) / 255.0
    # fallback simple gradient pattern
    arr = np.zeros((size, size, 3), dtype=np.float32)
    for y in range(size):
        for x in range(size):
            arr[y, x] = ((x / max(1, size - 1)), (y / max(1, size - 1)), ((x + y) / max(1, 2 * size - 2)))
    return arr
