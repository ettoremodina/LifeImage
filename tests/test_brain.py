"""Summary: Tests MLP forward output shape."""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parent.parent / 'src'
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from brains.mlp import MLP
from utils.rng import RNG
import numpy as np


def test_mlp_forward_shape():
    rng = RNG(1)
    mlp = MLP(10, [5], 4, rng)
    out = mlp.forward(np.zeros(10, dtype=np.float32))
    assert out.shape == (4,)

if __name__ == '__main__':
    test_mlp_forward_shape()
    print('ok')
