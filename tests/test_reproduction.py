"""Summary: Tests uniform crossover produces mixed weights."""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parent.parent / 'src'
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from brains.mlp import MLP
from evolution.reproduction import uniform_crossover
from utils.rng import RNG
import numpy as np


def test_uniform_crossover_mix():
    rng = RNG(2)
    a = MLP(4, [3], 2, rng)
    b = MLP(4, [3], 2, rng)
    child = uniform_crossover(a, b, rng)
    same_as_a = sum((cw == aw).all() for cw, aw in zip(child.weights, a.weights))
    same_as_b = sum((cw == bw).all() for cw, bw in zip(child.weights, b.weights))
    assert same_as_a + same_as_b < len(child.weights)

if __name__ == '__main__':
    test_uniform_crossover_mix()
    print('ok')
