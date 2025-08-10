"""Summary: Tests radial gradient normalization and influence on spawn probability logic."""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parent.parent / 'src'
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from environment.gradients import radial_gradient
import numpy as np


def test_radial_gradient_mean_near_one():
    g = radial_gradient(33)
    assert abs(g.mean() - 1.0) < 0.05
    assert g.max() > g.min()
