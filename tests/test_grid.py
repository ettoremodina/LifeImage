"""Summary: Tests grid food regeneration and placement invariants."""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parent.parent / 'src'
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from core.config import load_config
from core.simulation import Simulation
from pathlib import Path


def test_food_regeneration():
    cfg = load_config(Path("config") / "default.yaml")
    sim = Simulation(cfg)
    sim.grid.food[:] = 0
    sim.grid.tick_food()
    assert (sim.grid.food > 0).sum() <= cfg.max_food

if __name__ == '__main__':
    test_food_regeneration()
    print('ok')
