"""Summary: Tests one simulation step updates fitness periodically and maintains constraints."""
import sys, pathlib
root = pathlib.Path(__file__).resolve().parent.parent / 'src'
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from core.config import load_config
from core.simulation import Simulation
from pathlib import Path


def test_simulation_step():
    cfg = load_config(Path("config") / "default.yaml")
    cfg.ticks = 2
    sim = Simulation(cfg)
    pre_count = len(sim.organisms)
    sim.step()
    assert len(sim.organisms) >= pre_count
    assert sim.global_fitness.last_score >= -1.0

if __name__ == '__main__':
    test_simulation_step()
    print('ok')
