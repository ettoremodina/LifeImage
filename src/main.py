"""Summary: Entry point to run a simulation using default configuration."""
from __future__ import annotations
from pathlib import Path
import sys

# ensure src directory (parent) is on path when executed as script
ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from core.config import load_config
from core.simulation import Simulation


def main():
    cfg = load_config(Path("config") / "default.yaml")
    sim = Simulation(cfg)
    sim.run()
    print("Final organisms:", len(sim.organisms))
    print("Global fitness:", sim.global_fitness.last_score)


if __name__ == "__main__":
    main()
