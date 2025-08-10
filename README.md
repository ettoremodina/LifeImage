# LifeImage

Foundational phases (0-5) implemented: config loading, grid environment with probabilistic food, gradients (radial) for food spawn & move cost, organisms with sensing & basic MLP brain, reproduction (uniform crossover + mutation schedule), simple global image fitness placeholder, basic rendering of occupancy frames (saved in renders/).

## Quick Start

Install (editable + dev):
```bash
pip install -e .[dev]
```

Run simulation:
```bash
python -m src.main
```

Run tests:
```bash
pytest
```

## Next Steps
- Enhance global fitness integration into energy / selection
- Add logging & visualization
- Implement advanced selection and hybrid learning (deferred)
