# LifeImage: Evolutionary Simulation Engine

A high-performance evolutionary simulation where organisms evolve neural networks to approximate target images through spatial behavior and reproduction.

## Overview

LifeImage implements a 2D grid-based ecosystem where organisms with neural network "brains" evolve over time. The goal is for the population to spatially arrange itself to match a target image pattern, creating emergent behavior through evolutionary pressure.

## Core Architecture

### Simulation Loop: The `step()` Function

The main simulation loop (`Simulation.step()`) orchestrates one tick of the ecosystem. Each step consists of 8 distinct phases executed in sequence:

---

## Phase 1: Environment Management
```python
self.grid.tick_food()
sigma = self.mutation_schedule.current_sigma(self.tick_index)
alive_orgs = [o for o in self.organisms if o.alive()]
```

**Purpose**: Prepare the environment and filter active organisms for processing.

**Implementation**:
- **Food Regeneration**: Probabilistically spawn new food across empty grid cells
- **Mutation Schedule**: Calculate current mutation strength using linear decay
- **Organism Filter**: Create list of living organisms (energy > 0) for batch processing

**Performance**: O(grid_size²) for food spawning, O(n) for organism filtering

---

## Phase 2: Sensory Input Processing (Batched)
```python
if alive_orgs:
    senses = np.stack([o.sense(self.grid, self.cfg.sense_radius) for o in alive_orgs])
```

**Purpose**: Generate sensory inputs for all organisms simultaneously.

**Method**: 
- **Batch Creation**: Stack all sensory data into single numpy array
- **Sensory Window**: Each organism senses occupancy and food in radius around position
- **Feature Vector**: `[(2*radius+1)² * 2 + 1]` features per organism (occupancy + food + energy)

**Performance**: O(n × radius²) where n = alive organisms
**Memory**: Single contiguous array enables vectorized neural network processing

---

## Phase 3: Neural Network Inference (Brain Grouping)
```python
brain_groups = {}
for idx, org in enumerate(alive_orgs):
    brain_groups.setdefault(id(org.brain), []).append((idx, org))

logits_list = [None] * len(alive_orgs)
for _, group in brain_groups.items():
    idxs, group_orgs = zip(*group)
    out = group_orgs[0].brain.forward_batch(senses[list(idxs)])
    for j, bi in enumerate(idxs):
        logits_list[bi] = out[j]
```

**Purpose**: Efficiently compute neural network outputs for all organisms.

**Optimization Strategy**:
- **Brain Sharing Detection**: Group organisms by brain object ID
- **Batch Processing**: Process identical brains together in single forward pass
- **Memory Efficiency**: Avoid redundant computation for shared neural networks

**Performance**: O(unique_brains × batch_size × network_complexity)
**Parallelization**: Each brain group processes independently (future GPU acceleration ready)

---

## Phase 4: Action Processing
```python
new_orgs = []
for i, org in enumerate(alive_orgs):
    action = org.act(logits_list[i], stochastic=self.stochastic_actions)
    self._process_action(org, action, sigma, new_orgs)
```

**Purpose**: Execute organism decisions and handle consequences.

### Action Types:
1. **Movement (0-3)**: Cardinal directions with collision detection
2. **Stay (4)**: Remain in place with stay cost
3. **Reproduce (5)**: Attempt sexual reproduction with neighbor

**Implementation Details**:
- **Stochastic Actions**: Optional softmax sampling vs deterministic argmax
- **Collision System**: Grid-based movement validation
- **Energy Costs**: Movement and staying consume energy based on configuration
- **Gradient Costs**: Optional spatial cost modifiers via gradients

---

## Phase 5: Reproduction System (`_attempt_reproduction`)
```python
def _attempt_reproduction(self, org, sigma, new_orgs):
    # Energy threshold check
    # Partner finding (Moore neighborhood)
    # Genetic crossover + mutation
    # Spatial placement of offspring
    # Reproduction cost application
```

**Purpose**: Implement sexual reproduction with spatial constraints.

**Genetic Algorithm**:
- **Partner Selection**: Find neighbor with sufficient energy in Moore neighborhood (8-cell)
- **Uniform Crossover**: Random gene mixing between parent neural networks
- **Mutation**: Gaussian noise application with time-decaying strength
- **Spatial Placement**: Child placed in empty neighbor cell

**Population Control**:
- **Energy Thresholds**: Both parents must meet reproduction threshold
- **Spatial Constraints**: Requires empty adjacent cell
- **Population Limits**: Respects maximum organism count

**Cost Models**:
- `energy`: Deduct energy from both parents
- `death`: Kill both parents (extreme selection pressure)
- `none`: No reproduction cost

---

## Phase 6: Population Management
```python
self.organisms.extend(new_orgs)
self.organisms = cull_dead(self.organisms)
```

**Purpose**: Update population and remove deceased organisms.

**Implementation**:
- **Population Merge**: Add newborns to main population
- **Death Processing**: Remove organisms with energy ≤ 0
- **Grid Cleanup**: Clear occupancy data for dead organisms

**Performance**: O(n) scan and filter operation

---

## Phase 7: Fitness Evaluation & Energy Bonuses
```python
if self.tick_index % self.cfg.fitness_eval_interval == 0:
    self.global_fitness.evaluate(self.grid.occupancy)
    if self.tick_index % self.cfg.fitness_energy_bonus_interval == 0:
        bonus = max(0.0, self.global_fitness.last_score) * self.cfg.fitness_energy_bonus_scale
        per_org_bonus = bonus / max(1, len(self.organisms))
        for org in self.organisms:
            org.energy += per_org_bonus
```

**Purpose**: Measure evolutionary progress and provide fitness-based rewards.

**Fitness Function**:
- **Global Image Comparison**: Compare organism spatial distribution to target image
- **Binary Thresholding**: Convert image brightness to occupancy expectation
- **MSE Scoring**: 1 - mean_squared_error between actual and expected occupancy

**Energy Bonus System**:
- **Proportional Rewards**: Higher fitness → more energy for all organisms
- **Population Sharing**: Bonus distributed equally among all living organisms
- **Evolutionary Pressure**: Encourages behaviors that improve global fitness

---

## Phase 8: Visualization & Logging
```python
if self.tick_index % self.cfg.render_interval == 0:
    self._handle_rendering()
if self.tick_index % self.cfg.log_interval == 0:
    self._log_metrics()
```

**Purpose**: Generate visual output and collect performance metrics.

### Rendering Modes:
1. **Live Video**: Direct frame appending to video writer (no intermediate files)
2. **Compact Video**: Store position data, render on demand (memory efficient)
3. **PNG Frames**: Traditional frame-by-frame image generation

### Metrics Collection:
- **Population**: Current organism count
- **Energy**: Average energy across population
- **Fitness**: Current global fitness score
- **Food**: Total food count and average energy value

**Performance Optimization**:
- **Configurable Intervals**: Reduce rendering/logging frequency for speed
- **Auto-upscaling**: Small grids automatically upscaled for visibility
- **Memory Management**: Compact storage avoids disk I/O bottlenecks

---

## Performance Characteristics

### Computational Complexity
- **Overall**: O(n × radius² + unique_brains × network_size + grid²)
- **Bottlenecks**: Neural network inference, sensory processing
- **Scaling**: Near-linear with population size due to batching optimizations

### Memory Usage
- **Batch Arrays**: Temporary sensory and logits arrays
- **Compact Video**: Position-only storage vs full frame images
- **Brain Sharing**: Multiple organisms can reference same neural network

### Parallelization Opportunities
- **Brain Batching**: Different brain groups can process in parallel
- **Spatial Partitioning**: Grid operations can be spatially divided
- **GPU Acceleration**: Batch operations ready for CUDA/OpenCL

---

## Configuration

Key parameters controlling simulation behavior:

```yaml
# Population & Evolution
max_organisms: 100
reproduction_threshold: 1000.0
mutation_initial_sigma: 0.2
mutation_decay_ticks: 1000

# Environment
grid_size: 30
food_spawn_probability: 0.01
max_food: 200

# Performance
render_interval: 10
log_interval: 10
stochastic_actions: true

# Video Output
video_mode: compact  # live | compact | post
video_fps: 8
```

---

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

## Usage

```python
from core.config import load_config
from core.simulation import Simulation

cfg = load_config("config/default.yaml")
sim = Simulation(cfg)
sim.run()  # Execute full simulation with video generation
```

The simulation automatically generates:
- **Video output**: High-resolution visualization of evolution
- **Metrics CSV**: Detailed performance logs
- **Console output**: Real-time progress information

---

## Technical Features

- ✅ **Batched Neural Processing**: Efficient brain computation
- ✅ **Memory-Efficient Video**: Compact position-based rendering
- ✅ **Auto-Resolution Scaling**: Small grids upscaled automatically
- ✅ **Configurable Everything**: YAML-driven parameter control
- ✅ **Multiple Video Modes**: Live, compact, and traditional rendering
- ✅ **Spatial Evolutionary Pressure**: Fitness-based energy bonuses
- ✅ **Gradient Environments**: Optional spatial cost/food modifiers
