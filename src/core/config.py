"""Summary: Load and expose configuration values from YAML with simple dataclass-like access."""
from __future__ import annotations
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class SimulationConfig:
    grid_size: int
    ticks: int
    master_seed: int
    fitness_eval_interval: int
    max_organisms: int
    sense_radius: int
    mutation_initial_sigma: float
    mutation_min_sigma: float
    mutation_decay_ticks: int
    reproduction_threshold: float
    reproduction_cost_mode: str
    reproduction_energy_cost: float
    metabolism_cost: float
    move_cost: float
    stay_cost: float
    food_spawn_probability: float
    max_food: int
    food_energy_min: float
    food_energy_max: float
    brain_hidden_layers: List[int]
    action_reproduce_index: int
    target_image: str
    environment_gradients_enabled: bool
    environment_food_spawn_gradient: str
    environment_move_cost_gradient: str
    render_interval: int
    render_output_dir: str
    fitness_energy_bonus_interval: int
    fitness_energy_bonus_scale: float
    log_interval: int
    log_output_csv: str
    video_output_path: str
    video_fps: int
    cleanup_frames_after_video: bool
    video_mode: str
    write_frame_images: bool
    live_video_output_path: str
    live_video_fps: int
    stochastic_actions: bool


def load_config(path: str | Path) -> SimulationConfig:
    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f)
    sim = data["simulation"]
    return SimulationConfig(**sim)
