"""Summary: Main simulation loop integrating grid, organisms, brains, reproduction, mutation, and fitness."""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
from utils.rng import RNG
from core.config import SimulationConfig
from environment.grid import Grid
from environment.gradients import build_gradient
from organisms.organism import Organism
from brains.mlp import MLP
from evolution.reproduction import uniform_crossover
from evolution.mutation import MutationSchedule, mutate_uniform
from evolution.selection import cull_dead
from fitness.global_image import GlobalImageFitness
from utils.image_loader import load_or_create_target
from visuals.renderer import render_occupancy, render_entities, frames_to_video, LiveVideoWriter
from pathlib import Path
from utils.metrics import MetricsLogger


class Simulation:
    def __init__(self, cfg: SimulationConfig):
        self.stochastic_actions = getattr(cfg, 'stochastic_actions', False)
        self.cfg = cfg
        self.rng = RNG(cfg.master_seed)
        self.grid = Grid(cfg.grid_size, cfg.max_food, cfg.food_spawn_probability, cfg.food_energy_min, cfg.food_energy_max, self.rng)
        self.organisms: List[Organism] = []
        self.next_org_id = 0
        self.mutation_schedule = MutationSchedule(cfg.mutation_initial_sigma, cfg.mutation_min_sigma, cfg.mutation_decay_ticks)
        self.target_image = load_or_create_target(cfg.target_image, cfg.grid_size, self.rng)
        self.global_fitness = GlobalImageFitness(self.target_image)
        self.input_size = ((2 * cfg.sense_radius + 1) ** 2) * 2 + 1
        self.output_size = 6
        if cfg.environment_gradients_enabled:
            self.grid.spawn_gradient = build_gradient(cfg.environment_food_spawn_gradient, cfg.grid_size)
            self.grid.move_cost_gradient = build_gradient(cfg.environment_move_cost_gradient, cfg.grid_size)
        self._seed_initial_population()
        self.tick_index = 0
        self.render_dir = Path(cfg.render_output_dir)
        if not self.render_dir.exists():
            self.render_dir.mkdir(parents=True, exist_ok=True)
        self.logger = MetricsLogger(Path(cfg.log_output_csv))
        self.last_logged_tick = -1
        self.live_video: LiveVideoWriter | None = None
        if self.cfg.video_mode == 'live':
            self.live_video = LiveVideoWriter(Path(self.cfg.live_video_output_path), self.cfg.live_video_fps)

    def _seed_initial_population(self):
        needed = min(self.cfg.max_organisms // 10, 500)
        for _ in range(needed):
            x = self.rng.randint(0, self.cfg.grid_size - 1)
            y = self.rng.randint(0, self.cfg.grid_size - 1)
            if not self.grid.is_empty(x, y):
                continue
            brain = MLP(self.input_size, self.cfg.brain_hidden_layers, self.output_size, self.rng)
            org = Organism(self.next_org_id, x, y, energy=self.cfg.reproduction_threshold, brain=brain)
            self.next_org_id += 1
            self.grid.place(org.id, x, y)
            self.organisms.append(org)

    def step(self):
        self.grid.tick_food()
        sigma = self.mutation_schedule.current_sigma(self.tick_index)
        alive_orgs = [o for o in self.organisms if o.alive()]
        # batch sensing
        senses = [o.sense(self.grid, self.cfg.sense_radius) for o in alive_orgs]
        if senses:
            batch = np.stack(senses, axis=0)
            # currently each organism has its own brain; batching per unique brain groups
            # group organisms by brain object id to batch per brain
            brain_groups = {}
            for idx, org in enumerate(alive_orgs):
                brain_groups.setdefault(id(org.brain), []).append((idx, org))
            logits_list = [None] * len(alive_orgs)
            for _, group in brain_groups.items():
                idxs, group_orgs = zip(*group)
                sub_batch = batch[list(idxs)]
                out = group_orgs[0].brain.forward_batch(sub_batch)
                for j, bi in enumerate(idxs):
                    logits_list[bi] = out[j]
        new_orgs = []
        for i, org in enumerate(alive_orgs):
            logits = logits_list[i]
            action = org.act(logits, stochastic=self.stochastic_actions)
            moved = False
            if action in (0,1,2,3):
                dx, dy = {0:(0,-1),1:(0,1),2:(-1,0),3:(1,0)}[action]
                nx, ny = org.x + dx, org.y + dy
                new_pos = self.grid.move(org.id, (org.x, org.y), (nx, ny))
                moved = new_pos != (org.x, org.y)
                org.x, org.y = new_pos
                if moved:
                    move_cost = self.cfg.move_cost
                    if self.grid.move_cost_gradient is not None:
                        move_cost *= self.grid.move_cost_gradient[org.y, org.x]
                    org.energy -= move_cost
            elif action == 4:  # stay
                org.energy -= self.cfg.stay_cost
            elif action == self.cfg.action_reproduce_index:
                if org.energy >= self.cfg.reproduction_threshold:
                    # find partner in Moore neighborhood meeting criteria
                    partner = None
                    for nx, ny in self.grid.moore_neighbors(org.x, org.y):
                        oid = self.grid.occupancy[ny, nx]
                        if oid != -1:
                            p = next((o for o in self.organisms if o.id == oid), None)
                            if p and p.energy >= self.cfg.reproduction_threshold:
                                partner = p
                                break
                    if partner and len(self.organisms) + len(new_orgs) < self.cfg.max_organisms:
                        child_brain = uniform_crossover(org.brain, partner.brain, self.rng)
                        mutate_uniform(child_brain, sigma, self.rng)
                        # place child in empty neighbor cell
                        placed = False
                        for nx, ny in self.grid.moore_neighbors(org.x, org.y):
                            if self.grid.is_empty(nx, ny):
                                child = Organism(self.next_org_id, nx, ny, energy=self.cfg.reproduction_threshold/2, brain=child_brain)
                                self.next_org_id += 1
                                self.grid.place(child.id, nx, ny)
                                new_orgs.append(child)
                                placed = True
                                break
                        if placed:
                            if self.cfg.reproduction_cost_mode == 'energy':
                                org.energy -= self.cfg.reproduction_energy_cost
                                partner.energy -= self.cfg.reproduction_energy_cost
                            elif self.cfg.reproduction_cost_mode == 'death':
                                org.energy = 0
                                partner.energy = 0
            # consume food
            gained = self.grid.consume_food(org.x, org.y)
            if gained:
                org.energy += gained
            # metabolism
            org.energy -= self.cfg.metabolism_cost
        self.organisms.extend(new_orgs)
        # cull
        self.organisms = cull_dead(self.organisms)
        fitness_updated = False
        if (self.tick_index % self.cfg.fitness_eval_interval) == 0:
            self.global_fitness.evaluate(self.grid.occupancy)
            fitness_updated = True
        # apply energy bonus proportional to fitness (simple scaling)
        if fitness_updated and (self.tick_index % self.cfg.fitness_energy_bonus_interval) == 0:
            bonus = max(0.0, self.global_fitness.last_score) * self.cfg.fitness_energy_bonus_scale
            for org in self.organisms:
                org.energy += bonus / max(1, len(self.organisms))
        if (self.tick_index % self.cfg.render_interval) == 0:
            if self.cfg.video_mode == 'live' and self.live_video is not None:
                frame = render_entities(self.grid.occupancy, self.grid.food)
                self.live_video.append(frame)
            else:
                if self.cfg.write_frame_images:
                    out_file = self.render_dir / f"frame_{self.tick_index:06d}.png"
                    render_occupancy(self.grid.occupancy, self.target_image, out_file)
        if (self.tick_index % self.cfg.log_interval) == 0 and self.tick_index != self.last_logged_tick:
            self.logger.log({
                'tick': self.tick_index,
                'organisms': len(self.organisms),
                'fitness': self.global_fitness.last_score,
                'avg_energy': sum(o.energy for o in self.organisms)/max(1,len(self.organisms)),
            })
            self.last_logged_tick = self.tick_index
        self.tick_index += 1

    def run(self, ticks: int | None = None):
        t = ticks if ticks is not None else self.cfg.ticks
        for _ in range(t):
            self.step()
        self.logger.close()
        if self.live_video is not None:
            self.live_video.close()
            print(f"Live video written to {self.live_video.path}")
        elif self.cfg.video_mode == 'post' and self.cfg.write_frame_images:
            print(f"Building video from frames in {self.render_dir}")
            frames_to_video(
                self.render_dir,
                'frame_*.png',
                Path(self.cfg.video_output_path),
                fps=self.cfg.video_fps,
                delete_frames=self.cfg.cleanup_frames_after_video,
            )
