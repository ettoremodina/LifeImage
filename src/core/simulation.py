"""Summary: Main simulation loop integrating grid, organisms, brains, reproduction, mutation, and fitness."""
from __future__ import annotations
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
from utils.rng import RNG
from core.config import SimulationConfig
from environment.grid import Grid
from environment.gradients import build_gradient
from organisms.organism import Organism
from brains.mlp import MLP
from evolution.reproduction import uniform_crossover
from evolution.mutation import MutationSchedule, mutate_uniform
from evolution.selection import cull_dead, species_based_selection, assign_initial_species
from fitness.global_image import GlobalImageFitness
from utils.image_loader import load_or_create_target
from visuals.renderer_new import render_occupancy, render_entities, simple_video_from_frames, LiveVideoWriter, CompactVideoWriter
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
        self.input_size = ((2 * cfg.sense_radius + 1) ** 2) * 2 + 3
        self.output_size = 6
        if cfg.environment_gradients_enabled:
            self.grid.spawn_gradient = build_gradient(cfg.environment_food_spawn_gradient, cfg.grid_size)
            self.grid.move_cost_gradient = build_gradient(cfg.environment_move_cost_gradient, cfg.grid_size)
        self._seed_initial_population()
        # Assign initial species to organisms
        assign_initial_species(self.organisms, self.cfg.selection_num_species, self.rng)
        # Update colors based on species
        for org in self.organisms:
            org._update_color_based_on_categories()
        self.tick_index = 0
        self.render_dir = Path(cfg.render_output_dir)
        if not self.render_dir.exists():
            self.render_dir.mkdir(parents=True, exist_ok=True)
        self.logger = MetricsLogger(Path(cfg.log_output_csv))
        self.last_logged_tick = -1
        self.live_video: LiveVideoWriter | None = None
        self.compact_video: CompactVideoWriter | None = None
        if self.cfg.video_mode == 'live':
            self.live_video = LiveVideoWriter(Path(self.cfg.live_video_output_path), self.cfg.live_video_fps)
        elif self.cfg.video_mode == 'compact':
            self.compact_video = CompactVideoWriter(Path(self.cfg.video_output_path), self.cfg.video_fps, cfg.grid_size)

    def _seed_initial_population(self):
        needed = self.cfg.max_organisms // 2
        for _ in range(needed):
            x = self.rng.randint(0, self.cfg.grid_size - 1)
            y = self.rng.randint(0, self.cfg.grid_size - 1)
            if not self.grid.is_empty(x, y):
                continue
            brain = MLP(self.input_size, self.cfg.brain_hidden_layers, self.output_size, self.rng)
            org = Organism(self.next_org_id, x, y, energy=self.cfg.initial_energy, brain=brain)
            # Initial organisms are not marked as "just_born" since they're seeds
            self.next_org_id += 1
            self.grid.place(org.id, x, y)
            self.organisms.append(org)

    def step(self):
        self.grid.tick_food()
        sigma = self.mutation_schedule.current_sigma(self.tick_index)
        alive_orgs = [o for o in self.organisms if o.alive()]
        
        # update organism categories and colors
        # for org in alive_orgs:
        #     org.tick_update()
        #     # add energy-based categories
        #     if org.energy > self.cfg.reproduction_threshold * 1.5:
        #         org.categories.add_tag("high_energy")
        #     elif org.energy < self.cfg.reproduction_threshold * 0.3:
        #         org.categories.add_tag("low_energy")
        
        # batch processing
        if alive_orgs:
            senses = np.stack([o.sense(self.grid, self.cfg.sense_radius) for o in alive_orgs])
            brain_groups = {}
            for idx, org in enumerate(alive_orgs):
                brain_groups.setdefault(id(org.brain), []).append((idx, org))
            
            logits_list = [None] * len(alive_orgs)
            for _, group in brain_groups.items():
                idxs, group_orgs = zip(*group)
                out = group_orgs[0].brain.forward_batch(senses[list(idxs)])
                for j, bi in enumerate(idxs):
                    logits_list[bi] = out[j]
        
        # process actions
        new_orgs = []
        for i, org in enumerate(alive_orgs):
            action = org.act(logits_list[i], stochastic=self.stochastic_actions)
            self._process_action(org, action, sigma, new_orgs)
        
        self.organisms.extend(new_orgs)
        self.organisms = cull_dead(self.organisms)
        
        # fitness calculations and evaluations
        if self.tick_index % self.cfg.fitness_calculation_interval == 0:
            self.global_fitness.accumulate_fitness(self.grid.occupancy)
            self.global_fitness.accumulate_species_fitness(self.organisms, self.grid.occupancy)
        
        if self.tick_index % self.cfg.fitness_evaluation_interval == 0:
            # Evaluate accumulated fitness
            fitness_score = self.global_fitness.evaluate_and_reset()
            species_fitness = self.global_fitness.evaluate_species_and_reset()
            
            # Perform species-based selection using accumulated fitness
            if len(self.organisms) > 1:
                self.organisms = species_based_selection(
                    self.organisms, 
                    species_fitness,  # Pass accumulated species fitness
                    self.cfg.selection_num_species,
                    self.rng
                )
                # Update grid occupancy after selection
                self._update_grid_after_selection()
                # Update colors for any organisms with new species
                for org in self.organisms:
                    org._update_color_based_on_categories()
        
        # legacy fitness and bonuses (for backward compatibility)
        # if self.tick_index % self.cfg.fitness_eval_interval == 0:
        #     self.global_fitness.evaluate(self.grid.occupancy)
        #     if self.tick_index % self.cfg.fitness_energy_bonus_interval == 0:
        #         bonus = max(0.0, self.global_fitness.last_score) * self.cfg.fitness_energy_bonus_scale
        #         per_org_bonus = bonus / max(1, len(self.organisms))
        #         for org in self.organisms:
        #             org.energy += per_org_bonus
        
        # rendering and logging
        if self.tick_index % self.cfg.render_interval == 0:
            self._handle_rendering()
        if self.tick_index % self.cfg.log_interval == 0 and self.tick_index != self.last_logged_tick:
            self._log_metrics()
        
        self.tick_index += 1
    
    def _process_action(self, org, action, sigma, new_orgs):
        """Process single organism action."""
        if action in (0, 1, 2, 3):  # movement
            dx, dy = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}[action]
            new_pos = self.grid.move(org.id, (org.x, org.y), (org.x + dx, org.y + dy))
            if new_pos != (org.x, org.y):
                org.x, org.y = new_pos
                move_cost = self.cfg.move_cost
                if self.grid.move_cost_gradient is not None:
                    move_cost *= self.grid.move_cost_gradient[org.y, org.x]
                org.energy -= move_cost
        elif action == 4:  # stay
            org.energy -= self.cfg.stay_cost
        elif action == self.cfg.action_reproduce_index:  # reproduce
            self._attempt_reproduction(org, sigma, new_orgs)
        
        # consume food and metabolism
        gained = self.grid.consume_food(org.x, org.y)
        if gained:
            org.energy += gained
        org.energy -= self.cfg.metabolism_cost
    
    def _attempt_reproduction(self, org, sigma, new_orgs):
        """Handle reproduction logic."""
        if org.energy < self.cfg.reproduction_threshold:
            return
        
        # find partner
        partner = None
        for nx, ny in self.grid.moore_neighbors(org.x, org.y):
            oid = self.grid.occupancy[ny, nx]
            if oid != -1:
                p = next((o for o in self.organisms if o.id == oid), None)
                if p and p.energy >= self.cfg.reproduction_threshold:
                    partner = p
                    break
        
        if not partner or len(self.organisms) + len(new_orgs) >= self.cfg.max_organisms:
            return
        
        # create child
        child_brain = uniform_crossover(org.brain, partner.brain, self.rng)
        child_brain = mutate_uniform(child_brain, sigma, self.rng)
        
        # place child
        for nx, ny in self.grid.moore_neighbors(org.x, org.y):
            if self.grid.is_empty(nx, ny):
                child = Organism(self.next_org_id, nx, ny, 
                               energy=self.cfg.initial_energy, brain=child_brain,
                               species=org.species)  # Inherit species from parent
                # mark newborn with categories
                # child.categories.add_tag("just_born", duration=3)  # yellow for 3 ticks
                # child.categories.add_tag("offspring")  # permanent tag for tracking
                child._update_color_based_on_categories()  # ensure color is set immediately         
                # mark parents as reproducers
                # org.categories.add_tag("reproducer", duration=5)
                # partner.categories.add_tag("reproducer", duration=5)
                # org._update_color_based_on_categories()
                # partner._update_color_based_on_categories()
                
                self.next_org_id += 1
                self.grid.place(child.id, nx, ny)
                new_orgs.append(child)
                
                # apply reproduction costs
                if self.cfg.reproduction_cost_mode == 'energy':
                    org.energy -= self.cfg.reproduction_energy_cost
                    partner.energy -= self.cfg.reproduction_energy_cost
                elif self.cfg.reproduction_cost_mode == 'death':
                    org.energy = 0
                    partner.energy = 0
                break
    
    def _update_grid_after_selection(self):
        """Update grid occupancy after organisms have been removed by selection."""
        # Clear grid
        self.grid.occupancy.fill(-1)
        
        # Re-place remaining organisms
        for org in self.organisms:
            self.grid.place(org.id, org.x, org.y)
    
    def _handle_rendering(self):
        """Handle rendering based on video mode."""
        if self.cfg.video_mode == 'live' and self.live_video:
            frame = render_entities(self.grid.occupancy, self.grid.food, self.organisms)
            self.live_video.append(frame)
        elif self.cfg.video_mode == 'compact' and self.compact_video:
            self.compact_video.append_state(self.grid.occupancy, self.grid.food, self.organisms)
        elif self.cfg.write_frame_images:
            out_file = self.render_dir / f"frame_{self.tick_index:06d}.png"
            render_occupancy(self.grid.occupancy, self.target_image, out_file)
    
    def _log_metrics(self):
        """Log simulation metrics."""
        total_food = np.sum(self.grid.food > 0)
        avg_food_value = np.mean(self.grid.food[self.grid.food > 0]) if total_food > 0 else 0.0
        self.logger.log({
            'tick': self.tick_index,
            'organisms': len(self.organisms),
            'fitness': self.global_fitness.last_score,
            'avg_energy': sum(o.energy for o in self.organisms) / max(1, len(self.organisms)),
            'total_food': int(total_food),
            'avg_food_value': float(avg_food_value),
        })
        self.last_logged_tick = self.tick_index

    def run(self, ticks: int | None = None):
        t = ticks if ticks is not None else self.cfg.ticks
        for _ in tqdm(range(t)):
            self.step()
        self.logger.close()
        if self.live_video is not None:
            self.live_video.close()
            print(f"Live video written to {self.live_video.path}")
        elif self.compact_video is not None:
            self.compact_video.save_video()
            print(f"Compact video written to {self.compact_video.path}")
        elif self.cfg.video_mode == 'post' and self.cfg.write_frame_images:
            print(f"Building video from frames in {self.render_dir}")
            simple_video_from_frames(self.render_dir, Path(self.cfg.video_output_path), fps=self.cfg.video_fps)
