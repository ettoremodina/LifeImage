"""Summary: 2D grid managing organism occupancy and food with probabilistic regeneration."""
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple


class Grid:
    def __init__(self, size: int, max_food: int, food_spawn_probability: float, food_energy_min: float, food_energy_max: float, rng):
        self.size = size
        self.max_food = max_food
        self.food_spawn_probability = food_spawn_probability
        self.food_energy_min = food_energy_min
        self.food_energy_max = food_energy_max
        self.rng = rng
        self.occupancy = -np.ones((size, size), dtype=np.int32)
        self.food = np.zeros((size, size), dtype=np.float32)
        self.spawn_gradient = None  # optional multiplier map (size,size)
        self.move_cost_gradient = None  # optional multiplier map (size,size)
        self._seed_initial_food()

    def inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    def is_empty(self, x: int, y: int) -> bool:
        return self.occupancy[y, x] == -1

    def place(self, org_id: int, x: int, y: int) -> bool:
        if self.inside(x, y) and self.is_empty(x, y):
            self.occupancy[y, x] = org_id
            return True
        return False

    def remove(self, x: int, y: int):
        if self.inside(x, y):
            self.occupancy[y, x] = -1

    def move(self, org_id: int, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> Tuple[int, int]:
        fx, fy = from_pos
        tx, ty = to_pos
        if not self.inside(tx, ty) or not self.is_empty(tx, ty):
            return from_pos
        if self.occupancy[fy, fx] == org_id:
            self.occupancy[fy, fx] = -1
            self.occupancy[ty, tx] = org_id
            return (tx, ty)
        return from_pos

    def tick_food(self):
        current_food_cells = np.count_nonzero(self.food > 0)
        if current_food_cells >= self.max_food:
            return
        empties = np.where(self.food == 0)
        for y, x in zip(empties[0], empties[1]):
            base_p = self.food_spawn_probability
            if self.spawn_gradient is not None:
                base_p *= self.spawn_gradient[y, x]
            if self.rng.random.random() < base_p:
                self.food[y, x] = self.rng.uniform(self.food_energy_min, self.food_energy_max)
                current_food_cells += 1
                if current_food_cells >= self.max_food:
                    break

    def consume_food(self, x: int, y: int) -> float:
        if self.food[y, x] > 0:
            val = float(self.food[y, x])
            self.food[y, x] = 0
            return val
        return 0.0

    def neighbor_coords(self, x: int, y: int):
        for dx, dy in ((0,-1),(0,1),(-1,0),(1,0)):
            nx, ny = x+dx, y+dy
            if self.inside(nx, ny):
                yield nx, ny

    def moore_neighbors(self, x: int, y: int):
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dx==0 and dy==0: continue
                nx, ny = x+dx, y+dy
                if self.inside(nx, ny):
                    yield nx, ny

    def _seed_initial_food(self):
        """Place initial food on the grid at start."""
        initial_food_count = min(self.max_food // 4, self.size * self.size // 20)  # 25% of max or 5% of grid
        for _ in range(initial_food_count):
            x = self.rng.randint(0, self.size - 1)
            y = self.rng.randint(0, self.size - 1)
            if self.food[y, x] == 0:  # empty cell
                self.food[y, x] = self.rng.uniform(self.food_energy_min, self.food_energy_max)
