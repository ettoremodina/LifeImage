"""Summary: Enhanced global image fitness with cumulative evaluation and species-based selection."""
from __future__ import annotations
import numpy as np
from typing import List, Dict, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from organisms.organism import Organism


class GlobalImageFitness:
    def __init__(self, target_image: np.ndarray):
        self.target = target_image
        self.last_score = 0.0
        self.accumulated_scores: List[float] = []
        self.evaluation_count = 0
        # Track accumulated fitness per species
        self.species_accumulated_scores: Dict[int, List[float]] = defaultdict(list)
        self.last_species_fitness: Dict[int, float] = {}

    def calculate_fitness(self, occupancy: np.ndarray) -> float:
        """Calculate fitness score for current state."""
        bright = self.target.mean(axis=2)
        desired = (bright > 0.5).astype(np.float32)
        occupied = (occupancy != -1).astype(np.float32)
        diff = (desired - occupied) ** 2
        score = 1.0 - diff.mean()
        return score

    def calculate_species_fitness(self, species_organisms: List['Organism'], 
                                grid_occupancy: np.ndarray) -> float:
        """
        Calculate fitness for a specific species based on how well its organisms match the target pattern.
        
        Args:
            species_organisms: All organisms in this species
            grid_occupancy: Current grid state
        
        Returns:
            Species fitness score (0.0 to 1.0, higher is better)
        """
        if not species_organisms:
            return 0.0
        
        # Create a mask of positions occupied by this species
        species_occupancy = np.full_like(grid_occupancy, -1)
        for org in species_organisms:
            if 0 <= org.x < grid_occupancy.shape[1] and 0 <= org.y < grid_occupancy.shape[0]:
                species_occupancy[org.y, org.x] = org.id
        
        # Calculate fitness similar to calculate_fitness but only for this species
        bright = self.target.mean(axis=2)
        desired = (bright > 0.5).astype(np.float32)
        occupied_by_species = (species_occupancy != -1).astype(np.float32)
        
        # Calculate how well this species matches the target
        diff = (desired - occupied_by_species) ** 2
        fitness = 1.0 - diff.mean()
        
        return fitness

    def calculate_all_species_fitness(self, organisms: List['Organism'], 
                                    grid_occupancy: np.ndarray) -> Dict[int, float]:
        """
        Calculate fitness for all species present in the organism list.
        
        Args:
            organisms: List of all organisms
            grid_occupancy: Current grid state
        
        Returns:
            Dictionary mapping species_id to fitness score
        """
        # Group organisms by species
        species_groups: Dict[int, List['Organism']] = defaultdict(list)
        for org in organisms:
            species_groups[org.species].append(org)
        
        # Calculate fitness for each species
        species_fitness = {}
        for species_id, species_organisms in species_groups.items():
            fitness = self.calculate_species_fitness(species_organisms, grid_occupancy)
            species_fitness[species_id] = fitness
        
        return species_fitness

    def accumulate_fitness(self, occupancy: np.ndarray):
        """Accumulate a fitness calculation."""
        score = self.calculate_fitness(occupancy)
        self.accumulated_scores.append(score)

    def accumulate_species_fitness(self, organisms: List['Organism'], grid_occupancy: np.ndarray):
        """Accumulate fitness calculations for each species."""
        species_fitness = self.calculate_all_species_fitness(organisms, grid_occupancy)
        for species_id, fitness in species_fitness.items():
            self.species_accumulated_scores[species_id].append(fitness)

    def evaluate_and_reset(self) -> float:
        """Evaluate accumulated fitness and reset for next period."""
        if not self.accumulated_scores:
            self.last_score = 0.0
        else:
            self.last_score = np.mean(self.accumulated_scores)
        
        self.accumulated_scores.clear()
        self.evaluation_count += 1
        return self.last_score

    def evaluate_species_and_reset(self) -> Dict[int, float]:
        """Evaluate accumulated species fitness and reset for next period."""
        self.last_species_fitness.clear()
        
        for species_id, scores in self.species_accumulated_scores.items():
            if scores:
                self.last_species_fitness[species_id] = np.mean(scores)
            else:
                self.last_species_fitness[species_id] = 0.0
        
        # Clear accumulated scores for next period
        self.species_accumulated_scores.clear()
        
        return self.last_species_fitness.copy()

    def evaluate(self, occupancy: np.ndarray) -> float:
        """Legacy method for backward compatibility."""
        self.last_score = self.calculate_fitness(occupancy)
        return self.last_score
