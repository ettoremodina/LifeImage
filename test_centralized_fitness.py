"""Summary: Quick verification test for the species-based fitness system with centralized fitness calculation."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

import numpy as np
from fitness.global_image import GlobalImageFitness
from organisms.organism import Organism
from brains.mlp import MLP
from utils.rng import RNG
from evolution.selection import species_based_selection, assign_initial_species


def test_centralized_species_fitness():
    """Test that species fitness calculation works through GlobalImageFitness."""
    print("Testing centralized species fitness system...")
    
    rng = RNG(42)
    
    # Create a simple target pattern (cross shape)
    target_image = np.zeros((5, 5, 3))
    target_image[2, :] = 1.0  # Horizontal line
    target_image[:, 2] = 1.0  # Vertical line
    
    # Create fitness calculator
    fitness_calc = GlobalImageFitness(target_image)
    
    # Create organisms with 2 species
    organisms = []
    brain = MLP(5, [4], 3, rng)
    
    # Species 0: positioned to match the target well
    organisms.append(Organism(0, 2, 0, energy=100, brain=brain, species=0))  # Top of cross
    organisms.append(Organism(1, 2, 2, energy=100, brain=brain, species=0))  # Center
    organisms.append(Organism(2, 2, 4, energy=100, brain=brain, species=0))  # Bottom of cross
    
    # Species 1: positioned poorly (corners)
    organisms.append(Organism(3, 0, 0, energy=100, brain=brain, species=1))  # Corner
    organisms.append(Organism(4, 4, 4, energy=100, brain=brain, species=1))  # Corner
    
    # Create grid occupancy
    grid_occupancy = np.full((5, 5), -1)
    for org in organisms:
        grid_occupancy[org.y, org.x] = org.id
    
    # Test individual species fitness calculation
    species_0_organisms = [org for org in organisms if org.species == 0]
    species_1_organisms = [org for org in organisms if org.species == 1]
    
    fitness_0 = fitness_calc.calculate_species_fitness(species_0_organisms, grid_occupancy)
    fitness_1 = fitness_calc.calculate_species_fitness(species_1_organisms, grid_occupancy)
    
    print(f"Species 0 fitness (should be high): {fitness_0:.3f}")
    print(f"Species 1 fitness (should be low): {fitness_1:.3f}")
    
    # Species 0 should have better fitness (positioned on target)
    assert fitness_0 > fitness_1, f"Species 0 ({fitness_0}) should have better fitness than species 1 ({fitness_1})"
    
    # Test all species fitness calculation
    all_fitness = fitness_calc.calculate_all_species_fitness(organisms, grid_occupancy)
    print(f"All species fitness: {all_fitness}")
    
    assert len(all_fitness) == 2
    assert all_fitness[0] == fitness_0
    assert all_fitness[1] == fitness_1
    
    # Test selection (should eliminate species 1)
    survivors = species_based_selection(organisms, fitness_calc, grid_occupancy, num_species=2)
    
    surviving_species = set(org.species for org in survivors)
    print(f"Surviving species: {surviving_species}")
    print(f"Number of survivors: {len(survivors)}")
    
    # Should only have species 0 left
    assert len(surviving_species) == 1
    assert 0 in surviving_species
    assert len(survivors) == 3
    
    print("✓ All tests passed! Species-based fitness system working correctly with centralized calculation.")


if __name__ == "__main__":
    test_centralized_species_fitness()
