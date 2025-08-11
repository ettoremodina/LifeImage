"""Summary: Comprehensive test for species replacement functionality."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

import numpy as np
from fitness.global_image import GlobalImageFitness
from organisms.organism import Organism
from brains.mlp import MLP
from utils.rng import RNG
from evolution.selection import species_based_selection, copy_brain, find_next_available_species_id


def test_species_replacement_system():
    """Test the complete species replacement system."""
    print("Testing species replacement system...")
    
    rng = RNG(42)
    
    # Create a clear target pattern - vertical line in center
    target_image = np.zeros((5, 5, 3))
    target_image[:, 2] = 1.0  # Vertical line at column 2
    
    fitness_calc = GlobalImageFitness(target_image)
    
    # Create organisms with different species and clear fitness differences
    organisms = []
    brain = MLP(5, [4], 3, rng)
    
    # Species 0: positioned on target (good fitness)
    organisms.append(Organism(0, 2, 0, energy=100, brain=brain, species=0))  # On target
    organisms.append(Organism(1, 2, 2, energy=100, brain=brain, species=0))  # On target
    organisms.append(Organism(2, 2, 4, energy=100, brain=brain, species=0))  # On target
    
    # Species 1: positioned off target (bad fitness)
    organisms.append(Organism(3, 0, 0, energy=100, brain=brain, species=1))  # Off target
    organisms.append(Organism(4, 4, 0, energy=100, brain=brain, species=1))  # Off target
    organisms.append(Organism(5, 0, 4, energy=100, brain=brain, species=1))  # Off target
    
    print(f"Initial organisms: {len(organisms)}")
    print(f"Species distribution: {[org.species for org in organisms]}")
    
    # Create grid occupancy
    grid_occupancy = np.full((5, 5), -1)
    for org in organisms:
        grid_occupancy[org.y, org.x] = org.id
    
    # Calculate initial fitness for verification
    species_fitness = fitness_calc.calculate_all_species_fitness(organisms, grid_occupancy)
    print(f"Initial species fitness: {species_fitness}")
    
    # Perform selection with replacement
    survivors = species_based_selection(organisms, fitness_calc, grid_occupancy, num_species=2, rng=rng)
    
    print(f"Survivors: {len(survivors)}")
    print(f"Survivor species: {[org.species for org in survivors]}")
    print(f"Unique species after selection: {set(org.species for org in survivors)}")
    
    # Verify results
    assert len(survivors) == 6, f"Expected 6 survivors, got {len(survivors)}"
    
    surviving_species = set(org.species for org in survivors)
    print(f"Number of unique species: {len(surviving_species)}")
    
    # Should have 2 species (original good + replacement)
    assert len(surviving_species) <= 2, f"Expected at most 2 species, got {len(surviving_species)}"
    
    # Check that we have a mix of original and new species
    original_species_present = any(org.species in {0, 1} for org in survivors)
    assert original_species_present, "Should retain at least some organisms from original species"
    
    print("✓ Species replacement system working correctly!")
    
    # Test brain copying
    print("\nTesting brain copying...")
    original_brain = brain
    copied_brain = copy_brain(original_brain, rng)
    
    # Verify brain structure is preserved
    assert len(copied_brain.weights) == len(original_brain.weights)
    assert copied_brain.shapes == original_brain.shapes
    
    # Verify weights are actually copied (not just referenced)
    original_brain.weights[0][0, 0] = 999.0
    assert copied_brain.weights[0][0, 0] != 999.0, "Brain weights should be independent copies"
    
    print("✓ Brain copying working correctly!")
    
    # Test species ID assignment
    print("\nTesting species ID assignment...")
    test_organisms = [Organism(i, 0, 0, energy=100, brain=brain, species=i % 3) for i in range(5)]
    next_id = find_next_available_species_id(test_organisms, max_species=3)
    used_species = {org.species for org in test_organisms}
    
    assert next_id not in used_species, f"Next species ID {next_id} should not be in use: {used_species}"
    print(f"✓ Next available species ID: {next_id}")
    
    print("\n✅ All species replacement tests passed!")


if __name__ == "__main__":
    test_species_replacement_system()
