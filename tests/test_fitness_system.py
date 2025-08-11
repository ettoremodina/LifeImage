"""Summary: Tests for enhanced fitness system with cumulative evaluation."""
import numpy as np
import sys, pathlib
root = pathlib.Path(__file__).resolve().parent.parent / 'src'
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
from fitness.global_image import GlobalImageFitness
from organisms.organism import Organism
from brains.mlp import MLP
from utils.rng import RNG


def test_basic_fitness_calculation():
    """Test basic fitness calculation functionality."""
    target = np.ones((4, 4, 3)) * 0.8  # Bright target
    fitness = GlobalImageFitness(target)
    
    # Empty occupancy (all -1)
    occupancy = np.full((4, 4), -1)
    score = fitness.calculate_fitness(occupancy)
    assert 0.0 <= score <= 1.0
    
    # Full occupancy (all 0)
    occupancy = np.zeros((4, 4))
    score = fitness.calculate_fitness(occupancy)
    assert score == 1.0  # Perfect match


def test_cumulative_fitness():
    """Test cumulative fitness accumulation and evaluation."""
    target = np.ones((2, 2, 3)) * 0.8
    fitness = GlobalImageFitness(target)
    
    # Accumulate some scores
    occupancy1 = np.full((2, 2), -1)  # Empty
    occupancy2 = np.zeros((2, 2))     # Full
    
    fitness.accumulate_fitness(occupancy1)
    fitness.accumulate_fitness(occupancy2)
    
    # Should have 2 accumulated scores
    assert len(fitness.accumulated_scores) == 2
    
    # Evaluate and reset
    avg_score = fitness.evaluate_and_reset()
    assert len(fitness.accumulated_scores) == 0
    assert fitness.evaluation_count == 1
    assert 0.0 <= avg_score <= 1.0


def test_legacy_compatibility():
    """Test that legacy evaluate method still works."""
    target = np.ones((2, 2, 3)) * 0.6
    fitness = GlobalImageFitness(target)
    
    occupancy = np.zeros((2, 2))
    score = fitness.evaluate(occupancy)
    
    assert score == fitness.last_score
    assert 0.0 <= score <= 1.0


def test_species_fitness_calculation():
    """Test species-specific fitness calculation."""
    rng = RNG(42)
    target = np.ones((3, 3, 3)) * 0.8  # Bright target everywhere
    fitness = GlobalImageFitness(target)
    
    # Create organisms for one species
    brain = MLP(3, [2], 2, rng)
    organisms = [
        Organism(0, 0, 0, energy=100, brain=brain, species=0),
        Organism(1, 1, 1, energy=100, brain=brain, species=0),
        Organism(2, 2, 2, energy=100, brain=brain, species=0)
    ]
    
    # Create grid occupancy
    grid_occupancy = np.full((3, 3), -1)
    for org in organisms:
        grid_occupancy[org.y, org.x] = org.id
    
    species_fitness = fitness.calculate_species_fitness(organisms, grid_occupancy)
    assert 0.0 <= species_fitness <= 1.0


def test_all_species_fitness():
    """Test calculation of fitness for all species at once."""
    rng = RNG(42)
    target = np.ones((3, 3, 3)) * 0.8
    fitness = GlobalImageFitness(target)
    
    # Create organisms from multiple species
    brain = MLP(3, [2], 2, rng)
    organisms = [
        Organism(0, 0, 0, energy=100, brain=brain, species=0),
        Organism(1, 1, 0, energy=100, brain=brain, species=0),
        Organism(2, 0, 1, energy=100, brain=brain, species=1),
        Organism(3, 2, 2, energy=100, brain=brain, species=1)
    ]
    
    # Create grid occupancy
    grid_occupancy = np.full((3, 3), -1)
    for org in organisms:
        grid_occupancy[org.y, org.x] = org.id
    
    all_fitness = fitness.calculate_all_species_fitness(organisms, grid_occupancy)
    
    assert len(all_fitness) == 2  # Two species
    assert 0 in all_fitness and 1 in all_fitness
    assert all(0.0 <= score <= 1.0 for score in all_fitness.values())
