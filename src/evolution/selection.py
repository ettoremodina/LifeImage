"""Summary: Selection utilities including species-based selection and energy-based culling."""
from __future__ import annotations
import numpy as np
from typing import List, Dict, TYPE_CHECKING
from collections import defaultdict

if TYPE_CHECKING:
    from organisms.organism import Organism
    from fitness.global_image import GlobalImageFitness


def cull_dead(organisms: List['Organism']) -> List['Organism']:
    """Remove dead organisms."""
    return [o for o in organisms if o.alive()]


def species_based_selection(organisms: List['Organism'], species_fitness: Dict[int, float],
                          num_species: int = 2, rng=None) -> List['Organism']:
    """
    Calculate fitness at species level, eliminate the weakest species, and replace it 
    with new organisms based on the best species genome.
    
    Args:
        organisms: List of organisms to select from
        species_fitness: Dictionary mapping species_id to accumulated fitness scores
        num_species: Number of species expected (weakest will be eliminated)
        rng: Random number generator for creating new organisms
    
    Returns:
        List of surviving organisms (including new ones from best species)
    """
    if len(organisms) <= 1 or num_species <= 1:
        return organisms
    
    # Group organisms by species
    species_groups: Dict[int, List['Organism']] = defaultdict(list)
    for org in organisms:
        species_groups[org.species].append(org)
    
    available_species = list(species_groups.keys())
    
    if len(available_species) < num_species:
        # Not enough species, keep all
        return organisms
    
    # Sort species by fitness (higher is better)
    species_fitness_list = []
    for species_id in available_species:
        fitness = species_fitness.get(species_id, 0.0)
        species_fitness_list.append((species_id, fitness))
    
    sorted_species = sorted(species_fitness_list, key=lambda x: x[1], reverse=True)
    
    # Get best and worst species
    best_species_id = sorted_species[0][0]
    worst_species_id = sorted_species[-1][0]
    
    # Find organisms from best and worst species
    best_species_organisms = species_groups[best_species_id]
    worst_species_organisms = species_groups[worst_species_id]
    
    # Keep all but the worst species
    surviving_species_ids = {species_id for species_id, _ in sorted_species[:-1]}
    survivors = [org for org in organisms if org.species in surviving_species_ids]
    
    # Create new organisms to replace the eliminated species
    if rng is not None and best_species_organisms and worst_species_organisms:
        new_species_id = find_next_available_species_id(organisms, num_species)
        
        # Create new organisms based on the best species genome
        for eliminated_org in worst_species_organisms:
            # Pick a random organism from the best species as template
            template_org = rng.choice(best_species_organisms)
            
            # Create new organism with copied brain and new species ID
            from brains.mlp import MLP
            new_brain = copy_brain(template_org.brain, rng)
            
            new_org = type(eliminated_org)(
                eliminated_org.id,  # Keep same ID for grid consistency
                eliminated_org.x,   # Keep same position
                eliminated_org.y,
                energy=eliminated_org.energy,  # Keep same energy
                brain=new_brain,
                species=new_species_id
            )
            
            # Copy categories if they exist
            if hasattr(eliminated_org, 'categories'):
                new_org.categories = eliminated_org.categories
            
            survivors.append(new_org)
    
    return survivors


def find_next_available_species_id(organisms: List['Organism'], max_species: int) -> int:
    """Find the next available species ID."""
    used_species = {org.species for org in organisms}
    for species_id in range(max_species * 2):  # Look beyond max_species for safety
        if species_id not in used_species:
            return species_id
    return max_species  # Fallback


def copy_brain(original_brain, rng):
    """Create a copy of a brain with the same architecture and weights."""
    # Use the existing clone method if available
    if hasattr(original_brain, 'clone'):
        return original_brain.clone()
    
    # Fallback: create new brain and copy weights manually
    from brains.mlp import MLP
    
    # Extract architecture from shapes
    input_size = original_brain.shapes[0][0]
    hidden_layers = [shape[1] for shape in original_brain.shapes[:-1]]
    output_size = original_brain.shapes[-1][1]
    
    # Create new brain with same architecture
    new_brain = MLP(input_size, hidden_layers, output_size, rng)
    
    # Copy weights
    new_brain.weights = [w.copy() for w in original_brain.weights]
    new_brain.shapes = original_brain.shapes.copy()
    
    return new_brain


def assign_initial_species(organisms: List['Organism'], num_species: int, rng) -> None:
    """
    Assign initial species to organisms randomly.
    
    Args:
        organisms: List of organisms to assign species to
        num_species: Number of species to create
        rng: Random number generator
    """
    for org in organisms:
        org.species = rng.randint(0, num_species - 1)
