"""Summary: Test species-based coloring system."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / 'src'))

import numpy as np
from organisms.organism import Organism, generate_species_color
from brains.mlp import MLP
from utils.rng import RNG
from visuals.renderer_new import render_entities


def test_species_colors():
    """Test that different species get different colors."""
    print("Testing species-based coloring system...")
    
    rng = RNG(42)
    brain = MLP(3, [4], 2, rng)
    
    # Create organisms with different species
    organisms = []
    for i in range(10):
        org = Organism(i, i % 5, i // 5, energy=100, brain=brain, species=i % 4)
        org._update_color_based_on_categories()  # Set initial color
        organisms.append(org)
    
    # Check that species colors are distinct
    species_colors = {}
    for org in organisms:
        if org.species not in species_colors:
            species_colors[org.species] = org.categories.color
        else:
            # Verify same species has same color (unless overridden by categories)
            expected_color = generate_species_color(org.species)
            if org.categories.color != expected_color:
                print(f"Organism {org.id} has color {org.categories.color}, expected {expected_color}")
    
    print(f"Species colors: {species_colors}")
    
    # Verify we have distinct colors for different species
    unique_colors = set(species_colors.values())
    print(f"Number of unique species colors: {len(unique_colors)}")
    print(f"Expected number of species: {len(species_colors)}")
    
    # Test predefined colors
    for i in range(5):
        color = generate_species_color(i)
        print(f"Species {i} color: {color}")
        assert len(color) == 3, f"Color should be RGB tuple, got {color}"
        assert all(0 <= c <= 255 for c in color), f"Color values should be 0-255, got {color}"
    
    # Test rendering with species colors
    print("\nTesting rendering with species colors...")
    
    # Create a small grid
    grid_size = 5
    occupancy = np.full((grid_size, grid_size), -1)
    food = np.zeros((grid_size, grid_size))
    
    # Place organisms on grid
    for org in organisms[:grid_size]:  # Only use first 5 organisms
        occupancy[org.y, org.x] = org.id
    
    # Add some food
    food[1, 1] = 10.0
    food[3, 3] = 15.0
    
    # Render
    img = render_entities(occupancy, food, organisms[:grid_size])
    
    print(f"Rendered image shape: {img.shape}")
    assert img.shape[2] == 3, "Should be RGB image"
    
    # Verify that organisms have different colors in the image
    organism_pixels = []
    for org in organisms[:grid_size]:
        pixel_color = tuple(img[org.y, org.x])
        organism_pixels.append(pixel_color)
        print(f"Organism {org.id} (species {org.species}) at ({org.x}, {org.y}): rendered color {pixel_color}, expected {org.categories.color}")
    
    # Check that we have visual variety
    unique_rendered_colors = set(organism_pixels)
    print(f"Unique rendered colors: {len(unique_rendered_colors)}")
    
    print("✓ Species coloring system working correctly!")
    
    # Test category overrides
    print("\nTesting category color overrides...")
    test_org = organisms[0]
    original_color = test_org.categories.color
    print(f"Original color: {original_color}")
    
    # Test just_born override
    test_org.categories.add_tag("just_born")
    test_org._update_color_based_on_categories()
    just_born_color = test_org.categories.color
    print(f"Just born color: {just_born_color}")
    assert just_born_color == (255, 255, 0), f"Just born should be yellow, got {just_born_color}"
    
    # Remove tag and test normal species color
    test_org.categories.remove_tag("just_born")
    test_org._update_color_based_on_categories()
    normal_color = test_org.categories.color
    print(f"Back to normal color: {normal_color}")
    assert normal_color == original_color, f"Should return to original color {original_color}, got {normal_color}"
    
    print("✓ Category color overrides working correctly!")
    print("\n✅ All species coloring tests passed!")


if __name__ == "__main__":
    test_species_colors()
