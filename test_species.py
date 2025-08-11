"""Summary: Quick test script for species-based fitness system."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import numpy as np
from PIL import Image
from core.config import SimulationConfig
from core.simulation import Simulation


def test_species_system():
    """Test the species-based fitness system with a simple simulation."""
    print("Testing species-based fitness system...")
    
    # Create minimal config
    config_data = {
        'grid_size': 8,
        'ticks': 25,
        'master_seed': 42,
        'fitness_eval_interval': 5,
        'fitness_calculation_interval': 2,
        'fitness_evaluation_interval': 10,
        'selection_num_species': 2,
        'max_organisms': 8,
        'sense_radius': 1,
        'mutation_initial_sigma': 0.1,
        'mutation_min_sigma': 0.01,
        'mutation_decay_ticks': 100,
        'initial_energy': 1000.0,
        'reproduction_threshold': 50.0,
        'reproduction_cost_mode': 'energy',
        'reproduction_energy_cost': 10.0,
        'metabolism_cost': 1.0,
        'move_cost': 1.0,
        'stay_cost': 0.5,
        'food_spawn_probability': 0.01,
        'max_food': 10,
        'food_energy_min': 5.0,
        'food_energy_max': 15.0,
        'brain_hidden_layers': [4],
        'action_reproduce_index': 5,
        'target_image': 'assets/target/test_species.png',
        'environment_gradients_enabled': False,
        'environment_food_spawn_gradient': 'none',
        'environment_move_cost_gradient': 'none',
        'render_interval': 100,
        'render_output_dir': 'test_renders',
        'fitness_energy_bonus_interval': 10,
        'fitness_energy_bonus_scale': 0.0,  # No bonus as requested
        'log_interval': 5,
        'log_output_csv': 'test_species_metrics.csv',
        'video_output_path': 'test_video.mp4',
        'video_fps': 1,
        'cleanup_frames_after_video': True,
        'video_mode': 'compact',
        'write_frame_images': False,
        'live_video_output_path': 'test_live.mp4',
        'live_video_fps': 1,
        'stochastic_actions': False
    }
    
    cfg = SimulationConfig(**config_data)
    
    # Create test target image
    target_dir = Path('assets/target')
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a simple cross pattern
    target_img = np.zeros((8, 8, 3), dtype=np.uint8)
    target_img[3:5, :] = 255  # Horizontal line
    target_img[:, 3:5] = 255  # Vertical line
    
    img = Image.fromarray(target_img, 'RGB')
    img.save('assets/target/test_species.png')
    
    # Run simulation
    sim = Simulation(cfg)
    
    print(f"Initial organisms: {len(sim.organisms)}")
    print(f"Species distribution: {[org.species for org in sim.organisms]}")
    
    # Run for a few steps
    for i in range(25):
        sim.step()
        if i % 5 == 0:
            species_counts = {}
            for org in sim.organisms:
                species_counts[org.species] = species_counts.get(org.species, 0) + 1
            print(f"Tick {i}: {len(sim.organisms)} organisms, species: {species_counts}")
            if hasattr(sim.global_fitness, 'evaluation_count'):
                print(f"  Fitness evaluations: {sim.global_fitness.evaluation_count}")
    
    # Verify fitness system is working
    assert sim.global_fitness.evaluation_count > 0, "No fitness evaluations performed"
    assert hasattr(sim.global_fitness, 'accumulated_scores'), "Accumulated scores not found"
    
    print("✓ Species-based fitness system working correctly!")
    
    # Clean up
    Path('assets/target/test_species.png').unlink(missing_ok=True)
    Path('test_species_metrics.csv').unlink(missing_ok=True)


if __name__ == "__main__":
    test_species_system()
