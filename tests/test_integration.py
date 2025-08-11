"""Summary: Integration test for new fitness system in simulation."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

import numpy as np
from core.config import SimulationConfig
from core.simulation import Simulation
from pathlib import Path


def test_simulation_with_new_fitness():
    """Test that simulation runs with new fitness parameters."""
    # Create minimal config
    config_data = {
        'grid_size': 8,
        'ticks': 20,
        'master_seed': 42,
        'fitness_eval_interval': 5,
        'fitness_calculation_interval': 2,
        'fitness_evaluation_interval': 10,
        'selection_num_species': 2,
        'max_organisms': 10,
        'sense_radius': 1,
        'mutation_initial_sigma': 0.1,
        'mutation_min_sigma': 0.01,
        'mutation_decay_ticks': 100,
        'initial_energy': 100.0,
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
        'target_image': 'assets/target/test.png',
        'environment_gradients_enabled': False,
        'environment_food_spawn_gradient': 'none',
        'environment_move_cost_gradient': 'none',
        'render_interval': 100,
        'render_output_dir': 'test_renders',
        'fitness_energy_bonus_interval': 10,
        'fitness_energy_bonus_scale': 1.0,
        'log_interval': 10,
        'log_output_csv': 'test_metrics.csv',
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
    
    # Create a simple target pattern
    target_img = np.zeros((8, 8, 3), dtype=np.uint8)
    target_img[2:6, 2:6] = 255  # White square in center
    
    from PIL import Image
    img = Image.fromarray(target_img, 'RGB')
    img.save('assets/target/test.png')
    
    # Run simulation
    sim = Simulation(cfg)
    initial_organism_count = len(sim.organisms)
    
    # Run for a few steps
    sim.run(ticks=15)
    
    # Verify fitness system is working
    assert sim.global_fitness.evaluation_count > 0
    assert hasattr(sim.global_fitness, 'accumulated_scores')
    
    # Clean up
    Path('assets/target/test.png').unlink(missing_ok=True)
    Path('test_metrics.csv').unlink(missing_ok=True)


if __name__ == "__main__":
    test_simulation_with_new_fitness()
    print("Integration test passed!")
