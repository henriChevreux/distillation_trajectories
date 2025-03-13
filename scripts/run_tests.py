#!/usr/bin/env python3
"""
Script to run tests on diffusion models.
Currently supports testing the teacher model's performance.
"""

import os
import sys
import argparse
import torch
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config
from models import SimpleUNet
from utils.diffusion import get_diffusion_params, p_sample_loop
from data.dataset import get_data_loader

def test_teacher_model(config, num_samples=16, use_cpu=False, test_timesteps=100):
    """
    Test the teacher model by generating samples and saving them
    
    Args:
        config: Configuration object
        num_samples: Number of samples to generate
        use_cpu: Whether to force CPU usage
        test_timesteps: Number of timesteps to use for testing (lower = faster)
    """
    # Set up device
    if use_cpu:
        print("\nForcing CPU usage")
        device = torch.device('cpu')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Using {test_timesteps} timesteps for testing (original model uses {config.timesteps})")
    
    # Load teacher model - try best model first, then fall back to final
    teacher_best_path = os.path.join(config.teacher_models_dir, 'model_best.pt')
    teacher_final_path = os.path.join(config.teacher_models_dir, 'model_final.pt')
    
    if os.path.exists(teacher_best_path):
        teacher_path = teacher_best_path
        print("Using best teacher model")
    elif os.path.exists(teacher_final_path):
        teacher_path = teacher_final_path
        print("Using final teacher model")
    else:
        raise FileNotFoundError(f"Teacher model not found at {teacher_best_path} or {teacher_final_path}")
    
    # Create and load teacher model
    print("Loading teacher model...")
    teacher_model = SimpleUNet(config).to(device)
    teacher_model.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher_model.eval()
    
    # Create test directory
    test_output_dir = os.path.join(config.output_dir, 'tests', 'teacher')
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Get diffusion parameters with test timesteps
    diffusion_params = get_diffusion_params(test_timesteps, config)
    
    print(f"\nGenerating {num_samples} samples...")
    
    # Generate samples
    with torch.no_grad():
        samples = p_sample_loop(
            model=teacher_model,
            shape=(num_samples, config.channels, config.teacher_image_size, config.teacher_image_size),
            timesteps=test_timesteps,  # Use test timesteps here
            diffusion_params=diffusion_params,
            device=device,
            config=config,
            track_trajectory=True  # Save intermediate steps
        )
    
    # Save final samples
    grid = (samples + 1) / 2
    grid = torch.clamp(grid, 0, 1)
    grid = torchvision.utils.make_grid(grid, nrow=int(num_samples**0.5))
    plt.figure(figsize=(20, 20))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.savefig(os.path.join(test_output_dir, 'teacher_samples.png'))
    plt.close()
    
    print(f"\nTest results saved to: {test_output_dir}")
    print("Generated samples saved as 'teacher_samples.png'")

def main():
    parser = argparse.ArgumentParser(
        description="Run tests on diffusion models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--num-samples", type=int, default=16,
                        help="Number of samples to generate")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU usage")
    parser.add_argument("--timesteps", type=int, default=100,
                        help="Number of timesteps for testing (lower = faster, but may affect quality)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config()
    config.create_directories()
    
    print("\n" + "="*80)
    print("RUNNING TEACHER MODEL TESTS")
    print("="*80)
    
    # Run tests
    test_teacher_model(config, args.num_samples, args.cpu, args.timesteps)
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main() 