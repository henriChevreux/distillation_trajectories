#!/usr/bin/env python3
"""
Script to generate trajectories and create radar plots for different model sizes.
This script combines trajectory generation from run_trajectory_comparison.py with
radar plot visualization from model_comparisons.py.
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from config.config import Config
from models import SimpleUNet, StudentUNet, DiffusionUNet
from analysis.trajectory_engine import generate_trajectory
from analysis.metrics.trajectory_metrics import compute_trajectory_metrics
from analysis.metrics.model_comparisons import create_radar_plot_grid, create_composite_radar_plot

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate trajectories and create radar plots for different model sizes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    parser.add_argument('--teacher_model', type=str, default='model_epoch_1.pt',
                        help='Path to teacher model relative to models directory')
    parser.add_argument('--size_factors', type=str, default='0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0',
                        help='Comma-separated list of size factors to compare')
    parser.add_argument('--timesteps', type=int, default=50,
                        help='Number of timesteps for the diffusion process')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='analysis/trajectory_radar',
                        help='Directory to save analysis results')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of trajectory samples to generate')
    
    return parser.parse_args()

def main():
    """Main function to run the trajectory radar analysis"""
    args = parse_args()
    
    # Initialize configuration
    config = Config()
    config.timesteps = args.timesteps
    
    # Create necessary directories
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up model comparison directory
    model_comparisons_dir = os.path.join(output_dir, 'model_comparisons')
    os.makedirs(model_comparisons_dir, exist_ok=True)
    config.model_comparisons_dir = model_comparisons_dir
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Parse size factors
    size_factors = [float(sf) for sf in args.size_factors.split(',')]
    
    # Load teacher model
    teacher_model_path = os.path.join(project_root, 'output', 'models', 'teacher', args.teacher_model)
    print(f"Loading teacher model from {teacher_model_path}")
    
    if not os.path.exists(teacher_model_path):
        print(f"ERROR: Teacher model not found at {teacher_model_path}")
        sys.exit(1)
    
    # Initialize teacher model
    teacher_model = DiffusionUNet(config, size_factor=1.0).to(device)
    
    teacher_state_dict = torch.load(teacher_model_path, map_location=device)
    teacher_model.load_state_dict(teacher_state_dict)
    teacher_model.eval()
    
    print(f"Teacher model loaded successfully")
    
    # Dictionary to store metrics for each size factor
    metrics_by_size = {}
    
    # Generate random noise samples
    print(f"Generating {args.num_samples} random noise samples...")
    noise_samples = []
    for i in range(args.num_samples):
        # Set seed for reproducibility
        seed = 42 + i
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate random noise
        noise = torch.randn(1, config.channels, config.image_size, config.image_size)
        noise_samples.append((noise, seed))
    
    # Process each size factor
    for size_factor in size_factors:
        print(f"\n{'='*80}")
        print(f"Processing size factor: {size_factor}")
        print(f"{'='*80}")
        
        # Load student model
        student_model_path = os.path.join(
            project_root, 'output', 'models', 'students', 
            f'size_{size_factor}', 'model_epoch_5.pt'
        )
        
        print(f"Loading student model from {student_model_path}")
        
        if not os.path.exists(student_model_path):
            print(f"WARNING: Student model not found at {student_model_path}")
            print(f"Skipping size factor {size_factor}")
            continue
        
        # Initialize student model
        student_model = DiffusionUNet(config, size_factor=size_factor).to(device)
        
        student_state_dict = torch.load(student_model_path, map_location=device)
        student_model.load_state_dict(student_state_dict)
        student_model.eval()
        
        print(f"Student model loaded successfully")
        
        # List to store metrics for each sample
        sample_metrics = []
        
        # Process each noise sample
        for i, (noise, seed) in enumerate(noise_samples):
            print(f"\nProcessing sample {i+1}/{len(noise_samples)}...")
            
            # Generate teacher trajectory
            print(f"Generating teacher trajectory...")
            teacher_trajectory = generate_trajectory(teacher_model, noise, config.timesteps, device, seed=seed)
            
            # Generate student trajectory
            print(f"Generating student trajectory...")
            student_trajectory = generate_trajectory(student_model, noise, config.timesteps, device, seed=seed)
            
            # Compute metrics
            print(f"Computing metrics...")
            metrics = compute_trajectory_metrics(teacher_trajectory, student_trajectory, config)
            
            # Add metrics to list
            sample_metrics.append(metrics)
            
            # Print key metrics
            print(f"  Endpoint distance: {metrics['endpoint_distance']:.4f}")
            print(f"  MSE: {metrics['mse']:.4f}")
            print(f"  Path length similarity: {metrics['path_length_similarity']:.4f}")
            print(f"  Directional consistency: {metrics['mean_directional_consistency']:.4f}")
            print(f"  Distribution similarity: {metrics['distribution_similarity']:.4f}")
        
        # Compute average metrics across samples
        avg_metrics = {}
        for key in sample_metrics[0].keys():
            if isinstance(sample_metrics[0][key], (int, float)) and not isinstance(sample_metrics[0][key], bool):
                avg_metrics[key] = sum(m[key] for m in sample_metrics) / len(sample_metrics)
        
        # Store average metrics for this size factor
        metrics_by_size[size_factor] = avg_metrics
        
        # Print average metrics
        print(f"\nAverage metrics for size factor {size_factor}:")
        print(f"  Endpoint distance: {avg_metrics['endpoint_distance']:.4f}")
        print(f"  MSE: {avg_metrics['mse']:.4f}")
        print(f"  Path length similarity: {avg_metrics['path_length_similarity']:.4f}")
        print(f"  Directional consistency: {avg_metrics['mean_directional_consistency']:.4f}")
        print(f"  Distribution similarity: {avg_metrics['distribution_similarity']:.4f}")
    
    # Create radar plots
    print("\nCreating radar plots...")
    
    # Create grid of individual radar plots
    radar_plot_path = create_radar_plot_grid(metrics_by_size, config, model_comparisons_dir)
    print(f"Radar plot grid saved to: {radar_plot_path}")
    
    # Create a single composite radar plot with all models
    composite_path = create_composite_radar_plot(metrics_by_size, config, model_comparisons_dir)
    print(f"Composite radar plot saved to: {composite_path}")
    
    # Save metrics data for later use by other scripts
    metrics_file = os.path.join(output_dir, "metrics_by_size.npy")
    np.save(metrics_file, metrics_by_size)
    print(f"Metrics data saved to: {metrics_file}")
    
    print("\nTrajectory radar analysis completed")
    print(f"Results saved in {output_dir}")

if __name__ == "__main__":
    main() 