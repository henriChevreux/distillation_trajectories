#!/usr/bin/env python
"""
Script to visualize trajectory metrics for student models using new pure trajectory comparison metrics
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from config.config import Config
from models import load_model
from utils.diffusion import get_diffusion_params
from utils.trajectory_manager import generate_trajectories_with_disk_storage
from analysis.metrics.trajectory_metrics import compute_trajectory_metrics, visualize_metrics
from analysis.metrics.model_comparisons import create_radar_plot_grid, create_composite_radar_plot
from data.dataset import get_real_images

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize trajectory metrics for student models")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for visualization (default: config.output_dir/trajectory_metrics)")
    parser.add_argument("--size_factors", type=str, default="0.1,0.3,0.5,0.7,1.0", help="Comma-separated list of size factors to analyze")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate trajectories for")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for trajectory generation")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    config = Config()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(config.output_dir, "trajectory_metrics")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse size factors
    size_factors = [float(sf) for sf in args.size_factors.split(",")]
    
    # Get samples for trajectory generation
    print(f"Getting {args.num_samples} samples for trajectory generation...")
    fixed_samples = get_real_images(config, args.num_samples)
    
    # Load teacher model
    print("Loading teacher model...")
    teacher_model = load_model(config, device)
    teacher_model.eval()
    
    # Process each size factor
    metrics_by_size = {}
    
    for size_factor in size_factors:
        print(f"\nProcessing size factor {size_factor}...")
        
        # Skip teacher model (size 1.0) if it's in the list twice
        if size_factor == 1.0 and 1.0 in metrics_by_size:
            continue
        
        # Load student model or use teacher model if size_factor is 1.0
        if size_factor == 1.0:
            print("Using teacher model as size 1.0...")
            student_model = teacher_model
        else:
            student_model_path = os.path.join(config.models_dir, f"student_model_size_{size_factor}.pt")
            
            if not os.path.exists(student_model_path):
                print(f"Student model file not found: {student_model_path}")
                print(f"Skipping size factor {size_factor}")
                continue
                
            print(f"Loading student model: {student_model_path}")
            student_model = load_model(config, device, size_factor=size_factor)
            student_model.eval()
        
        # Generate trajectories
        print(f"Generating trajectories for size factor {size_factor}...")
        
        teacher_trajectories = []
        student_trajectories = []
        
        # Create a subdirectory for this size factor
        size_output_dir = os.path.join(output_dir, f"size_{size_factor}")
        os.makedirs(size_output_dir, exist_ok=True)
        
        # Generate trajectories
        for i in range(args.num_samples):
            print(f"  Generating trajectory {i+1}/{args.num_samples}...")
            
            # Use a single sample
            sample = fixed_samples[i:i+1].to(device)
            
            # Generate trajectories for this sample
            teacher_traj, student_traj = generate_trajectories_with_disk_storage(
                teacher_model, student_model, config, size_factor, 1, sample)
            
            teacher_trajectories.append(teacher_traj[0])
            student_trajectories.append(student_traj[0])
            
            # Compute metrics for this sample
            metrics = compute_trajectory_metrics(teacher_traj[0], student_traj[0], config)
            
            # Visualize metrics for this sample
            sample_output_dir = os.path.join(size_output_dir, f"sample_{i+1}")
            os.makedirs(sample_output_dir, exist_ok=True)
            visualize_metrics(metrics, sample_output_dir, size_factor, f"_sample_{i+1}")
        
        # Aggregate metrics across samples
        aggregate_metrics = {}
        metric_keys = [
            'path_length_similarity', 'endpoint_distance', 'efficiency_similarity',
            'mean_velocity_similarity', 'mean_directional_consistency', 
            'distribution_similarity', 'mean_wasserstein', 'mean_position_difference'
        ]
        
        # Initialize aggregate metrics
        for key in metric_keys:
            aggregate_metrics[key] = 0.0
        
        # Sum metrics across samples
        for i in range(args.num_samples):
            sample_metrics = compute_trajectory_metrics(teacher_trajectories[i], student_trajectories[i], config)
            for key in metric_keys:
                if key in sample_metrics:
                    aggregate_metrics[key] += sample_metrics[key]
        
        # Average metrics
        for key in metric_keys:
            aggregate_metrics[key] /= args.num_samples
        
        # Store metrics for radar plot
        metrics_by_size[size_factor] = aggregate_metrics
        
        # Save aggregate metrics to file
        with open(os.path.join(size_output_dir, "aggregate_metrics.txt"), "w") as f:
            f.write(f"Aggregate metrics for size factor {size_factor} (averaged over {args.num_samples} samples):\n\n")
            for key, value in aggregate_metrics.items():
                f.write(f"{key}: {value:.6f}\n")
    
    # Create radar plots
    print("\nCreating radar plots...")
    create_radar_plot_grid(metrics_by_size, config, output_dir)
    create_composite_radar_plot(metrics_by_size, config, output_dir)
    
    print(f"\nVisualization complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main() 