#!/usr/bin/env python
"""
Quick script to check if trajectory metrics are being correctly calculated
"""

import os
import sys
import pickle
import torch
import numpy as np

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from models import SimpleUNet, StudentUNet
from utils.diffusion import get_diffusion_params, p_sample_loop
from analysis.metrics.trajectory_metrics import compute_trajectory_metrics

def load_teacher_model(config, device):
    """Load the teacher model with the appropriate architecture"""
    # Create teacher model
    model = SimpleUNet(config)
    
    # Load the model weights - use the latest epoch
    teacher_model_path = os.path.join(config.models_dir, "teacher", "model_epoch_35.pt")
    if not os.path.exists(teacher_model_path):
        print(f"Teacher model file not found: {teacher_model_path}")
        sys.exit(1)
        
    model.load_state_dict(torch.load(teacher_model_path, map_location=device))
    model = model.to(device)
    return model

def load_student_model(config, device, size_factor):
    """Load a student model with the specified size factor"""
    # Create student model
    model = StudentUNet(config, size_factor=size_factor)
    
    # Load the model weights - use the latest epoch (15)
    student_model_path = os.path.join(config.models_dir, "students", f"size_{size_factor}", "model_epoch_15.pt")
    if not os.path.exists(student_model_path):
        print(f"Student model file not found: {student_model_path}")
        sys.exit(1)
        
    model.load_state_dict(torch.load(student_model_path, map_location=device))
    model = model.to(device)
    return model

def generate_trajectory(model, config, device, track_trajectory=True):
    """
    Generate a trajectory using the p_sample_loop function
    
    Args:
        model: The diffusion model
        config: Configuration object
        device: Device to use
        track_trajectory: Whether to track the trajectory
        
    Returns:
        List of (image, timestep) pairs
    """
    # Set model to eval mode
    model.eval()
    
    # Get diffusion parameters
    diffusion_params = get_diffusion_params(config.timesteps, config)
    
    # Set shape for generated samples
    shape = (1, config.channels, config.image_size, config.image_size)
    
    # Generate trajectory
    with torch.no_grad():
        # Use p_sample_loop with track_trajectory=True to get the full trajectory
        samples, trajectory = p_sample_loop(
            model, 
            shape, 
            config.timesteps, 
            diffusion_params, 
            device=device, 
            config=config, 
            track_trajectory=track_trajectory
        )
    
    return trajectory

def main():
    # Load config
    config = Config()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load teacher model
    print("Loading teacher model...")
    teacher_model = load_teacher_model(config, device)
    teacher_model.eval()
    
    # Test with a small size factor
    size_factor = 0.1
    
    # Check if student model exists
    student_model_path = os.path.join(config.models_dir, "students", f"size_{size_factor}", "model_epoch_15.pt")
    
    if not os.path.exists(student_model_path):
        print(f"Student model file not found: {student_model_path}")
        print("Please specify a different size factor or train the model first.")
        sys.exit(1)
    
    print(f"Loading student model: {student_model_path}")
    student_model = load_student_model(config, device, size_factor)
    student_model.eval()
    
    # Generate trajectories
    print("Generating teacher trajectory...")
    teacher_traj = generate_trajectory(teacher_model, config, device)
    
    print("Generating student trajectory...")
    student_traj = generate_trajectory(student_model, config, device)
    
    # Print trajectory information
    print("\nTrajectory Information:")
    print(f"Teacher trajectory length: {len(teacher_traj)}")
    print(f"Student trajectory length: {len(student_traj)}")
    
    if len(teacher_traj) > 0:
        print(f"Teacher trajectory first item type: {type(teacher_traj[0])}")
        if isinstance(teacher_traj[0], tuple):
            print(f"Teacher trajectory first item shape: {teacher_traj[0][0].shape}")
            print(f"Teacher trajectory first timestep: {teacher_traj[0][1]}")
            print(f"Teacher trajectory first image min/max: {teacher_traj[0][0].min().item():.4f}/{teacher_traj[0][0].max().item():.4f}")
            print(f"Teacher trajectory last image min/max: {teacher_traj[-1][0].min().item():.4f}/{teacher_traj[-1][0].max().item():.4f}")
        else:
            print(f"Teacher trajectory first item shape: {teacher_traj[0].shape}")
            print(f"Teacher trajectory first image min/max: {teacher_traj[0].min().item():.4f}/{teacher_traj[0].max().item():.4f}")
            print(f"Teacher trajectory last image min/max: {teacher_traj[-1].min().item():.4f}/{teacher_traj[-1].max().item():.4f}")
    
    if len(student_traj) > 0:
        print(f"Student trajectory first item type: {type(student_traj[0])}")
        if isinstance(student_traj[0], tuple):
            print(f"Student trajectory first item shape: {student_traj[0][0].shape}")
            print(f"Student trajectory first timestep: {student_traj[0][1]}")
            print(f"Student trajectory first image min/max: {student_traj[0][0].min().item():.4f}/{student_traj[0][0].max().item():.4f}")
            print(f"Student trajectory last image min/max: {student_traj[-1][0].min().item():.4f}/{student_traj[-1][0].max().item():.4f}")
        else:
            print(f"Student trajectory first item shape: {student_traj[0].shape}")
            print(f"Student trajectory first image min/max: {student_traj[0].min().item():.4f}/{student_traj[0].max().item():.4f}")
            print(f"Student trajectory last image min/max: {student_traj[-1].min().item():.4f}/{student_traj[-1].max().item():.4f}")
    
    # Check for NaN values in trajectories
    if len(teacher_traj) > 0:
        if isinstance(teacher_traj[0], tuple):
            has_teacher_nans = any(torch.isnan(item[0]).any().item() for item in teacher_traj)
        else:
            has_teacher_nans = any(torch.isnan(item).any().item() for item in teacher_traj)
        print(f"Teacher trajectory has NaN values: {has_teacher_nans}")
    
    if len(student_traj) > 0:
        if isinstance(student_traj[0], tuple):
            has_student_nans = any(torch.isnan(item[0]).any().item() for item in student_traj)
        else:
            has_student_nans = any(torch.isnan(item).any().item() for item in student_traj)
        print(f"Student trajectory has NaN values: {has_student_nans}")
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_trajectory_metrics(teacher_traj, student_traj, config)
    
    # Print all metrics to verify our new metrics are present
    print("\nTrajectory Metrics:")
    print("-" * 50)
    
    # Original metrics
    print("Original metrics:")
    keys = ['endpoint_distance', 'teacher_path_length', 'student_path_length', 
            'path_length_ratio', 'teacher_efficiency', 'student_efficiency', 'mean_wasserstein']
    for key in keys:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.6f}")
    
    # New metrics
    print("\nNew metrics:")
    new_keys = ['path_length_similarity', 'efficiency_similarity', 'mean_velocity_similarity', 
                'mean_directional_consistency', 'mean_position_difference', 'distribution_similarity']
    for key in new_keys:
        if key in metrics:
            print(f"  {key}: {metrics[key]:.6f}")
        else:
            print(f"  {key}: NOT FOUND")
    
    print("\nDone!")

if __name__ == "__main__":
    main() 