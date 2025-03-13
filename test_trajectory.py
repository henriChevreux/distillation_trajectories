#!/usr/bin/env python3
"""
Test script to generate a single trajectory using the teacher and student models.
This script is used to debug trajectory generation issues.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

# Add the project root directory to the Python path
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config.config import Config
from models import SimpleUNet, StudentUNet
from utils.trajectory_manager import TrajectoryManager

def visualize_trajectory(trajectory, title, save_path):
    """
    Visualize a trajectory as a grid of images
    
    Args:
        trajectory: List of (image, timestep) pairs
        title: Title for the plot
        save_path: Path to save the visualization
    """
    # Extract images from trajectory
    images = [item[0] for item in trajectory]
    
    # Create a grid of images
    num_images = len(images)
    rows = int(np.sqrt(num_images))
    cols = int(np.ceil(num_images / rows))
    
    # Convert images to range [0, 1] for visualization
    normalized_images = []
    for img in images:
        # Normalize each image to [0, 1]
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            normalized_img = (img - img_min) / (img_max - img_min)
        else:
            normalized_img = torch.zeros_like(img)
        normalized_images.append(normalized_img)
    
    # Create grid
    grid = make_grid(torch.cat(normalized_images), nrow=cols)
    
    # Save grid
    save_image(grid, save_path)
    print(f"Saved trajectory visualization to {save_path}")

def main():
    # Create configuration
    config = Config()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else 
                         "cpu")
    print(f"Using device: {device}")
    
    # Load teacher model
    teacher_model_path = os.path.join("output", "models", "teacher", "model_epoch_1.pt")
    teacher_model = SimpleUNet(config).to(device)
    teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=device))
    print(f"Loaded teacher model from {teacher_model_path}")
    
    # Load student model with size factor 0.1
    size_factor = 0.1
    student_model_path = os.path.join("output", "models", "students", f"size_{size_factor}", "model_epoch_1.pt")
    student_model = StudentUNet(config, size_factor=size_factor, architecture_type='small').to(device)
    student_model.load_state_dict(torch.load(student_model_path, map_location=device))
    print(f"Loaded student model from {student_model_path}")
    
    # Create trajectory manager
    trajectory_manager = TrajectoryManager(
        teacher_model=teacher_model,
        student_model=student_model,
        config=config
    )
    
    # Generate a single trajectory
    print("Generating trajectory...")
    seed = 42  # Use a fixed seed for reproducibility
    teacher_trajectory, student_trajectory = trajectory_manager.generate_trajectory(seed=seed)
    
    # Print trajectory information
    print(f"Teacher trajectory length: {len(teacher_trajectory)}")
    print(f"Student trajectory length: {len(student_trajectory)}")
    
    # Print shape information for the first image in each trajectory
    if teacher_trajectory:
        print(f"Teacher image shape: {teacher_trajectory[0][0].shape}")
    if student_trajectory:
        print(f"Student image shape: {student_trajectory[0][0].shape}")
    
    # Visualize trajectories
    os.makedirs("debug", exist_ok=True)
    visualize_trajectory(teacher_trajectory, "Teacher Trajectory", "debug/teacher_trajectory.png")
    visualize_trajectory(student_trajectory, "Student Trajectory", "debug/student_trajectory.png")
    
    # Print distance between first and last images in each trajectory
    if teacher_trajectory:
        first_img = teacher_trajectory[0][0]
        last_img = teacher_trajectory[-1][0]
        distance = torch.norm(last_img - first_img).item()
        print(f"Teacher trajectory distance (first to last): {distance}")
    
    if student_trajectory:
        first_img = student_trajectory[0][0]
        last_img = student_trajectory[-1][0]
        distance = torch.norm(last_img - first_img).item()
        print(f"Student trajectory distance (first to last): {distance}")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    main() 