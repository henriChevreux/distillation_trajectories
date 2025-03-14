"""
Trajectory visualization script for diffusion models.
This script visualizes the denoising process in the trajectory,
showing how the image evolves from noise to a clear image.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from config.config import Config
from models import SimpleUNet
from analysis.trajectory_comparison import generate_trajectory

def visualize_denoising_process(trajectory, output_dir, size_factor=1.0):
    """
    Visualize the denoising process in the trajectory
    
    Args:
        trajectory: List of images representing the trajectory
        output_dir: Directory to save the visualization
        size_factor: Size factor of the model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Select a subset of timesteps to visualize
    num_timesteps = len(trajectory)
    
    # For a grid visualization, select evenly spaced timesteps
    if num_timesteps > 25:
        # Select 25 evenly spaced timesteps
        indices = np.linspace(0, num_timesteps - 1, 25, dtype=int)
    else:
        # Use all timesteps
        indices = range(num_timesteps)
    
    # Create a grid of images
    num_rows = 5
    num_cols = 5
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    
    for i, idx in enumerate(indices):
        if i >= num_rows * num_cols:
            break
            
        row = i // num_cols
        col = i % num_cols
        
        # Get image
        img = trajectory[idx].squeeze().numpy().transpose(1, 2, 0)
        
        # Normalize to [0, 1] for visualization
        img = (img - img.min()) / (img.max() - img.min())
        
        # Plot image
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"Step {idx}")
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"denoising_grid_size_{size_factor}.png"), dpi=300)
    plt.close()
    
    # Create an animation of the denoising process
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Function to update the plot for each frame
    def update(frame):
        ax.clear()
        img = trajectory[frame].squeeze().numpy().transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min())
        ax.imshow(img)
        ax.set_title(f"Denoising Step {frame}/{num_timesteps-1}")
        ax.axis('off')
        return [ax]
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=range(0, num_timesteps, max(1, num_timesteps // 50)), interval=100)
    
    # Save animation
    ani.save(os.path.join(output_dir, f"denoising_animation_size_{size_factor}.gif"), writer='pillow', dpi=100)
    plt.close()
    
    # Create a side-by-side comparison of key frames
    key_frames = [0, num_timesteps // 4, num_timesteps // 2, 3 * num_timesteps // 4, num_timesteps - 1]
    
    fig, axes = plt.subplots(1, len(key_frames), figsize=(15, 5))
    
    for i, idx in enumerate(key_frames):
        # Get image
        img = trajectory[idx].squeeze().numpy().transpose(1, 2, 0)
        
        # Normalize to [0, 1] for visualization
        img = (img - img.min()) / (img.max() - img.min())
        
        # Plot image
        axes[i].imshow(img)
        axes[i].set_title(f"Step {idx}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"denoising_key_frames_size_{size_factor}.png"), dpi=300)
    plt.close()

def main():
    """Main function to run the trajectory visualization"""
    # Initialize configuration
    config = Config()
    config.analysis_dir = "analysis"
    config.timesteps = 50
    
    # Create necessary directories
    os.makedirs(config.analysis_dir, exist_ok=True)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load teacher model
    teacher_model_path = os.path.join(project_root, 'output', 'models', 'teacher', 'model_epoch_1.pt')
    print(f"Loading teacher model from {teacher_model_path}")
    
    if not os.path.exists(teacher_model_path):
        print(f"ERROR: Teacher model not found at {teacher_model_path}")
        sys.exit(1)
    
    # Initialize teacher model
    teacher_model = SimpleUNet(config).to(device)
    
    teacher_state_dict = torch.load(teacher_model_path, map_location=device)
    teacher_model.load_state_dict(teacher_state_dict)
    teacher_model.eval()
    
    print(f"Teacher model loaded successfully")
    print(f"Teacher model dimensions: {teacher_model.dims}")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Generate random noise
    noise = torch.randn(1, config.channels, config.image_size, config.image_size)
    
    # Generate trajectory
    print("Generating teacher trajectory...")
    teacher_trajectory = generate_trajectory(teacher_model, noise, config.timesteps, device)
    
    # Visualize denoising process
    print("Visualizing denoising process...")
    output_dir = os.path.join(config.analysis_dir, "trajectory_visualization")
    visualize_denoising_process(teacher_trajectory, output_dir)
    
    print(f"Visualization saved in {output_dir}")

if __name__ == "__main__":
    main() 