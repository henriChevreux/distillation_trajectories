#!/usr/bin/env python3
"""
Script to visualize trajectory comparisons for different guidance weights.
This script generates visualizations showing how trajectories change with increasing guidance scales.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from sklearn.decomposition import PCA

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from config.config import Config
from models import DiffusionUNet
from analysis.enhanced_trajectory_comparison import generate_trajectory
from utils.diffusion import get_diffusion_params

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Visualize trajectory comparisons for different guidance weights',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    parser.add_argument('--teacher_model', type=str, default='model_epoch_10.pt',
                        help='Path to teacher model relative to models directory')
    parser.add_argument('--size_factor', type=float, default=0.5,
                        help='Size factor of the student model to visualize')
    parser.add_argument('--guidance_scales', type=str, default='1.0,2.0,5.0,10.0,20.0,50.0,100.0',
                        help='Comma-separated list of guidance scales to visualize')
    parser.add_argument('--timesteps', type=int, default=50,
                        help='Number of timesteps for the diffusion process')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to average over for more stable results')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='analysis/cfg_trajectory_visualization',
                        help='Directory to save visualization results')
    
    return parser.parse_args()

def visualize_trajectories_pca(trajectories, guidance_scales, output_dir, size_factor, model_type):
    """
    Visualize trajectories using PCA for dimensionality reduction
    
    Args:
        trajectories: Dictionary mapping guidance scales to trajectories
        guidance_scales: List of guidance scales
        output_dir: Directory to save the visualization
        size_factor: Size factor of the model
        model_type: 'Teacher' or 'Student'
    """
    print(f"Visualizing {model_type} trajectories with PCA...")
    
    # Convert trajectories to feature vectors
    def process_trajectory(traj):
        features = [t.cpu().numpy().reshape(-1) for t in traj]
        return np.stack(features)
    
    # Get a reference trajectory (with no CFG) for PCA
    reference_trajectory = trajectories[guidance_scales[0]]
    print(f"Reference trajectory length: {len(reference_trajectory)}")
    print(f"Reference trajectory shape: {reference_trajectory[0].shape}")
    
    reference_features = process_trajectory(reference_trajectory)
    print(f"Reference features shape: {reference_features.shape}")
    
    # Fit PCA on reference trajectory
    pca = PCA(n_components=3)  # Use 3 components for 3D visualization
    pca.fit(reference_features)
    
    # Create 2D figure with more space for the legend
    fig_2d, ax_2d = plt.subplots(figsize=(16, 12))
    
    # Create 3D figure
    fig_3d = plt.figure(figsize=(16, 14))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    # Create a colormap for guidance scales
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(guidance_scales), max(guidance_scales))
    
    # Plot trajectories for each guidance scale
    for g_scale in guidance_scales:
        # Get trajectory for this guidance scale
        traj = trajectories[g_scale]
        
        # Process and project trajectory
        features = process_trajectory(traj)
        traj_pca = pca.transform(features)
        
        # Get color for this guidance scale
        color = cmap(norm(g_scale))
        
        # Plot 2D trajectory (first two components)
        ax_2d.plot(traj_pca[:, 0], traj_pca[:, 1],
                '-o', color=color, alpha=0.8, markersize=4,
                label=f'w={g_scale}')
        
        # Plot 3D trajectory
        ax_3d.plot(traj_pca[:, 0], traj_pca[:, 1], traj_pca[:, 2],
                '-o', color=color, alpha=0.8, markersize=4,
                label=f'w={g_scale}')
        
        # Add markers for start and end points in 3D
        ax_3d.scatter(traj_pca[0, 0], traj_pca[0, 1], traj_pca[0, 2], 
                    color=color, s=100, marker='o', edgecolor='black', linewidth=1.5)
        ax_3d.scatter(traj_pca[-1, 0], traj_pca[-1, 1], traj_pca[-1, 2], 
                    color=color, s=100, marker='*', edgecolor='black', linewidth=1.5)
    
    # Add legend to 2D plot with more space
    ax_2d.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10)
    
    # Add colorbar to 2D plot
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig_2d.colorbar(sm, ax=ax_2d, label='Guidance Scale', pad=0.01)
    
    # Add labels and title to 2D plot
    ax_2d.set_title(f'{model_type} Trajectories with Different Guidance Scales (2D)\n(Size Factor: {size_factor})')
    ax_2d.set_xlabel('First Principal Component')
    ax_2d.set_ylabel('Second Principal Component')
    
    # Add labels and title to 3D plot
    ax_3d.set_title(f'{model_type} Trajectories with Different Guidance Scales (3D)\n(Size Factor: {size_factor})')
    ax_3d.set_xlabel('First Principal Component')
    ax_3d.set_ylabel('Second Principal Component')
    ax_3d.set_zlabel('Third Principal Component')
    
    # Add legend to 3D plot with more space
    ax_3d.legend(bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10)
    
    # Adjust layout for 2D plot with more space for the legend
    plt.figure(fig_2d.number)
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    
    # Save 2D plot
    output_path_2d = os.path.join(output_dir, f'{model_type.lower()}_trajectories_pca_2d_size_{size_factor}.png')
    print(f"Saving 2D PCA visualization to {output_path_2d}")
    fig_2d.savefig(output_path_2d)
    
    # Adjust layout for 3D plot with more space for the legend
    plt.figure(fig_3d.number)
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    
    # Save 3D plot
    output_path_3d = os.path.join(output_dir, f'{model_type.lower()}_trajectories_pca_3d_size_{size_factor}.png')
    print(f"Saving 3D PCA visualization to {output_path_3d}")
    fig_3d.savefig(output_path_3d)
    
    # Create interactive 3D plot with multiple viewing angles
    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        ax_3d.view_init(elev=30, azim=angle)
        output_path_angle = os.path.join(output_dir, f'{model_type.lower()}_trajectories_pca_3d_angle_{angle}_size_{size_factor}.png')
        fig_3d.savefig(output_path_angle)
    
    plt.close(fig_2d)
    plt.close(fig_3d)

def visualize_final_images(trajectories, guidance_scales, output_dir, size_factor, model_type):
    """
    Visualize final images for different guidance scales
    
    Args:
        trajectories: Dictionary mapping guidance scales to trajectories
        guidance_scales: List of guidance scales
        output_dir: Directory to save the visualization
        size_factor: Size factor of the model
        model_type: 'Teacher' or 'Student'
    """
    print(f"Visualizing {model_type} final images...")
    
    # Create figure
    fig, axes = plt.subplots(1, len(guidance_scales), figsize=(16, 4))
    
    # Handle the case where there's only one guidance scale
    if len(guidance_scales) == 1:
        axes = [axes]
    
    # Plot final image for each guidance scale
    for i, g_scale in enumerate(guidance_scales):
        # Get final image for this guidance scale
        final_img = trajectories[g_scale][-1].squeeze().cpu().numpy()
        print(f"Final image shape for {model_type}, scale {g_scale}: {final_img.shape}")
        
        # Transpose image to have channels as the last dimension (C, H, W) -> (H, W, C)
        if final_img.shape[0] == 3:  # If channels first (3, H, W)
            final_img = np.transpose(final_img, (1, 2, 0))
            print(f"Transposed image shape: {final_img.shape}")
        
        # Plot image
        axes[i].imshow(final_img)
        axes[i].set_title(f'w={g_scale}')
        axes[i].axis('off')
    
    # Add overall title
    fig.suptitle(f'{model_type} Final Images with Different Guidance Scales (Size Factor: {size_factor})')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_dir, f'{model_type.lower()}_final_images_size_{size_factor}.png')
    print(f"Saving final images to {output_path}")
    plt.savefig(output_path)
    plt.close()

def visualize_trajectory_comparison(teacher_trajectories, student_trajectories, guidance_scales, output_dir, size_factor):
    """
    Visualize comparison between teacher and student trajectories for different guidance scales
    
    Args:
        teacher_trajectories: Dictionary mapping guidance scales to teacher trajectories
        student_trajectories: Dictionary mapping guidance scales to student trajectories
        guidance_scales: List of guidance scales
        output_dir: Directory to save the visualization
        size_factor: Size factor of the student model
    """
    print(f"Visualizing teacher vs student trajectory comparison...")
    
    # Convert trajectories to feature vectors
    def process_trajectory(traj):
        features = [t.cpu().numpy().reshape(-1) for t in traj]
        return np.stack(features)
    
    # Get a reference trajectory (teacher with no CFG) for PCA
    reference_trajectory = teacher_trajectories[guidance_scales[0]]
    print(f"Reference trajectory length: {len(reference_trajectory)}")
    print(f"Reference trajectory shape: {reference_trajectory[0].shape}")
    
    reference_features = process_trajectory(reference_trajectory)
    print(f"Reference features shape: {reference_features.shape}")
    
    # Fit PCA on reference trajectory
    pca = PCA(n_components=3)  # Use 3 components for 3D visualization
    pca.fit(reference_features)
    
    # Create 2D figure with more space for the legend
    fig_2d, ax_2d = plt.subplots(figsize=(16, 12))
    
    # Create 3D figure
    fig_3d = plt.figure(figsize=(16, 14))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    # Create a colormap for guidance scales
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(guidance_scales), max(guidance_scales))
    
    # Plot trajectories for each guidance scale
    for g_scale in guidance_scales:
        # Get trajectories for this guidance scale
        teacher_traj = teacher_trajectories[g_scale]
        student_traj = student_trajectories[g_scale]
        
        # Process and project trajectories
        teacher_features = process_trajectory(teacher_traj)
        student_features = process_trajectory(student_traj)
        
        teacher_pca = pca.transform(teacher_features)
        student_pca = pca.transform(student_features)
        
        # Get color for this guidance scale
        color = cmap(norm(g_scale))
        
        # Plot 2D teacher trajectory
        ax_2d.plot(teacher_pca[:, 0], teacher_pca[:, 1],
                '-o', color=color, alpha=0.8, markersize=4,
                label=f'Teacher (w={g_scale})')
        
        # Plot 2D student trajectory
        ax_2d.plot(student_pca[:, 0], student_pca[:, 1],
                '--s', color=color, alpha=0.8, markersize=4,
                label=f'Student (w={g_scale})')
        
        # Plot 3D teacher trajectory
        ax_3d.plot(teacher_pca[:, 0], teacher_pca[:, 1], teacher_pca[:, 2],
                '-o', color=color, alpha=0.8, markersize=4,
                label=f'Teacher (w={g_scale})')
        
        # Plot 3D student trajectory
        ax_3d.plot(student_pca[:, 0], student_pca[:, 1], student_pca[:, 2],
                '--s', color=color, alpha=0.8, markersize=4,
                label=f'Student (w={g_scale})')
        
        # Add markers for start and end points in 3D
        # Teacher start and end
        ax_3d.scatter(teacher_pca[0, 0], teacher_pca[0, 1], teacher_pca[0, 2], 
                    color=color, s=100, marker='o', edgecolor='black', linewidth=1.5)
        ax_3d.scatter(teacher_pca[-1, 0], teacher_pca[-1, 1], teacher_pca[-1, 2], 
                    color=color, s=100, marker='*', edgecolor='black', linewidth=1.5)
        
        # Student start and end
        ax_3d.scatter(student_pca[0, 0], student_pca[0, 1], student_pca[0, 2], 
                    color=color, s=100, marker='s', edgecolor='black', linewidth=1.5)
        ax_3d.scatter(student_pca[-1, 0], student_pca[-1, 1], student_pca[-1, 2], 
                    color=color, s=100, marker='D', edgecolor='black', linewidth=1.5)
    
    # Create custom legend with one entry per guidance scale for 2D plot
    legend_elements_2d = []
    for g_scale in guidance_scales:
        color = cmap(norm(g_scale))
        # Teacher (solid line)
        legend_elements_2d.append(plt.Line2D([0], [0], color=color, lw=2, ls='-', marker='o', markersize=4,
                                         label=f'Teacher (w={g_scale})'))
        # Student (dashed line)
        legend_elements_2d.append(plt.Line2D([0], [0], color=color, lw=2, ls='--', marker='s', markersize=4,
                                         label=f'Student (w={g_scale})'))
    
    # Add legend outside the 2D plot with more space
    ax_2d.legend(handles=legend_elements_2d, bbox_to_anchor=(1.15, 1), loc='upper left', fontsize=10)
    
    # Add colorbar to 2D plot
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig_2d.colorbar(sm, ax=ax_2d, label='Guidance Scale', pad=0.01)
    
    # Add labels and title to 2D plot
    ax_2d.set_title(f'Teacher vs Student Trajectory Comparison with CFG (2D)\n(Student Size Factor: {size_factor})')
    ax_2d.set_xlabel('First Principal Component')
    ax_2d.set_ylabel('Second Principal Component')
    
    # Add labels and title to 3D plot
    ax_3d.set_title(f'Teacher vs Student Trajectory Comparison with CFG (3D)\n(Student Size Factor: {size_factor})')
    ax_3d.set_xlabel('First Principal Component')
    ax_3d.set_ylabel('Second Principal Component')
    ax_3d.set_zlabel('Third Principal Component')
    
    # Create custom legend for 3D plot
    legend_elements_3d = [
        plt.Line2D([0], [0], color='black', lw=2, ls='-', marker='o', markersize=4, label='Teacher'),
        plt.Line2D([0], [0], color='black', lw=2, ls='--', marker='s', markersize=4, label='Student'),
        plt.Line2D([0], [0], color='gray', lw=0, marker='o', markersize=8, label='Start'),
        plt.Line2D([0], [0], color='gray', lw=0, marker='*', markersize=8, label='End (Teacher)'),
        plt.Line2D([0], [0], color='gray', lw=0, marker='D', markersize=8, label='End (Student)')
    ]
    
    # Add legend to 3D plot
    ax_3d.legend(handles=legend_elements_3d, bbox_to_anchor=(1.15, 1), loc='upper left')
    
    # Adjust layout for 2D plot with more space for the legend
    plt.figure(fig_2d.number)
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    
    # Save 2D plot
    output_path_2d = os.path.join(output_dir, f'teacher_student_trajectory_comparison_2d_size_{size_factor}.png')
    print(f"Saving 2D trajectory comparison to {output_path_2d}")
    fig_2d.savefig(output_path_2d)
    
    # Adjust layout for 3D plot
    plt.figure(fig_3d.number)
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    
    # Save 3D plot
    output_path_3d = os.path.join(output_dir, f'teacher_student_trajectory_comparison_3d_size_{size_factor}.png')
    print(f"Saving 3D trajectory comparison to {output_path_3d}")
    fig_3d.savefig(output_path_3d)
    
    # Create interactive 3D plot with multiple viewing angles
    for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        ax_3d.view_init(elev=30, azim=angle)
        output_path_angle = os.path.join(output_dir, f'teacher_student_trajectory_comparison_3d_angle_{angle}_size_{size_factor}.png')
        fig_3d.savefig(output_path_angle)
    
    plt.close(fig_2d)
    plt.close(fig_3d)

def main():
    """Main function to visualize trajectory comparisons"""
    args = parse_args()
    
    # Load configuration
    config = Config()
    config.timesteps = args.timesteps
    
    # Set up output directory
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Convert guidance scales to list of floats
    guidance_scales = [float(gs) for gs in args.guidance_scales.split(',')]
    print(f"Guidance scales: {guidance_scales}")
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load teacher model
    teacher_path = os.path.join(config.teacher_models_dir, args.teacher_model)
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(f"Teacher model not found at {teacher_path}")
    
    print(f"Loading teacher model from {teacher_path}")
    teacher_model = DiffusionUNet(config, size_factor=1.0)
    teacher_model.load_state_dict(torch.load(teacher_path, map_location=device))
    teacher_model.eval()
    teacher_model = teacher_model.to(device)
    
    # Load student model
    size_factor = args.size_factor
    size_dir = os.path.join(config.student_models_dir, f'size_{size_factor}')
    if not os.path.exists(size_dir):
        raise FileNotFoundError(f"No models found for size factor {size_factor}")
    
    # Find latest model file
    model_files = [f for f in os.listdir(size_dir) if f.startswith('model_epoch_') and f.endswith('.pt')]
    if not model_files:
        raise FileNotFoundError(f"No model files found in {size_dir}")
    
    latest_model = max(model_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
    student_path = os.path.join(size_dir, latest_model)
    
    print(f"Loading student model from {student_path}")
    student_model = DiffusionUNet(config, size_factor=size_factor)
    student_model.load_state_dict(torch.load(student_path, map_location=device))
    student_model.eval()
    student_model = student_model.to(device)
    
    # Set base seed for reproducibility
    base_seed = args.seed
    num_samples = args.num_samples
    print(f"Generating {num_samples} samples for averaging")
    
    # Dictionary to store averaged trajectories
    avg_teacher_trajectories = {gs: [] for gs in guidance_scales}
    avg_student_trajectories = {gs: [] for gs in guidance_scales}
    
    # Generate trajectories for each sample
    for sample_idx in range(num_samples):
        # Set seed for this sample
        seed = base_seed + sample_idx
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        print(f"\nGenerating sample {sample_idx+1}/{num_samples}")
        
        # Generate random noise
        noise = torch.randn(1, config.channels, config.image_size, config.image_size)
        
        # Generate trajectories for each guidance scale
        for gs in guidance_scales:
            print(f"  Generating trajectories with guidance scale {gs}...")
            
            print("    Generating teacher trajectory...")
            teacher_traj = generate_trajectory(teacher_model, noise, config.timesteps, device, seed=seed, guidance_scale=gs)
            avg_teacher_trajectories[gs].append(teacher_traj)
            
            print("    Generating student trajectory...")
            student_traj = generate_trajectory(student_model, noise, config.timesteps, device, seed=seed, guidance_scale=gs)
            avg_student_trajectories[gs].append(student_traj)
    
    # Average the trajectories
    print("\nAveraging trajectories across samples...")
    teacher_trajectories = {}
    student_trajectories = {}
    
    for gs in guidance_scales:
        # Average teacher trajectories
        teacher_avg = []
        for t in range(len(avg_teacher_trajectories[gs][0])):
            # Stack the same timestep from all samples
            timestep_tensors = torch.stack([traj[t] for traj in avg_teacher_trajectories[gs]])
            # Average across samples
            avg_tensor = torch.mean(timestep_tensors, dim=0)
            teacher_avg.append(avg_tensor)
        teacher_trajectories[gs] = teacher_avg
        
        # Average student trajectories
        student_avg = []
        for t in range(len(avg_student_trajectories[gs][0])):
            # Stack the same timestep from all samples
            timestep_tensors = torch.stack([traj[t] for traj in avg_student_trajectories[gs]])
            # Average across samples
            avg_tensor = torch.mean(timestep_tensors, dim=0)
            student_avg.append(avg_tensor)
        student_trajectories[gs] = student_avg
    
    # Visualize trajectories
    print("\nVisualizing averaged trajectories...")
    
    try:
        # Visualize teacher trajectories
        visualize_trajectories_pca(teacher_trajectories, guidance_scales, output_dir, size_factor, "Teacher")
        
        # Visualize student trajectories
        visualize_trajectories_pca(student_trajectories, guidance_scales, output_dir, size_factor, "Student")
        
        # Visualize final images
        visualize_final_images(teacher_trajectories, guidance_scales, output_dir, size_factor, "Teacher")
        visualize_final_images(student_trajectories, guidance_scales, output_dir, size_factor, "Student")
        
        # Visualize trajectory comparison
        visualize_trajectory_comparison(teacher_trajectories, student_trajectories, guidance_scales, output_dir, size_factor)
        
        print(f"\nTrajectory visualization completed")
        print(f"Results saved in {output_dir}")
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 