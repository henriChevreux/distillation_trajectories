"""
3D latent space visualization for diffusion model trajectories
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

def generate_latent_space_visualization(teacher_trajectory, student_trajectory, config, size_factor=None):
    """
    Generate 3D latent space visualization of teacher and student trajectories
    
    Args:
        teacher_trajectory: List of teacher trajectories (or a single trajectory)
        student_trajectory: List of student trajectories (or a single trajectory)
        config: Configuration object
        size_factor: Size factor of the student model
    
    Returns:
        Path to the directory containing the visualizations
    """
    # Create output directory
    output_dir = config.latent_space_dir
    if size_factor is not None:
        output_dir = os.path.join(output_dir, f"size_{size_factor}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating 3D latent space visualization for size factor {size_factor}...")
    
    # Handle case where a list of trajectories is passed
    if isinstance(teacher_trajectory, list) and isinstance(teacher_trajectory[0], list):
        # Just use the first trajectory for visualization
        teacher_traj = teacher_trajectory[0]
        student_traj = student_trajectory[0]
    else:
        teacher_traj = teacher_trajectory
        student_traj = student_trajectory
    
    # Extract images from trajectories
    teacher_images = [item[0] for item in teacher_traj]
    student_images = [item[0] for item in student_traj]
    
    # Flatten images for dimensionality reduction
    teacher_flat = [img.flatten().cpu().numpy() for img in teacher_images]
    student_flat = [img.flatten().cpu().numpy() for img in student_images]
    
    # Combine teacher and student data for joint embedding
    combined_data = np.vstack([teacher_flat, student_flat])
    
    # Perform PCA to get 3D representation
    try:
        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(combined_data)
        
        # Split results back into teacher and student
        teacher_pca = pca_result[:len(teacher_flat)]
        student_pca = pca_result[len(teacher_flat):]
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create colormaps for the trajectories
        teacher_colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(teacher_pca)))
        student_colors = plt.cm.Oranges(np.linspace(0.3, 1.0, len(student_pca)))
        
        # Plot teacher trajectory
        for i in range(len(teacher_pca) - 1):
            # Plot points
            ax.scatter(teacher_pca[i, 0], teacher_pca[i, 1], teacher_pca[i, 2], 
                      color=teacher_colors[i], marker='o', s=50, alpha=0.7)
            # Plot connecting lines
            ax.plot([teacher_pca[i, 0], teacher_pca[i+1, 0]], 
                   [teacher_pca[i, 1], teacher_pca[i+1, 1]], 
                   [teacher_pca[i, 2], teacher_pca[i+1, 2]], 
                   color='blue', alpha=0.5, linewidth=1.5)
        
        # Plot final teacher point
        ax.scatter(teacher_pca[-1, 0], teacher_pca[-1, 1], teacher_pca[-1, 2], 
                  color=teacher_colors[-1], marker='*', s=200, alpha=0.7, label='Teacher End')
        
        # Highlight teacher starting point
        ax.scatter(teacher_pca[0, 0], teacher_pca[0, 1], teacher_pca[0, 2], 
                  color='blue', marker='D', s=100, alpha=1.0, label='Teacher Start')
        
        # Plot student trajectory
        for i in range(len(student_pca) - 1):
            # Plot points
            ax.scatter(student_pca[i, 0], student_pca[i, 1], student_pca[i, 2], 
                      color=student_colors[i], marker='o', s=50, alpha=0.7)
            # Plot connecting lines
            ax.plot([student_pca[i, 0], student_pca[i+1, 0]], 
                   [student_pca[i, 1], student_pca[i+1, 1]], 
                   [student_pca[i, 2], student_pca[i+1, 2]], 
                   color='orange', alpha=0.5, linewidth=1.5)
        
        # Plot final student point
        ax.scatter(student_pca[-1, 0], student_pca[-1, 1], student_pca[-1, 2], 
                  color=student_colors[-1], marker='*', s=200, alpha=0.7, label='Student End')
        
        # Highlight student starting point
        ax.scatter(student_pca[0, 0], student_pca[0, 1], student_pca[0, 2], 
                  color='orange', marker='D', s=100, alpha=1.0, label='Student Start')
        
        # Add labels and title
        ax.set_xlabel(f'Principal Component 1 (Variance: {pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'Principal Component 2 (Variance: {pca.explained_variance_ratio_[1]:.2%})')
        ax.set_zlabel(f'Principal Component 3 (Variance: {pca.explained_variance_ratio_[2]:.2%})')
        ax.set_title(f'3D Latent Space Trajectory (Size Factor: {size_factor})')
        
        # Add legend
        ax.legend()
        
        # Save static figure
        plt.savefig(os.path.join(output_dir, '3d_latent_space.png'), dpi=300, bbox_inches='tight')
        
        # Create 360-degree rotation animation (multiple views)
        views = []
        for angle in range(0, 360, 45):  # 8 views
            ax.view_init(elev=20, azim=angle)
            plt.savefig(os.path.join(output_dir, f'3d_latent_space_angle_{angle}.png'), dpi=300, bbox_inches='tight')
            views.append(angle)
        
        plt.close()
        
        # Create a "from-above" view
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        
        # Plot teacher trajectory
        for i in range(len(teacher_pca) - 1):
            # Plot points
            ax.scatter(teacher_pca[i, 0], teacher_pca[i, 1], 
                      color=teacher_colors[i], marker='o', s=50, alpha=0.7)
            # Plot connecting lines
            ax.plot([teacher_pca[i, 0], teacher_pca[i+1, 0]], 
                   [teacher_pca[i, 1], teacher_pca[i+1, 1]], 
                   color='blue', alpha=0.5, linewidth=1.5)
        
        # Plot final teacher point
        ax.scatter(teacher_pca[-1, 0], teacher_pca[-1, 1], 
                  color=teacher_colors[-1], marker='*', s=200, alpha=0.7, label='Teacher End')
        
        # Highlight teacher starting point
        ax.scatter(teacher_pca[0, 0], teacher_pca[0, 1], 
                  color='blue', marker='D', s=100, alpha=1.0, label='Teacher Start')
        
        # Plot student trajectory
        for i in range(len(student_pca) - 1):
            # Plot points
            ax.scatter(student_pca[i, 0], student_pca[i, 1], 
                      color=student_colors[i], marker='o', s=50, alpha=0.7)
            # Plot connecting lines
            ax.plot([student_pca[i, 0], student_pca[i+1, 0]], 
                   [student_pca[i, 1], student_pca[i+1, 1]], 
                   color='orange', alpha=0.5, linewidth=1.5)
        
        # Plot final student point
        ax.scatter(student_pca[-1, 0], student_pca[-1, 1], 
                  color=student_colors[-1], marker='*', s=200, alpha=0.7, label='Student End')
        
        # Highlight student starting point
        ax.scatter(student_pca[0, 0], student_pca[0, 1], 
                  color='orange', marker='D', s=100, alpha=1.0, label='Student Start')
        
        # Add labels and title
        ax.set_xlabel(f'Principal Component 1 (Variance: {pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'Principal Component 2 (Variance: {pca.explained_variance_ratio_[1]:.2%})')
        ax.set_title(f'2D Latent Space Trajectory (Size Factor: {size_factor})')
        
        # Add legend
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save 2D figure
        plt.savefig(os.path.join(output_dir, '2d_latent_space.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Latent space visualization completed for size factor {size_factor}")
        return os.path.abspath(output_dir)
    
    except Exception as e:
        print(f"Error generating latent space visualization: {e}")
        return output_dir 