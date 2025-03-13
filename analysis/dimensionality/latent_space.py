"""
Latent space visualization for diffusion trajectories
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

def extract_trajectory_features(trajectories):
    """
    Extract features from trajectories for dimensionality reduction
    
    Args:
        trajectories: List of trajectories, where each trajectory is a list of (image, timestep) pairs
        
    Returns:
        Flattened features for each trajectory point and corresponding timesteps
    """
    # Extract images from trajectories
    all_images = []
    all_timesteps = []
    
    for trajectory in trajectories:
        for img, timestep in trajectory:
            # Convert to numpy and flatten
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            
            # Flatten the image
            img_flat = img.reshape(1, -1)
            all_images.append(img_flat)
            all_timesteps.append(timestep)
    
    # Stack all flattened images
    features = np.vstack(all_images)
    timesteps = np.array(all_timesteps)
    
    return features, timesteps

def create_3d_trajectory_plot(teacher_features, student_features, teacher_timesteps, student_timesteps, output_dir, size_factor):
    """
    Create a 3D plot of trajectories
    
    Args:
        teacher_features: Features from teacher trajectories
        student_features: Features from student trajectories
        teacher_timesteps: Timesteps from teacher trajectories
        student_timesteps: Timesteps from student trajectories
        output_dir: Directory to save visualizations
        size_factor: Size factor of the student model for labeling
    """
    # Combine features for PCA fitting
    combined_features = np.vstack([teacher_features, student_features])
    
    # Fit PCA
    pca = PCA(n_components=3)
    combined_pca = pca.fit_transform(combined_features)
    
    # Split back into teacher and student
    teacher_pca = combined_pca[:len(teacher_features)]
    student_pca = combined_pca[len(teacher_features):]
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create colormaps
    teacher_cmap = plt.cm.viridis
    student_cmap = plt.cm.plasma
    
    # Normalize timesteps to [0, 1] for colormap
    max_timestep = max(np.max(teacher_timesteps), np.max(student_timesteps))
    norm_teacher_timesteps = teacher_timesteps / max_timestep
    norm_student_timesteps = student_timesteps / max_timestep
    
    # Plot teacher trajectory
    sc1 = ax.scatter(
        teacher_pca[:, 0], teacher_pca[:, 1], teacher_pca[:, 2],
        c=norm_teacher_timesteps, cmap=teacher_cmap, marker='o', s=50, alpha=0.7, label='Teacher'
    )
    
    # Plot student trajectory
    sc2 = ax.scatter(
        student_pca[:, 0], student_pca[:, 1], student_pca[:, 2],
        c=norm_student_timesteps, cmap=student_cmap, marker='x', s=50, alpha=0.7, label='Student'
    )
    
    # Add colorbar
    cbar = plt.colorbar(sc1, ax=ax, shrink=0.7)
    cbar.set_label('Normalized Timestep')
    
    # Add labels and title
    ax.set_xlabel(f'PC1 (Var: {pca.explained_variance_ratio_[0]:.2f})')
    ax.set_ylabel(f'PC2 (Var: {pca.explained_variance_ratio_[1]:.2f})')
    ax.set_zlabel(f'PC3 (Var: {pca.explained_variance_ratio_[2]:.2f})')
    ax.set_title(f'3D PCA of Diffusion Trajectories (Size Factor: {size_factor})')
    
    # Add legend
    ax.legend()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f'3d_pca_trajectories_size_{size_factor}.png'), dpi=300, bbox_inches='tight')
    
    # Create an animated version that rotates the plot
    def rotate(angle):
        ax.view_init(elev=20, azim=angle)
        return [sc1, sc2]
    
    # Create animation
    ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, 5), interval=100, blit=True)
    ani.save(os.path.join(output_dir, f'3d_pca_trajectories_size_{size_factor}.gif'), writer='pillow', fps=10, dpi=100)
    
    plt.close()

def create_trajectory_comparison_plot(teacher_features, student_features, teacher_timesteps, student_timesteps, output_dir, size_factor):
    """
    Create a comparison plot of teacher and student trajectories
    
    Args:
        teacher_features: Features from teacher trajectories
        student_features: Features from student trajectories
        teacher_timesteps: Timesteps from teacher trajectories
        student_timesteps: Timesteps from student trajectories
        output_dir: Directory to save visualizations
        size_factor: Size factor of the student model for labeling
    """
    # Combine features for PCA fitting
    combined_features = np.vstack([teacher_features, student_features])
    
    # Fit PCA
    pca = PCA(n_components=2)
    combined_pca = pca.fit_transform(combined_features)
    
    # Split back into teacher and student
    teacher_pca = combined_pca[:len(teacher_features)]
    student_pca = combined_pca[len(teacher_features):]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create colormaps
    teacher_cmap = plt.cm.viridis
    student_cmap = plt.cm.plasma
    
    # Normalize timesteps to [0, 1] for colormap
    max_timestep = max(np.max(teacher_timesteps), np.max(student_timesteps))
    norm_teacher_timesteps = teacher_timesteps / max_timestep
    norm_student_timesteps = student_timesteps / max_timestep
    
    # Plot teacher trajectory
    sc1 = ax.scatter(
        teacher_pca[:, 0], teacher_pca[:, 1],
        c=norm_teacher_timesteps, cmap=teacher_cmap, marker='o', s=50, alpha=0.7, label='Teacher'
    )
    
    # Plot student trajectory
    sc2 = ax.scatter(
        student_pca[:, 0], student_pca[:, 1],
        c=norm_student_timesteps, cmap=student_cmap, marker='x', s=50, alpha=0.7, label='Student'
    )
    
    # Connect points in sequence for teacher
    for i in range(len(teacher_pca) - 1):
        ax.plot(
            [teacher_pca[i, 0], teacher_pca[i+1, 0]],
            [teacher_pca[i, 1], teacher_pca[i+1, 1]],
            'b-', alpha=0.3
        )
    
    # Connect points in sequence for student
    for i in range(len(student_pca) - 1):
        ax.plot(
            [student_pca[i, 0], student_pca[i+1, 0]],
            [student_pca[i, 1], student_pca[i+1, 1]],
            'r-', alpha=0.3
        )
    
    # Add colorbar
    cbar = plt.colorbar(sc1, ax=ax, shrink=0.7)
    cbar.set_label('Normalized Timestep')
    
    # Add labels and title
    ax.set_xlabel(f'Principal Component 1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2f})')
    ax.set_ylabel(f'Principal Component 2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2f})')
    ax.set_title(f'PCA of Diffusion Trajectories (Size Factor: {size_factor})')
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f'pca_trajectory_comparison_size_{size_factor}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_tsne_visualization(teacher_features, student_features, teacher_timesteps, student_timesteps, output_dir, size_factor):
    """
    Create a t-SNE visualization of trajectories
    
    Args:
        teacher_features: Features from teacher trajectories
        student_features: Features from student trajectories
        teacher_timesteps: Timesteps from teacher trajectories
        student_timesteps: Timesteps from student trajectories
        output_dir: Directory to save visualizations
        size_factor: Size factor of the student model for labeling
    """
    # Combine features
    combined_features = np.vstack([teacher_features, student_features])
    
    # Reduce dimensionality with PCA first to speed up t-SNE
    if combined_features.shape[1] > 50:
        pca = PCA(n_components=50)
        combined_features = pca.fit_transform(combined_features)
    
    # Fit t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_features) // 5))
    combined_tsne = tsne.fit_transform(combined_features)
    
    # Split back into teacher and student
    teacher_tsne = combined_tsne[:len(teacher_features)]
    student_tsne = combined_tsne[len(teacher_features):]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create colormaps
    teacher_cmap = plt.cm.viridis
    student_cmap = plt.cm.plasma
    
    # Normalize timesteps to [0, 1] for colormap
    max_timestep = max(np.max(teacher_timesteps), np.max(student_timesteps))
    norm_teacher_timesteps = teacher_timesteps / max_timestep
    norm_student_timesteps = student_timesteps / max_timestep
    
    # Plot teacher trajectory
    sc1 = ax.scatter(
        teacher_tsne[:, 0], teacher_tsne[:, 1],
        c=norm_teacher_timesteps, cmap=teacher_cmap, marker='o', s=50, alpha=0.7, label='Teacher'
    )
    
    # Plot student trajectory
    sc2 = ax.scatter(
        student_tsne[:, 0], student_tsne[:, 1],
        c=norm_student_timesteps, cmap=student_cmap, marker='x', s=50, alpha=0.7, label='Student'
    )
    
    # Connect points in sequence for teacher
    for i in range(len(teacher_tsne) - 1):
        ax.plot(
            [teacher_tsne[i, 0], teacher_tsne[i+1, 0]],
            [teacher_tsne[i, 1], teacher_tsne[i+1, 1]],
            'b-', alpha=0.3
        )
    
    # Connect points in sequence for student
    for i in range(len(student_tsne) - 1):
        ax.plot(
            [student_tsne[i, 0], student_tsne[i+1, 0]],
            [student_tsne[i, 1], student_tsne[i+1, 1]],
            'r-', alpha=0.3
        )
    
    # Add colorbar
    cbar = plt.colorbar(sc1, ax=ax, shrink=0.7)
    cbar.set_label('Normalized Timestep')
    
    # Add labels and title
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.set_title(f't-SNE of Diffusion Trajectories (Size Factor: {size_factor})')
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f'tsne_trajectory_comparison_size_{size_factor}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_latent_space_visualization(teacher_trajectories, student_trajectories, config, output_dir=None, size_factor=None):
    """
    Generate latent space visualization for trajectories
    
    Args:
        teacher_trajectories: List of teacher trajectories
        student_trajectories: List of student trajectories
        config: Configuration object
        output_dir: Directory to save visualizations
        size_factor: Size factor of the student model for labeling
        
    Returns:
        Dictionary of visualization results
    """
    if output_dir is None:
        output_dir = os.path.join(config.analysis_dir, "latent_space", f"size_{size_factor}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating latent space visualization for size factor {size_factor}...")
    
    # Extract features from trajectories
    teacher_features, teacher_timesteps = extract_trajectory_features(teacher_trajectories)
    student_features, student_timesteps = extract_trajectory_features(student_trajectories)
    
    # Check if we have enough data
    if len(teacher_features) < 3 or len(student_features) < 3:
        print("  Not enough trajectory data for latent space visualization.")
        return {"status": "insufficient_data"}
    
    # Create 3D trajectory plot
    create_3d_trajectory_plot(teacher_features, student_features, teacher_timesteps, student_timesteps, output_dir, size_factor)
    
    # Create trajectory comparison plot
    create_trajectory_comparison_plot(teacher_features, student_features, teacher_timesteps, student_timesteps, output_dir, size_factor)
    
    # Create t-SNE visualization
    create_tsne_visualization(teacher_features, student_features, teacher_timesteps, student_timesteps, output_dir, size_factor)
    
    # Calculate and save metrics
    results = {
        "teacher_feature_dim": teacher_features.shape[1],
        "student_feature_dim": student_features.shape[1],
        "teacher_trajectory_count": len(teacher_trajectories),
        "student_trajectory_count": len(student_trajectories),
        "teacher_points_count": len(teacher_features),
        "student_points_count": len(student_features)
    }
    
    # Save metrics
    with open(os.path.join(output_dir, f"latent_space_metrics_size_{size_factor}.txt"), "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    return results 