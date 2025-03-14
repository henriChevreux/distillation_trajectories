"""
Dimensionality reduction analysis for diffusion trajectories
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.colors import LinearSegmentedColormap

def extract_trajectory_features(trajectories):
    """
    Extract features from trajectories for dimensionality reduction
    
    Args:
        trajectories: List of trajectories, where each trajectory is a list of (image, timestep) pairs
        
    Returns:
        Flattened features for each trajectory point
    """
    # Extract images from trajectories
    all_images = []
    all_timesteps = []
    
    # Check if trajectories is empty
    if not trajectories:
        print("  Warning: Empty trajectories provided. Returning default empty arrays.")
        return np.array([]).reshape(0, 1), np.array([])
    
    # Print trajectory information for debugging
    print(f"  Processing {len(trajectories)} trajectories")
    
    for i, trajectory in enumerate(trajectories):
        if not trajectory:  # Skip empty trajectories
            print(f"  Warning: Trajectory {i} is empty, skipping")
            continue
            
        print(f"  Trajectory {i} has {len(trajectory)} points")
        for j, (img, timestep) in enumerate(trajectory):
            # Convert to numpy and flatten
            if isinstance(img, torch.Tensor):
                try:
                    img = img.cpu().numpy()
                except Exception as e:
                    print(f"  Error converting tensor to numpy for trajectory {i}, point {j}: {e}")
                    continue
            
            # Print shape information for debugging
            if j == 0:  # Only print for first point to avoid clutter
                print(f"  Image shape for trajectory {i}: {img.shape}")
            
            # Flatten the image
            try:
                img_flat = img.reshape(1, -1)
                all_images.append(img_flat)
                all_timesteps.append(timestep)
            except Exception as e:
                print(f"  Error flattening image for trajectory {i}, point {j}: {e}")
                continue
    
    # Check if we have any valid images
    if not all_images:
        print("  Warning: No valid images found in trajectories. Returning empty arrays.")
        return np.array([]).reshape(0, 1), np.array([])
    
    # Stack all flattened images
    try:
        features = np.vstack(all_images)
        timesteps = np.array(all_timesteps)
    except Exception as e:
        print(f"  Error stacking features: {e}")
        # Return empty arrays as fallback
        return np.array([]).reshape(0, 1), np.array([])
        
    print(f"  Extracted features shape: {features.shape}")
    return features, timesteps

def perform_pca(teacher_features, student_features, teacher_timesteps, student_timesteps, output_dir, size_factor):
    """
    Perform PCA on trajectory features
    
    Args:
        teacher_features: Features from teacher trajectories
        student_features: Features from student trajectories
        teacher_timesteps: Timesteps from teacher trajectories
        student_timesteps: Timesteps from student trajectories
        output_dir: Directory to save visualizations
        size_factor: Size factor of the student model for labeling
    """
    print("  Performing PCA...")
    
    # Check if the features have different dimensions
    if teacher_features.shape[1] != student_features.shape[1]:
        print(f"  Warning: Feature dimensions don't match - teacher: {teacher_features.shape[1]}, student: {student_features.shape[1]}")
        
        # Determine which has more features (usually the teacher)
        if teacher_features.shape[1] > student_features.shape[1]:
            # Pad student features with zeros
            padding_size = teacher_features.shape[1] - student_features.shape[1]
            student_features_padded = np.pad(student_features, ((0, 0), (0, padding_size)), 'constant', constant_values=0)
            print(f"  Padded student features from {student_features.shape} to {student_features_padded.shape}")
            student_features = student_features_padded
        else:
            # Pad teacher features with zeros
            padding_size = student_features.shape[1] - teacher_features.shape[1]
            teacher_features_padded = np.pad(teacher_features, ((0, 0), (0, padding_size)), 'constant', constant_values=0)
            print(f"  Padded teacher features from {teacher_features.shape} to {teacher_features_padded.shape}")
            teacher_features = teacher_features_padded
    
    # Combine features for PCA fitting
    combined_features = np.vstack([teacher_features, student_features])
    
    # Fit PCA
    pca = PCA(n_components=2)
    combined_pca = pca.fit_transform(combined_features)
    
    # Split back into teacher and student
    teacher_pca = combined_pca[:len(teacher_features)]
    student_pca = combined_pca[len(teacher_features):]
    
    # Create colormap based on timesteps
    max_timestep = max(np.max(teacher_timesteps), np.max(student_timesteps))
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot teacher trajectories
    sc1 = plt.scatter(teacher_pca[:, 0], teacher_pca[:, 1], c=teacher_timesteps, 
                     cmap='viridis', marker='o', alpha=0.7, label='Teacher')
    
    # Plot student trajectories
    sc2 = plt.scatter(student_pca[:, 0], student_pca[:, 1], c=student_timesteps, 
                     cmap='plasma', marker='x', alpha=0.7, label='Student')
    
    # Add colorbar
    cbar = plt.colorbar(sc1)
    cbar.set_label('Timestep')
    
    # Add labels and title
    plt.xlabel(f'Principal Component 1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2f})')
    plt.ylabel(f'Principal Component 2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2f})')
    plt.title(f'PCA of Diffusion Trajectories (Size Factor: {size_factor})')
    plt.legend()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f'pca_trajectories_size_{size_factor}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a 3D PCA plot if we have enough components
    if combined_features.shape[1] >= 3:
        pca = PCA(n_components=3)
        combined_pca = pca.fit_transform(combined_features)
        
        # Split back into teacher and student
        teacher_pca = combined_pca[:len(teacher_features)]
        student_pca = combined_pca[len(teacher_features):]
        
        # Create 3D figure
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot teacher trajectories
        sc1 = ax.scatter(teacher_pca[:, 0], teacher_pca[:, 1], teacher_pca[:, 2], 
                        c=teacher_timesteps, cmap='viridis', marker='o', alpha=0.7, label='Teacher')
        
        # Plot student trajectories
        sc2 = ax.scatter(student_pca[:, 0], student_pca[:, 1], student_pca[:, 2], 
                        c=student_timesteps, cmap='plasma', marker='x', alpha=0.7, label='Student')
        
        # Add colorbar
        cbar = plt.colorbar(sc1)
        cbar.set_label('Timestep')
        
        # Add labels and title
        ax.set_xlabel(f'PC1 (Var: {pca.explained_variance_ratio_[0]:.2f})')
        ax.set_ylabel(f'PC2 (Var: {pca.explained_variance_ratio_[1]:.2f})')
        ax.set_zlabel(f'PC3 (Var: {pca.explained_variance_ratio_[2]:.2f})')
        ax.set_title(f'3D PCA of Diffusion Trajectories (Size Factor: {size_factor})')
        plt.legend()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f'pca_3d_trajectories_size_{size_factor}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def perform_tsne(teacher_features, student_features, teacher_timesteps, student_timesteps, output_dir, size_factor):
    """
    Perform t-SNE on trajectory features
    
    Args:
        teacher_features: Features from teacher trajectories
        student_features: Features from student trajectories
        teacher_timesteps: Timesteps from teacher trajectories
        student_timesteps: Timesteps from student trajectories
        output_dir: Directory to save visualizations
        size_factor: Size factor of the student model for labeling
    """
    print("  Performing t-SNE...")
    
    # Combine features for t-SNE fitting
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
    plt.figure(figsize=(12, 10))
    
    # Plot teacher trajectories
    sc1 = plt.scatter(teacher_tsne[:, 0], teacher_tsne[:, 1], c=teacher_timesteps, 
                     cmap='viridis', marker='o', alpha=0.7, label='Teacher')
    
    # Plot student trajectories
    sc2 = plt.scatter(student_tsne[:, 0], student_tsne[:, 1], c=student_timesteps, 
                     cmap='plasma', marker='x', alpha=0.7, label='Student')
    
    # Add colorbar
    cbar = plt.colorbar(sc1)
    cbar.set_label('Timestep')
    
    # Add labels and title
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title(f't-SNE of Diffusion Trajectories (Size Factor: {size_factor})')
    plt.legend()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f'tsne_trajectories_size_{size_factor}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def visualize_trajectory_comparison(teacher_trajectories, student_trajectories, output_dir, size_factor):
    """
    Visualize a comparison of teacher and student trajectories
    
    Args:
        teacher_trajectories: List of teacher trajectories
        student_trajectories: List of student trajectories
        output_dir: Directory to save visualizations
        size_factor: Size factor of the student model for labeling
    """
    print("  Visualizing trajectory comparison...")
    
    # Select a single trajectory for visualization
    teacher_traj = teacher_trajectories[0]
    student_traj = student_trajectories[0]
    
    # Get number of timesteps
    n_timesteps = len(teacher_traj)
    
    # Select a subset of timesteps to visualize
    # Ensure we're showing the trajectory from noise (t=high) to clean (t=0)
    timesteps_to_show = min(10, n_timesteps)
    indices = np.linspace(0, n_timesteps - 1, timesteps_to_show, dtype=int)
    
    # Create figure
    fig, axes = plt.subplots(2, timesteps_to_show, figsize=(20, 5))
    fig.suptitle(f'Trajectory Comparison (Size Factor: {size_factor})', fontsize=16)
    
    # Plot teacher trajectory
    for i, idx in enumerate(indices):
        img, timestep = teacher_traj[idx]
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        # Reshape if needed
        if len(img.shape) == 4:  # [B, C, H, W]
            img = img[0]
        
        # Convert to appropriate format for display
        if img.shape[0] == 1:  # Grayscale
            img = img[0]
            axes[0, i].imshow(img, cmap='gray')
        else:  # RGB
            img = np.transpose(img, (1, 2, 0))
            # Normalize to [0, 1]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            axes[0, i].imshow(img)
        
        axes[0, i].set_title(f't={timestep}')
        axes[0, i].axis('off')
    
    # Plot student trajectory
    for i, idx in enumerate(indices):
        img, timestep = student_traj[idx]
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        
        # Reshape if needed
        if len(img.shape) == 4:  # [B, C, H, W]
            img = img[0]
        
        # Convert to appropriate format for display
        if img.shape[0] == 1:  # Grayscale
            img = img[0]
            axes[1, i].imshow(img, cmap='gray')
        else:  # RGB
            img = np.transpose(img, (1, 2, 0))
            # Normalize to [0, 1]
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            axes[1, i].imshow(img)
        
        axes[1, i].set_title(f't={timestep}')
        axes[1, i].axis('off')
    
    # Add row labels
    axes[0, 0].set_ylabel('Teacher', fontsize=14)
    axes[1, 0].set_ylabel('Student', fontsize=14)
    
    # Add a note about the diffusion process direction
    plt.figtext(0.5, 0.01, 'Diffusion Process: Noise (left) → Clean Image (right)', 
                ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'trajectory_comparison_size_{size_factor}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def dimensionality_reduction_analysis(teacher_trajectories, student_trajectories, config, output_dir=None, size_factor=None):
    """
    Perform dimensionality reduction analysis on trajectories
    
    Args:
        teacher_trajectories: List of teacher trajectories
        student_trajectories: List of student trajectories
        config: Configuration object
        output_dir: Directory to save visualizations
        size_factor: Size factor of the student model for labeling
        
    Returns:
        Dictionary of analysis results
    """
    if output_dir is None:
        output_dir = os.path.join(config.analysis_dir, "dimensionality", f"size_{size_factor}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Performing dimensionality reduction analysis for size factor {size_factor}...")
    
    # Check for empty trajectory lists
    if not teacher_trajectories or not student_trajectories:
        print(f"  WARNING: Empty trajectory list provided - teacher: {len(teacher_trajectories)}, student: {len(student_trajectories)}")
        print("  Skipping dimensionality reduction analysis.")
        return {"status": "skipped", "reason": "empty_trajectories"}
    
    # Print trajectory information for debugging
    print(f"  Teacher trajectories: {len(teacher_trajectories)}")
    print(f"  Student trajectories: {len(student_trajectories)}")
    
    # Extract features from trajectories
    teacher_features, teacher_timesteps = extract_trajectory_features(teacher_trajectories)
    student_features, student_timesteps = extract_trajectory_features(student_trajectories)
    
    # Check if we have enough data
    if len(teacher_features) < 2 or len(student_features) < 2:
        print("  Not enough trajectory data for dimensionality reduction analysis.")
        print(f"  Teacher features: {teacher_features.shape}, Student features: {student_features.shape}")
        return {"status": "skipped", "reason": "insufficient_data"}
    
    # Check if one of the arrays is empty
    if teacher_features.size == 0 or student_features.size == 0:
        print("  One of the feature sets is empty. Skipping dimensionality reduction.")
        print(f"  Teacher features: {teacher_features.shape}, Student features: {student_features.shape}")
        return {"status": "skipped", "reason": "empty_features"}
        
    # Create visualizations
    try:
        perform_pca(teacher_features, student_features, teacher_timesteps, student_timesteps, output_dir, size_factor)
        perform_tsne(teacher_features, student_features, teacher_timesteps, student_timesteps, output_dir, size_factor)
        
        # If we have a teacher and student trajectory, visualize one of them
        if teacher_trajectories and student_trajectories:
            visualize_trajectory_comparison(teacher_trajectories, student_trajectories, output_dir, size_factor)
        
        return {"status": "success"}
    except Exception as e:
        print(f"  Error during dimensionality reduction: {e}")
        # Print more detailed error information
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)} 