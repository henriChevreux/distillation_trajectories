"""
Simple trajectory comparison visualization for diffusion models.
This script generates a visualization comparing the trajectories of teacher and student models
from a random starting point.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

from utils.diffusion import get_diffusion_params

def generate_trajectory(model, noise, timesteps, device, seed=None):
    """
    Generate a trajectory from a given noise sample
    
    Args:
        model: The diffusion model
        noise: Starting noise sample
        timesteps: Number of timesteps in the diffusion process
        device: Device to run on
        seed: Random seed for reproducibility
        
    Returns:
        List of images representing the trajectory
    """
    model.eval()
    trajectory = []
    
    # Make a copy of the noise to avoid modifying the original
    x = noise.clone().to(device)
    
    # Get diffusion parameters
    diffusion_params = get_diffusion_params(timesteps)
    
    # Calculate alphas from betas (not directly provided by get_diffusion_params)
    alphas = 1.0 - diffusion_params['betas']
    
    # Record the starting point
    trajectory.append(x.detach().cpu())
    
    # Set seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Denoise step by step
    with torch.no_grad():
        for t in tqdm(range(timesteps - 1, -1, -1), desc="Generating trajectory"):
            t_tensor = torch.tensor([t], device=device)
            
            # Predict noise
            noise_pred = model(x, t_tensor)
            
            # Update x
            if t > 0:
                # Sample noise for the next step - use deterministic noise if seed is provided
                if seed is not None:
                    # Set seed for this specific timestep to ensure reproducibility
                    # but allow different noise at different timesteps
                    step_seed = seed + t
                    torch.manual_seed(step_seed)
                    np.random.seed(step_seed)
                
                noise = torch.randn_like(x)
                
                # Get alpha values
                alpha_t = alphas[t]
                alpha_t_prev = alphas[t-1] if t > 0 else torch.tensor(1.0, device=device)
                
                # Compute coefficients
                c1 = torch.sqrt(alpha_t_prev) / torch.sqrt(alpha_t)
                c2 = torch.sqrt(1 - alpha_t_prev) - torch.sqrt(alpha_t_prev / alpha_t) * torch.sqrt(1 - alpha_t)
                
                # Update x
                x = c1 * x - c2 * noise_pred
                
                # Add noise for the next step
                sigma_t = torch.sqrt(1 - alpha_t_prev) * torch.sqrt(1 - alpha_t / alpha_t_prev)
                x = x + sigma_t * noise
            
            # Record the current state
            trajectory.append(x.detach().cpu())
    
    return trajectory

def visualize_trajectories(teacher_trajectory, student_trajectory, output_dir, size_factor):
    """
    Visualize the trajectories using PCA
    
    Args:
        teacher_trajectory: List of images from teacher model
        student_trajectory: List of images from student model
        output_dir: Directory to save the visualization
        size_factor: Size factor of the student model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert trajectories to numpy arrays
    teacher_features = []
    student_features = []
    
    for img in teacher_trajectory:
        # Flatten the image
        teacher_features.append(img.reshape(-1).numpy())
    
    for img in student_trajectory:
        # Flatten the image
        student_features.append(img.reshape(-1).numpy())
    
    teacher_features = np.array(teacher_features)
    student_features = np.array(student_features)
    
    # Check if the features have different dimensions
    if teacher_features.shape[1] != student_features.shape[1]:
        print(f"Feature dimensions don't match - teacher: {teacher_features.shape[1]}, student: {student_features.shape[1]}")
        
        # Determine which has more features (usually the teacher)
        if teacher_features.shape[1] > student_features.shape[1]:
            # Pad student features with zeros
            padding_size = teacher_features.shape[1] - student_features.shape[1]
            student_features_padded = np.pad(student_features, ((0, 0), (0, padding_size)), 'constant', constant_values=0)
            print(f"Padded student features from {student_features.shape} to {student_features_padded.shape}")
            student_features = student_features_padded
        else:
            # Pad teacher features with zeros
            padding_size = student_features.shape[1] - teacher_features.shape[1]
            teacher_features_padded = np.pad(teacher_features, ((0, 0), (0, padding_size)), 'constant', constant_values=0)
            print(f"Padded teacher features from {teacher_features.shape} to {teacher_features_padded.shape}")
            teacher_features = teacher_features_padded
    
    # Calculate MSE between trajectories to verify they are similar
    mse_values = []
    for t_feat, s_feat in zip(teacher_features, student_features):
        mse = np.mean((t_feat - s_feat) ** 2)
        mse_values.append(mse)
    mean_mse = np.mean(mse_values)
    print(f"Mean MSE between trajectories before PCA: {mean_mse:.10f}")
    
    # Fit PCA on teacher trajectory only to establish a reference frame
    pca = PCA(n_components=2)
    teacher_pca = pca.fit_transform(teacher_features)
    
    # Transform student trajectory using the same PCA model
    student_pca = pca.transform(student_features)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create timestep arrays
    teacher_timesteps = np.arange(len(teacher_trajectory))
    student_timesteps = np.arange(len(student_trajectory))
    
    # Plot teacher trajectory
    plt.scatter(teacher_pca[:, 0], teacher_pca[:, 1], c=teacher_timesteps, 
               cmap='viridis', marker='o', alpha=0.7, label='Teacher')
    
    # Connect points in sequence for teacher
    for i in range(len(teacher_pca) - 1):
        plt.plot(
            [teacher_pca[i, 0], teacher_pca[i+1, 0]],
            [teacher_pca[i, 1], teacher_pca[i+1, 1]],
            'b-', alpha=0.3
        )
    
    # Plot student trajectory
    plt.scatter(student_pca[:, 0], student_pca[:, 1], c=student_timesteps, 
               cmap='plasma', marker='x', alpha=0.7, label='Student')
    
    # Connect points in sequence for student
    for i in range(len(student_pca) - 1):
        plt.plot(
            [student_pca[i, 0], student_pca[i+1, 0]],
            [student_pca[i, 1], student_pca[i+1, 1]],
            'r-', alpha=0.3
        )
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Timestep')
    
    # Add labels and title
    plt.xlabel(f'Principal Component 1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2f})')
    plt.ylabel(f'Principal Component 2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2f})')
    plt.title(f'PCA of Diffusion Trajectories (Size Factor: {size_factor})')
    plt.legend()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f'simple_trajectory_comparison_size_{size_factor}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a 3D PCA plot if we have enough components
    if teacher_features.shape[1] >= 3:
        # Fit 3D PCA on teacher trajectory only
        pca3d = PCA(n_components=3)
        teacher_pca3d = pca3d.fit_transform(teacher_features)
        
        # Transform student trajectory using the same PCA model
        student_pca3d = pca3d.transform(student_features)
        
        # Create 3D figure
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot teacher trajectory
        sc1 = ax.scatter(teacher_pca3d[:, 0], teacher_pca3d[:, 1], teacher_pca3d[:, 2], 
                        c=teacher_timesteps, cmap='viridis', marker='o', alpha=0.7, label='Teacher')
        
        # Connect points in sequence for teacher
        for i in range(len(teacher_pca3d) - 1):
            ax.plot(
                [teacher_pca3d[i, 0], teacher_pca3d[i+1, 0]],
                [teacher_pca3d[i, 1], teacher_pca3d[i+1, 1]],
                [teacher_pca3d[i, 2], teacher_pca3d[i+1, 2]],
                'b-', alpha=0.3
            )
        
        # Plot student trajectory
        sc2 = ax.scatter(student_pca3d[:, 0], student_pca3d[:, 1], student_pca3d[:, 2], 
                        c=student_timesteps, cmap='plasma', marker='x', alpha=0.7, label='Student')
        
        # Connect points in sequence for student
        for i in range(len(student_pca3d) - 1):
            ax.plot(
                [student_pca3d[i, 0], student_pca3d[i+1, 0]],
                [student_pca3d[i, 1], student_pca3d[i+1, 1]],
                [student_pca3d[i, 2], student_pca3d[i+1, 2]],
                'r-', alpha=0.3
            )
        
        # Add colorbar
        cbar = plt.colorbar(sc1)
        cbar.set_label('Timestep')
        
        # Add labels and title
        ax.set_xlabel(f'PC1 (Var: {pca3d.explained_variance_ratio_[0]:.2f})')
        ax.set_ylabel(f'PC2 (Var: {pca3d.explained_variance_ratio_[1]:.2f})')
        ax.set_zlabel(f'PC3 (Var: {pca3d.explained_variance_ratio_[2]:.2f})')
        ax.set_title(f'3D PCA of Diffusion Trajectories (Size Factor: {size_factor})')
        plt.legend()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f'simple_3d_trajectory_comparison_size_{size_factor}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def compare_trajectories(teacher_model, student_model, config, size_factor=1.0):
    """
    Compare trajectories of teacher and student models
    
    Args:
        teacher_model: The teacher diffusion model
        student_model: The student diffusion model
        config: Configuration object
        size_factor: Size factor of the student model
    """
    # Set device
    device = next(teacher_model.parameters()).device
    
    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate random noise
    noise = torch.randn(1, config.channels, config.image_size, config.image_size)
    
    print(f"Generating teacher trajectory for size factor {size_factor}...")
    teacher_trajectory = generate_trajectory(teacher_model, noise, config.timesteps, device, seed=seed)
    
    print(f"Generating student trajectory for size factor {size_factor}...")
    student_trajectory = generate_trajectory(student_model, noise, config.timesteps, device, seed=seed)
    
    # Create output directory
    output_dir = os.path.join(config.analysis_dir, "trajectory_comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Visualizing trajectories for size factor {size_factor}...")
    visualize_trajectories(teacher_trajectory, student_trajectory, output_dir, size_factor)
    
    print(f"Trajectory comparison completed for size factor {size_factor}")
    
    return {
        "teacher_trajectory_length": len(teacher_trajectory),
        "student_trajectory_length": len(student_trajectory)
    }

if __name__ == "__main__":
    # This script is meant to be imported and used by run_analysis.py
    print("This script is meant to be imported and used by run_analysis.py") 