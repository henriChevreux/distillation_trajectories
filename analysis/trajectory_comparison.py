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
import matplotlib.patches as mpatches
from matplotlib.path import Path

from utils.diffusion import get_diffusion_params
from analysis.metrics.trajectory_metrics import compute_trajectory_metrics

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

def create_radar_plot(metrics, output_dir, size_factor):
    """
    Create a radar plot for trajectory metrics
    
    Args:
        metrics: Dictionary of metrics
        output_dir: Directory to save the visualization
        size_factor: Size factor of the student model
    """
    # Define the metrics to include in the radar plot
    radar_metrics = [
        ('point_by_point_similarity', 'Point-by-Point\nSimilarity'),
        ('log_mse_similarity', 'Log MSE\nSimilarity'),
        ('weighted_directional_consistency', 'Weighted Directional\nConsistency'),
        ('path_alignment', 'Path\nAlignment')
    ]
    
    # Extract values for each metric
    values = []
    labels = []
    
    for key, label in radar_metrics:
        if key in metrics:
            # Ensure value is between 0 and 1
            value = max(0, min(1, metrics[key]))
            values.append(value)
            labels.append(label)
            print(f"  {label}: {value:.4f}")
    
    # Create radar plot
    fig = plt.figure(figsize=(10, 10))
    
    # Create a figure with two subplots - radar plot and text summary
    gs = plt.GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0, 0], polar=True)
    
    # Number of metrics
    N = len(values)
    
    # Angle of each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Values for each metric
    values += values[:1]  # Close the loop
    
    # Draw the plot
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='#3498db')
    ax.fill(angles, values, alpha=0.25, color='#3498db')
    
    # Set the angle labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    
    # Add grid lines
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.grid(True)
    
    # Add title with traditional metrics for comparison
    title = f'Enhanced Trajectory Metrics (Size Factor: {size_factor})'
    plt.title(title, size=15, y=1.1)
    
    # Add a text box with traditional metrics
    traditional_metrics_text = (
        f"Traditional Metrics:\n"
        f"Path Length Similarity: {metrics.get('path_length_similarity', 0):.4f}\n"
        f"MSE: {metrics.get('mse', 0):.6f}\n"
        f"Mean Directional Consistency: {metrics.get('mean_directional_consistency', 0):.4f}\n"
        f"Distribution Similarity: {metrics.get('distribution_similarity', 0):.4f}"
    )
    
    # Add text box to the figure
    plt.figtext(0.5, 0.01, traditional_metrics_text, ha='center', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Save the radar plot
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust layout to make room for text
    plt.savefig(os.path.join(output_dir, f'enhanced_metrics_radar_size_{size_factor}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Radar plot saved to {os.path.join(output_dir, f'enhanced_metrics_radar_size_{size_factor}.png')}")

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
    
    # Compute enhanced metrics and create radar plot
    print("\nComputing enhanced trajectory metrics...")
    metrics = compute_trajectory_metrics(teacher_trajectory, student_trajectory)
    
    # Create radar plot with enhanced metrics
    create_radar_plot(metrics, output_dir, size_factor)
    
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
        
    Returns:
        Tuple of (teacher_trajectory, student_trajectory)
    """
    # Set device
    device = next(teacher_model.parameters()).device
    
    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate random noise
    noise = torch.randn(1, config.channels, config.image_size, config.image_size)
    
    # Generate trajectories
    print("\nGenerating trajectories...")
    print("Generating teacher trajectory...")
    teacher_trajectory = generate_trajectory(teacher_model, noise, config.timesteps, device, seed=seed)
    print("Generating student trajectory...")
    student_trajectory = generate_trajectory(student_model, noise, config.timesteps, device, seed=seed)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'trajectory_comparison')
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize trajectories
    print("\nVisualizing trajectories...")
    visualize_trajectories(teacher_trajectory, student_trajectory, output_dir, size_factor)
    
    # Compute and print enhanced metrics
    print("\nEnhanced Trajectory Metrics Summary:")
    metrics = compute_trajectory_metrics(teacher_trajectory, student_trajectory)
    
    # Print the new metrics with explanations
    print(f"  Point-by-Point Similarity: {metrics.get('point_by_point_similarity', 0):.4f}")
    print(f"    (Measures direct correspondence between points in trajectories)")
    
    print(f"  Log MSE Similarity: {metrics.get('log_mse_similarity', 0):.4f}")
    print(f"    (Logarithmic transformation of MSE to better highlight differences)")
    
    print(f"  Weighted Directional Consistency: {metrics.get('weighted_directional_consistency', 0):.4f}")
    print(f"    (Directional consistency weighted by step magnitude)")
    
    print(f"  Path Alignment: {metrics.get('path_alignment', 0):.4f}")
    print(f"    (Measures how closely student follows teacher's exact path)")
    
    # Also print traditional metrics for comparison
    print("\nTraditional Metrics (for comparison):")
    print(f"  Path Length Similarity: {metrics.get('path_length_similarity', 0):.4f}")
    print(f"    (Only compares total path lengths, not actual paths)")
    
    print(f"  MSE: {metrics.get('mse', 0):.6f}")
    print(f"  MSE Similarity (1-MSE): {1.0 - metrics.get('mse', 0):.6f}")
    print(f"    (Very close to 1.0 for all models, masking differences)")
    
    print(f"  Mean Directional Consistency: {metrics.get('mean_directional_consistency', 0):.4f}")
    print(f"    (Doesn't account for magnitude of movements)")
    
    print(f"  Distribution Similarity: {metrics.get('distribution_similarity', 0):.4f}")
    print(f"    (Overall distribution similarity, not path-specific)")
    
    # Print a summary comparison
    print("\nMetric Comparison Summary:")
    print(f"  Size Factor: {size_factor}")
    print(f"  Traditional Path Length Similarity: {metrics.get('path_length_similarity', 0):.4f}")
    print(f"  Enhanced Path Alignment: {metrics.get('path_alignment', 0):.4f}")
    print(f"  Difference: {metrics.get('path_length_similarity', 0) - metrics.get('path_alignment', 0):.4f}")
    
    # Save metrics to a file for later reference
    metrics_file = os.path.join(output_dir, f'metrics_summary_size_{size_factor}.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Trajectory Metrics Summary for Size Factor {size_factor}\n")
        f.write(f"=============================================\n\n")
        
        f.write("Enhanced Metrics:\n")
        f.write(f"  Point-by-Point Similarity: {metrics.get('point_by_point_similarity', 0):.4f}\n")
        f.write(f"  Log MSE Similarity: {metrics.get('log_mse_similarity', 0):.4f}\n")
        f.write(f"  Weighted Directional Consistency: {metrics.get('weighted_directional_consistency', 0):.4f}\n")
        f.write(f"  Path Alignment: {metrics.get('path_alignment', 0):.4f}\n\n")
        
        f.write("Traditional Metrics:\n")
        f.write(f"  Path Length Similarity: {metrics.get('path_length_similarity', 0):.4f}\n")
        f.write(f"  MSE: {metrics.get('mse', 0):.6f}\n")
        f.write(f"  MSE Similarity (1-MSE): {1.0 - metrics.get('mse', 0):.6f}\n")
        f.write(f"  Mean Directional Consistency: {metrics.get('mean_directional_consistency', 0):.4f}\n")
        f.write(f"  Distribution Similarity: {metrics.get('distribution_similarity', 0):.4f}\n\n")
        
        f.write("Raw Measurements:\n")
        f.write(f"  Teacher Path Length: {metrics.get('teacher_path_length', 0):.4f}\n")
        f.write(f"  Student Path Length: {metrics.get('student_path_length', 0):.4f}\n")
        f.write(f"  Path Length Ratio: {metrics.get('path_length_ratio', 0):.4f}\n")
    
    print(f"\nDetailed metrics saved to: {metrics_file}")
    
    return teacher_trajectory, student_trajectory

if __name__ == "__main__":
    # This script is meant to be imported and used by run_analysis.py
    print("This script is meant to be imported and used by run_analysis.py") 