"""
Classifier-Free Guidance (CFG) trajectory comparison script.
This script extends the trajectory comparison functionality to include CFG,
allowing visualization of how different guidance scales affect the trajectories
of both teacher and student models.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from config.config import Config
from models import SimpleUNet, StudentUNet
from utils.diffusion import get_diffusion_params

def extract(a, t, x_shape):
    """Extract coefficients at specified timesteps t"""
    b, *_ = t.shape
    
    # Ensure t is within bounds
    t = torch.clamp(t, 0, a.shape[0] - 1)
    
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def generate_cfg_trajectory(model, noise, timesteps, guidance_scale, device, seed=None):
    """
    Generate a trajectory using classifier-free guidance
    
    Args:
        model: The diffusion model
        noise: Starting noise sample
        timesteps: Number of timesteps in the diffusion process
        guidance_scale: The CFG guidance scale (w)
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
    
    # Record the starting point
    trajectory.append(x.detach().cpu())
    
    # Set seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Reverse diffusion process with CFG
    for i in tqdm(reversed(range(timesteps)), desc='Generating trajectory'):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        
        # Get two predictions: one conditioned (c) and one unconditioned (uc)
        with torch.no_grad():
            # Concatenate the same input twice for efficiency
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            
            # First half will be unconditioned (negative conditioning/empty token)
            # Second half will be conditioned (positive conditioning/class token)
            c = torch.cat([torch.zeros(1, 1), torch.ones(1, 1)]).to(device)
            
            # Get both predictions in a single forward pass
            pred_all = model(x_in, t_in, c)
            pred_uncond, pred_cond = pred_all.chunk(2)
            
            # Apply classifier-free guidance
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            
            # Get diffusion parameters for this timestep
            sqrt_one_minus_alphas_cumprod_t = extract(
                diffusion_params['sqrt_one_minus_alphas_cumprod'], t, x.shape
            )
            sqrt_recip_alphas_t = extract(diffusion_params['sqrt_recip_alphas'], t, x.shape)
            
            # Direction pointing to x_t
            pred_original_direction = (1. - sqrt_one_minus_alphas_cumprod_t) * pred
            
            # For the final step, completely eliminate noise and variance
            if i == 0:
                # Final deterministic step
                x = sqrt_recip_alphas_t * (x - pred_original_direction)
            else:
                # Regular step with noise
                noise = torch.randn_like(x)
                betas_t = extract(diffusion_params['betas'], t, x.shape)
                noise_scale = torch.sqrt(betas_t)
                x = sqrt_recip_alphas_t * (x - pred_original_direction) + noise * noise_scale
        
        # Record the current state
        trajectory.append(x.detach().cpu())
    
    return trajectory

def generate_trajectory_without_cfg(model, noise, timesteps, device, seed=None):
    """
    Generate a trajectory without using classifier-free guidance
    """
    model.eval()
    trajectory = []
    
    # Make a copy of the noise to avoid modifying the original
    x = noise.clone().to(device)
    
    # Get diffusion parameters
    diffusion_params = get_diffusion_params(timesteps)
    
    # Record the starting point
    trajectory.append(x.detach().cpu())
    
    # Set seed for reproducibility if provided
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Reverse diffusion process without CFG
    for i in tqdm(reversed(range(timesteps)), desc='Generating trajectory'):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        
        # Get prediction without conditioning
        with torch.no_grad():
            pred = model(x, t)
            
            # Get diffusion parameters for this timestep
            sqrt_one_minus_alphas_cumprod_t = extract(
                diffusion_params['sqrt_one_minus_alphas_cumprod'], t, x.shape
            )
            sqrt_recip_alphas_t = extract(diffusion_params['sqrt_recip_alphas'], t, x.shape)
            
            # Direction pointing to x_t
            pred_original_direction = (1. - sqrt_one_minus_alphas_cumprod_t) * pred
            
            # For the final step, completely eliminate noise and variance
            if i == 0:
                # Final deterministic step
                x = sqrt_recip_alphas_t * (x - pred_original_direction)
            else:
                # Regular step with noise
                noise = torch.randn_like(x)
                betas_t = extract(diffusion_params['betas'], t, x.shape)
                noise_scale = torch.sqrt(betas_t)
                x = sqrt_recip_alphas_t * (x - pred_original_direction) + noise * noise_scale
        
        # Record the current state
        trajectory.append(x.detach().cpu())
    
    return trajectory

def compare_cfg_trajectories(teacher_model, student_model, config, guidance_scales=[1.0, 3.0, 5.0, 7.0], size_factor=1.0):
    """
    Compare trajectories of teacher and student models with and without CFG
    """
    # Set device
    device = next(teacher_model.parameters()).device
    
    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate random noise
    noise = torch.randn(1, config.channels, config.image_size, config.image_size)
    
    # Generate trajectories for each guidance scale
    teacher_trajectories = {}
    student_trajectories = {}
    teacher_no_cfg_trajectories = {}
    student_no_cfg_trajectories = {}
    
    # First generate trajectories without CFG (only need to do this once)
    print("\nGenerating trajectories without CFG...")
    print("Generating teacher trajectory...")
    teacher_no_cfg = generate_trajectory_without_cfg(teacher_model, noise, config.timesteps, device, seed=seed)
    print("Generating student trajectory...")
    student_no_cfg = generate_trajectory_without_cfg(student_model, noise, config.timesteps, device, seed=seed)
    
    # Store no-CFG trajectories for each scale (for easier plotting)
    for w in guidance_scales:
        teacher_no_cfg_trajectories[w] = teacher_no_cfg
        student_no_cfg_trajectories[w] = student_no_cfg
    
    # Generate trajectories with CFG for each scale
    for w in guidance_scales:
        print(f"\nGenerating trajectories with guidance scale {w}...")
        
        print("Generating teacher trajectory...")
        teacher_traj = generate_cfg_trajectory(teacher_model, noise, config.timesteps, w, device, seed=seed)
        teacher_trajectories[w] = teacher_traj
        
        print("Generating student trajectory...")
        student_traj = generate_cfg_trajectory(student_model, noise, config.timesteps, w, device, seed=seed)
        student_trajectories[w] = student_traj
    
    # Create output directory
    output_dir = os.path.join(config.analysis_dir, "cfg_trajectory_comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize trajectories with and without CFG
    visualize_cfg_vs_no_cfg_trajectories(
        teacher_trajectories, student_trajectories,
        teacher_no_cfg_trajectories, student_no_cfg_trajectories,
        output_dir, guidance_scales, size_factor
    )
    
    # Visualize final images
    visualize_cfg_vs_no_cfg_final_images(
        teacher_trajectories, student_trajectories,
        teacher_no_cfg_trajectories, student_no_cfg_trajectories,
        output_dir, guidance_scales, size_factor
    )
    
    # Add the new combined visualization
    visualize_combined_cfg_trajectories(
        teacher_trajectories, student_trajectories,
        output_dir, guidance_scales, size_factor
    )
    
    # Add the new similarity visualization
    visualize_cfg_similarity_metrics(
        teacher_trajectories, student_trajectories,
        teacher_no_cfg_trajectories, student_no_cfg_trajectories,
        output_dir, guidance_scales, size_factor
    )
    
    # Add the teacher-student similarity comparison
    visualize_teacher_student_similarity(
        teacher_trajectories, student_trajectories,
        teacher_no_cfg_trajectories, student_no_cfg_trajectories,
        output_dir, guidance_scales, size_factor
    )
    
    return {
        "teacher_trajectories": teacher_trajectories,
        "student_trajectories": student_trajectories,
        "teacher_no_cfg_trajectories": teacher_no_cfg_trajectories,
        "student_no_cfg_trajectories": student_no_cfg_trajectories
    }

def visualize_cfg_vs_no_cfg_trajectories(teacher_trajectories, student_trajectories,
                                       teacher_no_cfg_trajectories, student_no_cfg_trajectories,
                                       output_dir, guidance_scales, size_factor):
    """
    Visualize trajectories with and without CFG for each guidance scale
    """
    # Create a separate plot for each guidance scale
    for g_scale in guidance_scales:
        # Get trajectories for this guidance scale
        teacher_traj = teacher_trajectories[g_scale]
        student_traj = student_trajectories[g_scale]
        teacher_no_cfg = teacher_no_cfg_trajectories[g_scale]
        student_no_cfg = student_no_cfg_trajectories[g_scale]
        
        # Convert trajectories to feature vectors
        def process_trajectory(traj):
            features = [t.cpu().numpy().reshape(-1) for t in traj]
            return np.stack(features)
        
        teacher_features = process_trajectory(teacher_traj)
        student_features = process_trajectory(student_traj)
        teacher_no_cfg_features = process_trajectory(teacher_no_cfg)
        student_no_cfg_features = process_trajectory(student_no_cfg)
        
        # Fit PCA on teacher's no-CFG trajectory only (for consistency with regular trajectory comparison)
        pca = PCA(n_components=2)
        teacher_no_cfg_pca = pca.fit_transform(teacher_no_cfg_features)
        
        # Project all other trajectories into the same PCA space
        teacher_pca = pca.transform(teacher_features)
        student_pca = pca.transform(student_features)
        student_no_cfg_pca = pca.transform(student_no_cfg_features)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot trajectories
        plt.plot(teacher_pca[:, 0], teacher_pca[:, 1],
                '-o', color='blue', alpha=0.7, markersize=4,
                label=f'Teacher (CFG w={g_scale})')
        plt.plot(student_pca[:, 0], student_pca[:, 1],
                '--s', color='red', alpha=0.7, markersize=4,
                label=f'Student (CFG w={g_scale})')
        plt.plot(teacher_no_cfg_pca[:, 0], teacher_no_cfg_pca[:, 1],
                '-o', color='lightblue', alpha=0.7, markersize=4,
                label='Teacher (No CFG)')
        plt.plot(student_no_cfg_pca[:, 0], student_no_cfg_pca[:, 1],
                '--s', color='lightcoral', alpha=0.7, markersize=4,
                label='Student (No CFG)')
        
        plt.title(f'Trajectory Comparison (Guidance Scale {g_scale})\nStudent Size Factor: {size_factor}')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.legend()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f'cfg_vs_no_cfg_trajectories_w{g_scale}_size_{size_factor}.png'),
                    bbox_inches='tight', dpi=300)
        plt.close()

def visualize_cfg_vs_no_cfg_final_images(teacher_trajectories, student_trajectories,
                                       teacher_no_cfg_trajectories, student_no_cfg_trajectories,
                                       output_dir, guidance_scales, size_factor):
    """
    Visualize final images with and without CFG for each guidance scale
    """
    num_guidance_scales = len(guidance_scales)
    fig, axes = plt.subplots(4, num_guidance_scales, figsize=(4 * num_guidance_scales, 16))
    
    for i, guidance_scale in enumerate(guidance_scales):
        # Get final images for this guidance scale
        t_img_cfg = teacher_trajectories[guidance_scale][-1]  # Last image in trajectory
        s_img_cfg = student_trajectories[guidance_scale][-1]  # Last image in trajectory
        t_img_no_cfg = teacher_no_cfg_trajectories[guidance_scale][-1]  # Last image in trajectory
        s_img_no_cfg = student_no_cfg_trajectories[guidance_scale][-1]  # Last image in trajectory
        
        # Convert tensors to numpy arrays with correct normalization
        def process_image(img):
            return img.squeeze(0).permute(1, 2, 0).clamp(-1, 1).cpu().numpy() * 0.5 + 0.5
        
        t_img_cfg = process_image(t_img_cfg)
        s_img_cfg = process_image(s_img_cfg)
        t_img_no_cfg = process_image(t_img_no_cfg)
        s_img_no_cfg = process_image(s_img_no_cfg)
        
        # Plot images
        axes[0, i].imshow(t_img_cfg)
        axes[0, i].set_title(f'Teacher (CFG w={guidance_scale})')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(s_img_cfg)
        axes[1, i].set_title(f'Student (CFG w={guidance_scale})')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(t_img_no_cfg)
        axes[2, i].set_title('Teacher (No CFG)')
        axes[2, i].axis('off')
        
        axes[3, i].imshow(s_img_no_cfg)
        axes[3, i].set_title('Student (No CFG)')
        axes[3, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cfg_vs_no_cfg_final_images_size_{size_factor}.png'))
    plt.close()

def visualize_combined_cfg_trajectories(teacher_trajectories, student_trajectories,
                                      output_dir, guidance_scales, size_factor):
    """
    Visualize all CFG trajectories in a single plot with a color gradient
    
    This creates a combined visualization showing how different guidance scales
    affect the trajectories of both teacher and student models.
    """
    # Convert trajectories to feature vectors
    def process_trajectory(traj):
        features = [t.cpu().numpy().reshape(-1) for t in traj]
        return np.stack(features)
    
    # Get a reference trajectory (teacher with no CFG) for PCA
    reference_trajectory = teacher_trajectories[guidance_scales[0]]
    reference_features = process_trajectory(reference_trajectory)
    
    # Fit PCA on reference trajectory
    pca = PCA(n_components=2)
    reference_pca = pca.fit_transform(reference_features)
    
    # Create figure with a specific size and layout for the colorbar
    fig, ax = plt.subplots(figsize=(12, 10))
    
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
        
        # Plot teacher trajectory
        ax.plot(teacher_pca[:, 0], teacher_pca[:, 1],
                '-o', color=color, alpha=0.8, markersize=4,
                label=f'Teacher (w={g_scale})')
        
        # Plot student trajectory
        ax.plot(student_pca[:, 0], student_pca[:, 1],
                '--s', color=color, alpha=0.8, markersize=4,
                label=f'Student (w={g_scale})')
    
    # Create custom legend with one entry per guidance scale
    legend_elements = []
    for g_scale in guidance_scales:
        color = cmap(norm(g_scale))
        # Teacher (solid line)
        legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, ls='-', marker='o', markersize=4,
                                         label=f'Teacher (w={g_scale})'))
        # Student (dashed line)
        legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, ls='--', marker='s', markersize=4,
                                         label=f'Student (w={g_scale})'))
    
    # Add legend outside the plot
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label='Guidance Scale')
    
    # Add labels and title
    ax.set_title(f'Trajectory Comparison with CFG\n(Student Size Factor: {size_factor})')
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    
    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Save plot
    plt.savefig(os.path.join(output_dir, f'combined_cfg_trajectories_size_{size_factor}.png'),
               bbox_inches='tight', dpi=300)
    plt.close()

def visualize_cfg_similarity_metrics(teacher_trajectories, student_trajectories,
                                   teacher_no_cfg_trajectories, student_no_cfg_trajectories,
                                   output_dir, guidance_scales, size_factor):
    """
    Visualize the impact of CFG on trajectory similarity
    
    This function calculates and visualizes various similarity metrics between
    trajectories with and without CFG across different guidance scales.
    """
    # Initialize lists to store metrics
    cosine_similarities_teacher = []
    cosine_similarities_student = []
    euclidean_distances_teacher = []
    euclidean_distances_student = []
    
    # Process trajectories to feature vectors
    def process_trajectory(traj):
        features = [t.cpu().numpy().reshape(-1) for t in traj]
        return np.stack(features)
    
    # Calculate cosine similarity between two feature matrices
    def cosine_similarity_trajectories(traj1, traj2):
        # Ensure both trajectories have the same number of steps
        min_steps = min(len(traj1), len(traj2))
        similarities = []
        
        for i in range(min_steps):
            # Flatten the images to 1D vectors
            vec1 = traj1[i].reshape(1, -1)
            vec2 = traj2[i].reshape(1, -1)
            
            # Calculate cosine similarity
            dot_product = np.sum(vec1 * vec2)
            norm1 = np.sqrt(np.sum(vec1 ** 2))
            norm2 = np.sqrt(np.sum(vec2 ** 2))
            
            if norm1 * norm2 > 0:  # Avoid division by zero
                similarity = dot_product / (norm1 * norm2)
            else:
                similarity = 0
                
            similarities.append(similarity)
        
        return np.array(similarities)
    
    # Calculate Euclidean distance between two feature matrices
    def euclidean_distance_trajectories(traj1, traj2):
        # Ensure both trajectories have the same number of steps
        min_steps = min(len(traj1), len(traj2))
        distances = []
        
        for i in range(min_steps):
            # Flatten the images to 1D vectors
            vec1 = traj1[i].reshape(-1)
            vec2 = traj2[i].reshape(-1)
            
            # Calculate Euclidean distance
            distance = np.sqrt(np.sum((vec1 - vec2) ** 2))
            distances.append(distance)
        
        return np.array(distances)
    
    # Calculate metrics for each guidance scale
    for g_scale in guidance_scales:
        # Get trajectories for this guidance scale
        teacher_traj = process_trajectory(teacher_trajectories[g_scale])
        student_traj = process_trajectory(student_trajectories[g_scale])
        teacher_no_cfg = process_trajectory(teacher_no_cfg_trajectories[g_scale])
        student_no_cfg = process_trajectory(student_no_cfg_trajectories[g_scale])
        
        # Calculate cosine similarity
        teacher_cosine = cosine_similarity_trajectories(teacher_traj, teacher_no_cfg)
        student_cosine = cosine_similarity_trajectories(student_traj, student_no_cfg)
        
        # Calculate Euclidean distance
        teacher_euclidean = euclidean_distance_trajectories(teacher_traj, teacher_no_cfg)
        student_euclidean = euclidean_distance_trajectories(student_traj, student_no_cfg)
        
        # Store average metrics
        cosine_similarities_teacher.append(np.mean(teacher_cosine))
        cosine_similarities_student.append(np.mean(student_cosine))
        euclidean_distances_teacher.append(np.mean(teacher_euclidean))
        euclidean_distances_student.append(np.mean(student_euclidean))
    
    # Create figure for similarity metrics across guidance scales
    plt.figure(figsize=(12, 10))
    
    # Create subplot for cosine similarity
    plt.subplot(2, 1, 1)
    plt.plot(guidance_scales, cosine_similarities_teacher, '-o', label='Teacher', color='blue')
    plt.plot(guidance_scales, cosine_similarities_student, '--s', label='Student', color='red')
    plt.title(f'Average Cosine Similarity Between CFG and No-CFG Trajectories\n(Student Size Factor: {size_factor})')
    plt.xlabel('Guidance Scale')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Create subplot for Euclidean distance
    plt.subplot(2, 1, 2)
    plt.plot(guidance_scales, euclidean_distances_teacher, '-o', label='Teacher', color='blue')
    plt.plot(guidance_scales, euclidean_distances_student, '--s', label='Student', color='red')
    plt.title(f'Average Euclidean Distance Between CFG and No-CFG Trajectories\n(Student Size Factor: {size_factor})')
    plt.xlabel('Guidance Scale')
    plt.ylabel('Euclidean Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cfg_similarity_metrics_size_{size_factor}.png'), dpi=300)
    plt.close()
    
    # Create figure for step-by-step similarity metrics for each guidance scale
    for g_scale in guidance_scales:
        plt.figure(figsize=(12, 10))
        
        # Get trajectories for this guidance scale
        teacher_traj = process_trajectory(teacher_trajectories[g_scale])
        student_traj = process_trajectory(student_trajectories[g_scale])
        teacher_no_cfg = process_trajectory(teacher_no_cfg_trajectories[g_scale])
        student_no_cfg = process_trajectory(student_no_cfg_trajectories[g_scale])
        
        # Calculate step-by-step metrics
        teacher_cosine = cosine_similarity_trajectories(teacher_traj, teacher_no_cfg)
        student_cosine = cosine_similarity_trajectories(student_traj, student_no_cfg)
        teacher_euclidean = euclidean_distance_trajectories(teacher_traj, teacher_no_cfg)
        student_euclidean = euclidean_distance_trajectories(student_traj, student_no_cfg)
        
        # Create x-axis for timesteps (use natural order for diffusion process)
        timesteps = np.arange(len(teacher_cosine))
        
        # Create subplot for cosine similarity
        plt.subplot(2, 1, 1)
        plt.plot(timesteps, teacher_cosine, '-', label='Teacher', color='blue')
        plt.plot(timesteps, student_cosine, '--', label='Student', color='red')
        plt.title(f'Cosine Similarity Between CFG (w={g_scale}) and No-CFG Trajectories\n(Student Size Factor: {size_factor})')
        plt.xlabel('Diffusion Timestep')
        plt.ylabel('Cosine Similarity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Create subplot for Euclidean distance
        plt.subplot(2, 1, 2)
        plt.plot(timesteps, teacher_euclidean, '-', label='Teacher', color='blue')
        plt.plot(timesteps, student_euclidean, '--', label='Student', color='red')
        plt.title(f'Euclidean Distance Between CFG (w={g_scale}) and No-CFG Trajectories\n(Student Size Factor: {size_factor})')
        plt.xlabel('Diffusion Timestep')
        plt.ylabel('Euclidean Distance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'cfg_similarity_timesteps_w{g_scale}_size_{size_factor}.png'), dpi=300)
        plt.close()
    
    # Create a visualization of trajectory divergence in PCA space
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Fit PCA on teacher's no-CFG trajectory
    reference_trajectory = teacher_no_cfg_trajectories[guidance_scales[0]]
    reference_features = process_trajectory(reference_trajectory)
    pca = PCA(n_components=2)
    pca.fit(reference_features)
    
    # Create a colormap for guidance scales
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(guidance_scales), max(guidance_scales))
    
    # Plot the divergence between CFG and no-CFG trajectories
    for g_scale in guidance_scales:
        # Get trajectories for this guidance scale
        teacher_traj = process_trajectory(teacher_trajectories[g_scale])
        student_traj = process_trajectory(student_trajectories[g_scale])
        teacher_no_cfg = process_trajectory(teacher_no_cfg_trajectories[g_scale])
        student_no_cfg = process_trajectory(student_no_cfg_trajectories[g_scale])
        
        # Project trajectories to PCA space
        teacher_cfg_pca = pca.transform(teacher_traj)
        student_cfg_pca = pca.transform(student_traj)
        teacher_no_cfg_pca = pca.transform(teacher_no_cfg)
        student_no_cfg_pca = pca.transform(student_no_cfg)
        
        # Calculate divergence vectors
        teacher_divergence = teacher_cfg_pca - teacher_no_cfg_pca
        student_divergence = student_cfg_pca - student_no_cfg_pca
        
        # Get color for this guidance scale
        color = cmap(norm(g_scale))
        
        # Plot divergence vectors at selected timesteps (every 5 steps)
        step_interval = max(1, len(teacher_divergence) // 10)  # Show about 10 arrows
        for i in range(0, len(teacher_divergence), step_interval):
            # Teacher divergence
            ax.arrow(teacher_no_cfg_pca[i, 0], teacher_no_cfg_pca[i, 1],
                     teacher_divergence[i, 0], teacher_divergence[i, 1],
                     color=color, alpha=0.7, width=0.005,
                     head_width=0.05, head_length=0.1,
                     length_includes_head=True)
            
            # Student divergence
            ax.arrow(student_no_cfg_pca[i, 0], student_no_cfg_pca[i, 1],
                     student_divergence[i, 0], student_divergence[i, 1],
                     color=color, alpha=0.7, width=0.005, linestyle='--',
                     head_width=0.05, head_length=0.1,
                     length_includes_head=True)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label='Guidance Scale')
    
    # Add legend for teacher vs student
    ax.plot([], [], '-', color='black', label='Teacher')
    ax.plot([], [], '--', color='black', label='Student')
    ax.legend()
    
    ax.set_title(f'Trajectory Divergence Due to CFG\n(Student Size Factor: {size_factor})')
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'cfg_trajectory_divergence_size_{size_factor}.png'), dpi=300)
    plt.close()

def visualize_teacher_student_similarity(teacher_trajectories, student_trajectories,
                                       teacher_no_cfg_trajectories, student_no_cfg_trajectories,
                                       output_dir, guidance_scales, size_factor):
    """
    Visualize the similarity between teacher and student models under different guidance scales
    
    This function calculates and visualizes how the similarity between teacher and student
    trajectories changes with different guidance scales.
    """
    # Process trajectories to feature vectors
    def process_trajectory(traj):
        features = [t.cpu().numpy().reshape(-1) for t in traj]
        return np.stack(features)
    
    # Calculate cosine similarity between two feature matrices
    def cosine_similarity_trajectories(traj1, traj2):
        # Ensure both trajectories have the same number of steps
        min_steps = min(len(traj1), len(traj2))
        similarities = []
        
        for i in range(min_steps):
            # Flatten the images to 1D vectors
            vec1 = traj1[i].reshape(1, -1)
            vec2 = traj2[i].reshape(1, -1)
            
            # Calculate cosine similarity
            dot_product = np.sum(vec1 * vec2)
            norm1 = np.sqrt(np.sum(vec1 ** 2))
            norm2 = np.sqrt(np.sum(vec2 ** 2))
            
            if norm1 * norm2 > 0:  # Avoid division by zero
                similarity = dot_product / (norm1 * norm2)
            else:
                similarity = 0
                
            similarities.append(similarity)
        
        return np.array(similarities)
    
    # Calculate Euclidean distance between two feature matrices
    def euclidean_distance_trajectories(traj1, traj2):
        # Ensure both trajectories have the same number of steps
        min_steps = min(len(traj1), len(traj2))
        distances = []
        
        for i in range(min_steps):
            # Flatten the images to 1D vectors
            vec1 = traj1[i].reshape(-1)
            vec2 = traj2[i].reshape(-1)
            
            # Calculate Euclidean distance
            distance = np.sqrt(np.sum((vec1 - vec2) ** 2))
            distances.append(distance)
        
        return np.array(distances)
    
    # Initialize lists to store metrics
    cosine_similarities_cfg = []
    cosine_similarities_no_cfg = []
    euclidean_distances_cfg = []
    euclidean_distances_no_cfg = []
    
    # Calculate metrics for each guidance scale
    for g_scale in guidance_scales:
        # Get trajectories for this guidance scale
        teacher_traj = process_trajectory(teacher_trajectories[g_scale])
        student_traj = process_trajectory(student_trajectories[g_scale])
        teacher_no_cfg = process_trajectory(teacher_no_cfg_trajectories[g_scale])
        student_no_cfg = process_trajectory(student_no_cfg_trajectories[g_scale])
        
        # Calculate cosine similarity between teacher and student
        cosine_cfg = cosine_similarity_trajectories(teacher_traj, student_traj)
        cosine_no_cfg = cosine_similarity_trajectories(teacher_no_cfg, student_no_cfg)
        
        # Calculate Euclidean distance between teacher and student
        euclidean_cfg = euclidean_distance_trajectories(teacher_traj, student_traj)
        euclidean_no_cfg = euclidean_distance_trajectories(teacher_no_cfg, student_no_cfg)
        
        # Store average metrics
        cosine_similarities_cfg.append(np.mean(cosine_cfg))
        cosine_similarities_no_cfg.append(np.mean(cosine_no_cfg))
        euclidean_distances_cfg.append(np.mean(euclidean_cfg))
        euclidean_distances_no_cfg.append(np.mean(euclidean_no_cfg))
    
    # Create figure for similarity metrics across guidance scales
    plt.figure(figsize=(12, 10))
    
    # Create subplot for cosine similarity
    plt.subplot(2, 1, 1)
    plt.plot(guidance_scales, cosine_similarities_cfg, '-o', label='With CFG', color='blue')
    plt.plot(guidance_scales, [cosine_similarities_no_cfg[0]] * len(guidance_scales), '--', label='Without CFG', color='red')
    plt.title(f'Average Cosine Similarity Between Teacher and Student Trajectories\n(Student Size Factor: {size_factor})')
    plt.xlabel('Guidance Scale')
    plt.ylabel('Cosine Similarity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Create subplot for Euclidean distance
    plt.subplot(2, 1, 2)
    plt.plot(guidance_scales, euclidean_distances_cfg, '-o', label='With CFG', color='blue')
    plt.plot(guidance_scales, [euclidean_distances_no_cfg[0]] * len(guidance_scales), '--', label='Without CFG', color='red')
    plt.title(f'Average Euclidean Distance Between Teacher and Student Trajectories\n(Student Size Factor: {size_factor})')
    plt.xlabel('Guidance Scale')
    plt.ylabel('Euclidean Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'teacher_student_similarity_metrics_size_{size_factor}.png'), dpi=300)
    plt.close()
    
    # Create figure for step-by-step similarity metrics for each guidance scale
    for g_scale in guidance_scales:
        plt.figure(figsize=(12, 10))
        
        # Get trajectories for this guidance scale
        teacher_traj = process_trajectory(teacher_trajectories[g_scale])
        student_traj = process_trajectory(student_trajectories[g_scale])
        teacher_no_cfg = process_trajectory(teacher_no_cfg_trajectories[g_scale])
        student_no_cfg = process_trajectory(student_no_cfg_trajectories[g_scale])
        
        # Calculate step-by-step metrics
        cosine_cfg = cosine_similarity_trajectories(teacher_traj, student_traj)
        cosine_no_cfg = cosine_similarity_trajectories(teacher_no_cfg, student_no_cfg)
        euclidean_cfg = euclidean_distance_trajectories(teacher_traj, student_traj)
        euclidean_no_cfg = euclidean_distance_trajectories(teacher_no_cfg, student_no_cfg)
        
        # Create x-axis for timesteps (use natural order for diffusion process)
        timesteps = np.arange(len(cosine_cfg))
        
        # Create subplot for cosine similarity
        plt.subplot(2, 1, 1)
        plt.plot(timesteps, cosine_cfg, '-', label=f'With CFG (w={g_scale})', color='blue')
        plt.plot(timesteps, cosine_no_cfg, '--', label='Without CFG', color='red')
        plt.title(f'Cosine Similarity Between Teacher and Student Trajectories\n(Student Size Factor: {size_factor})')
        plt.xlabel('Diffusion Timestep')
        plt.ylabel('Cosine Similarity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Create subplot for Euclidean distance
        plt.subplot(2, 1, 2)
        plt.plot(timesteps, euclidean_cfg, '-', label=f'With CFG (w={g_scale})', color='blue')
        plt.plot(timesteps, euclidean_no_cfg, '--', label='Without CFG', color='red')
        plt.title(f'Euclidean Distance Between Teacher and Student Trajectories\n(Student Size Factor: {size_factor})')
        plt.xlabel('Diffusion Timestep')
        plt.ylabel('Euclidean Distance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'teacher_student_similarity_timesteps_w{g_scale}_size_{size_factor}.png'), dpi=300)
        plt.close()
    
    # Create a visualization comparing the relative impact of CFG on teacher vs student
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Calculate the ratio of similarity change due to CFG
    cosine_ratio = [cfg / no_cfg for cfg, no_cfg in zip(cosine_similarities_cfg, [cosine_similarities_no_cfg[0]] * len(guidance_scales))]
    euclidean_ratio = [cfg / no_cfg for cfg, no_cfg in zip(euclidean_distances_cfg, [euclidean_distances_no_cfg[0]] * len(guidance_scales))]
    
    # Create subplot for cosine similarity ratio
    axs[0].plot(guidance_scales, cosine_ratio, '-o', color='purple')
    axs[0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    axs[0].set_title(f'Relative Change in Teacher-Student Similarity Due to CFG\n(Student Size Factor: {size_factor})')
    axs[0].set_xlabel('Guidance Scale')
    axs[0].set_ylabel('Cosine Similarity Ratio\n(CFG / No CFG)')
    axs[0].grid(True, alpha=0.3)
    
    # Create subplot for Euclidean distance ratio
    axs[1].plot(guidance_scales, euclidean_ratio, '-o', color='purple')
    axs[1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    axs[1].set_xlabel('Guidance Scale')
    axs[1].set_ylabel('Euclidean Distance Ratio\n(CFG / No CFG)')
    axs[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'teacher_student_similarity_ratio_size_{size_factor}.png'), dpi=300)
    plt.close() 