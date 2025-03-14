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
    
    # Calculate alphas from betas
    alphas = 1.0 - diffusion_params['betas']
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
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
        
        # Apply the prediction
        alpha = alphas[i]
        alpha_cumprod = alphas_cumprod[i]
        beta = diffusion_params['betas'][i]
        
        # Only add noise if not at the final step
        if i > 0:
            noise = torch.randn_like(x)
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * pred) + torch.sqrt(beta) * noise
        else:
            # For the final step, don't add noise to get a deterministic output
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * pred)
        
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
    
    # Calculate alphas from betas
    alphas = 1.0 - diffusion_params['betas']
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
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
        
        # Apply the prediction
        alpha = alphas[i]
        alpha_cumprod = alphas_cumprod[i]
        beta = diffusion_params['betas'][i]
        
        # Only add noise if not at the final step
        if i > 0:
            noise = torch.randn_like(x)
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * pred) + torch.sqrt(beta) * noise
        else:
            # For the final step, don't add noise to get a deterministic output
            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * pred)
        
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
        
        # Combine features for PCA
        all_features = np.concatenate([
            teacher_features,
            student_features,
            teacher_no_cfg_features,
            student_no_cfg_features
        ], axis=0)
        
        # Fit PCA
        pca = PCA(n_components=2)
        all_pca = pca.fit_transform(all_features)
        
        # Split back into separate trajectories
        n_steps = len(teacher_traj)
        teacher_pca = all_pca[:n_steps]
        student_pca = all_pca[n_steps:2*n_steps]
        teacher_no_cfg_pca = all_pca[2*n_steps:3*n_steps]
        student_no_cfg_pca = all_pca[3*n_steps:]
        
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
        t_img_cfg = teacher_trajectories[guidance_scale][-1][0]
        s_img_cfg = student_trajectories[guidance_scale][-1][0]
        t_img_no_cfg = teacher_no_cfg_trajectories[guidance_scale][-1][0]
        s_img_no_cfg = student_no_cfg_trajectories[guidance_scale][-1][0]
        
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