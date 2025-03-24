"""
Enhanced trajectory comparison visualization for diffusion models.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

from utils.diffusion import get_diffusion_params

def extract(a, t, x_shape):
    """Extract coefficients at specified timesteps t"""
    b, *_ = t.shape
    
    # Ensure t is within bounds
    t = torch.clamp(t, 0, a.shape[0] - 1)
    
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def generate_trajectory(model, noise, timesteps, device, seed=None, guidance_scale=None):
    """
    Generate a trajectory from a given noise sample, with optional CFG support
    
    Args:
        model: The diffusion model
        noise: Starting noise sample
        timesteps: Number of timesteps in the diffusion process
        device: Device to run on
        seed: Random seed for reproducibility
        guidance_scale: Optional CFG guidance scale (w). If None, no CFG is used.
        
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
            
            # Predict noise - with or without CFG
            if guidance_scale is not None and guidance_scale > 1.0:
                # Use CFG approach
                # Concatenate the same input twice for efficiency
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t_tensor] * 2)
                
                # First half will be unconditioned (negative conditioning/empty token)
                # Second half will be conditioned (positive conditioning/class token)
                c = torch.cat([torch.zeros(1, 1), torch.ones(1, 1)]).to(device)
                
                # Get both predictions in a single forward pass
                pred_all = model(x_in, t_in, c)
                pred_uncond, pred_cond = pred_all.chunk(2)
                
                # Apply classifier-free guidance
                noise_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            else:
                # Standard approach without CFG
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

def compare_trajectories(teacher_model, student_model, config, guidance_scales=[1.0, 3.0, 5.0], size_factor=1.0, num_samples=3):
    """
    Compare trajectories of teacher and student models with and without CFG
    
    Args:
        teacher_model: The teacher diffusion model
        student_model: The student diffusion model
        config: Configuration object
        guidance_scales: List of guidance scales to evaluate (1.0 means no CFG)
        size_factor: Size factor of the student model
        num_samples: Number of noise samples to average over
        
    Returns:
        Dictionary of metrics for each guidance scale
    """
    from analysis.metrics.trajectory_metrics import compute_trajectory_metrics
    
    # Set device
    device = next(teacher_model.parameters()).device
    
    # Initialize dictionaries to store metrics
    teacher_metrics = {gs: [] for gs in guidance_scales}
    student_metrics = {gs: [] for gs in guidance_scales}
    
    # Generate trajectories for multiple noise samples
    for sample_idx in range(num_samples):
        # Set seed for reproducibility but different for each sample
        seed = 42 + sample_idx
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate random noise
        noise = torch.randn(1, config.channels, config.image_size, config.image_size)
        
        # Generate trajectories for each guidance scale
        for gs in guidance_scales:
            print(f"\nGenerating trajectories with guidance scale {gs} (sample {sample_idx+1}/{num_samples})...")
            
            print("Generating teacher trajectory...")
            teacher_traj = generate_trajectory(teacher_model, noise, config.timesteps, device, seed=seed, guidance_scale=gs)
            
            print("Generating student trajectory...")
            student_traj = generate_trajectory(student_model, noise, config.timesteps, device, seed=seed, guidance_scale=gs)
            
            # Compute metrics between teacher and student
            metrics = compute_trajectory_metrics(teacher_traj, student_traj, config)
            teacher_metrics[gs].append(metrics)
            student_metrics[gs].append(metrics)
    
    # Average metrics across samples
    avg_teacher_metrics = {gs: {} for gs in guidance_scales}
    avg_student_metrics = {gs: {} for gs in guidance_scales}
    
    # Average metrics for each guidance scale
    for gs in guidance_scales:
        for key in teacher_metrics[gs][0].keys():
            if isinstance(teacher_metrics[gs][0][key], (int, float)) and not isinstance(teacher_metrics[gs][0][key], bool):
                avg_teacher_metrics[gs][key] = sum(m[key] for m in teacher_metrics[gs]) / len(teacher_metrics[gs])
                avg_student_metrics[gs][key] = sum(m[key] for m in student_metrics[gs]) / len(student_metrics[gs])
    
    return {
        'teacher_metrics': avg_teacher_metrics,
        'student_metrics': avg_student_metrics
    } 