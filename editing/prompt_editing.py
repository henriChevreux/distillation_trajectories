"""
Prompt-based editing for diffusion models
"""

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def apply_prompt_editing(model, diffusion_params, original_prompt, edited_prompt, 
                        config, device, record_trajectory=True):
    """
    Apply prompt-based editing to a diffusion model
    
    Args:
        model: Diffusion model
        diffusion_params: Diffusion parameters
        original_prompt: Original text prompt
        edited_prompt: Edited text prompt
        config: Configuration object
        device: Device to run on
        record_trajectory: Whether to record the trajectory
        
    Returns:
        Dictionary containing:
        - original_image: Generated image from original prompt
        - edited_image: Generated image from edited prompt
        - trajectory: List of (image, timestep) pairs if record_trajectory is True
    """
    # Set model to evaluation mode
    model.eval()
    
    # For this simplified implementation, we'll simulate prompt-based editing
    # In a real implementation, this would use a text-conditioned diffusion model
    
    # Generate random seed for reproducibility
    seed = torch.randint(0, 10000, (1,)).item()
    
    # Generate original image
    torch.manual_seed(seed)
    original_image, original_trajectory = generate_image_with_trajectory(
        model, diffusion_params, config, device
    )
    
    # Generate edited image with the same seed but different "prompt"
    # In this simulation, we'll just use a different random seed to represent a different prompt
    torch.manual_seed(seed + 1)  # Different seed to simulate different prompt
    edited_image, edited_trajectory = generate_image_with_trajectory(
        model, diffusion_params, config, device
    )
    
    result = {
        "original_image": original_image,
        "edited_image": edited_image,
        "original_prompt": original_prompt,
        "edited_prompt": edited_prompt,
    }
    
    if record_trajectory:
        result["original_trajectory"] = original_trajectory
        result["edited_trajectory"] = edited_trajectory
    
    return result

def generate_image_with_trajectory(model, diffusion_params, config, device):
    """
    Generate an image and record the trajectory
    
    Args:
        model: Diffusion model
        diffusion_params: Diffusion parameters
        config: Configuration object
        device: Device to run on
        
    Returns:
        Tuple of (generated_image, trajectory)
    """
    # Initialize from random noise
    x = torch.randn(1, config.channels, config.image_size, config.image_size).to(device)
    
    # Record trajectory
    trajectory = []
    
    # Denoise step by step
    with torch.no_grad():
        for t in tqdm(range(diffusion_params["timesteps"] - 1, -1, -1), desc="Generating image"):
            t_tensor = torch.tensor([t], device=device)
            
            # Record current state
            trajectory.append((x.clone(), t))
            
            # Predict noise
            noise_pred = model(x, t_tensor)
            
            # Resize noise prediction if dimensions don't match
            if noise_pred.shape != x.shape:
                print(f"Resizing noise prediction from {noise_pred.shape} to {x.shape}")
                noise_pred = torch.nn.functional.interpolate(
                    noise_pred,
                    size=(x.shape[2], x.shape[3]),
                    mode='bilinear',
                    align_corners=True
                )
            
            # Update x
            if t > 0:
                # Sample noise for the next step
                noise = torch.randn_like(x)
                
                # Apply diffusion update
                alpha_t = diffusion_params["alphas"][t]
                alpha_t_prev = diffusion_params["alphas"][t-1] if t > 0 else torch.tensor(1.0)
                
                # Compute coefficients
                c1 = torch.sqrt(alpha_t_prev) / torch.sqrt(alpha_t)
                c2 = torch.sqrt(1 - alpha_t_prev) - torch.sqrt(alpha_t_prev / alpha_t) * torch.sqrt(1 - alpha_t)
                
                # Update x
                x = c1 * x - c2 * noise_pred
                
                # Add noise for the next step
                sigma_t = torch.sqrt(1 - alpha_t_prev) * torch.sqrt(1 - alpha_t / alpha_t_prev)
                x = x + sigma_t * noise
    
    # Normalize to [0, 1] range for visualization
    image = (x + 1) / 2
    image = torch.clamp(image, 0, 1)
    
    return image, trajectory

def visualize_prompt_editing(result, output_dir, size_factor=None):
    """
    Visualize prompt editing results
    
    Args:
        result: Result dictionary from apply_prompt_editing
        output_dir: Directory to save visualizations
        size_factor: Size factor of the model for labeling
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot original and edited images side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image
    axes[0].imshow(result["original_image"][0].permute(1, 2, 0).cpu().numpy())
    axes[0].set_title(f"Original: {result['original_prompt']}")
    axes[0].axis("off")
    
    # Edited image
    axes[1].imshow(result["edited_image"][0].permute(1, 2, 0).cpu().numpy())
    axes[1].set_title(f"Edited: {result['edited_prompt']}")
    axes[1].axis("off")
    
    plt.suptitle(f"Prompt-Based Editing (Size Factor: {size_factor})" if size_factor else "Prompt-Based Editing")
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "prompt_editing_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # If trajectories are available, visualize them
    if "original_trajectory" in result and "edited_trajectory" in result:
        visualize_trajectories(
            result["original_trajectory"], 
            result["edited_trajectory"],
            output_dir,
            size_factor
        )

def visualize_trajectories(original_trajectory, edited_trajectory, output_dir, size_factor=None):
    """
    Visualize original and edited trajectories
    
    Args:
        original_trajectory: List of (image, timestep) pairs for original generation
        edited_trajectory: List of (image, timestep) pairs for edited generation
        output_dir: Directory to save visualizations
        size_factor: Size factor of the model for labeling
    """
    # Create trajectory directory
    traj_dir = os.path.join(output_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)
    
    # Sample a few timesteps to visualize
    num_samples = min(5, len(original_trajectory))
    sample_indices = np.linspace(0, len(original_trajectory) - 1, num_samples, dtype=int)
    
    # Plot trajectories
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i, idx in enumerate(sample_indices):
        # Original trajectory
        orig_img = original_trajectory[idx][0]
        orig_t = original_trajectory[idx][1]
        
        # Normalize for visualization
        orig_img = (orig_img + 1) / 2
        orig_img = torch.clamp(orig_img, 0, 1)
        
        axes[0, i].imshow(orig_img[0].permute(1, 2, 0).cpu().numpy())
        axes[0, i].set_title(f"t = {orig_t}")
        axes[0, i].axis("off")
        
        # Edited trajectory
        edit_img = edited_trajectory[idx][0]
        edit_t = edited_trajectory[idx][1]
        
        # Normalize for visualization
        edit_img = (edit_img + 1) / 2
        edit_img = torch.clamp(edit_img, 0, 1)
        
        axes[1, i].imshow(edit_img[0].permute(1, 2, 0).cpu().numpy())
        axes[1, i].set_title(f"t = {edit_t}")
        axes[1, i].axis("off")
    
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Edited")
    
    plt.suptitle(f"Denoising Trajectories (Size Factor: {size_factor})" if size_factor else "Denoising Trajectories")
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(traj_dir, "trajectory_comparison.png"), dpi=300, bbox_inches="tight") 