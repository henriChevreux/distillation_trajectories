"""
Classifier-free guidance editing for diffusion models
"""

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def apply_classifier_free_guidance(model, diffusion_params, prompt, config, device, 
                                  guidance_scale=3.0, record_trajectory=True):
    """
    Apply classifier-free guidance to a diffusion model
    
    Args:
        model: Diffusion model
        diffusion_params: Diffusion parameters
        prompt: Text prompt for conditional generation
        config: Configuration object
        device: Device to run on
        guidance_scale: Scale factor for guidance (higher = stronger adherence to prompt)
        record_trajectory: Whether to record the trajectory
        
    Returns:
        Dictionary containing:
        - guided_image: Generated image with guidance
        - unguided_image: Generated image without guidance
        - trajectory: List of (image, timestep) pairs if record_trajectory is True
    """
    # Set model to evaluation mode
    model.eval()
    
    # Generate random seed for reproducibility
    seed = torch.randint(0, 10000, (1,)).item()
    
    # Generate unguided image (unconditional)
    torch.manual_seed(seed)
    unguided_image, unguided_trajectory = generate_image_with_cfg(
        model, diffusion_params, config, device, 
        guidance_scale=1.0,  # No guidance
        conditional=False
    )
    
    # Generate guided image with the same seed but with guidance
    torch.manual_seed(seed)
    guided_image, guided_trajectory = generate_image_with_cfg(
        model, diffusion_params, config, device,
        guidance_scale=guidance_scale,
        conditional=True
    )
    
    result = {
        "unguided_image": unguided_image,
        "guided_image": guided_image,
        "prompt": prompt,
        "guidance_scale": guidance_scale
    }
    
    if record_trajectory:
        result["unguided_trajectory"] = unguided_trajectory
        result["guided_trajectory"] = guided_trajectory
    
    return result

def generate_image_with_cfg(model, diffusion_params, config, device, guidance_scale=3.0, conditional=True):
    """
    Generate an image using classifier-free guidance and record the trajectory
    
    Args:
        model: Diffusion model
        diffusion_params: Diffusion parameters
        config: Configuration object
        device: Device to run on
        guidance_scale: Scale factor for guidance (higher = stronger adherence to prompt)
        conditional: Whether to use conditional generation
        
    Returns:
        Tuple of (generated_image, trajectory)
    """
    # Initialize from random noise
    x = torch.randn(1, config.channels, config.image_size, config.image_size).to(device)
    
    # Record trajectory
    trajectory = []
    
    # Denoise step by step
    with torch.no_grad():
        for t in tqdm(range(diffusion_params["timesteps"] - 1, -1, -1), desc="Generating image with CFG"):
            t_tensor = torch.tensor([t], device=device)
            
            # Record current state
            trajectory.append((x.clone(), t))
            
            # For classifier-free guidance, we need both conditional and unconditional predictions
            # In a real implementation with text conditioning, we would pass the text embedding
            # Here we simulate it by using a condition flag
            
            # Unconditional prediction (no prompt)
            unconditional_noise_pred = model(x, t_tensor)
            
            if conditional and guidance_scale > 1.0:
                # Conditional prediction (with prompt)
                # In a real implementation, this would use text conditioning
                # Here we simulate it by adding a small bias to represent the prompt influence
                conditional_noise_pred = model(x, t_tensor)
                
                # Apply a small modification to simulate text conditioning
                # In a real implementation, the model would actually use the text embedding
                conditional_bias = torch.randn_like(conditional_noise_pred) * 0.1
                conditional_noise_pred = conditional_noise_pred + conditional_bias
                
                # Apply classifier-free guidance formula:
                # noise_pred = unconditional_pred + guidance_scale * (conditional_pred - unconditional_pred)
                noise_pred = unconditional_noise_pred + guidance_scale * (conditional_noise_pred - unconditional_noise_pred)
            else:
                # If not using guidance or guidance_scale is 1.0, just use the unconditional prediction
                noise_pred = unconditional_noise_pred
            
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

def visualize_cfg_editing(result, output_dir, size_factor=None):
    """
    Visualize classifier-free guidance editing results
    
    Args:
        result: Result dictionary from apply_classifier_free_guidance
        output_dir: Directory to save visualizations
        size_factor: Size factor of the model for labeling
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot unguided and guided images side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Unguided image
    axes[0].imshow(result["unguided_image"][0].permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Unguided (No Prompt)")
    axes[0].axis("off")
    
    # Guided image
    axes[1].imshow(result["guided_image"][0].permute(1, 2, 0).cpu().numpy())
    axes[1].set_title(f"Guided (Prompt: {result['prompt']}, Scale: {result['guidance_scale']})")
    axes[1].axis("off")
    
    plt.suptitle(f"Classifier-Free Guidance (Size Factor: {size_factor})" if size_factor else "Classifier-Free Guidance")
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "cfg_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # If trajectories are available, visualize them
    if "unguided_trajectory" in result and "guided_trajectory" in result:
        visualize_cfg_trajectories(
            result["unguided_trajectory"], 
            result["guided_trajectory"],
            output_dir,
            result["guidance_scale"],
            size_factor
        )

def visualize_cfg_trajectories(unguided_trajectory, guided_trajectory, output_dir, guidance_scale, size_factor=None):
    """
    Visualize unguided and guided trajectories
    
    Args:
        unguided_trajectory: List of (image, timestep) pairs for unguided generation
        guided_trajectory: List of (image, timestep) pairs for guided generation
        output_dir: Directory to save visualizations
        guidance_scale: Scale factor used for guidance
        size_factor: Size factor of the model for labeling
    """
    # Create trajectory directory
    traj_dir = os.path.join(output_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)
    
    # Sample a few timesteps to visualize
    num_samples = min(5, len(unguided_trajectory))
    sample_indices = np.linspace(0, len(unguided_trajectory) - 1, num_samples, dtype=int)
    
    # Plot trajectories
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i, idx in enumerate(sample_indices):
        # Unguided trajectory
        unguided_img = unguided_trajectory[idx][0]
        unguided_t = unguided_trajectory[idx][1]
        
        # Normalize for visualization
        unguided_img = (unguided_img + 1) / 2
        unguided_img = torch.clamp(unguided_img, 0, 1)
        
        axes[0, i].imshow(unguided_img[0].permute(1, 2, 0).cpu().numpy())
        axes[0, i].set_title(f"t = {unguided_t}")
        axes[0, i].axis("off")
        
        # Guided trajectory
        guided_img = guided_trajectory[idx][0]
        guided_t = guided_trajectory[idx][1]
        
        # Normalize for visualization
        guided_img = (guided_img + 1) / 2
        guided_img = torch.clamp(guided_img, 0, 1)
        
        axes[1, i].imshow(guided_img[0].permute(1, 2, 0).cpu().numpy())
        axes[1, i].set_title(f"t = {guided_t}")
        axes[1, i].axis("off")
    
    axes[0, 0].set_ylabel("Unguided")
    axes[1, 0].set_ylabel(f"Guided (Scale: {guidance_scale})")
    
    plt.suptitle(f"CFG Denoising Trajectories (Size Factor: {size_factor})" if size_factor else "CFG Denoising Trajectories")
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(traj_dir, "cfg_trajectory_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()

def visualize_guidance_scale_comparison(model, diffusion_params, prompt, config, device, output_dir, 
                                       guidance_scales=[1.0, 2.0, 5.0, 7.5, 10.0], size_factor=None):
    """
    Visualize the effect of different guidance scales
    
    Args:
        model: Diffusion model
        diffusion_params: Diffusion parameters
        prompt: Text prompt for conditional generation
        config: Configuration object
        device: Device to run on
        output_dir: Directory to save visualizations
        guidance_scales: List of guidance scale values to compare
        size_factor: Size factor of the model for labeling
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate random seed for reproducibility
    seed = torch.randint(0, 10000, (1,)).item()
    
    # Generate images with different guidance scales
    images = []
    
    for scale in guidance_scales:
        torch.manual_seed(seed)  # Use same seed for fair comparison
        image, _ = generate_image_with_cfg(
            model, diffusion_params, config, device,
            guidance_scale=scale,
            conditional=True
        )
        images.append(image)
    
    # Plot images with different guidance scales
    fig, axes = plt.subplots(1, len(guidance_scales), figsize=(15, 5))
    
    for i, (scale, image) in enumerate(zip(guidance_scales, images)):
        axes[i].imshow(image[0].permute(1, 2, 0).cpu().numpy())
        axes[i].set_title(f"Scale: {scale}")
        axes[i].axis("off")
    
    plt.suptitle(f"Effect of Guidance Scale (Size Factor: {size_factor})" if size_factor else "Effect of Guidance Scale")
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "guidance_scale_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    return images 