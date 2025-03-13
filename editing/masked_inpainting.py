"""
Masked inpainting for diffusion models
"""

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def apply_masked_inpainting(model, diffusion_params, original_image, mask, 
                           config, device, record_trajectory=True):
    """
    Apply masked inpainting to a diffusion model
    
    Args:
        model: Diffusion model
        diffusion_params: Diffusion parameters
        original_image: Original image to inpaint
        mask: Binary mask where 1 indicates areas to inpaint
        config: Configuration object
        device: Device to run on
        record_trajectory: Whether to record the trajectory
        
    Returns:
        Dictionary containing:
        - original_image: Original image
        - inpainted_image: Inpainted image
        - mask: Binary mask
        - trajectory: List of (image, timestep) pairs if record_trajectory is True
    """
    # Set model to evaluation mode
    model.eval()
    
    # For this simplified implementation, we'll simulate masked inpainting
    # In a real implementation, this would use a proper inpainting algorithm
    
    # If no original image is provided, generate one
    if original_image is None:
        # Generate random seed for reproducibility
        seed = torch.randint(0, 10000, (1,)).item()
        torch.manual_seed(seed)
        
        # Generate original image
        original_image, _ = generate_image(model, diffusion_params, config, device)
    
    # If no mask is provided, create a random rectangular mask
    if mask is None:
        mask = create_random_mask(config.image_size, config.image_size)
    
    # Convert mask to tensor and move to device
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, device=device).float()
    
    # Ensure mask has the right shape
    if len(mask.shape) == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif len(mask.shape) == 3:
        mask = mask.unsqueeze(0)
    
    # Expand mask to match image channels
    mask = mask.expand(-1, config.channels, -1, -1)
    
    # Apply inpainting
    inpainted_image, trajectory = inpaint_with_trajectory(
        model, diffusion_params, original_image, mask, config, device
    )
    
    result = {
        "original_image": original_image,
        "inpainted_image": inpainted_image,
        "mask": mask,
    }
    
    if record_trajectory:
        result["trajectory"] = trajectory
    
    return result

def create_random_mask(height, width, min_size=0.2, max_size=0.5):
    """
    Create a random rectangular mask
    
    Args:
        height: Image height
        width: Image width
        min_size: Minimum size of mask as fraction of image
        max_size: Maximum size of mask as fraction of image
        
    Returns:
        Binary mask where 1 indicates areas to inpaint
    """
    # Create empty mask
    mask = np.zeros((height, width))
    
    # Determine mask size
    mask_h = int(np.random.uniform(min_size, max_size) * height)
    mask_w = int(np.random.uniform(min_size, max_size) * width)
    
    # Determine mask position
    mask_y = np.random.randint(0, height - mask_h)
    mask_x = np.random.randint(0, width - mask_w)
    
    # Fill mask
    mask[mask_y:mask_y+mask_h, mask_x:mask_x+mask_w] = 1
    
    return mask

def generate_image(model, diffusion_params, config, device):
    """
    Generate an image without recording the trajectory
    
    Args:
        model: Diffusion model
        diffusion_params: Diffusion parameters
        config: Configuration object
        device: Device to run on
        
    Returns:
        Generated image
    """
    # Initialize from random noise
    x = torch.randn(1, config.channels, config.image_size, config.image_size).to(device)
    
    # Denoise step by step
    with torch.no_grad():
        for t in range(diffusion_params["timesteps"] - 1, -1, -1):
            t_tensor = torch.tensor([t], device=device)
            
            # Predict noise
            noise_pred = model(x, t_tensor)
            
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
    
    return image, None

def inpaint_with_trajectory(model, diffusion_params, original_image, mask, config, device):
    """
    Inpaint an image and record the trajectory
    
    Args:
        model: Diffusion model
        diffusion_params: Diffusion parameters
        original_image: Original image to inpaint
        mask: Binary mask where 1 indicates areas to inpaint
        config: Configuration object
        device: Device to run on
        
    Returns:
        Tuple of (inpainted_image, trajectory)
    """
    # Initialize from random noise in masked regions, keep original elsewhere
    x = torch.randn(1, config.channels, config.image_size, config.image_size).to(device)
    
    # Scale original image to [-1, 1] for model input
    original_scaled = 2 * original_image - 1
    
    # Apply mask: keep original in non-masked areas, use noise in masked areas
    x = mask * x + (1 - mask) * original_scaled
    
    # Record trajectory
    trajectory = []
    
    # Denoise step by step
    with torch.no_grad():
        for t in tqdm(range(diffusion_params["timesteps"] - 1, -1, -1), desc="Inpainting"):
            t_tensor = torch.tensor([t], device=device)
            
            # Record current state
            trajectory.append((x.clone(), t))
            
            # Predict noise
            noise_pred = model(x, t_tensor)
            
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
                x_updated = c1 * x - c2 * noise_pred
                
                # Add noise for the next step
                sigma_t = torch.sqrt(1 - alpha_t_prev) * torch.sqrt(1 - alpha_t / alpha_t_prev)
                x_updated = x_updated + sigma_t * noise
                
                # Apply mask: keep original in non-masked areas, use updated in masked areas
                x = mask * x_updated + (1 - mask) * original_scaled
    
    # Normalize to [0, 1] range for visualization
    image = (x + 1) / 2
    image = torch.clamp(image, 0, 1)
    
    return image, trajectory

def visualize_inpainting(result, output_dir, size_factor=None):
    """
    Visualize inpainting results
    
    Args:
        result: Result dictionary from apply_masked_inpainting
        output_dir: Directory to save visualizations
        size_factor: Size factor of the model for labeling
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot original, mask, and inpainted images side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(result["original_image"][0].permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Mask
    mask_display = result["mask"][0, 0].cpu().numpy()
    axes[1].imshow(mask_display, cmap="gray")
    axes[1].set_title("Inpainting Mask")
    axes[1].axis("off")
    
    # Inpainted image
    axes[2].imshow(result["inpainted_image"][0].permute(1, 2, 0).cpu().numpy())
    axes[2].set_title("Inpainted Image")
    axes[2].axis("off")
    
    plt.suptitle(f"Masked Inpainting (Size Factor: {size_factor})" if size_factor else "Masked Inpainting")
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "inpainting_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # If trajectory is available, visualize it
    if "trajectory" in result:
        visualize_inpainting_trajectory(
            result["trajectory"],
            result["mask"],
            output_dir,
            size_factor
        )

def visualize_inpainting_trajectory(trajectory, mask, output_dir, size_factor=None):
    """
    Visualize inpainting trajectory
    
    Args:
        trajectory: List of (image, timestep) pairs
        mask: Binary mask where 1 indicates areas to inpaint
        output_dir: Directory to save visualizations
        size_factor: Size factor of the model for labeling
    """
    # Create trajectory directory
    traj_dir = os.path.join(output_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)
    
    # Sample a few timesteps to visualize
    num_samples = min(5, len(trajectory))
    sample_indices = np.linspace(0, len(trajectory) - 1, num_samples, dtype=int)
    
    # Plot trajectory
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    for i, idx in enumerate(sample_indices):
        # Get image and timestep
        img = trajectory[idx][0]
        t = trajectory[idx][1]
        
        # Normalize for visualization
        img = (img + 1) / 2
        img = torch.clamp(img, 0, 1)
        
        # Display image
        axes[i].imshow(img[0].permute(1, 2, 0).cpu().numpy())
        axes[i].set_title(f"t = {t}")
        axes[i].axis("off")
    
    plt.suptitle(f"Inpainting Trajectory (Size Factor: {size_factor})" if size_factor else "Inpainting Trajectory")
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(traj_dir, "inpainting_trajectory.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # Also visualize the masked region evolution
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    # Get mask as numpy array
    mask_np = mask[0, 0].cpu().numpy()
    
    for i, idx in enumerate(sample_indices):
        # Get image and timestep
        img = trajectory[idx][0]
        t = trajectory[idx][1]
        
        # Normalize for visualization
        img = (img + 1) / 2
        img = torch.clamp(img, 0, 1)
        
        # Convert to numpy
        img_np = img[0].permute(1, 2, 0).cpu().numpy()
        
        # Create a masked version to highlight the inpainted region
        masked_img = img_np.copy()
        
        # Add a red tint to the masked region
        for c in range(3):
            channel = masked_img[:, :, c]
            if c == 0:  # Red channel
                channel[mask_np > 0.5] = 1.0
            else:  # Green and Blue channels
                channel[mask_np > 0.5] *= 0.5
        
        # Display image
        axes[i].imshow(masked_img)
        axes[i].set_title(f"t = {t}")
        axes[i].axis("off")
    
    plt.suptitle(f"Masked Region Evolution (Size Factor: {size_factor})" if size_factor else "Masked Region Evolution")
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(traj_dir, "masked_region_evolution.png"), dpi=300, bbox_inches="tight")
    plt.close() 