"""
Latent space manipulation for diffusion models
"""

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

def apply_latent_manipulation(model, diffusion_params, direction, strength, 
                             config, device, num_samples=5, record_trajectory=True):
    """
    Apply latent space manipulation to a diffusion model
    
    Args:
        model: Diffusion model
        diffusion_params: Diffusion parameters
        direction: Direction in latent space to manipulate (can be 'random' or a specific vector)
        strength: Strength of manipulation
        config: Configuration object
        device: Device to run on
        num_samples: Number of samples to generate
        record_trajectory: Whether to record the trajectory
        
    Returns:
        Dictionary containing:
        - original_images: List of original generated images
        - manipulated_images: List of manipulated images
        - direction: Direction vector used for manipulation
        - strength: Strength of manipulation
        - trajectories: List of trajectories if record_trajectory is True
    """
    # Set model to evaluation mode
    model.eval()
    
    # For this simplified implementation, we'll simulate latent space manipulation
    # In a real implementation, this would use a proper latent space manipulation algorithm
    
    # Generate random seed for reproducibility
    seed = torch.randint(0, 10000, (1,)).item()
    
    original_images = []
    manipulated_images = []
    trajectories = []
    
    # Generate latent direction if not provided
    if direction == 'random' or direction is None:
        # Create a random direction vector in latent space
        latent_dim = config.channels * config.image_size * config.image_size
        direction = torch.randn(latent_dim, device=device)
        # Normalize the direction vector
        direction = direction / torch.norm(direction)
    
    # Generate samples
    for i in range(num_samples):
        # Set seed for reproducibility but different for each sample
        torch.manual_seed(seed + i)
        
        # Generate original image
        original_image, original_latents, original_trajectory = generate_image_with_latents(
            model, diffusion_params, config, device
        )
        
        # Apply latent manipulation
        manipulated_image, manipulated_trajectory = manipulate_latent(
            model, diffusion_params, original_latents, direction, strength, config, device
        )
        
        original_images.append(original_image)
        manipulated_images.append(manipulated_image)
        
        if record_trajectory:
            trajectories.append({
                'original': original_trajectory,
                'manipulated': manipulated_trajectory
            })
    
    result = {
        'original_images': original_images,
        'manipulated_images': manipulated_images,
        'direction': direction,
        'strength': strength
    }
    
    if record_trajectory:
        result['trajectories'] = trajectories
    
    return result

def generate_image_with_latents(model, diffusion_params, config, device):
    """
    Generate an image and record the latent trajectory
    
    Args:
        model: Diffusion model
        diffusion_params: Diffusion parameters
        config: Configuration object
        device: Device to run on
        
    Returns:
        Tuple of (generated_image, final_latent, trajectory)
    """
    # Initialize from random noise
    x = torch.randn(1, config.channels, config.image_size, config.image_size).to(device)
    
    # Record trajectory
    trajectory = []
    
    # Denoise step by step
    with torch.no_grad():
        for t in range(diffusion_params["timesteps"] - 1, -1, -1):
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
                x = c1 * x - c2 * noise_pred
                
                # Add noise for the next step
                sigma_t = torch.sqrt(1 - alpha_t_prev) * torch.sqrt(1 - alpha_t / alpha_t_prev)
                x = x + sigma_t * noise
    
    # Get final latent representation
    final_latent = x.clone()
    
    # Normalize to [0, 1] range for visualization
    image = (x + 1) / 2
    image = torch.clamp(image, 0, 1)
    
    return image, final_latent, trajectory

def manipulate_latent(model, diffusion_params, latent, direction, strength, config, device):
    """
    Manipulate a latent representation and generate a new image
    
    Args:
        model: Diffusion model
        diffusion_params: Diffusion parameters
        latent: Latent representation to manipulate
        direction: Direction in latent space to manipulate
        strength: Strength of manipulation
        config: Configuration object
        device: Device to run on
        
    Returns:
        Tuple of (manipulated_image, trajectory)
    """
    # Reshape direction if needed
    if direction.dim() == 1:
        direction = direction.reshape(latent.shape)
    
    # Apply manipulation to latent
    manipulated_latent = latent + strength * direction
    
    # Initialize x with manipulated latent
    x = manipulated_latent.clone()
    
    # Record trajectory
    trajectory = []
    
    # Denoise step by step
    with torch.no_grad():
        for t in tqdm(range(diffusion_params["timesteps"] // 2, -1, -1), desc="Manipulating latent"):
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
                x = c1 * x - c2 * noise_pred
                
                # Add noise for the next step
                sigma_t = torch.sqrt(1 - alpha_t_prev) * torch.sqrt(1 - alpha_t / alpha_t_prev)
                x = x + sigma_t * noise
    
    # Normalize to [0, 1] range for visualization
    image = (x + 1) / 2
    image = torch.clamp(image, 0, 1)
    
    return image, trajectory

def find_semantic_directions(model, diffusion_params, config, device, num_samples=100):
    """
    Find semantic directions in latent space using PCA
    
    Args:
        model: Diffusion model
        diffusion_params: Diffusion parameters
        config: Configuration object
        device: Device to run on
        num_samples: Number of samples to generate for PCA
        
    Returns:
        Dictionary of semantic directions
    """
    # Generate multiple samples and collect their latent representations
    latents = []
    
    for i in tqdm(range(num_samples), desc="Generating samples for PCA"):
        # Set seed for reproducibility but different for each sample
        torch.manual_seed(i)
        
        # Generate image and get latent
        _, latent, _ = generate_image_with_latents(model, diffusion_params, config, device)
        
        # Flatten latent
        latent_flat = latent.reshape(-1).cpu().numpy()
        latents.append(latent_flat)
    
    # Stack latents
    latents = np.stack(latents)
    
    # Apply PCA to find principal components
    pca = PCA(n_components=10)
    pca.fit(latents)
    
    # Extract principal components as semantic directions
    directions = {}
    for i in range(10):
        direction = torch.tensor(pca.components_[i], device=device).float()
        directions[f"pca_{i}"] = direction
    
    return directions

def visualize_latent_manipulation(result, output_dir, size_factor=None):
    """
    Visualize latent manipulation results
    
    Args:
        result: Result dictionary from apply_latent_manipulation
        output_dir: Directory to save visualizations
        size_factor: Size factor of the model for labeling
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get number of samples
    num_samples = len(result["original_images"])
    
    # Plot original and manipulated images side by side
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    
    # Handle single sample case
    if num_samples == 1:
        axes = axes.reshape(1, 2)
    
    for i in range(num_samples):
        # Original image
        axes[i, 0].imshow(result["original_images"][i][0].permute(1, 2, 0).cpu().numpy())
        axes[i, 0].set_title(f"Original (Sample {i+1})")
        axes[i, 0].axis("off")
        
        # Manipulated image
        axes[i, 1].imshow(result["manipulated_images"][i][0].permute(1, 2, 0).cpu().numpy())
        axes[i, 1].set_title(f"Manipulated (Sample {i+1})")
        axes[i, 1].axis("off")
    
    plt.suptitle(f"Latent Space Manipulation (Size Factor: {size_factor}, Strength: {result['strength']})" 
                if size_factor else f"Latent Space Manipulation (Strength: {result['strength']})")
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "latent_manipulation_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    # If trajectories are available, visualize them
    if "trajectories" in result:
        visualize_manipulation_trajectories(
            result["trajectories"],
            output_dir,
            size_factor
        )

def visualize_manipulation_trajectories(trajectories, output_dir, size_factor=None):
    """
    Visualize manipulation trajectories
    
    Args:
        trajectories: List of trajectory dictionaries
        output_dir: Directory to save visualizations
        size_factor: Size factor of the model for labeling
    """
    # Create trajectory directory
    traj_dir = os.path.join(output_dir, "trajectories")
    os.makedirs(traj_dir, exist_ok=True)
    
    # Visualize each trajectory pair
    for i, trajectory_pair in enumerate(trajectories):
        original_trajectory = trajectory_pair["original"]
        manipulated_trajectory = trajectory_pair["manipulated"]
        
        # Sample a few timesteps to visualize
        num_samples = min(5, len(manipulated_trajectory))
        sample_indices = np.linspace(0, len(manipulated_trajectory) - 1, num_samples, dtype=int)
        
        # Plot trajectories
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
        
        for j, idx in enumerate(sample_indices):
            # Original trajectory - find closest timestep
            orig_t = manipulated_trajectory[idx][1]
            orig_idx = min(range(len(original_trajectory)), 
                          key=lambda k: abs(original_trajectory[k][1] - orig_t))
            orig_img = original_trajectory[orig_idx][0]
            
            # Normalize for visualization
            orig_img = (orig_img + 1) / 2
            orig_img = torch.clamp(orig_img, 0, 1)
            
            axes[0, j].imshow(orig_img[0].permute(1, 2, 0).cpu().numpy())
            axes[0, j].set_title(f"t = {original_trajectory[orig_idx][1]}")
            axes[0, j].axis("off")
            
            # Manipulated trajectory
            manip_img = manipulated_trajectory[idx][0]
            manip_t = manipulated_trajectory[idx][1]
            
            # Normalize for visualization
            manip_img = (manip_img + 1) / 2
            manip_img = torch.clamp(manip_img, 0, 1)
            
            axes[1, j].imshow(manip_img[0].permute(1, 2, 0).cpu().numpy())
            axes[1, j].set_title(f"t = {manip_t}")
            axes[1, j].axis("off")
        
        axes[0, 0].set_ylabel("Original")
        axes[1, 0].set_ylabel("Manipulated")
        
        plt.suptitle(f"Manipulation Trajectory {i+1} (Size Factor: {size_factor})" 
                    if size_factor else f"Manipulation Trajectory {i+1}")
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(traj_dir, f"manipulation_trajectory_{i+1}.png"), dpi=300, bbox_inches="tight")
        plt.close() 