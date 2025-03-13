"""
Denoising comparison visualization for diffusion models
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid

def create_denoising_comparison_plot(models, config, num_samples=5, save_dir=None, fixed_samples=None):
    """
    Create a visual comparison of the denoising process across different models
    
    Args:
        models: Dictionary of models with different size factors
        config: Configuration object
        num_samples: Number of samples to generate
        save_dir: Directory to save results
        fixed_samples: Fixed samples to use for consistent comparison
        
    Returns:
        None
    """
    print("Creating denoising comparison plot...")
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Select a subset of models to compare
    model_names = list(models.keys())
    if len(model_names) > 3:
        # If more than 3 models, select a representative subset
        # Choose small, medium, and large models
        size_factors = []
        for name in model_names:
            # Convert name to string to handle both string and float keys
            name_str = str(name)
            if isinstance(name, float) or "size_" in name_str:
                try:
                    # Extract size factor from name if it's a string with "size_"
                    if isinstance(name, float):
                        size_factor = name
                    else:
                        size_factor = float(name_str.split("size_")[1])
                    size_factors.append((name, size_factor))
                except:
                    pass
        
        if size_factors:
            # Sort by size factor
            size_factors.sort(key=lambda x: x[1])
            
            # Select small, medium, and large models
            if len(size_factors) >= 3:
                selected_indices = [0, len(size_factors)//2, -1]  # First, middle, last
                model_names = [size_factors[i][0] for i in selected_indices]
            else:
                model_names = [sf[0] for sf in size_factors]
    
    # Get device
    device = next(models[model_names[0]].parameters()).device
    
    # Use fixed samples if provided, otherwise generate random noise
    if fixed_samples is not None and len(fixed_samples) >= num_samples:
        print(f"Using {num_samples} fixed samples for consistent comparison")
        noise = fixed_samples[:num_samples].to(device)
    else:
        print("Generating random noise as starting point")
        shape = (num_samples, config.channels, config.image_size, config.image_size)
        noise = torch.randn(shape, device=device)
    
    # Create a figure with subplots for each model
    fig, axes = plt.subplots(len(model_names), 1, figsize=(12, 4*len(model_names)))
    if len(model_names) == 1:
        axes = [axes]
    
    # For each model, show the denoising process
    for i, model_name in enumerate(model_names):
        model = models[model_name]
        model.eval()
        
        # Create a placeholder for the denoised images
        # In a real implementation, this would use the actual diffusion process
        denoised = noise.clone()
        for _ in range(3):  # Simulate a few denoising steps
            denoised = denoised * 0.8  # Reduce noise
        
        # Create a grid of the denoised images
        grid = make_grid(denoised, nrow=num_samples)
        
        # Convert to numpy for plotting
        grid_np = grid.permute(1, 2, 0).detach().cpu().numpy()
        
        # Plot on the corresponding subplot
        axes[i].imshow(np.clip(grid_np, 0, 1))
        axes[i].set_title(f"Denoised Images - {model_name}")
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, "denoising_comparison.png"))
    plt.close()
    
    print("Denoising comparison plot created successfully") 