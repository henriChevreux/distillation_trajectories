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
        fixed_samples: Fixed samples to use for consistent comparison (ignored, using random noise instead)
        
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
    
    # Always use random noise for consistent comparison
    print("Generating random noise as starting point")
    # Set a fixed seed for reproducibility
    torch.manual_seed(42)
    shape = (num_samples, config.channels, config.image_size, config.image_size)
    noise = torch.randn(shape, device=device)
    
    # Ensure all models use the same image resolution
    # Get the image size from the config
    image_size = config.image_size
    print(f"Using consistent image size of {image_size}x{image_size} for all models")
    
    # Resize noise if needed
    if noise.shape[2] != image_size or noise.shape[3] != image_size:
        noise = torch.nn.functional.interpolate(
            noise, size=(image_size, image_size), mode='bilinear', align_corners=True
        )
        print(f"Resized noise to {noise.shape}")
    
    # Number of timesteps to visualize
    num_viz_steps = 5
    
    # Create a figure with subplots for each model and timestep
    fig, axes = plt.subplots(len(model_names), num_viz_steps, figsize=(15, 4*len(model_names)))
    if len(model_names) == 1:
        axes = [axes]
    
    # Calculate timesteps to visualize (from noisy to clean)
    timesteps = torch.linspace(config.timesteps-1, 0, num_viz_steps).long().to(device)
    
    # For each model, show the denoising process at different timesteps
    for i, model_name in enumerate(model_names):
        model = models[model_name]
        model.eval()
        
        # Start with the same noise for all models
        x = noise.clone()
        
        with torch.no_grad():
            # Perform the actual diffusion process
            for j, t in enumerate(timesteps):
                # Create a batch of the same timestep
                t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
                
                # Get model prediction
                model_output = model(x, t_batch)
                
                # Ensure the output has the same resolution as the input
                if model_output.shape[2] != image_size or model_output.shape[3] != image_size:
                    model_output = torch.nn.functional.interpolate(
                        model_output, size=(image_size, image_size), mode='bilinear', align_corners=True
                    )
                    print(f"Resized model output for {model_name} to {model_output.shape}")
                
                # For visualization purposes, we'll normalize the output
                normalized = (model_output + 1) / 2  # Normalize from [-1,1] to [0,1]
                
                # Create a grid of the images at this timestep
                grid = make_grid(normalized, nrow=1)
                
                # Convert to numpy for plotting
                grid_np = grid.permute(1, 2, 0).detach().cpu().numpy()
                
                # Plot on the corresponding subplot
                axes[i][j].imshow(np.clip(grid_np, 0, 1))
                axes[i][j].set_title(f"t={t.item()}")
                axes[i][j].axis('off')
                
                # Update x for the next timestep if not the last one
                if j < num_viz_steps - 1:
                    # Use the model output as the input for the next timestep
                    x = model_output
        
        # Add row label
        fig.text(0.01, 0.5 + (i - len(model_names)/2 + 0.5) / len(model_names), 
                 f"Model: {model_name}", va='center', ha='left', rotation='vertical')
    
    # Add column labels
    for j in range(num_viz_steps):
        fig.text(0.1 + (j + 0.5) / num_viz_steps, 0.01, 
                 f"Timestep {timesteps[j].item()}", va='bottom', ha='center')
    
    # Add title
    fig.suptitle("Denoising Process Comparison", fontsize=16)
    
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, "denoising_comparison.png"))
    plt.close()
    
    print("Denoising comparison plot created successfully") 