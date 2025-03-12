#!/usr/bin/env python3
"""
Script to test the Consistency Trajectory Model (CTM).
This script initializes CTM models of different sizes and runs inference with different timestep configurations.
"""

import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from PIL import Image
import time

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config
from models import ConsistencyTrajectoryModel, SimpleUNet
from utils.diffusion import q_sample

# Replace the import with our own implementation
def get_diffusion_params(timesteps, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """Initialize diffusion parameters (betas, alphas, etc.)"""
    # Linear beta schedule
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
    
    # Calculations from original DDPM implementation
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), alphas_cumprod[:-1]])
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    
    # Return all parameters as a dictionary
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance,
    }

def save_image_grid(tensor, filename, nrow=4):
    """Save a grid of images from a batch tensor."""
    # Normalize to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # Make grid and convert to numpy
    grid = make_grid(tensor, nrow=nrow, padding=2).permute(1, 2, 0).cpu().numpy()
    
    # Save as image
    plt.figure(figsize=(10, 10))
    plt.imshow(grid)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved image grid to {filename}")

def sample_ctm(model, config, shape, timesteps=20, use_dpm=True):
    """Generate samples using the CTM model."""
    device = next(model.parameters()).device
    
    # Generate random noise
    x_T = torch.randn(shape, device=device)
    
    # Sample using the model
    print(f"Sampling with {'DPM-solver' if use_dpm else 'standard'} mode, {timesteps} steps")
    
    # Time the sampling process
    start_time = time.time()
    samples = model.sample(shape, timesteps=timesteps, x_T=x_T, use_dpm=use_dpm)
    elapsed_time = time.time() - start_time
    
    print(f"Sampling completed in {elapsed_time:.2f} seconds")
    
    return samples

def add_noise_to_images(images, diffusion_params, t):
    """Add noise to images according to diffusion process."""
    device = images.device
    batch_size = images.shape[0]
    
    # Create timestep tensor
    t_tensor = torch.ones(batch_size, dtype=torch.long, device=device) * t
    
    # Add noise
    noise = torch.randn_like(images)
    
    # Get diffusion parameters for timestep t
    sqrt_alpha_cumprod = diffusion_params["sqrt_alphas_cumprod"][t]
    sqrt_one_minus_alpha_cumprod = diffusion_params["sqrt_one_minus_alphas_cumprod"][t]
    
    # Add noise
    noisy_images = sqrt_alpha_cumprod * images + sqrt_one_minus_alpha_cumprod * noise
    
    return noisy_images, noise

def test_ctm_denoising(model, config, clean_images, diffusion_params, t_to_test=None):
    """Test CTM's ability to denoise images at different timesteps."""
    device = next(model.parameters()).device
    results = []
    
    # Default timesteps to test
    if t_to_test is None:
        max_timestep = len(diffusion_params["betas"]) - 1
        t_to_test = [int(max_timestep * f) for f in [0.1, 0.3, 0.5, 0.7, 0.9]]
    
    for t in t_to_test:
        print(f"Testing denoising at timestep {t}")
        
        # Add noise to the clean images
        noisy_images, noise = add_noise_to_images(clean_images, diffusion_params, t)
        
        # Prepare timestep tensor
        batch_size = clean_images.shape[0]
        t_tensor = torch.ones(batch_size, dtype=torch.long, device=device) * t
        
        # Run the model in single-step denoising mode
        with torch.no_grad():
            pred = model(noisy_images, t_tensor)
            denoised_image = pred['sample']
        
        # Run the model in trajectory mode (jumping from t to 0)
        with torch.no_grad():
            t_end = torch.zeros(batch_size, dtype=torch.long, device=device)
            pred_traj = model(noisy_images, t_tensor, t_end)
            traj_denoised = pred_traj['sample']
        
        # Collect results
        results.append({
            't': t,
            'clean': clean_images,
            'noisy': noisy_images,
            'denoised_single': denoised_image,
            'denoised_trajectory': traj_denoised
        })
    
    return results

def visualize_denoising_results(results, output_dir):
    """Visualize denoising results at different timesteps."""
    os.makedirs(output_dir, exist_ok=True)
    
    for result in results:
        t = result['t']
        
        # Calculate PSNR for each method
        def calculate_psnr(pred, target):
            mse = torch.mean((pred - target) ** 2).item()
            return -10 * np.log10(mse)
        
        psnr_single = calculate_psnr(result['denoised_single'], result['clean'])
        psnr_traj = calculate_psnr(result['denoised_trajectory'], result['clean'])
        
        # Create a grid with all images
        grid = torch.cat([
            result['clean'],  # Original
            result['noisy'],  # Noisy
            result['denoised_single'],  # Single-step denoising
            result['denoised_trajectory']  # Trajectory denoising
        ])
        
        # Save grid
        filename = os.path.join(output_dir, f"denoise_t{t}.png")
        save_image_grid(grid, filename, nrow=4)
        
        print(f"Timestep {t}:")
        print(f"  PSNR (Single-step): {psnr_single:.2f} dB")
        print(f"  PSNR (Trajectory): {psnr_traj:.2f} dB")
        
        # If trajectory is better, highlight it
        if psnr_traj > psnr_single:
            print(f"  TRAJECTORY BETTER BY: {psnr_traj - psnr_single:.2f} dB")
        else:
            print(f"  Single-step better by: {psnr_single - psnr_traj:.2f} dB")

def test_ctm_sampling(model, config, output_dir, num_samples=4, default_steps=10, use_dpm=True):
    """Test CTM's ability to generate samples with different numbers of timesteps."""
    device = next(model.parameters()).device
    
    # Sample shape
    shape = (num_samples, config.channels, config.teacher_image_size, config.teacher_image_size)
    
    # Ensure same starting noise for all sampling methods
    torch.manual_seed(42)
    x_T = torch.randn(shape, device=device)
    
    # Use command-line specified defaults for the primary sampling method
    primary_method = {"name": f"{'dpm' if use_dpm else 'standard'}_{default_steps}", 
                     "steps": default_steps, 
                     "dpm": use_dpm, 
                     "description": f"{'DPM-solver' if use_dpm else 'Standard'} ({default_steps} steps)"}
    
    # Test different sampling configurations
    sampling_configs = [
        # Include the primary method first (from command line args)
        primary_method,
        # Include other methods for comparison
        {"name": "dpm_20", "steps": 20, "dpm": True, "description": "DPM-solver (20 steps)"},
        {"name": "dpm_5", "steps": 5, "dpm": True, "description": "DPM-solver (5 steps)"},
        {"name": "standard_50", "steps": 50, "dpm": False, "description": "Standard (50 steps)"},
    ]
    
    # Remove duplicates if the primary method overlaps with predefined methods
    unique_configs = []
    seen_names = set()
    for cfg in sampling_configs:
        if cfg["name"] not in seen_names:
            unique_configs.append(cfg)
            seen_names.add(cfg["name"])
    
    sampling_configs = unique_configs
    
    # Run all sampling configurations and track time
    all_samples = []
    all_times = []
    
    for cfg in sampling_configs:
        print(f"Testing sampling: {cfg['description']} ({cfg['steps']} steps)")
        
        # Time the sampling process
        start_time = time.time()
        
        # Sample
        samples = model.sample(shape, timesteps=cfg['steps'], x_T=x_T.clone(), use_dpm=cfg['dpm'])
        
        # Record time
        elapsed_time = time.time() - start_time
        all_times.append(elapsed_time)
        
        # Save samples
        output_path = os.path.join(output_dir, f"samples_{cfg['name']}.png")
        save_image_grid(samples, output_path, nrow=2)
        
        print(f"  Sampling time: {elapsed_time:.2f} seconds")
        
        # Add to collection
        all_samples.append(samples)
    
    # Create a big grid with all samples for comparison
    big_grid = torch.cat(all_samples)
    big_grid_path = os.path.join(output_dir, "all_sampling_methods.png")
    save_image_grid(big_grid, big_grid_path, nrow=num_samples)
    
    # Print sampling time summary
    print("\nSampling time summary:")
    for i, cfg in enumerate(sampling_configs):
        print(f"  {cfg['description']}: {all_times[i]:.2f} seconds")

def load_or_generate_test_images(config, device, num_images=4):
    """Load test images from dataset or generate random ones."""
    try:
        # Try to load real dataset images
        from data.dataset import get_data_loader
        
        # Get a dataloader using the correct parameters
        dataloader = get_data_loader(
            config,  # Pass the entire config object
            image_size=config.teacher_image_size
        )
        
        # Get first batch
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]  # Assuming images are the first item
            else:
                images = batch
            
            # Normalize to [-1, 1] if needed
            if images.max() > 1.0:
                images = images / 127.5 - 1
                
            return images.to(device)[:num_images]
            
    except (ImportError, FileNotFoundError, AttributeError) as e:
        print(f"Could not load real images: {e}")
        print("Generating random test images...")
        
        # Generate random images (checkerboard pattern for visual testing)
        images = []
        for i in range(num_images):
            # Create a checkerboard pattern
            h, w = config.teacher_image_size, config.teacher_image_size
            y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w))
            
            # Add some circles and patterns
            r = torch.sqrt(x**2 + y**2)
            z = torch.sin(4 * (x + y + torch.sin(2*r)))
            
            # Make it 3-channel
            image = torch.stack([z, torch.sin(4*x), torch.sin(4*y)], dim=0)
            images.append(image)
        
        return torch.stack(images).to(device)

def main():
    parser = argparse.ArgumentParser(description="Test Consistency Trajectory Model")
    parser.add_argument("--size", type=float, default=0.2, help="Model size factor")
    parser.add_argument("--output", type=str, default="output/ctm_test", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda or cpu)")
    parser.add_argument("--sample-steps", type=int, default=20, help="Default number of steps for sampling tests")
    parser.add_argument("--use-dpm", action="store_true", default=True, help="Use DPM-solver for sampling by default")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load config
    config = Config()
    
    # Force device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Initialize CTM model
    print(f"Initializing CTM model with size factor {args.size}")
    ctm_model = ConsistencyTrajectoryModel(config, size_factor=args.size)
    ctm_model.to(device)
    ctm_model.eval()
    
    # Get diffusion parameters
    diffusion_params = get_diffusion_params(config.timesteps, device=device)
    
    # Load or generate test images
    print("Loading test images")
    test_images = load_or_generate_test_images(config, device, num_images=4)
    
    # Test directory
    test_dir = os.path.join(args.output, f"size_{args.size}")
    os.makedirs(test_dir, exist_ok=True)
    
    # Save original test images
    save_image_grid(test_images, os.path.join(test_dir, "test_images.png"))
    
    # Test denoising capabilities
    print("\nTesting denoising capabilities...")
    denoising_results = test_ctm_denoising(ctm_model, config, test_images, diffusion_params)
    
    # Visualize denoising results
    print("\nVisualizing denoising results...")
    denoising_dir = os.path.join(test_dir, "denoising")
    visualize_denoising_results(denoising_results, denoising_dir)
    
    # Test sampling with different step sizes
    print("\nTesting sampling capabilities...")
    sampling_dir = os.path.join(test_dir, "sampling")
    os.makedirs(sampling_dir, exist_ok=True)
    test_ctm_sampling(ctm_model, config, sampling_dir, default_steps=args.sample_steps, use_dpm=args.use_dpm)
    
    print("\nTest complete! Results are saved in:", test_dir)

if __name__ == "__main__":
    main() 