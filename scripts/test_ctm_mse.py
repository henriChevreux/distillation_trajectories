#!/usr/bin/env python3
"""
Script to generate MSE heatmaps for Consistency Trajectory Models (CTM).
This extends the original MSE heatmap analysis to include CTM models.
"""

import os
import sys
import torch
import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config
from models import ConsistencyTrajectoryModel, SimpleUNet, StudentUNet
from utils.diffusion import q_sample
from analysis.mse_heatmap import calculate_mse_for_model, save_heatmap, save_normalized_heatmap

def calculate_mse_for_ctm(model, config, diffusion_params, image_size, test_samples=None, is_teacher=False,
                        use_trajectory=True):
    """
    Calculate the reconstruction MSE for a CTM model at different timesteps.
    
    This function is similar to calculate_mse_for_model but adapted for CTM models,
    which can use either single-step or trajectory mode.
    """
    try:
        # Set device
        device = next(model.parameters()).device
        print(f"  Using device: {device}")
        
        # Ensure diffusion parameters are on the same device as the model
        diff_params = {}
        for key, value in diffusion_params.items():
            diff_params[key] = value.to(device)
        
        # Process test data at the correct image size
        print(f"  Preparing test data with image size {image_size}")
        
        # Use provided test samples or generate random ones if not provided
        if test_samples is not None:
            # Resize test samples to the correct image size if needed
            if test_samples.shape[2] != image_size or test_samples.shape[3] != image_size:
                print(f"  Resizing test samples from {test_samples.shape[2]}x{test_samples.shape[3]} to {image_size}x{image_size}")
                test_samples = torch.nn.functional.interpolate(test_samples, size=(image_size, image_size), mode='bilinear')
            
            # Use a subset of samples to avoid memory issues
            num_test_samples = min(100, len(test_samples))
            reference_images = test_samples[:num_test_samples].to(device)
            print(f"  Using {num_test_samples} provided test samples")
        else:
            # If no test samples provided, create random ones (fallback)
            print("  No test samples provided, using random noise as reference images")
            num_test_samples = 2  # Use minimal samples in fallback mode
            torch.manual_seed(42)  # Set fixed seed for reproducibility
            reference_images = torch.randn(num_test_samples, 3, image_size, image_size).to(device)
        
        # Define timesteps to test
        # Calculate various timesteps to test for reconstruction MSE
        timesteps_to_test = []
        max_steps = len(diff_params["betas"]) - 1
        
        # Test 5 evenly spaced timesteps for a more comprehensive analysis
        for t_fraction in [0, 0.1, 0.2, 0.3, 0.4]:
            timestep = max(min(int(max_steps * t_fraction), max_steps), 0)
            timesteps_to_test.append(timestep)
        
        print(f"  Testing {len(timesteps_to_test)} timesteps with {len(reference_images)} samples...")
        print(f"  Using {'trajectory' if use_trajectory else 'single-step'} mode")
        
        total_mse = 0.0
        
        with torch.no_grad():
            for t in timesteps_to_test:
                print(f"  Denoising from timestep {t}/{max_steps}")
                t_tensor = torch.ones(len(reference_images), dtype=torch.long).to(device) * t
                
                # Add noise to the clean images
                torch.manual_seed(43)  # Set fixed seed for reproducibility
                noise = torch.randn_like(reference_images).to(device)
                
                # Safely access diffusion parameters
                try:
                    sqrt_alpha_cumprod = diff_params["sqrt_alphas_cumprod"][t]
                    sqrt_one_minus_alpha_cumprod = diff_params["sqrt_one_minus_alphas_cumprod"][t]
                except (IndexError, KeyError) as e:
                    print(f"  ERROR: Invalid diffusion parameter access: {e}")
                    return np.nan
                
                noisy_images = sqrt_alpha_cumprod * reference_images + sqrt_one_minus_alpha_cumprod * noise
                
                # Predict using the CTM model
                try:
                    if use_trajectory:
                        # Use trajectory mode to go directly from t to 0
                        t_end = torch.zeros_like(t_tensor)
                        pred = model(noisy_images, t_tensor, t_end)
                        denoised_images = pred['sample']
                    else:
                        # Use single-step mode (traditional diffusion)
                        pred = model(noisy_images, t_tensor)
                        denoised_images = pred['sample']
                    
                    # Verify that output has the correct shape
                    if denoised_images.shape != reference_images.shape:
                        print(f"  ERROR: Model output shape {denoised_images.shape} doesn't match expected shape {reference_images.shape}")
                        continue
                    
                    # Clamp values to valid image range
                    denoised_images = torch.clamp(denoised_images, -1.0, 1.0)
                    
                    # Calculate MSE between reconstructed images and original reference images
                    mse = F.mse_loss(denoised_images, reference_images)
                    total_mse += mse.item()
                    
                    print(f"    Image reconstruction MSE at timestep {t}: {mse.item():.4e}")
                except RuntimeError as e:
                    print(f"  ERROR during model prediction: {str(e)}")
                    if "CUDA" in str(e):
                        print("  This may be a GPU memory issue. Try running on CPU or with smaller batch size.")
                    return np.nan
        
        # Average MSE across timesteps
        avg_mse = total_mse / len(timesteps_to_test)
        print(f"  Average image reconstruction MSE across {len(timesteps_to_test)} timesteps: {avg_mse:.4e}")
        return avg_mse
        
    except Exception as e:
        print(f"Error calculating reconstruction MSE: {str(e)}")
        import traceback
        traceback.print_exc()
        return np.nan

def create_ctm_mse_heatmap(config, ctm_dir, test_samples=None):
    """
    Create a heatmap grid of MSE values comparing different CTM models.
    
    Args:
        config: Configuration object
        ctm_dir: Directory containing trained CTM models
        test_samples: Optional test samples for consistent evaluation
    """
    # Set output directory for results
    output_dir = os.path.join(config.output_dir, "analysis", "comparative", "ctm_mse")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cpu")
    print("Using CPU for better memory stability")
    
    # Load teacher model
    teacher_path = os.path.join(config.teacher_models_dir, "model_epoch_1.pt")
    print(f"Loading teacher model from {teacher_path}")
    
    # Create teacher model instance
    teacher_model = SimpleUNet(config)
    # Load state dict
    teacher_checkpoint = torch.load(teacher_path, map_location=device)
    teacher_model.load_state_dict(teacher_checkpoint)
    teacher_model.to(device)
    teacher_model.eval()
    
    # Initialize diffusion parameters
    print("Initializing diffusion parameters...")
    diffusion_params = get_diffusion_params(config.timesteps, device=device)
    
    # Get available CTM models
    size_factors = []
    model_paths = []
    
    # Search for CTM models in the ctm_dir
    if os.path.exists(ctm_dir):
        print(f"Searching for CTM models in {ctm_dir}")
        for size_dir in os.listdir(ctm_dir):
            if size_dir.startswith("size_"):
                try:
                    size_factor = float(size_dir.split("_")[1])
                    model_path = os.path.join(ctm_dir, size_dir, "model.pt")
                    
                    if os.path.exists(model_path):
                        size_factors.append(size_factor)
                        model_paths.append(model_path)
                        print(f"Found CTM model with size factor {size_factor}")
                except ValueError:
                    pass
    
    if not size_factors:
        print("No CTM models found. Using default size factors for placeholder.")
        size_factors = [0.01, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Sort size factors
    size_factors = sorted(size_factors)
    
    # Get image size factors
    image_size_factors = config.student_image_size_factors.copy()
    
    # Testing mode: use a limited set of sizes if specified
    if hasattr(config, "mse_image_size_factors_limit") and config.mse_image_size_factors_limit:
        if len(image_size_factors) > 2:
            image_size_factors = [min(image_size_factors), image_size_factors[len(image_size_factors)//2], max(image_size_factors)]
        print(f"Using limited set of {len(image_size_factors)} image size factors for testing: {image_size_factors}")
    
    print(f"\nTESTING MODE: Using {len(size_factors)} CTM model size(s) and {len(image_size_factors)} image size(s)")
    print(f"CTM model sizes: {size_factors}")
    print(f"Image sizes: {image_size_factors}")
    
    # Calculate teacher MSE
    teacher_image_size = config.teacher_image_size
    print(f"\nCalculating Teacher model image reconstruction MSE (image size: {teacher_image_size})")
    teacher_mse = calculate_mse_for_model(teacher_model, config, diffusion_params, teacher_image_size, test_samples, is_teacher=True)
    print(f"Teacher image reconstruction MSE: {teacher_mse}")
    
    # Calculate CTM MSE for each combination of size and image size
    results_trajectory = np.zeros((len(size_factors) + 1, len(image_size_factors)))  # +1 for teacher
    results_single = np.zeros((len(size_factors) + 1, len(image_size_factors)))  # +1 for teacher
    
    # First row is for the teacher model (same for both result matrices)
    for j, image_size_factor in enumerate(image_size_factors):
        student_image_size = int(teacher_image_size * image_size_factor)
        print(f"\nProcessing teacher model with image_size={student_image_size}")
        
        # Calculate MSE for teacher at this image size
        teacher_mse_at_size = calculate_mse_for_model(teacher_model, config, diffusion_params, student_image_size, test_samples, is_teacher=True)
        
        # Store the relative MSE (compared to original teacher size)
        if not np.isnan(teacher_mse_at_size) and not np.isnan(teacher_mse):
            relative_mse = teacher_mse_at_size / teacher_mse
            results_trajectory[0, j] = relative_mse
            results_single[0, j] = relative_mse
            print(f"Teacher image reconstruction MSE at image size {student_image_size}: {teacher_mse_at_size} (compared to original: {relative_mse:.2f}x)")
        else:
            results_trajectory[0, j] = np.nan
            results_single[0, j] = np.nan
            print(f"Teacher image reconstruction MSE at image size {student_image_size}: {teacher_mse_at_size} (compared to original: nanx)")
    
    # Now process CTM models
    for i, size_factor in enumerate(tqdm(size_factors, desc="CTM model sizes")):
        for j, image_size_factor in enumerate(image_size_factors):
            student_image_size = int(teacher_image_size * image_size_factor)
            print(f"\nProcessing CTM model: size={size_factor}, image_size={image_size_factor}")
            
            # Find model path for this size factor
            model_path = None
            if i < len(model_paths):
                model_path = model_paths[i]
            
            if model_path and os.path.exists(model_path):
                # Load CTM model
                print(f"Loading CTM model from {model_path}")
                ctm_model = ConsistencyTrajectoryModel(config, size_factor=size_factor)
                ctm_model.load_state_dict(torch.load(model_path, map_location=device))
                ctm_model.to(device)
                ctm_model.eval()
                
                # Calculate MSE in trajectory mode
                print("Evaluating in trajectory mode")
                trajectory_mse = calculate_mse_for_ctm(ctm_model, config, diffusion_params, student_image_size, 
                                                     test_samples, use_trajectory=True)
                
                # Calculate MSE in single-step mode
                print("Evaluating in single-step mode")
                single_mse = calculate_mse_for_ctm(ctm_model, config, diffusion_params, student_image_size, 
                                                 test_samples, use_trajectory=False)
                
                if not np.isnan(trajectory_mse) and not np.isnan(teacher_mse):
                    relative_trajectory_mse = trajectory_mse / teacher_mse
                    results_trajectory[i + 1, j] = relative_trajectory_mse  # +1 to account for teacher row
                    print(f"CTM trajectory MSE: {trajectory_mse} (compared to teacher: {relative_trajectory_mse:.2f}x)")
                else:
                    results_trajectory[i + 1, j] = np.nan
                    print(f"CTM trajectory MSE: {trajectory_mse} (compared to teacher: nanx)")
                
                if not np.isnan(single_mse) and not np.isnan(teacher_mse):
                    relative_single_mse = single_mse / teacher_mse
                    results_single[i + 1, j] = relative_single_mse  # +1 to account for teacher row
                    print(f"CTM single-step MSE: {single_mse} (compared to teacher: {relative_single_mse:.2f}x)")
                else:
                    results_single[i + 1, j] = np.nan
                    print(f"CTM single-step MSE: {single_mse} (compared to teacher: nanx)")
            else:
                # Model not found, just use NaN
                results_trajectory[i + 1, j] = np.nan
                results_single[i + 1, j] = np.nan
                print(f"CTM model with size={size_factor} not found, using NaN for MSE values")
    
    # Check if we have valid results
    if np.all(np.isnan(results_trajectory)) and np.all(np.isnan(results_single)):
        raise ValueError("No valid MSE results calculated")
    
    # Save the results
    print("Saving CTM trajectory mode results")
    trajectory_dir = os.path.join(output_dir, "trajectory_mode")
    os.makedirs(trajectory_dir, exist_ok=True)
    save_heatmap(results_trajectory, ["Teacher"] + size_factors, image_size_factors, config, 
                 output_dir=trajectory_dir, title="CTM Trajectory Mode - Image Reconstruction MSE")
    
    print("Saving CTM single-step mode results")
    single_dir = os.path.join(output_dir, "single_step_mode")
    os.makedirs(single_dir, exist_ok=True)
    save_heatmap(results_single, ["Teacher"] + size_factors, image_size_factors, config,
                output_dir=single_dir, title="CTM Single-step Mode - Image Reconstruction MSE")
    
    # Compare trajectory vs. single-step
    print("Creating comparison between trajectory and single-step modes")
    if not np.all(np.isnan(results_trajectory)) and not np.all(np.isnan(results_single)):
        # Calculate improvement of trajectory over single-step
        improvement = results_single / results_trajectory
        improvement_dir = os.path.join(output_dir, "improvement")
        os.makedirs(improvement_dir, exist_ok=True)
        
        # Create improved save_heatmap version with custom title for this comparison
        save_comparison_heatmap(improvement, ["Teacher"] + size_factors, image_size_factors, config,
                               output_dir=improvement_dir)
    
    return results_trajectory, results_single

def save_comparison_heatmap(results, size_factors, image_size_factors, config, output_dir):
    """Save a heatmap showing the improvement of trajectory mode over single-step mode."""
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Reverse order of size factors and results for display (from large to small)
    reversed_size_factors = size_factors.copy()
    reversed_results = results.copy()
    
    if len(size_factors) > 1:
        # Keep "Teacher" at the top if it exists, and reverse the rest
        if isinstance(size_factors[0], str) and size_factors[0] == "Teacher":
            reversed_size_factors = [size_factors[0]] + list(reversed(size_factors[1:]))
            reversed_results = np.vstack([results[0:1], results[1:][::-1]])
        else:
            reversed_size_factors = list(reversed(size_factors))
            reversed_results = results[::-1]
    
    # Create y-tick labels, handling both string and numeric types
    y_labels = []
    for sf in reversed_size_factors:
        if isinstance(sf, str):
            y_labels.append(sf)  # Use the string directly (e.g., "Teacher")
        else:
            y_labels.append(f"{sf:.2f}x")  # Format numeric size factors
    
    # Create a diverging colormap centered at 1.0
    # Values > 1.0 indicate trajectory is better (green)
    # Values < 1.0 indicate single-step is better (purple)
    cmap = sns.diverging_palette(260, 120, s=80, l=55, as_cmap=True)
    
    # Plot heatmap
    ax = sns.heatmap(
        reversed_results, 
        annot=True, 
        fmt=".2f", 
        cmap=cmap,
        xticklabels=[f"{factor:.4f}x" for factor in image_size_factors],
        yticklabels=y_labels,
        vmin=0.5,  # Values below 1.0 mean single-step is better
        vmax=1.5,  # Values above 1.0 mean trajectory is better
        center=1.0,  # Center the colormap at 1.0 (equal performance)
    )
    
    # Set labels
    plt.title("Improvement of Trajectory Mode over Single-step Mode")
    plt.xlabel("Image Size Factor")
    plt.ylabel("Model Size Factor")
    
    # Add colorbar label
    cbar = ax.collections[0].colorbar
    cbar.set_label("Ratio of Single-step MSE / Trajectory MSE\n(>1.0 means trajectory is better)")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectory_vs_single_step.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "trajectory_vs_single_step.pdf"))
    plt.close()
    
    # Save raw data
    np.save(os.path.join(output_dir, "trajectory_vs_single_step.npy"), results)
    
    # Also save as CSV
    with open(os.path.join(output_dir, "trajectory_vs_single_step.csv"), "w") as f:
        # Write header
        f.write("model_size,")
        f.write(",".join([f"img_{factor}" for factor in image_size_factors]))
        f.write("\n")
        
        # Write data
        for i, size_factor in enumerate(size_factors):
            if isinstance(size_factor, str):
                f.write(f"{size_factor}")  # Use the string directly for labels
            else:
                f.write(f"{size_factor}")  # Use original numeric value for data
                
            for j in range(len(image_size_factors)):
                f.write(f",{results[i, j]}")
            f.write("\n")
    
    print(f"Saved trajectory vs. single-step comparison to {output_dir}")

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

def main():
    parser = argparse.ArgumentParser(description="Generate MSE heatmaps for CTM models")
    parser.add_argument("--ctm-dir", type=str, default="output/ctm_models", help="Directory containing CTM models")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cuda or cpu)")
    args = parser.parse_args()
    
    # Load config
    config = Config()
    
    # Override device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create CTM MSE heatmap
    create_ctm_mse_heatmap(config, args.ctm_dir)

if __name__ == "__main__":
    main() 