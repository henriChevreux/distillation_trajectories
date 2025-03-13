#!/usr/bin/env python3
"""
Script to generate a heatmap visualization of MSE across different model sizes and image sizes.
"""

import os
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import traceback
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config
from models import SimpleUNet, StudentUNet
from utils.diffusion import get_diffusion_params, q_sample, p_sample_loop, p_sample
from data.dataset import get_data_loader

def calculate_mse_for_model(model, config, diffusion_params, image_size, test_samples=None, is_teacher=False):
    """Calculate the reconstruction MSE for a given model at different timesteps"""
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
        
        total_mse = 0.0
        
        with torch.no_grad():
            for t in timesteps_to_test:
                print(f"  Denoising from timestep {t}/{max_steps} ({len(timesteps_to_test)}/{len(timesteps_to_test)})")
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
                
                # Predict noise using the model
                try:
                    predicted_noise = model(noisy_images, t_tensor)
                    
                    # Verify that predicted noise has the correct shape
                    if predicted_noise.shape != noise.shape:
                        print(f"  ERROR: Model output shape {predicted_noise.shape} doesn't match expected shape {noise.shape}")
                        continue
                    
                    # Use the predicted noise to reconstruct the image (one-step denoising)
                    # x_0 = (x_t - sqrt(1-alpha_cumprod_t) * predicted_noise) / sqrt(alpha_cumprod_t)
                    reconstructed_images = (noisy_images - sqrt_one_minus_alpha_cumprod * predicted_noise) / sqrt_alpha_cumprod
                    
                    # Clamp values to valid image range [-1, 1] or [0, 1] depending on your data normalization
                    reconstructed_images = torch.clamp(reconstructed_images, -1.0, 1.0)
                    
                    # Calculate MSE between reconstructed images and original reference images
                    mse = F.mse_loss(reconstructed_images, reference_images)
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
        traceback.print_exc()
        return np.nan

def create_mse_heatmap(config, test_samples=None):
    """Create a heatmap grid of MSE values comparing different models"""
    # Add backwards compatibility for models using teacher_steps instead of timesteps
    if hasattr(config, "teacher_steps") and not hasattr(config, "timesteps"):
        config.timesteps = config.teacher_steps
        print(f"Added teacher_steps={config.timesteps} to config")
        
    if hasattr(config, "student_steps") and not hasattr(config, "student_timesteps"):
        config.student_timesteps = config.student_steps
        print(f"Added student_steps={config.student_timesteps} to config")
    
    # Set output directory if not already set
    if not hasattr(config, "output_dir"):
        config.output_dir = os.path.join(os.getcwd(), "output")
        print(f"Added output_dir={config.output_dir} to config")
    
    # Set directories if not already set
    if not hasattr(config, "teacher_models_dir"):
        config.teacher_models_dir = os.path.join(config.output_dir, "models", "teacher")
    if not hasattr(config, "student_models_dir"):
        config.student_models_dir = os.path.join(config.output_dir, "models", "students")
    
    # Force CPU usage for better debug stability
    device = torch.device("cpu")
    print("Forcing CPU usage for better debug stability")
    
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
    # Ensure diffusion parameters are on CPU
    diffusion_params = initialize_diffusion_parameters(config.timesteps, device=device)
    
    # Get size factors and image size factors
    size_factors = config.student_size_factors.copy()
    image_size_factors = config.student_image_size_factors.copy()
    
    # Testing mode: use a limited set of sizes
    # For MSE test, we use a small subset of sizes to speed up testing
    # Include the smallest, largest, and one middle size if more than 2 sizes
    if hasattr(config, "mse_size_factors_limit") and config.mse_size_factors_limit:
        if len(size_factors) > 2:
            size_factors = [min(size_factors), size_factors[len(size_factors)//2], max(size_factors)]
        print(f"Using limited set of {len(size_factors)} size factors for testing: {size_factors}")
    
    if hasattr(config, "mse_image_size_factors_limit") and config.mse_image_size_factors_limit:
        if len(image_size_factors) > 2:
            image_size_factors = [min(image_size_factors), image_size_factors[len(image_size_factors)//2], max(image_size_factors)]
        print(f"Using limited set of {len(image_size_factors)} image size factors for testing: {image_size_factors}")
    
    print(f"\nTESTING MODE: Using {len(size_factors)} model size(s) and {len(image_size_factors)} image size(s)")
    print(f"Model sizes: {size_factors}")
    print(f"Image sizes: {image_size_factors}")
    
    # Check if all the required models exist
    check_model_files(config, size_factors, image_size_factors)
    
    # Calculate teacher MSE
    teacher_image_size = config.teacher_image_size
    print(f"\nCalculating Teacher model image reconstruction MSE (image size: {teacher_image_size})")
    print(f"(Measuring the model's ability to reconstruct original images from noisy inputs)")
    teacher_mse = calculate_mse_for_model(teacher_model, config, diffusion_params, teacher_image_size, test_samples, is_teacher=True)
    print(f"Teacher image reconstruction MSE: {teacher_mse}")
    
    # Calculate student MSE for each combination of size and image size
    results = np.zeros((len(size_factors) + 1, len(image_size_factors)))  # +1 for teacher
    
    # First row is for the teacher model
    for j, image_size_factor in enumerate(image_size_factors):
        student_image_size = int(teacher_image_size * image_size_factor)
        print(f"\nProcessing teacher model with image_size={student_image_size}")
        
        # Calculate MSE for teacher at this image size
        teacher_mse_at_size = calculate_mse_for_model(teacher_model, config, diffusion_params, student_image_size, test_samples, is_teacher=True)
        
        # Store the relative MSE (compared to original teacher size)
        if not np.isnan(teacher_mse_at_size) and not np.isnan(teacher_mse):
            relative_mse = teacher_mse_at_size / teacher_mse
            results[0, j] = relative_mse
            print(f"Teacher image reconstruction MSE at image size {student_image_size}: {teacher_mse_at_size} (compared to original: {relative_mse:.2f}x)")
        else:
            results[0, j] = np.nan
            print(f"Teacher image reconstruction MSE at image size {student_image_size}: {teacher_mse_at_size} (compared to original: nanx)")
    
    # Now process student models
    for i, size_factor in enumerate(tqdm(size_factors, desc="Model sizes")):
        for j, image_size_factor in enumerate(image_size_factors):
            student_image_size = int(teacher_image_size * image_size_factor)
            print(f"\nProcessing student model: size={size_factor}, image_size={image_size_factor}")
            
            # Determine architecture type based on size factor
            arch_type = get_architecture_type(size_factor)
            print(f"Using architecture type: {arch_type} for size factor {size_factor}")
            
            # Determine hidden dimensions
            teacher_hidden_dims = [64, 128, 256, 512]  # Default UNet hidden dimensions
            student_hidden_dims = get_student_hidden_dims(size_factor, arch_type, teacher_hidden_dims)
            print(f"Teacher hidden dims: {teacher_hidden_dims}")
            print(f"Student hidden dims: {student_hidden_dims}")
            
            # Load student model
            student_path = os.path.join(config.student_models_dir, f"size_{size_factor}_img_{image_size_factor}", "model.pt")
            student_model = StudentUNet(config, size_factor)
            student_checkpoint = torch.load(student_path, map_location=device)
            student_model.load_state_dict(student_checkpoint)
            student_model.to(device)
            student_model.eval()
            
            # Calculate MSE
            student_mse = calculate_mse_for_model(student_model, config, diffusion_params, student_image_size, test_samples)
            
            if not np.isnan(student_mse) and not np.isnan(teacher_mse):
                relative_mse = student_mse / teacher_mse
                results[i + 1, j] = relative_mse  # +1 to account for teacher row
                print(f"Student image reconstruction MSE: {student_mse} (compared to teacher: {relative_mse:.2f}x)")
            else:
                results[i + 1, j] = np.nan
                print(f"Student image reconstruction MSE: {student_mse} (compared to teacher: nanx)")
    
    # Check if we have valid results
    if np.all(np.isnan(results)):
        raise ValueError("No valid MSE results calculated")
        
    # Save the results
    save_heatmap(results, ["Teacher"] + size_factors, image_size_factors, config)
    
    # Return results
    return results

# Function to initialize diffusion parameters
def initialize_diffusion_parameters(timesteps, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
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

# Add the missing model loading functions
def load_teacher_model(config):
    """Load the teacher model with the appropriate architecture"""
    # The SimpleUNet takes a config object
    from dataclasses import dataclass
    
    @dataclass
    class ModelConfig:
        channels: int = 3  # RGB images
        hidden_dims: list = None
        image_size: int = 256
        time_embedding_dim: int = 256
    
    # Create a model config
    model_config = ModelConfig()
    model_config.hidden_dims = [64, 128, 256, 512]  # Default UNet dimensions
    model_config.image_size = config.image_size
    
    # Create teacher model
    model = SimpleUNet(model_config)
    return model

def load_student_model(config, size_factor, image_size_factor):
    """Load a student model with the specified size factor and image size factor"""
    # The StudentUNet takes a config object and a size factor
    from dataclasses import dataclass
    
    @dataclass
    class ModelConfig:
        channels: int = 3  # RGB images
        hidden_dims: list = None
        image_size: int = 256
        time_embedding_dim: int = 256
    
    # Create a model config
    model_config = ModelConfig()
    model_config.hidden_dims = [64, 128, 256, 512]  # Default UNet dimensions
    model_config.image_size = int(config.image_size * image_size_factor)
    
    # Determine architecture type based on size factor
    if size_factor < 0.1:
        arch_type = "tiny"
    elif size_factor < 0.3:
        arch_type = "small"
    elif size_factor < 0.7:
        arch_type = "medium"
    else:
        arch_type = "full"
    
    # Create student model
    model = StudentUNet(model_config, size_factor, arch_type)
    return model

def get_architecture_type(size_factor):
    """Determine the architecture type based on the size factor"""
    if size_factor < 0.1:
        return "tiny"
    elif size_factor < 0.5:
        return "medium"
    else:
        return "full"

def get_student_hidden_dims(size_factor, arch_type, teacher_dims):
    """Calculate the hidden dimensions of a student model"""
    if arch_type == "tiny":
        # Tiny models use a very simplified architecture
        return [int(max(3, dim * size_factor)) for dim in teacher_dims[:3]]
    else:
        # Medium and full models scale all dimensions
        return [int(max(4, dim * size_factor)) for dim in teacher_dims]

def check_model_files(config, size_factors, image_size_factors):
    """Check if all expected model files exist"""
    print("\nChecking for expected model files:")
    all_models_exist = True
    
    for size_factor in size_factors:
        for image_size_factor in image_size_factors:
            model_path = os.path.join(config.student_models_dir, f"size_{size_factor}_img_{image_size_factor}", "model.pt")
            
            if os.path.exists(model_path):
                print(f"  + Found: size_{size_factor}_img_{image_size_factor}/model.pt")
            else:
                print(f"  - Missing: size_{size_factor}_img_{image_size_factor}/model.pt")
                all_models_exist = False
    
    if all_models_exist:
        print("All expected model files are present.")
    else:
        print("Some model files are missing. MSE results may be incomplete.")
    
    return all_models_exist

def save_heatmap(results, size_factors, image_size_factors, config):
    """Save the MSE heatmap visualization"""
    # Create output directory if it doesn't exist
    output_dir = os.path.join(config.output_dir, "analysis", "comparative", "reconstruction_mse")
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Plot heatmap
    ax = sns.heatmap(
        reversed_results, 
        annot=True, 
        fmt=".2f", 
        cmap="viridis_r",
        xticklabels=[f"{factor:.4f}x" for factor in image_size_factors],
        yticklabels=y_labels,
        vmin=0.0,
        vmax=np.nanpercentile(reversed_results, 95) * 1.5  # Use 95th percentile as max
    )
    
    # Set labels
    plt.title("Image Reconstruction MSE Relative to Teacher Model (Full Size)")
    plt.xlabel("Image Size Factor")
    plt.ylabel("Model Size Factor")
    
    # Add colorbar label
    cbar = ax.collections[0].colorbar
    cbar.set_label("Relative Image MSE (lower is better)")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "reconstruction_mse_heatmap.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "reconstruction_mse_heatmap.pdf"))
    plt.close()
    
    # Save raw data - keep original ordering for data files
    np.save(os.path.join(output_dir, "mse_data.npy"), results)
    
    # Also save as CSV
    with open(os.path.join(output_dir, "mse_data.csv"), "w") as f:
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
    
    # Generate a best-normalized version of the heatmap
    save_normalized_heatmap(results, size_factors, image_size_factors, output_dir)
    
    print(f"Saved image reconstruction MSE heatmap and data to {output_dir}")

def save_normalized_heatmap(results, size_factors, image_size_factors, output_dir):
    """Save a normalized version of the heatmap where all values are relative to the best (lowest) MSE"""
    # Find the minimum non-NaN value in the results
    min_mse = np.nanmin(results)
    if np.isnan(min_mse):
        print("Warning: No valid MSE values found for normalization")
        return
    
    # Normalize all values relative to the best model
    normalized_results = results / min_mse
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Reverse order of size factors and results for display (from large to small)
    reversed_size_factors = size_factors.copy()
    reversed_normalized_results = normalized_results.copy()
    
    if len(size_factors) > 1:
        # Keep "Teacher" at the top if it exists, and reverse the rest
        if isinstance(size_factors[0], str) and size_factors[0] == "Teacher":
            reversed_size_factors = [size_factors[0]] + list(reversed(size_factors[1:]))
            reversed_normalized_results = np.vstack([normalized_results[0:1], normalized_results[1:][::-1]])
        else:
            reversed_size_factors = list(reversed(size_factors))
            reversed_normalized_results = normalized_results[::-1]
    
    # Create y-tick labels, handling both string and numeric types
    y_labels = []
    for sf in reversed_size_factors:
        if isinstance(sf, str):
            y_labels.append(sf)  # Use the string directly (e.g., "Teacher")
        else:
            y_labels.append(f"{sf:.2f}x")  # Format numeric size factors
    
    # Plot heatmap
    ax = sns.heatmap(
        reversed_normalized_results, 
        annot=True, 
        fmt=".2f", 
        cmap="viridis_r",
        xticklabels=[f"{factor:.4f}x" for factor in image_size_factors],
        yticklabels=y_labels,
        vmin=1.0,  # Start at 1.0 since this is the best score
        vmax=min(np.nanpercentile(reversed_normalized_results, 95), 2.0)  # Cap at 2x the best or 95th percentile
    )
    
    # Mark the best (lowest MSE) model with a box
    best_idx = np.where(reversed_normalized_results == 1.0)
    for i, j in zip(best_idx[0], best_idx[1]):
        # Draw a rectangle around the best value
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', lw=3))
    
    # Set labels
    plt.title("Image Reconstruction MSE Relative to Best Model")
    plt.xlabel("Image Size Factor")
    plt.ylabel("Model Size Factor")
    
    # Add colorbar label
    cbar = ax.collections[0].colorbar
    cbar.set_label("Relative Image MSE (1.0 = best model)")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "best_normalized_mse_heatmap.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "best_normalized_mse_heatmap.pdf"))
    plt.close()
    
    # Save raw data - keep original ordering for data files
    np.save(os.path.join(output_dir, "normalized_mse_data.npy"), normalized_results)
    
    # Also save as CSV
    with open(os.path.join(output_dir, "normalized_mse_data.csv"), "w") as f:
        # Write header
        f.write("model_size,")
        f.write(",".join([f"img_{factor}" for factor in image_size_factors]))
        f.write("\n")
        
        # Write data
        for i, size_factor in enumerate(size_factors):
            if isinstance(size_factor, str):
                f.write(f"{size_factor}")
            else:
                f.write(f"{size_factor}")
                
            for j in range(len(image_size_factors)):
                f.write(f",{normalized_results[i, j]}")
            f.write("\n")
    
    print(f"Saved best-normalized image reconstruction MSE heatmap to {output_dir}")

def main():
    """Main function to generate the MSE heatmap"""
    try:
        # Display diagnostic information
        print("\n====== MSE Heatmap Generation ======")
        print(f"Python version: {sys.version}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"NumPy version: {np.__version__}")
        print(f"Current working directory: {os.getcwd()}")
        
        config = Config()
        
        # Print key configuration values
        print("\nKey configuration values:")
        print(f"- Dataset: {config.dataset if hasattr(config, 'dataset') else 'Not specified'}")
        print(f"- Teacher image size: {config.teacher_image_size if hasattr(config, 'teacher_image_size') else 'Not specified'}")
        print(f"- Batch size: {config.batch_size if hasattr(config, 'batch_size') else 'Not specified'}")
        print(f"- Student size factors: {config.student_size_factors if hasattr(config, 'student_size_factors') else 'Not specified'}")
        print(f"- Student image size factors: {config.student_image_size_factors if hasattr(config, 'student_image_size_factors') else 'Not specified'}")
        
        # Ensure configuration has required attributes
        if not hasattr(config, 'batch_size'):
            config.batch_size = 32
            print(f"Setting default batch_size = {config.batch_size}")
            
        # Create necessary directories
        config.create_directories()
        
        # Generate the heatmap
        create_mse_heatmap(config)
        
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 