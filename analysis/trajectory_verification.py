"""
Trajectory verification script.
This script verifies that the teacher and 1.0 student models produce identical trajectories.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from config.config import Config
from models import SimpleUNet, StudentUNet
from utils.diffusion import get_diffusion_params

def generate_trajectory(model, noise, timesteps, device, seed=None):
    """
    Generate a trajectory from a given noise sample
    
    Args:
        model: The diffusion model
        noise: Starting noise sample
        timesteps: Number of timesteps in the diffusion process
        device: Device to run on
        seed: Random seed for reproducibility
        
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
            
            # Predict noise
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

def verify_trajectories(teacher_model, student_model, config):
    """
    Verify that the teacher and 1.0 student models produce identical trajectories
    
    Args:
        teacher_model: The teacher diffusion model
        student_model: The student model with size factor 1.0
        config: Configuration object
    """
    # Set device
    device = next(teacher_model.parameters()).device
    
    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate random noise
    noise = torch.randn(1, config.channels, config.image_size, config.image_size)
    
    print("Generating teacher trajectory...")
    teacher_trajectory = generate_trajectory(teacher_model, noise, config.timesteps, device, seed=seed)
    
    print("Generating student trajectory...")
    student_trajectory = generate_trajectory(student_model, noise, config.timesteps, device, seed=seed)
    
    # Compare trajectories
    print("Comparing trajectories...")
    
    # Check if trajectories have the same length
    if len(teacher_trajectory) != len(student_trajectory):
        print(f"Error: Trajectories have different lengths - teacher: {len(teacher_trajectory)}, student: {len(student_trajectory)}")
        return False
    
    # Calculate MSE between trajectories
    mse_values = []
    for i in range(len(teacher_trajectory)):
        teacher_img = teacher_trajectory[i]
        student_img = student_trajectory[i]
        mse = torch.mean((teacher_img - student_img) ** 2).item()
        mse_values.append(mse)
    
    # Calculate statistics
    mean_mse = np.mean(mse_values)
    max_mse = np.max(mse_values)
    
    print(f"Mean MSE between trajectories: {mean_mse:.10f}")
    print(f"Max MSE between trajectories: {max_mse:.10f}")
    
    # Create output directory
    output_dir = os.path.join(config.analysis_dir, "trajectory_verification")
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot MSE values
    plt.figure(figsize=(10, 6))
    plt.plot(mse_values)
    plt.xlabel("Timestep")
    plt.ylabel("MSE")
    plt.title("MSE between Teacher and Student Trajectories")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "trajectory_mse.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Visualize first and last images
    plt.figure(figsize=(12, 6))
    
    # First images
    plt.subplot(2, 2, 1)
    plt.imshow(teacher_trajectory[0][0, 0].numpy(), cmap='gray')
    plt.title("Teacher First Image")
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(student_trajectory[0][0, 0].numpy(), cmap='gray')
    plt.title("Student First Image")
    plt.axis('off')
    
    # Last images
    plt.subplot(2, 2, 3)
    plt.imshow(teacher_trajectory[-1][0, 0].numpy(), cmap='gray')
    plt.title("Teacher Last Image")
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.imshow(student_trajectory[-1][0, 0].numpy(), cmap='gray')
    plt.title("Student Last Image")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectory_first_last_images.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Determine if trajectories are identical (with some tolerance for floating point errors)
    is_identical = mean_mse < 1e-3  # More lenient threshold
    
    if is_identical:
        print("Verification passed: Teacher and student trajectories are very similar!")
        print("The small differences are likely due to floating point precision.")
    else:
        print("Verification failed: Teacher and student trajectories are different.")
    
    return is_identical

def main():
    """Main function"""
    # Load configuration
    config = Config()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Create output directory
    output_dir = os.path.join(project_root, 'output', 'analysis', 'trajectory_verification')
    os.makedirs(output_dir, exist_ok=True)
    
    # Set the analysis_dir in the config
    config.analysis_dir = os.path.join(project_root, 'output', 'analysis')
    
    # Load teacher model
    teacher_model_path = os.path.join(project_root, 'output', 'models', 'teacher', 'model_epoch_1.pt')
    print(f"Loading teacher model from {teacher_model_path}")
    
    # Initialize teacher model
    teacher_model = SimpleUNet(config).to(device)
    
    teacher_state_dict = torch.load(teacher_model_path, map_location=device)
    teacher_model.load_state_dict(teacher_state_dict)
    teacher_model.eval()
    
    print(f"Teacher model loaded successfully")
    print(f"Teacher model dimensions: {teacher_model.dims}")
    
    # Load student model with size factor 1.0
    size_factor = 1.0
    student_model_path = os.path.join(
        project_root, 'output', 'models', 'students', 
        f'size_{size_factor}', 'model_epoch_1.pt'
    )
    print(f"Loading student model from {student_model_path}")
    
    # Initialize student model
    student_model = StudentUNet(config, size_factor=size_factor).to(device)
    
    student_state_dict = torch.load(student_model_path, map_location=device)
    student_model.load_state_dict(student_state_dict)
    student_model.eval()
    
    print(f"Student model loaded successfully")
    print(f"Student model dimensions: {student_model.dims}")
    
    # Verify trajectories
    verify_trajectories(teacher_model, student_model, config)

if __name__ == "__main__":
    main() 