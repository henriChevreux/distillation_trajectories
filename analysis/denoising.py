import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import seaborn as sns
from tqdm import tqdm
import pandas as pd

def denormalize_image(image):
    """Convert normalized image back to [0, 1] range"""
    return (image.clamp(-1, 1) + 1) / 2

def plot_denoising_grid(model, config, device, num_samples=1, save_path=None):
    """
    Generate and plot a grid showing the denoising process at different timesteps.
    
    Args:
        model: The diffusion model (teacher or student)
        config: Configuration object
        device: Device to run the model on
        num_samples: Number of samples to generate
        save_path: Path to save the visualization
    """
    model.eval()
    
    # Generate random noise
    image_size = config.teacher_image_size
    x = torch.randn(num_samples, 3, image_size, image_size).to(device)
    
    # Select timesteps to visualize
    num_viz_steps = 8
    timesteps = torch.linspace(0, config.timesteps - 1, num_viz_steps).long()
    
    # Create figure
    fig, axes = plt.subplots(num_samples, num_viz_steps, figsize=(20, 4 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]
    
    # Generate and plot images
    with torch.no_grad():
        for t_idx, t in enumerate(timesteps):
            # Denoise image at current timestep
            noise_level = torch.ones(num_samples, device=device) * t
            denoised = model(x, noise_level)
            
            # Plot each sample
            for sample_idx in range(num_samples):
                img = denormalize_image(denoised[sample_idx])
                img = img.cpu().permute(1, 2, 0).numpy()
                
                ax = axes[sample_idx, t_idx]
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f't={t.item()}')
    
    plt.tight_layout()
    
    if save_path:
        # Create directory if it doesn't exist
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def compare_denoising_process(teacher_model, student_model, config, device, save_dir=None):
    """
    Compare the denoising process between teacher and student models.
    
    Args:
        teacher_model: The teacher diffusion model
        student_model: The student diffusion model
        config: Configuration object
        device: Device to run models on
        save_dir: Directory to save visualizations
    """
    teacher_model.eval()
    student_model.eval()
    
    # Generate same initial noise for both models
    image_size = config.teacher_image_size
    x = torch.randn(1, 3, image_size, image_size).to(device)
    x_student = x.clone()
    
    # Select timesteps to visualize
    num_viz_steps = 8
    timesteps = torch.linspace(0, config.timesteps - 1, num_viz_steps).long()
    
    # Create figure
    fig, axes = plt.subplots(2, num_viz_steps, figsize=(20, 8))
    
    # Generate and plot images
    with torch.no_grad():
        for t_idx, t in enumerate(timesteps):
            noise_level = torch.ones(1, device=device) * t
            
            # Denoise with teacher
            teacher_denoised = teacher_model(x, noise_level)
            teacher_img = denormalize_image(teacher_denoised[0])
            teacher_img = teacher_img.cpu().permute(1, 2, 0).numpy()
            
            # Denoise with student
            student_denoised = student_model(x_student, noise_level)
            student_img = denormalize_image(student_denoised[0])
            student_img = student_img.cpu().permute(1, 2, 0).numpy()
            
            # Plot
            axes[0, t_idx].imshow(teacher_img)
            axes[0, t_idx].axis('off')
            axes[0, t_idx].set_title(f'Teacher t={t.item()}')
            
            axes[1, t_idx].imshow(student_img)
            axes[1, t_idx].axis('off')
            axes[1, t_idx].set_title(f'Student t={t.item()}')
    
    plt.suptitle('Teacher vs Student Denoising Process Comparison')
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, 'denoising_comparison.png')
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def analyze_denoising_trajectory(teacher_model, student_model, config, device, save_dir=None):
    """
    Analyze and visualize the denoising trajectory differences between teacher and student models.
    
    Args:
        teacher_model: The teacher diffusion model
        student_model: The student diffusion model
        config: Configuration object
        device: Device to run models on
        save_dir: Directory to save analysis results
    """
    teacher_model.eval()
    student_model.eval()
    
    # Generate same initial noise
    image_size = config.teacher_image_size
    num_samples = 5
    trajectories = []
    
    for _ in tqdm(range(num_samples), desc="Analyzing trajectories"):
        x = torch.randn(1, 3, image_size, image_size).to(device)
        x_student = x.clone()
        
        # Record trajectories at more frequent intervals
        timesteps = torch.linspace(0, config.timesteps - 1, 50).long()
        teacher_trajectory = []
        student_trajectory = []
        
        with torch.no_grad():
            for t in timesteps:
                noise_level = torch.ones(1, device=device) * t
                
                # Get predictions
                teacher_pred = teacher_model(x, noise_level)
                student_pred = student_model(x_student, noise_level)
                
                # Calculate difference
                diff = torch.mean((teacher_pred - student_pred) ** 2).item()
                teacher_trajectory.append(teacher_pred.cpu())
                student_trajectory.append(student_pred.cpu())
                
                trajectories.append({
                    'timestep': t.item(),
                    'mse_diff': diff
                })
    
    # Plot trajectory differences
    df = pd.DataFrame(trajectories)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='timestep', y='mse_diff', ci='sd')
    plt.title('MSE Difference Between Teacher and Student Predictions')
    plt.xlabel('Timestep')
    plt.ylabel('MSE Difference')
    
    if save_dir:
        save_path = os.path.join(save_dir, 'trajectory_difference.png')
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save raw data
        df.to_csv(os.path.join(save_dir, 'trajectory_difference.csv'), index=False)
    else:
        plt.show()

def run_denoising_analysis(teacher_model, student_model, config, device):
    """
    Run comprehensive denoising analysis comparing teacher and student models.
    
    Args:
        teacher_model: The teacher diffusion model
        student_model: The student diffusion model
        config: Configuration object
        device: Device to run models on
    """
    save_dir = config.denoising_analysis_dir
    
    # Individual denoising processes
    plot_denoising_grid(
        teacher_model, 
        config, 
        device,
        num_samples=config.denoising_num_samples,
        save_path=os.path.join(save_dir, 'teacher_denoising.png')
    )
    
    plot_denoising_grid(
        student_model, 
        config, 
        device,
        num_samples=config.denoising_num_samples,
        save_path=os.path.join(save_dir, 'student_denoising.png')
    )
    
    # Side-by-side comparison
    compare_denoising_process(
        teacher_model,
        student_model,
        config,
        device,
        save_dir=save_dir
    )
    
    # Trajectory analysis
    analyze_denoising_trajectory(
        teacher_model,
        student_model,
        config,
        device,
        save_dir=save_dir
    ) 