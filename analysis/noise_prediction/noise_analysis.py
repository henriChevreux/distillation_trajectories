"""
Noise prediction analysis for diffusion models
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def generate_noise_samples(batch_size, channels, image_size, device):
    """
    Generate random noise samples
    
    Args:
        batch_size: Number of samples to generate
        channels: Number of channels in the images
        image_size: Size of the images
        device: Device to generate samples on
        
    Returns:
        Tensor of noise samples
    """
    return torch.randn(batch_size, channels, image_size, image_size).to(device)

def predict_noise(model, noisy_images, timesteps, device):
    """
    Predict noise using the model
    
    Args:
        model: Diffusion model
        noisy_images: Noisy images
        timesteps: Timesteps for diffusion process
        device: Device to run prediction on
        
    Returns:
        Predicted noise
    """
    model.eval()
    with torch.no_grad():
        return model(noisy_images, timesteps)

def calculate_noise_metrics(teacher_noise, student_noise):
    """
    Calculate metrics comparing teacher and student noise predictions
    
    Args:
        teacher_noise: Noise predicted by teacher model
        student_noise: Noise predicted by student model
        
    Returns:
        Dictionary of metrics
    """
    # Resize student noise to match teacher noise if sizes differ
    if teacher_noise.shape != student_noise.shape:
        print(f"  Resizing student noise from {student_noise.shape} to {teacher_noise.shape}")
        student_noise = torch.nn.functional.interpolate(
            student_noise, 
            size=(teacher_noise.shape[2], teacher_noise.shape[3]),
            mode='bilinear', 
            align_corners=True
        )
    
    # Calculate MSE
    mse = torch.mean((teacher_noise - student_noise) ** 2).item()
    
    # Calculate MAE
    mae = torch.mean(torch.abs(teacher_noise - student_noise)).item()
    
    # Calculate cosine similarity
    teacher_flat = teacher_noise.view(teacher_noise.size(0), -1)
    student_flat = student_noise.view(student_noise.size(0), -1)
    
    # Normalize
    teacher_norm = torch.nn.functional.normalize(teacher_flat, p=2, dim=1)
    student_norm = torch.nn.functional.normalize(student_flat, p=2, dim=1)
    
    # Calculate cosine similarity
    cosine_sim = torch.mean(torch.sum(teacher_norm * student_norm, dim=1)).item()
    
    return {
        "mse": mse,
        "mae": mae,
        "cosine_similarity": cosine_sim
    }

def visualize_noise_predictions(original_images, noisy_images, teacher_noise, student_noise, true_noise, timesteps, output_dir, size_factor):
    """
    Visualize noise predictions
    
    Args:
        original_images: Original clean images
        noisy_images: Noisy images
        teacher_noise: Noise predicted by teacher model
        student_noise: Noise predicted by student model
        true_noise: True noise added to images
        timesteps: Timesteps for diffusion process
        output_dir: Directory to save visualizations
        size_factor: Size factor of the student model for labeling
    """
    # Select a subset of images to visualize
    n_images = min(5, original_images.size(0))
    
    # Create figure
    fig, axes = plt.subplots(5, n_images, figsize=(n_images * 3, 15))
    fig.suptitle(f'Noise Prediction Comparison (Size Factor: {size_factor})', fontsize=16)
    
    # Row titles
    row_titles = ['Original', 'Noisy', 'True Noise', 'Teacher Pred', 'Student Pred']
    
    # Plot images
    for i in range(n_images):
        # Original image
        img = original_images[i].cpu().permute(1, 2, 0).numpy()
        img = (img + 1) / 2.0  # Convert from [-1, 1] to [0, 1]
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Image {i+1}')
        
        # Noisy image
        img = noisy_images[i].cpu().permute(1, 2, 0).numpy()
        img = (img + 1) / 2.0  # Convert from [-1, 1] to [0, 1]
        axes[1, i].imshow(img)
        axes[1, i].set_title(f't={timesteps[i].item()}')
        
        # True noise
        img = true_noise[i].cpu().permute(1, 2, 0).numpy()
        # Normalize noise for visualization
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axes[2, i].imshow(img)
        
        # Teacher predicted noise
        img = teacher_noise[i].cpu().permute(1, 2, 0).numpy()
        # Normalize noise for visualization
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axes[3, i].imshow(img)
        
        # Student predicted noise
        img = student_noise[i].cpu().permute(1, 2, 0).numpy()
        # Normalize noise for visualization
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        axes[4, i].imshow(img)
    
    # Add row labels
    for i, title in enumerate(row_titles):
        axes[i, 0].set_ylabel(title, fontsize=14)
    
    # Turn off axis for all subplots
    for ax in axes.flatten():
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'noise_prediction_comparison_size_{size_factor}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_noise_metrics_by_timestep(metrics_by_timestep, output_dir, size_factor):
    """
    Plot noise prediction metrics by timestep
    
    Args:
        metrics_by_timestep: Dictionary of metrics by timestep
        output_dir: Directory to save visualizations
        size_factor: Size factor of the student model for labeling
    """
    # Extract timesteps and metrics
    timesteps = sorted(metrics_by_timestep.keys())
    mse_values = [metrics_by_timestep[t]['mse'] for t in timesteps]
    mae_values = [metrics_by_timestep[t]['mae'] for t in timesteps]
    cosine_values = [metrics_by_timestep[t]['cosine_similarity'] for t in timesteps]
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    fig.suptitle(f'Noise Prediction Metrics by Timestep (Size Factor: {size_factor})', fontsize=16)
    
    # Plot MSE
    axes[0].plot(timesteps, mse_values, 'o-', color='blue', linewidth=2)
    axes[0].set_ylabel('MSE')
    axes[0].set_title('Mean Squared Error')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot MAE
    axes[1].plot(timesteps, mae_values, 'o-', color='green', linewidth=2)
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Mean Absolute Error')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # Plot Cosine Similarity
    axes[2].plot(timesteps, cosine_values, 'o-', color='red', linewidth=2)
    axes[2].set_ylabel('Cosine Similarity')
    axes[2].set_title('Cosine Similarity (higher is better)')
    axes[2].set_xlabel('Timestep')
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'noise_metrics_by_timestep_size_{size_factor}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_noise_prediction(teacher_model, student_model, config, output_dir=None, size_factor=None, fixed_samples=None):
    """
    Analyze noise prediction accuracy of models
    
    Args:
        teacher_model: Teacher diffusion model
        student_model: Student diffusion model
        config: Configuration object
        output_dir: Directory to save visualizations
        size_factor: Size factor of the student model for labeling
        fixed_samples: Fixed samples to use for analysis (for consistent comparison)
        
    Returns:
        Dictionary of analysis results
    """
    if output_dir is None:
        output_dir = os.path.join(config.analysis_dir, "noise_prediction", f"size_{size_factor}")
    
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Analyzing noise prediction for size factor {size_factor}...")
    
    # Get device
    device = next(teacher_model.parameters()).device
    
    # Set models to evaluation mode
    teacher_model.eval()
    student_model.eval()
    
    # Get test dataset or use fixed samples
    if fixed_samples is not None:
        print(f"Using {len(fixed_samples)} fixed samples for consistent comparison")
        images = fixed_samples.to(device)
    else:
        test_dataset = config.get_test_dataset()
        test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)
        images, _ = next(iter(test_loader))
        images = images.to(device)
    
    # Number of timesteps to analyze
    n_timesteps = 10
    timesteps_to_analyze = torch.linspace(0, config.timesteps - 1, n_timesteps, dtype=torch.long).to(device)
    
    # Store metrics by timestep
    metrics_by_timestep = {}
    
    # Analyze each timestep
    for t in timesteps_to_analyze:
        # Create batch of same timestep
        timesteps = torch.full((images.size(0),), t, device=device, dtype=torch.long)
        
        # Calculate diffusion parameters for this timestep
        beta_t = config.beta_start + (config.beta_end - config.beta_start) * t / config.timesteps
        alpha_t = 1.0 - beta_t
        
        # Calculate alpha_bar_t (cumulative product of alphas)
        alpha_bar_t = 1.0
        for i in range(t.item() + 1):
            beta_i = config.beta_start + (config.beta_end - config.beta_start) * i / config.timesteps
            alpha_i = 1.0 - beta_i
            alpha_bar_t *= alpha_i
        
        # Convert to tensors
        beta_t = torch.as_tensor(beta_t, device=device)
        alpha_t = torch.as_tensor(alpha_t, device=device)
        alpha_bar_t = torch.as_tensor(alpha_bar_t, device=device)
        
        # Generate noise
        noise = torch.randn_like(images)
        
        # Add noise to images
        noisy_images = torch.sqrt(alpha_bar_t) * images + torch.sqrt(1 - alpha_bar_t) * noise
        
        # Predict noise using teacher model
        teacher_noise = predict_noise(teacher_model, noisy_images, timesteps, device)
        
        # Predict noise using student model
        student_noise = predict_noise(student_model, noisy_images, timesteps, device)
        
        # Calculate metrics
        metrics = calculate_noise_metrics(teacher_noise, student_noise)
        
        # Store metrics
        metrics_by_timestep[t.item()] = metrics
        
        # Visualize first timestep
        if t.item() == timesteps_to_analyze[0].item():
            visualize_noise_predictions(
                images, noisy_images, teacher_noise, student_noise, noise, 
                timesteps, output_dir, size_factor
            )
    
    # Plot metrics by timestep
    plot_noise_metrics_by_timestep(metrics_by_timestep, output_dir, size_factor)
    
    # Calculate average metrics across all timesteps
    avg_mse = np.mean([metrics['mse'] for metrics in metrics_by_timestep.values()])
    avg_mae = np.mean([metrics['mae'] for metrics in metrics_by_timestep.values()])
    avg_cosine = np.mean([metrics['cosine_similarity'] for metrics in metrics_by_timestep.values()])
    
    # Save metrics
    results = {
        "avg_mse": avg_mse,
        "avg_mae": avg_mae,
        "avg_cosine_similarity": avg_cosine,
        "metrics_by_timestep": metrics_by_timestep
    }
    
    # Save metrics to file
    with open(os.path.join(output_dir, f"noise_metrics_size_{size_factor}.txt"), "w") as f:
        f.write(f"Average MSE: {avg_mse:.6f}\n")
        f.write(f"Average MAE: {avg_mae:.6f}\n")
        f.write(f"Average Cosine Similarity: {avg_cosine:.6f}\n\n")
        f.write("Metrics by Timestep:\n")
        for t, metrics in sorted(metrics_by_timestep.items()):
            f.write(f"  Timestep {t}:\n")
            f.write(f"    MSE: {metrics['mse']:.6f}\n")
            f.write(f"    MAE: {metrics['mae']:.6f}\n")
            f.write(f"    Cosine Similarity: {metrics['cosine_similarity']:.6f}\n")
    
    print(f"  Average MSE: {avg_mse:.6f}")
    print(f"  Average MAE: {avg_mae:.6f}")
    print(f"  Average Cosine Similarity: {avg_cosine:.6f}")
    
    return results 