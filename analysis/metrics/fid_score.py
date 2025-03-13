"""
FID score calculation for diffusion models
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import linalg
from torchvision.models import inception_v3
from torch.nn.functional import adaptive_avg_pool2d
import torchvision.transforms as transforms
import sys
from torch.utils.data import DataLoader

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

class InceptionModel:
    """Wrapper for InceptionV3 model to extract features"""
    def __init__(self, device):
        self.device = device
        self.model = inception_v3(pretrained=True, transform_input=False).to(device)
        self.model.eval()
        # Remove the final classification layer
        self.model.fc = torch.nn.Identity()
        # Register hook to get features from the last pooling layer
        self.features = None
        self.model.avgpool.register_forward_hook(self._get_features_hook())
    
    def _get_features_hook(self):
        def hook(module, input, output):
            self.features = output.detach()
        return hook
    
    def get_features(self, images):
        """Extract features from images using InceptionV3"""
        # Ensure images are in the right format for Inception
        # Inception expects images in range [0, 1] with shape [B, 3, 299, 299]
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Process images in batches to avoid memory issues
        batch_size = 32
        n_batches = int(np.ceil(len(images) / batch_size))
        features = []
        
        with torch.no_grad():
            for i in range(n_batches):
                batch = images[i * batch_size:(i + 1) * batch_size]
                # Ensure batch is in range [0, 1]
                batch = (batch + 1) / 2.0  # Convert from [-1, 1] to [0, 1]
                batch = transform(batch)
                _ = self.model(batch)
                features.append(self.features.squeeze(-1).squeeze(-1).cpu().numpy())
        
        return np.concatenate(features, axis=0)

def calculate_fid(features_1, features_2):
    """
    Calculate Fréchet Inception Distance between two sets of features
    
    Args:
        features_1: Features from first set of images
        features_2: Features from second set of images
        
    Returns:
        FID score
    """
    # Check if we have enough samples for a proper FID calculation
    if len(features_1) < 2 or len(features_2) < 2:
        print("  Warning: Not enough samples for a proper FID calculation.")
        print(f"  Number of samples in set 1: {len(features_1)}")
        print(f"  Number of samples in set 2: {len(features_2)}")
        print("  Returning a placeholder FID score of 999.0")
        return 999.0
    
    # Calculate mean and covariance
    mu1, sigma1 = features_1.mean(axis=0), np.cov(features_1, rowvar=False)
    mu2, sigma2 = features_2.mean(axis=0), np.cov(features_2, rowvar=False)
    
    # Calculate FID
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    # Check if covmean has complex parts due to numerical errors
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_and_visualize_fid(teacher_model, student_model, config, output_dir=None, size_factor=None, fixed_samples=None):
    """
    Calculate and visualize FID scores
    
    Args:
        teacher_model: Teacher diffusion model
        student_model: Student diffusion model
        config: Configuration object
        output_dir: Directory to save visualizations
        size_factor: Size factor of the student model for labeling
        fixed_samples: Fixed samples to use for analysis (for consistent comparison)
        
    Returns:
        Dictionary of FID results
    """
    if output_dir is None:
        output_dir = os.path.join(config.analysis_dir, "fid", f"size_{size_factor}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Calculating FID scores for size factor {size_factor}...")
    
    # Get device
    device = next(teacher_model.parameters()).device
    
    # Set models to evaluation mode
    teacher_model.eval()
    student_model.eval()
    
    # Number of samples to generate
    num_samples = config.num_samples if hasattr(config, 'num_samples') else 50
    
    # Generate samples from teacher model
    print("  Generating samples from teacher model...")
    teacher_samples = generate_samples(teacher_model, config, num_samples, device, fixed_samples=fixed_samples)
    
    # Generate samples from student model
    print("  Generating samples from student model...")
    student_samples = generate_samples(student_model, config, num_samples, device, fixed_samples=fixed_samples)
    
    # Initialize Inception model
    print("  Extracting features using InceptionV3...")
    inception_model = InceptionModel(device)
    
    # Extract features
    teacher_features = inception_model.get_features(teacher_samples)
    student_features = inception_model.get_features(student_samples)
    
    # Calculate FID
    print("  Calculating FID score...")
    fid_score = calculate_fid(teacher_features, student_features)
    
    print(f"  FID score for size factor {size_factor}: {fid_score:.4f}")
    
    # Visualize samples - determine how many samples to show
    num_samples_to_show = min(5, len(teacher_samples), len(student_samples))
    
    # Only create visualization if we have samples to show
    if num_samples_to_show > 0:
        fig, axes = plt.subplots(2, num_samples_to_show, figsize=(3 * num_samples_to_show, 6))
        fig.suptitle(f"Sample Comparison (Size Factor: {size_factor})", fontsize=16)
        
        # Handle the case where we only have one sample (axes would be 1D)
        if num_samples_to_show == 1:
            # Show teacher sample
            img = teacher_samples[0].cpu().permute(1, 2, 0).numpy()
            img = (img + 1) / 2.0  # Convert from [-1, 1] to [0, 1]
            axes[0].imshow(img)
            axes[0].set_title("Teacher")
            axes[0].axis("off")
            
            # Show student sample
            img = student_samples[0].cpu().permute(1, 2, 0).numpy()
            img = (img + 1) / 2.0  # Convert from [-1, 1] to [0, 1]
            axes[1].imshow(img)
            axes[1].set_title("Student")
            axes[1].axis("off")
        else:
            # Show teacher samples
            for i in range(num_samples_to_show):
                img = teacher_samples[i].cpu().permute(1, 2, 0).numpy()
                img = (img + 1) / 2.0  # Convert from [-1, 1] to [0, 1]
                axes[0, i].imshow(img)
                axes[0, i].set_title("Teacher")
                axes[0, i].axis("off")
            
            # Show student samples
            for i in range(num_samples_to_show):
                img = student_samples[i].cpu().permute(1, 2, 0).numpy()
                img = (img + 1) / 2.0  # Convert from [-1, 1] to [0, 1]
                axes[1, i].imshow(img)
                axes[1, i].set_title("Student")
                axes[1, i].axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"fid_samples_size_{size_factor}.png"), dpi=300)
        plt.close()
    
    # Save FID score
    with open(os.path.join(output_dir, f"fid_score_size_{size_factor}.txt"), "w") as f:
        f.write(f"FID Score: {fid_score:.4f}\n")
    
    return {"fid_score": fid_score}

def generate_samples(model, config, num_samples, device, fixed_samples=None):
    """
    Generate samples from a diffusion model
    
    Args:
        model: Diffusion model
        config: Configuration object
        num_samples: Number of samples to generate
        device: Device to generate samples on
        fixed_samples: Fixed samples to use as starting points (for consistent comparison)
        
    Returns:
        Tensor of generated samples
    """
    # Set model to evaluation mode
    model.eval()
    
    # Get image size from model if available
    image_size = config.image_size
    if hasattr(model, 'image_size'):
        image_size = model.image_size
    
    # Generate samples
    samples = []
    
    # Use fixed samples if provided
    if fixed_samples is not None:
        print(f"    Using {min(num_samples, len(fixed_samples))} fixed samples as starting points")
        # Use a subset of fixed samples
        starting_samples = fixed_samples[:num_samples]
        
        # Generate samples from fixed starting points
        with torch.no_grad():
            for i in range(len(starting_samples)):
                # Use the fixed sample as the starting point
                x = starting_samples[i:i+1].clone().to(device)
                
                # Resize if needed
                if x.shape[2] != image_size or x.shape[3] != image_size:
                    x = torch.nn.functional.interpolate(
                        x, size=(image_size, image_size), mode='bilinear', align_corners=True
                    )
                
                # Sample from the model
                sample = p_sample_loop(model, x, config)
                samples.append(sample)
    else:
        # Generate samples from random noise
        with torch.no_grad():
            for i in range(num_samples):
                # Generate random noise
                x = torch.randn(1, config.channels, image_size, image_size).to(device)
                
                # Sample from the model
                sample = p_sample_loop(model, x, config)
                samples.append(sample)
    
    # Concatenate samples
    samples = torch.cat(samples, dim=0)
    
    return samples

def p_sample_loop(model, x, config):
    """
    Sample from the model using the p_sample loop
    
    Args:
        model: Diffusion model
        x: Starting noise
        config: Configuration object
        
    Returns:
        Generated sample
    """
    model.eval()
    with torch.no_grad():
        for t in range(config.timesteps - 1, -1, -1):
            t_tensor = torch.tensor([t], device=x.device)
            
            # Predict noise
            noise_pred = model(x, t_tensor)
            
            # Ensure noise_pred has the same shape as x
            if noise_pred.shape != x.shape:
                print(f"  Resizing noise prediction from {noise_pred.shape} to {x.shape}")
                noise_pred = torch.nn.functional.interpolate(
                    noise_pred, 
                    size=(x.shape[2], x.shape[3]),
                    mode='bilinear', 
                    align_corners=True
                )
            
            # Calculate diffusion parameters
            beta_t = config.beta_start + (config.beta_end - config.beta_start) * t / config.timesteps
            alpha_t = 1.0 - beta_t
            
            # Calculate alpha_bar_t (cumulative product of alphas)
            alpha_bar_t = 1.0
            for i in range(t + 1):
                beta_i = config.beta_start + (config.beta_end - config.beta_start) * i / config.timesteps
                alpha_i = 1.0 - beta_i
                alpha_bar_t *= alpha_i
            
            # Convert to tensors
            beta_t = torch.as_tensor(beta_t, device=x.device)
            alpha_t = torch.as_tensor(alpha_t, device=x.device)
            alpha_bar_t = torch.as_tensor(alpha_bar_t, device=x.device)
            
            # Update x
            if t > 0:
                # Sample noise for the next step
                noise = torch.randn_like(x)
                
                # Update x
                x = (x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_t)
                x = x + torch.sqrt(beta_t) * noise
            else:
                # Final step
                x = (x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_t)
    
    return x 