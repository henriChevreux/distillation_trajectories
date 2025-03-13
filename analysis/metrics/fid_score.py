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

def calculate_and_visualize_fid(teacher_model, student_model, config, output_dir=None, size_factor=None):
    """
    Calculate and visualize FID scores
    
    Args:
        teacher_model: Teacher diffusion model
        student_model: Student diffusion model
        config: Configuration object
        output_dir: Directory to save visualizations
        size_factor: Size factor of the student model for labeling
        
    Returns:
        Dictionary of FID results
    """
    if output_dir is None:
        output_dir = os.path.join(config.analysis_dir, "fid_scores")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Calculating FID scores for size factor {size_factor}...")
    
    # Get device
    device = next(teacher_model.parameters()).device
    
    # Get test dataset
    test_dataset = config.get_test_dataset()
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)
    test_images, _ = next(iter(test_loader))
    test_images = test_images.to(device)
    
    # Set models to evaluation mode
    teacher_model.eval()
    student_model.eval()
    
    # Generate noisy images at different timesteps
    timesteps = [10, 20, 30, 40]  # Sample a few timesteps
    teacher_samples = []
    student_samples = []
    
    with torch.no_grad():
        for t in timesteps:
            # Create timestep tensor
            t_tensor = torch.full((len(test_images),), t, device=device, dtype=torch.long)
            
            # Add noise to images
            noise = torch.randn_like(test_images)
            alpha_cumprod = torch.tensor(1.0 - (config.beta_start + (config.beta_end - config.beta_start) * t / config.timesteps), device=device)
            noisy_images = torch.sqrt(alpha_cumprod) * test_images + torch.sqrt(1.0 - alpha_cumprod) * noise
            
            # Predict noise and denoise
            teacher_pred = teacher_model(noisy_images, t_tensor)
            student_pred = student_model(noisy_images, t_tensor)
            
            # Denoise images
            teacher_denoised = (noisy_images - torch.sqrt(1.0 - alpha_cumprod) * teacher_pred) / torch.sqrt(alpha_cumprod)
            student_denoised = (noisy_images - torch.sqrt(1.0 - alpha_cumprod) * student_pred) / torch.sqrt(alpha_cumprod)
            
            # Add to samples
            teacher_samples.append(teacher_denoised)
            student_samples.append(student_denoised)
    
    # Concatenate samples
    teacher_samples = torch.cat(teacher_samples, dim=0)
    student_samples = torch.cat(student_samples, dim=0)
    
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
    
    # Visualize samples
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle(f"Sample Comparison (Size Factor: {size_factor})", fontsize=16)
    
    # Show teacher samples
    for i in range(5):
        img = teacher_samples[i].cpu().permute(1, 2, 0).numpy()
        img = (img + 1) / 2.0  # Convert from [-1, 1] to [0, 1]
        axes[0, i].imshow(img)
        axes[0, i].set_title("Teacher")
        axes[0, i].axis("off")
    
    # Show student samples
    for i in range(5):
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