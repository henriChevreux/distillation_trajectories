"""
Fréchet Inception Distance (FID) calculation utility.
Implementation based on the CTM paper's evaluation metrics.
"""

import torch
import torch.nn as nn
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.models import inception_v3
import numpy as np
from scipy import linalg
from tqdm import tqdm

class InceptionStatistics(nn.Module):
    """Wrapper for Inception model to get statistics for FID calculation"""
    def __init__(self):
        super().__init__()
        self.model = inception_v3(pretrained=True, transform_input=False)
        self.model.fc = nn.Identity()  # Remove classification layer
        self.model.eval()
        
    def forward(self, x):
        # Ensure input is in [-1, 1]
        x = (x + 1) / 2  # Convert from [-1, 1] to [0, 1]
        x = x.clamp(0, 1)
        
        # Resize if needed (inception expects 299x299)
        if x.shape[-1] != 299:
            x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Get features before the classification layer
        with torch.no_grad():
            features = self.model(x)
        return features

def calculate_statistics(features):
    """Calculate mean and covariance statistics of features"""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Fréchet distance between two sets of statistics"""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, "Mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Covariance matrices have different dimensions"
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = f"FID calculation produced singular product; adding {eps} to diagonal of covariance matrices"
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    
    tr_covmean = np.trace(covmean)
    
    return (diff.dot(diff) + 
            np.trace(sigma1) + 
            np.trace(sigma2) - 
            2 * tr_covmean)

def get_features(model, dataloader, device, max_samples=None):
    """Extract features from data using inception model"""
    features = []
    samples_seen = 0
    
    with torch.no_grad():
        for batch, _ in tqdm(dataloader, desc="Extracting features"):
            if max_samples is not None and samples_seen >= max_samples:
                break
                
            batch = batch.to(device)
            batch_features = model(batch).cpu().numpy()
            features.append(batch_features)
            
            samples_seen += batch.shape[0]
            
    return np.concatenate(features, axis=0)

def calculate_fid(generated_samples, real_data_loader, device, max_samples=None):
    """
    Calculate FID score between generated samples and real data
    
    Args:
        generated_samples: Tensor of generated samples (N, C, H, W)
        real_data_loader: DataLoader for real data
        device: Device to use for inception model
        max_samples: Maximum number of samples to use (use None for all)
    
    Returns:
        fid_score: Calculated FID score
    """
    # Initialize inception model
    inception = InceptionStatistics().to(device)
    
    # Get features for generated samples
    generated_features = []
    for i in range(0, len(generated_samples), 32):  # Process in batches
        batch = generated_samples[i:i+32].to(device)
        features = inception(batch).cpu().numpy()
        generated_features.append(features)
    generated_features = np.concatenate(generated_features, axis=0)
    
    # Get features for real samples
    real_features = get_features(inception, real_data_loader, device, max_samples)
    
    if max_samples is not None:
        generated_features = generated_features[:max_samples]
        real_features = real_features[:max_samples]
    
    # Calculate statistics
    mu_gen, sigma_gen = calculate_statistics(generated_features)
    mu_real, sigma_real = calculate_statistics(real_features)
    
    # Calculate FID
    fid_score = calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
    
    return fid_score 