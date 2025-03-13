"""
Evaluation metrics for diffusion model editing
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.linalg import sqrtm
from torchvision.models import inception_v3
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

# Try to import LPIPS, but don't fail if it's not available
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available. Install with 'pip install lpips' for perceptual similarity metrics.")

def compute_lpips(image1, image2, device):
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity) between two images
    
    Args:
        image1: First image tensor (normalized to [0, 1])
        image2: Second image tensor (normalized to [0, 1])
        device: Device to run on
        
    Returns:
        LPIPS distance (lower means more similar)
    """
    if not LPIPS_AVAILABLE:
        print("LPIPS not available. Returning placeholder value.")
        return torch.tensor(0.5)
    
    # Initialize LPIPS model
    loss_fn = lpips.LPIPS(net='alex').to(device)
    
    # Scale images to [-1, 1] for LPIPS
    image1_scaled = 2 * image1 - 1
    image2_scaled = 2 * image2 - 1
    
    # Check if images are too small for LPIPS and resize if necessary
    # LPIPS with AlexNet requires images to be at least 32x32
    min_size = 32
    if image1_scaled.shape[2] < min_size or image1_scaled.shape[3] < min_size or \
       image2_scaled.shape[2] < min_size or image2_scaled.shape[3] < min_size:
        print(f"Resizing images from {image1_scaled.shape[2]}x{image1_scaled.shape[3]} to {min_size}x{min_size} for LPIPS calculation")
        image1_scaled = torch.nn.functional.interpolate(
            image1_scaled, 
            size=(min_size, min_size),
            mode='bilinear', 
            align_corners=True
        )
        image2_scaled = torch.nn.functional.interpolate(
            image2_scaled, 
            size=(min_size, min_size),
            mode='bilinear', 
            align_corners=True
        )
    
    # Compute distance
    with torch.no_grad():
        lpips_distance = loss_fn(image1_scaled, image2_scaled)
    
    return lpips_distance.item()

def compute_fid(real_images, generated_images, device, batch_size=8):
    """
    Compute Fréchet Inception Distance (FID) between two sets of images
    
    Args:
        real_images: List of real image tensors (normalized to [0, 1])
        generated_images: List of generated image tensors (normalized to [0, 1])
        device: Device to run on
        batch_size: Batch size for feature extraction
        
    Returns:
        FID score (lower means more similar)
    """
    # Check if we have enough images
    if len(real_images) < 2 or len(generated_images) < 2:
        print("Warning: Not enough images for FID calculation (need at least 2 images per set)")
        return 999.0  # Return a placeholder value
        
    try:
        # Load Inception model
        inception = inception_v3(pretrained=True, transform_input=False).to(device)
        inception.eval()
        
        # Remove final classification layer
        inception.fc = torch.nn.Identity()
        
        # Define preprocessing
        preprocess = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Check if images are too small and resize if necessary
        # Inception v3 requires images to be at least 75x75
        min_size = 75
        for i in range(len(real_images)):
            if real_images[i].shape[2] < min_size or real_images[i].shape[3] < min_size:
                print(f"Resizing real image from {real_images[i].shape[2]}x{real_images[i].shape[3]} to {min_size}x{min_size} for FID calculation")
                real_images[i] = torch.nn.functional.interpolate(
                    real_images[i], 
                    size=(min_size, min_size),
                    mode='bilinear', 
                    align_corners=True
                )
        
        for i in range(len(generated_images)):
            if generated_images[i].shape[2] < min_size or generated_images[i].shape[3] < min_size:
                print(f"Resizing generated image from {generated_images[i].shape[2]}x{generated_images[i].shape[3]} to {min_size}x{min_size} for FID calculation")
                generated_images[i] = torch.nn.functional.interpolate(
                    generated_images[i], 
                    size=(min_size, min_size),
                    mode='bilinear', 
                    align_corners=True
                )
        
        # Extract features for real images
        real_features = []
        with torch.no_grad():
            for i in range(0, len(real_images), batch_size):
                batch = torch.cat(real_images[i:i+batch_size])
                batch = preprocess(batch)
                features = inception(batch)
                real_features.append(features.cpu().numpy())
        
        real_features = np.concatenate(real_features)
        
        # Extract features for generated images
        gen_features = []
        with torch.no_grad():
            for i in range(0, len(generated_images), batch_size):
                batch = torch.cat(generated_images[i:i+batch_size])
                batch = preprocess(batch)
                features = inception(batch)
                gen_features.append(features.cpu().numpy())
        
        gen_features = np.concatenate(gen_features)
        
        # Calculate mean and covariance
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        
        mu_gen = np.mean(gen_features, axis=0)
        sigma_gen = np.cov(gen_features, rowvar=False)
        
        # Check if covariance matrices are valid
        if np.isnan(sigma_real).any() or np.isnan(sigma_gen).any():
            print("Warning: NaN values in covariance matrices, cannot compute FID")
            return 999.0
            
        if np.isinf(sigma_real).any() or np.isinf(sigma_gen).any():
            print("Warning: Infinite values in covariance matrices, cannot compute FID")
            return 999.0
            
        # Ensure matrices are positive semi-definite
        min_eig_real = np.min(np.linalg.eigvals(sigma_real))
        min_eig_gen = np.min(np.linalg.eigvals(sigma_gen))
        
        if min_eig_real < 0 or min_eig_gen < 0:
            print("Warning: Covariance matrices are not positive semi-definite, adding regularization")
            epsilon = max(0, -min_eig_real, -min_eig_gen) + 1e-6
            sigma_real += np.eye(sigma_real.shape[0]) * epsilon
            sigma_gen += np.eye(sigma_gen.shape[0]) * epsilon
        
        # Calculate FID
        diff = mu_real - mu_gen
        
        # Safely compute the square root of the product
        try:
            covmean = sqrtm(sigma_real.dot(sigma_gen))
            
            # Check for numerical issues
            if np.iscomplexobj(covmean):
                covmean = covmean.real
                
            fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
            return fid
            
        except ValueError as e:
            print(f"Error computing matrix square root: {e}")
            return 999.0
            
    except Exception as e:
        print(f"Error computing FID: {e}")
        return 999.0

def compute_trajectory_divergence(trajectory1, trajectory2):
    """
    Compute divergence between two trajectories in latent space
    
    Args:
        trajectory1: First trajectory as list of (image, timestep) pairs
        trajectory2: Second trajectory as list of (image, timestep) pairs
        
    Returns:
        Dictionary of trajectory divergence metrics
    """
    # Extract images from trajectories
    images1 = [item[0] for item in trajectory1]
    images2 = [item[0] for item in trajectory2]
    
    # Compute distances at each step
    distances = []
    for img1, img2 in zip(images1, images2):
        # Flatten images
        flat1 = img1.flatten()
        flat2 = img2.flatten()
        
        # Compute Euclidean distance
        dist = torch.norm(flat1 - flat2).item()
        distances.append(dist)
    
    # Compute cosine similarities
    similarities = []
    for img1, img2 in zip(images1, images2):
        # Flatten images
        flat1 = img1.flatten().cpu().numpy()
        flat2 = img2.flatten().cpu().numpy()
        
        # Reshape for cosine_similarity function
        flat1 = flat1.reshape(1, -1)
        flat2 = flat2.reshape(1, -1)
        
        # Compute cosine similarity
        sim = cosine_similarity(flat1, flat2)[0, 0]
        similarities.append(sim)
    
    # Compute trajectory length
    length1 = 0
    length2 = 0
    
    for i in range(1, len(images1)):
        length1 += torch.norm(images1[i] - images1[i-1]).item()
    
    for i in range(1, len(images2)):
        length2 += torch.norm(images2[i] - images2[i-1]).item()
    
    # Compute average and max divergence
    avg_distance = np.mean(distances)
    max_distance = np.max(distances)
    avg_similarity = np.mean(similarities)
    min_similarity = np.min(similarities)
    
    return {
        "distances": distances,
        "similarities": similarities,
        "avg_distance": avg_distance,
        "max_distance": max_distance,
        "avg_similarity": avg_similarity,
        "min_similarity": min_similarity,
        "length_ratio": length2 / length1 if length1 > 0 else float('inf')
    }

def visualize_metrics(metrics, output_dir, size_factor=None):
    """
    Visualize evaluation metrics
    
    Args:
        metrics: Dictionary of metrics
        output_dir: Directory to save visualizations
        size_factor: Size factor of the model for labeling
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot LPIPS distances
    if "lpips" in metrics:
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(metrics["lpips"])), metrics["lpips"])
        plt.axhline(y=np.mean(metrics["lpips"]), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(metrics["lpips"]):.4f}')
        plt.title(f"LPIPS Distances (Size Factor: {size_factor})" if size_factor else "LPIPS Distances")
        plt.xlabel("Sample")
        plt.ylabel("LPIPS Distance")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, "lpips_distances.png"), dpi=300, bbox_inches="tight")
        plt.close()
    
    # Plot FID scores
    if "fid" in metrics:
        plt.figure(figsize=(8, 5))
        plt.bar(["FID Score"], [metrics["fid"]])
        plt.title(f"FID Score (Size Factor: {size_factor})" if size_factor else "FID Score")
        plt.ylabel("FID")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, "fid_score.png"), dpi=300, bbox_inches="tight")
        plt.close()
    
    # Plot trajectory divergence
    if "trajectory_divergence" in metrics:
        # Plot distances
        plt.figure(figsize=(12, 6))
        plt.plot(metrics["trajectory_divergence"]["distances"])
        plt.axhline(y=metrics["trajectory_divergence"]["avg_distance"], color='r', linestyle='--',
                   label=f'Mean: {metrics["trajectory_divergence"]["avg_distance"]:.4f}')
        plt.title(f"Trajectory Distances (Size Factor: {size_factor})" if size_factor else "Trajectory Distances")
        plt.xlabel("Step")
        plt.ylabel("Distance")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, "trajectory_distances.png"), dpi=300, bbox_inches="tight")
        plt.close()
        
        # Plot similarities
        plt.figure(figsize=(12, 6))
        plt.plot(metrics["trajectory_divergence"]["similarities"])
        plt.axhline(y=metrics["trajectory_divergence"]["avg_similarity"], color='r', linestyle='--',
                   label=f'Mean: {metrics["trajectory_divergence"]["avg_similarity"]:.4f}')
        plt.title(f"Trajectory Similarities (Size Factor: {size_factor})" if size_factor else "Trajectory Similarities")
        plt.xlabel("Step")
        plt.ylabel("Cosine Similarity")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, "trajectory_similarities.png"), dpi=300, bbox_inches="tight")
        plt.close()
    
    # Create summary text file
    with open(os.path.join(output_dir, "metrics_summary.txt"), "w") as f:
        f.write(f"Evaluation Metrics Summary (Size Factor: {size_factor if size_factor else 'N/A'})\n")
        f.write("=" * 50 + "\n\n")
        
        if "lpips" in metrics:
            f.write(f"LPIPS Mean: {np.mean(metrics['lpips']):.4f}\n")
        
        if "fid" in metrics:
            f.write(f"FID Score: {metrics['fid']:.4f}\n")
        
        if "trajectory_divergence" in metrics:
            f.write("\nTrajectory Divergence:\n")
            f.write(f"  Average Distance: {metrics['trajectory_divergence']['avg_distance']:.4f}\n")
            f.write(f"  Maximum Distance: {metrics['trajectory_divergence']['max_distance']:.4f}\n")
            f.write(f"  Average Similarity: {metrics['trajectory_divergence']['avg_similarity']:.4f}\n")
            f.write(f"  Minimum Similarity: {metrics['trajectory_divergence']['min_similarity']:.4f}\n")
            f.write(f"  Length Ratio: {metrics['trajectory_divergence']['length_ratio']:.4f}\n") 