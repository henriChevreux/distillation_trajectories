import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import wasserstein_distance
from scipy.interpolate import interp1d
from matplotlib.colors import LinearSegmentedColormap
from scipy.linalg import sqrtm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.models as models
from mpl_toolkits.mplot3d import Axes3D
from diffusion_training import Config, SimpleUNet, StudentUNet, get_diffusion_params, p_sample_loop, q_sample
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

# Try to import umap, but don't fail if not available
try:
    import umap
except ImportError:
    print("UMAP not installed. Skipping UMAP visualization.")
    umap = None


# Determine device
#if torch.backends.mps.is_available():
#    device = torch.device("mps")
#    print("Using MPS acceleration")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA acceleration")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Number of threads for parallel processing
NUM_THREADS = min(multiprocessing.cpu_count(), 8)

@lru_cache(maxsize=1)
def get_inception_model():
    """
    Load and cache the Inception model for FID calculation.
    
    Returns:
        The Inception model in evaluation mode with fc layer replaced by Identity
    """
    inception_model = models.inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    
    # Set model to feature extraction mode
    inception_model.fc = nn.Identity()  # Replace the final fully connected layer with Identity
    inception_model.aux_logits = False
    
    return inception_model

def calculate_fid(real_images, generated_images):
    """
    Calculate Fréchet Inception Distance (FID) between real and generated images.
    
    Args:
        real_images: Tensor of real images, shape (N, C, H, W)
        generated_images: Tensor of generated images, shape (N, C, H, W)
        
    Returns:
        fid_score: The FID score (lower is better)
    """
    # Get cached Inception model
    inception_model = get_inception_model()
    
    # Function to extract features
    def extract_features(images):
        # Resize images to Inception input size if needed
        if images.shape[2] != 299 or images.shape[3] != 299:
            images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Ensure images are in the correct range [0, 1]
        images = (images + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        # Extract features
        features = []
        batch_size = 32  # Process in batches to avoid memory issues
        
        with torch.no_grad():
            for i in range(0, images.shape[0], batch_size):
                batch = images[i:i+batch_size].to(device)
                feat = inception_model(batch)
                features.append(feat.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    # Extract features for real and generated images
    real_features = extract_features(real_images)
    gen_features = extract_features(generated_images)
    
    # Calculate mean and covariance
    mu_real = np.mean(real_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    
    mu_gen = np.mean(gen_features, axis=0)
    sigma_gen = np.cov(gen_features, rowvar=False)
    
    # Calculate FID
    diff = mu_real - mu_gen
    
    # Product might be almost singular
    covmean, _ = sqrtm(sigma_real.dot(sigma_gen), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid_score = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    
    return fid_score

@lru_cache(maxsize=2)  # Cache for different datasets
def get_real_images(config, num_samples=100):
    """Get a batch of real images from the dataset"""
    if config.dataset == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((config.image_size, config.image_size)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        dataset = datasets.CIFAR10(
            root='data',
            train=False,  # Use test set for evaluation
            download=True,
            transform=transform
        )
        
    elif config.dataset == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((config.image_size, config.image_size)),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        dataset = datasets.MNIST(
            root='data',
            train=False,  # Use test set for evaluation
            download=True,
            transform=transform
        )
    else:
        raise ValueError(f"Dataset {config.dataset} not supported for FID calculation")
    
    # Create a DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=num_samples,
        shuffle=True
    )
    
    # Get a batch of images
    images, _ = next(iter(dataloader))
    return images

def generate_trajectories(teacher_model, student_model, config, num_samples=10):
    """
    Generate trajectories for both teacher and student models
    
    Args:
        teacher_model: The teacher diffusion model
        student_model: The student diffusion model
        config: Configuration object
        num_samples: Number of samples to generate
    
    Returns:
        teacher_trajectories: List of teacher trajectories
        student_trajectories: List of student trajectories
    """
    # Create directory for saving trajectories
    os.makedirs(config.trajectory_dir, exist_ok=True)
    
    # Get diffusion parameters
    teacher_params = get_diffusion_params(config.teacher_steps)
    student_params = get_diffusion_params(config.student_steps)
    
    # Generate samples with trajectory tracking
    teacher_trajectories = []
    student_trajectories = []
    
    # Prepare shape tuple for reuse
    shape = (1, config.channels, config.image_size, config.image_size)
    
    def generate_pair(seed):
        """Generate a pair of trajectories with the same seed"""
        # Use the same random seed for both models to start from the same noise
        torch.manual_seed(seed)
        
        # Generate teacher trajectory
        _, teacher_traj = p_sample_loop(
            teacher_model,
            shape=shape,
            timesteps=config.teacher_steps,
            diffusion_params=teacher_params,
            track_trajectory=True
        )
        
        # Reset seed to use the same initial noise
        torch.manual_seed(seed)
        
        # Generate student trajectory
        _, student_traj = p_sample_loop(
            student_model,
            shape=shape,
            timesteps=config.student_steps,
            diffusion_params=student_params,
            track_trajectory=True
        )
        
        return teacher_traj, student_traj
    
    print(f"Generating {num_samples} trajectories...")
    for i in tqdm(range(num_samples)):
        t_traj, s_traj = generate_pair(i)
        teacher_trajectories.append(t_traj)
        student_trajectories.append(s_traj)
    
    return teacher_trajectories, student_trajectories

def compute_trajectory_metrics(teacher_trajectories, student_trajectories, config):
    """
    Compute metrics to compare teacher and student trajectories
    
    Args:
        teacher_trajectories: List of teacher trajectories
        student_trajectories: List of student trajectories
        config: Configuration object
    
    Returns:
        metrics: Dictionary of metrics
    """
    metrics = {
        'wasserstein_distances': [],
        'endpoint_distances': [],
        'teacher_path_lengths': [],
        'student_path_lengths': [],
        'teacher_efficiency': [],
        'student_efficiency': []
    }
    
    def compute_metrics_for_pair(pair_idx):
        """Compute metrics for a single pair of trajectories"""
        teacher_traj = teacher_trajectories[pair_idx]
        student_traj = student_trajectories[pair_idx]
        
        # Flatten images for easier distance calculation
        teacher_flat = [t[0].reshape(-1).numpy() for t in teacher_traj]
        student_flat = [s[0].reshape(-1).numpy() for s in student_traj]
        
        # Compute Wasserstein distance between trajectories
        # We need to interpolate the student trajectory to match the teacher's timesteps
        if len(teacher_flat) != len(student_flat):
            # Create interpolation function for student trajectory
            student_timesteps = np.linspace(0, 1, len(student_flat))
            teacher_timesteps = np.linspace(0, 1, len(teacher_flat))
            
            # Vectorized interpolation
            student_values = np.array([s for s in student_flat])
            student_interp = np.zeros((len(teacher_timesteps), student_values.shape[1]))
            
            # Create interpolation function once
            for dim in range(0, student_values.shape[1], 1000):  # Process in chunks to avoid memory issues
                end_dim = min(dim + 1000, student_values.shape[1])
                chunk_values = student_values[:, dim:end_dim]
                interp_func = interp1d(student_timesteps, chunk_values, axis=0, kind='linear')
                student_interp[:, dim:end_dim] = interp_func(teacher_timesteps)
            
            # Compute Wasserstein distance
            w_dist = wasserstein_distance(np.concatenate(teacher_flat), student_interp.flatten())
        else:
            # If same number of timesteps, no interpolation needed
            w_dist = wasserstein_distance(np.concatenate(teacher_flat), np.concatenate(student_flat))
        
        # Compute endpoint distance (L2 norm between final images)
        endpoint_dist = np.linalg.norm(teacher_flat[-1] - student_flat[-1])
        
        # Compute path length using vectorized operations
        teacher_diffs = np.array(teacher_flat[1:]) - np.array(teacher_flat[:-1])
        student_diffs = np.array(student_flat[1:]) - np.array(student_flat[:-1])
        
        teacher_path_length = np.sum(np.sqrt(np.sum(teacher_diffs**2, axis=1)))
        student_path_length = np.sum(np.sqrt(np.sum(student_diffs**2, axis=1)))
        
        # Compute path efficiency (endpoint distance / path length)
        # This measures how direct the path is (higher is more efficient)
        start_end_dist_teacher = np.linalg.norm(teacher_flat[-1] - teacher_flat[0])
        start_end_dist_student = np.linalg.norm(student_flat[-1] - student_flat[0])
        
        teacher_efficiency = start_end_dist_teacher / teacher_path_length if teacher_path_length > 0 else 0
        student_efficiency = start_end_dist_student / student_path_length if student_path_length > 0 else 0
        
        return {
            'wasserstein': w_dist,
            'endpoint': endpoint_dist,
            'teacher_path': teacher_path_length,
            'student_path': student_path_length,
            'teacher_eff': teacher_efficiency,
            'student_eff': student_efficiency
        }
    
    # Process trajectory pairs in parallel
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        results = list(executor.map(compute_metrics_for_pair, range(len(teacher_trajectories))))
    
    # Collect results
    for result in results:
        metrics['wasserstein_distances'].append(result['wasserstein'])
        metrics['endpoint_distances'].append(result['endpoint'])
        metrics['teacher_path_lengths'].append(result['teacher_path'])
        metrics['student_path_lengths'].append(result['student_path'])
        metrics['teacher_efficiency'].append(result['teacher_eff'])
        metrics['student_efficiency'].append(result['student_eff'])
    
    return metrics

def visualize_metrics(metrics, config, suffix=""):
    """
    Visualize trajectory metrics
    
    Args:
        metrics: Dictionary of metrics
        config: Configuration object
        suffix: Suffix to add to filenames
    
    Returns:
        summary: Dictionary of summary statistics
    """
    # Create directory for saving visualizations
    metrics_dir = os.path.join(config.analysis_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Add suffix to filenames if provided
    suffix = suffix if suffix else ""
    
    # Compute summary statistics using numpy vectorized operations
    summary = {
        'avg_wasserstein': np.mean(metrics['wasserstein_distances']),
        'avg_endpoint_distance': np.mean(metrics['endpoint_distances']),
        'avg_teacher_path_length': np.mean(metrics['teacher_path_lengths']),
        'avg_student_path_length': np.mean(metrics['student_path_lengths']),
        'avg_teacher_efficiency': np.mean(metrics['teacher_efficiency']),
        'avg_student_efficiency': np.mean(metrics['student_efficiency'])
    }
    
    # Save summary to file
    with open(os.path.join(config.analysis_dir, 'metrics', f'summary{suffix}.txt'), 'w') as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot Wasserstein distances
    plt.subplot(2, 3, 1)
    plt.hist(metrics['wasserstein_distances'], bins=10, alpha=0.7)
    plt.axvline(summary['avg_wasserstein'], color='r', linestyle='--', 
                label=f'Mean: {summary["avg_wasserstein"]:.4f}')
    plt.title('Wasserstein Distances')
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.legend()
    
    # Plot endpoint distances
    plt.subplot(2, 3, 2)
    plt.hist(metrics['endpoint_distances'], bins=10, alpha=0.7)
    plt.axvline(summary['avg_endpoint_distance'], color='r', linestyle='--', 
                label=f'Mean: {summary["avg_endpoint_distance"]:.4f}')
    plt.title('Endpoint Distances')
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.legend()
    
    # Plot path lengths
    plt.subplot(2, 3, 3)
    plt.hist(metrics['teacher_path_lengths'], bins=10, alpha=0.7, label='Teacher')
    plt.hist(metrics['student_path_lengths'], bins=10, alpha=0.7, label='Student')
    plt.axvline(summary['avg_teacher_path_length'], color='b', linestyle='--', 
                label=f'Teacher Mean: {summary["avg_teacher_path_length"]:.4f}')
    plt.axvline(summary['avg_student_path_length'], color='orange', linestyle='--', 
                label=f'Student Mean: {summary["avg_student_path_length"]:.4f}')
    plt.title('Path Lengths')
    plt.xlabel('Length')
    plt.ylabel('Count')
    plt.legend()
    
    # Plot path efficiency
    plt.subplot(2, 3, 4)
    plt.hist(metrics['teacher_efficiency'], bins=10, alpha=0.7, label='Teacher')
    plt.hist(metrics['student_efficiency'], bins=10, alpha=0.7, label='Student')
    plt.axvline(summary['avg_teacher_efficiency'], color='b', linestyle='--', 
                label=f'Teacher Mean: {summary["avg_teacher_efficiency"]:.4f}')
    plt.axvline(summary['avg_student_efficiency'], color='orange', linestyle='--', 
                label=f'Student Mean: {summary["avg_student_efficiency"]:.4f}')
    plt.title('Path Efficiency')
    plt.xlabel('Efficiency')
    plt.ylabel('Count')
    plt.legend()
    
    # Plot path length vs endpoint distance
    plt.subplot(2, 3, 5)
    plt.scatter(metrics['teacher_path_lengths'], metrics['endpoint_distances'], 
                alpha=0.7, label='Teacher')
    plt.scatter(metrics['student_path_lengths'], metrics['endpoint_distances'], 
                alpha=0.7, label='Student')
    plt.title('Path Length vs Endpoint Distance')
    plt.xlabel('Path Length')
    plt.ylabel('Endpoint Distance')
    plt.legend()
    
    # Plot efficiency vs Wasserstein distance
    plt.subplot(2, 3, 6)
    plt.scatter(metrics['teacher_efficiency'], metrics['wasserstein_distances'], 
                alpha=0.7, label='Teacher')
    plt.scatter(metrics['student_efficiency'], metrics['wasserstein_distances'], 
                alpha=0.7, label='Student')
    plt.title('Efficiency vs Wasserstein Distance')
    plt.xlabel('Efficiency')
    plt.ylabel('Wasserstein Distance')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.analysis_dir, 'metrics', f'metrics_visualization{suffix}.png'))
    plt.close()
    
    return summary

def dimensionality_reduction_analysis(teacher_trajectories, student_trajectories, config, suffix=""):
    """
    Perform dimensionality reduction analysis on trajectories
    
    Args:
        teacher_trajectories: List of teacher trajectories
        student_trajectories: List of student trajectories
        config: Configuration object
        suffix: Suffix to add to filenames
    """
    # Create directory for saving visualizations
    os.makedirs(os.path.join(config.analysis_dir, 'dimensionality'), exist_ok=True)
    
    # Add suffix to filenames if provided
    suffix = suffix if suffix else ""
    
    # Take the first few trajectories for visualization
    num_vis = min(5, len(teacher_trajectories))
    
    # Prepare trajectory information
    teacher_points_per_traj = len(teacher_trajectories[0])
    student_points_per_traj = len(student_trajectories[0])
    
    # Pre-allocate lists with known sizes for better memory efficiency
    all_teacher_flat = []
    all_student_flat = []
    
    # Process trajectories in parallel
    def process_trajectory_pair(idx):
        teacher_traj = teacher_trajectories[idx]
        student_traj = student_trajectories[idx]
        
        teacher_flat = [t[0].reshape(-1).numpy() for t in teacher_traj]
        student_flat = [s[0].reshape(-1).numpy() for s in student_traj]
        
        return teacher_flat, student_flat
    
    # Process trajectories
    for i in range(num_vis):
        t_flat, s_flat = process_trajectory_pair(i)
        all_teacher_flat.extend(t_flat)
        all_student_flat.extend(s_flat)
    
    # Combine all points for dimensionality reduction
    all_points = np.vstack(all_teacher_flat + all_student_flat)
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_points)
    
    # Split back into teacher and student trajectories
    teacher_pca = pca_result[:len(all_teacher_flat)]
    student_pca = pca_result[len(all_teacher_flat):]
    
    # Reshape back into trajectories - use numpy operations for efficiency
    teacher_pca_traj = []
    student_pca_traj = []
    
    for i in range(num_vis):
        start_idx_teacher = i * teacher_points_per_traj
        end_idx_teacher = start_idx_teacher + teacher_points_per_traj
        
        start_idx_student = i * student_points_per_traj
        end_idx_student = start_idx_student + student_points_per_traj
        
        teacher_pca_traj.append(teacher_pca[start_idx_teacher:end_idx_teacher])
        student_pca_traj.append(student_pca[start_idx_student:end_idx_student])
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot PCA trajectories
    for i in range(num_vis):
        # Plot teacher trajectory
        plt.plot(teacher_pca_traj[i][:, 0], teacher_pca_traj[i][:, 1], 'b-', alpha=0.7)
        plt.scatter(teacher_pca_traj[i][0, 0], teacher_pca_traj[i][0, 1], c='blue', marker='o', s=100)
        plt.scatter(teacher_pca_traj[i][-1, 0], teacher_pca_traj[i][-1, 1], c='darkblue', marker='x', s=100)
        
        # Plot student trajectory
        plt.plot(student_pca_traj[i][:, 0], student_pca_traj[i][:, 1], 'r-', alpha=0.7)
        plt.scatter(student_pca_traj[i][0, 0], student_pca_traj[i][0, 1], c='orange', marker='o', s=100)
        plt.scatter(student_pca_traj[i][-1, 0], student_pca_traj[i][-1, 1], c='darkred', marker='x', s=100)
    
    # Add legend
    plt.scatter([], [], c='blue', marker='o', s=100, label='Teacher start')
    plt.scatter([], [], c='darkblue', marker='x', s=100, label='Teacher end')
    plt.scatter([], [], c='orange', marker='o', s=100, label='Student start')
    plt.scatter([], [], c='darkred', marker='x', s=100, label='Student end')
    plt.plot([], [], 'b-', alpha=0.7, label='Teacher trajectory')
    plt.plot([], [], 'r-', alpha=0.7, label='Student trajectory')
    
    plt.title('PCA of Diffusion Trajectories')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(config.analysis_dir, 'dimensionality', f'pca_trajectories{suffix}.png'))
    plt.close()
    
    # Apply t-SNE if available
    try:
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(all_points)
        
        # Split back into teacher and student trajectories
        teacher_tsne = tsne_result[:len(all_teacher_flat)]
        student_tsne = tsne_result[len(all_teacher_flat):]
        
        # Reshape back into trajectories
        teacher_tsne_traj = []
        student_tsne_traj = []
        
        for i in range(num_vis):
            start_idx_teacher = i * teacher_points_per_traj
            end_idx_teacher = start_idx_teacher + teacher_points_per_traj
            
            start_idx_student = i * student_points_per_traj
            end_idx_student = start_idx_student + student_points_per_traj
            
            teacher_tsne_traj.append(teacher_tsne[start_idx_teacher:end_idx_teacher])
            student_tsne_traj.append(student_tsne[start_idx_student:end_idx_student])
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot t-SNE trajectories
        for i in range(num_vis):
            # Plot teacher trajectory
            plt.plot(teacher_tsne_traj[i][:, 0], teacher_tsne_traj[i][:, 1], 'b-', alpha=0.7)
            plt.scatter(teacher_tsne_traj[i][0, 0], teacher_tsne_traj[i][0, 1], c='blue', marker='o', s=100)
            plt.scatter(teacher_tsne_traj[i][-1, 0], teacher_tsne_traj[i][-1, 1], c='darkblue', marker='x', s=100)
            
            # Plot student trajectory
            plt.plot(student_tsne_traj[i][:, 0], student_tsne_traj[i][:, 1], 'r-', alpha=0.7)
            plt.scatter(student_tsne_traj[i][0, 0], student_tsne_traj[i][0, 1], c='orange', marker='o', s=100)
            plt.scatter(student_tsne_traj[i][-1, 0], student_tsne_traj[i][-1, 1], c='darkred', marker='x', s=100)
        
        # Add legend
        plt.scatter([], [], c='blue', marker='o', s=100, label='Teacher start')
        plt.scatter([], [], c='darkblue', marker='x', s=100, label='Teacher end')
        plt.scatter([], [], c='orange', marker='o', s=100, label='Student start')
        plt.scatter([], [], c='darkred', marker='x', s=100, label='Student end')
        plt.plot([], [], 'b-', alpha=0.7, label='Teacher trajectory')
        plt.plot([], [], 'r-', alpha=0.7, label='Student trajectory')
        
        plt.title('t-SNE of Diffusion Trajectories')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(config.analysis_dir, 'dimensionality', f'tsne_trajectories{suffix}.png'))
        plt.close()
    except Exception as e:
        print(f"Error applying t-SNE: {e}")
    
    # Apply UMAP if available
    if umap is not None:
        try:
            umap_reducer = umap.UMAP(random_state=42)
            umap_result = umap_reducer.fit_transform(all_points)
            
            # Split back into teacher and student trajectories
            teacher_umap = umap_result[:len(all_teacher_flat)]
            student_umap = umap_result[len(all_teacher_flat):]
            
            # Reshape back into trajectories
            teacher_umap_traj = []
            student_umap_traj = []
            
            for i in range(num_vis):
                start_idx_teacher = i * teacher_points_per_traj
                end_idx_teacher = start_idx_teacher + teacher_points_per_traj
                
                start_idx_student = i * student_points_per_traj
                end_idx_student = start_idx_student + student_points_per_traj
                
                teacher_umap_traj.append(teacher_umap[start_idx_teacher:end_idx_teacher])
                student_umap_traj.append(student_umap[start_idx_student:end_idx_student])
            
            # Create visualization
            plt.figure(figsize=(15, 10))
            
            # Plot UMAP trajectories
            for i in range(num_vis):
                # Plot teacher trajectory
                plt.plot(teacher_umap_traj[i][:, 0], teacher_umap_traj[i][:, 1], 'b-', alpha=0.7)
                plt.scatter(teacher_umap_traj[i][0, 0], teacher_umap_traj[i][0, 1], c='blue', marker='o', s=100)
                plt.scatter(teacher_umap_traj[i][-1, 0], teacher_umap_traj[i][-1, 1], c='darkblue', marker='x', s=100)
                
                # Plot student trajectory
                plt.plot(student_umap_traj[i][:, 0], student_umap_traj[i][:, 1], 'r-', alpha=0.7)
                plt.scatter(student_umap_traj[i][0, 0], student_umap_traj[i][0, 1], c='orange', marker='o', s=100)
                plt.scatter(student_umap_traj[i][-1, 0], student_umap_traj[i][-1, 1], c='darkred', marker='x', s=100)
            
            # Add legend
            plt.scatter([], [], c='blue', marker='o', s=100, label='Teacher start')
            plt.scatter([], [], c='darkblue', marker='x', s=100, label='Teacher end')
            plt.scatter([], [], c='orange', marker='o', s=100, label='Student start')
            plt.scatter([], [], c='darkred', marker='x', s=100, label='Student end')
            plt.plot([], [], 'b-', alpha=0.7, label='Teacher trajectory')
            plt.plot([], [], 'r-', alpha=0.7, label='Student trajectory')
            
            plt.title('UMAP of Diffusion Trajectories')
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.savefig(os.path.join(config.analysis_dir, 'dimensionality', f'umap_trajectories{suffix}.png'))
            plt.close()
        except Exception as e:
            print(f"Error applying UMAP: {e}")

def analyze_noise_prediction(teacher_model, student_model, config, suffix=""):
    """
    Analyze noise prediction patterns of teacher and student models
    
    Args:
        teacher_model: The teacher diffusion model
        student_model: The student diffusion model
        config: Configuration object
        suffix: Suffix to add to filenames
    
    Returns:
        metrics: Dictionary of noise prediction metrics
    """
    # Create directory for saving visualizations
    os.makedirs(os.path.join(config.analysis_dir, 'noise_prediction'), exist_ok=True)
    
    # Add suffix to filenames if provided
    suffix = suffix if suffix else ""
    
    # Get diffusion parameters - cache these for reuse
    teacher_params = get_diffusion_params(config.teacher_steps)
    student_params = get_diffusion_params(config.student_steps)
    
    # Generate a batch of random images
    torch.manual_seed(42)  # For reproducibility
    batch_size = 10
    x_start = torch.randn(batch_size, config.channels, config.image_size, config.image_size).to(device)
    
    # Sample timesteps
    timesteps = [0, config.teacher_steps // 4, config.teacher_steps // 2, 3 * config.teacher_steps // 4, config.teacher_steps - 1]
    
    # Convert teacher timesteps to student timesteps - do this once
    student_timesteps = [int(t * config.student_steps / config.teacher_steps) for t in timesteps]
    
    # Initialize metrics
    metrics = {
        'teacher_noise_mse': [],
        'student_noise_mse': [],
        'noise_prediction_similarity': []
    }
    
    # Analyze noise prediction at different timesteps
    for t_teacher, t_student in zip(timesteps, student_timesteps):
        # Create tensor timesteps
        t_teacher_tensor = torch.full((batch_size,), t_teacher, device=device, dtype=torch.long)
        t_student_tensor = torch.full((batch_size,), t_student, device=device, dtype=torch.long)
        
        # Add noise to images - fix the duplicate comment
        x_noisy, true_noise = q_sample(x_start, t_teacher_tensor, teacher_params)
        
        # Get noise predictions from both models in a single batch for efficiency
        with torch.no_grad():
            teacher_pred = teacher_model(x_noisy, t_teacher_tensor)
            student_pred = student_model(x_noisy, t_student_tensor)
        
        # Compute metrics in a vectorized way
        teacher_mse = F.mse_loss(teacher_pred, true_noise).item()
        student_mse = F.mse_loss(student_pred, true_noise).item()
        
        # Compute similarity between teacher and student predictions
        similarity = F.cosine_similarity(teacher_pred.flatten(1), student_pred.flatten(1), dim=1).mean().item()
        
        # Store metrics
        metrics['teacher_noise_mse'].append(teacher_mse)
        metrics['student_noise_mse'].append(student_mse)
        metrics['noise_prediction_similarity'].append(similarity)
    
    # Create visualizations
    plt.figure(figsize=(15, 5))
    
    # Plot MSE
    plt.subplot(1, 3, 1)
    plt.plot(timesteps, metrics['teacher_noise_mse'], 'b-o', label='Teacher')
    plt.plot(timesteps, metrics['student_noise_mse'], 'r-o', label='Student')
    plt.title('Noise Prediction MSE')
    plt.xlabel('Timestep')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot similarity
    plt.subplot(1, 3, 2)
    plt.plot(timesteps, metrics['noise_prediction_similarity'], 'g-o')
    plt.title('Teacher-Student Prediction Similarity')
    plt.xlabel('Timestep')
    plt.ylabel('Cosine Similarity')
    plt.grid(True, alpha=0.3)
    
    # Plot MSE ratio
    plt.subplot(1, 3, 3)
    mse_ratio = [s / t if t > 0 else 1.0 for s, t in zip(metrics['student_noise_mse'], metrics['teacher_noise_mse'])]
    plt.plot(timesteps, mse_ratio, 'purple', marker='o')
    plt.axhline(y=1.0, color='gray', linestyle='--')
    plt.title('Student/Teacher MSE Ratio')
    plt.xlabel('Timestep')
    plt.ylabel('Ratio')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.analysis_dir, 'noise_prediction', f'noise_prediction{suffix}.png'))
    plt.close()
    
    # Save metrics to file
    with open(os.path.join(config.analysis_dir, 'noise_prediction', f'noise_metrics{suffix}.txt'), 'w') as f:
        f.write(f"Timesteps: {timesteps}\n")
        f.write(f"Teacher MSE: {metrics['teacher_noise_mse']}\n")
        f.write(f"Student MSE: {metrics['student_noise_mse']}\n")
        f.write(f"Similarity: {metrics['noise_prediction_similarity']}\n")
    
    return metrics

def analyze_attention_maps(teacher_model, student_model, config, suffix=""):
    """
    Analyze attention maps of teacher and student models
    
    Args:
        teacher_model: The teacher diffusion model
        student_model: The student diffusion model
        config: Configuration object
        suffix: Suffix to add to filenames
    
    Returns:
        metrics: Dictionary of attention map metrics
    """
    # Create directory for saving visualizations
    os.makedirs(os.path.join(config.analysis_dir, 'attention'), exist_ok=True)
    
    # Add suffix to filenames if provided
    suffix = suffix if suffix else ""
    
    # Since we don't have direct access to attention maps in this model architecture,
    # we'll use a proxy by analyzing the spatial patterns in the model's activations
    
    # Generate a batch of random images
    torch.manual_seed(42)  # For reproducibility
    batch_size = 4
    x_start = torch.randn(batch_size, config.channels, config.image_size, config.image_size).to(device)
    
    # Sample timesteps
    t_teacher = config.teacher_steps // 2
    t_student = config.student_steps // 2
    
    # Create tensor timesteps
    t_teacher_tensor = torch.full((batch_size,), t_teacher, device=device, dtype=torch.long)
    t_student_tensor = torch.full((batch_size,), t_student, device=device, dtype=torch.long)
    
    # Add noise to images
    x_noisy, _ = q_sample(x_start, t_teacher_tensor, get_diffusion_params(config.teacher_steps))
    
    # Get predictions from both models
    with torch.no_grad():
        teacher_pred = teacher_model(x_noisy, t_teacher_tensor)
        student_pred = student_model(x_noisy, t_student_tensor)
    
    # Compute spatial attention proxy by taking the absolute values of predictions
    teacher_attention = torch.abs(teacher_pred).mean(dim=1)  # Average over channels
    student_attention = torch.abs(student_pred).mean(dim=1)  # Average over channels
    
    # Normalize for visualization
    teacher_attention = (teacher_attention - teacher_attention.min()) / (teacher_attention.max() - teacher_attention.min() + 1e-8)
    student_attention = (student_attention - student_attention.min()) / (student_attention.max() - student_attention.min() + 1e-8)
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    for i in range(batch_size):
        # Plot original noisy image
        plt.subplot(batch_size, 3, i*3 + 1)
        img = x_noisy[i].cpu().permute(1, 2, 0)
        img = (img + 1) / 2  # Convert from [-1, 1] to [0, 1]
        plt.imshow(img)
        plt.title(f'Noisy Image {i+1}')
        plt.axis('off')
        
        # Plot teacher attention
        plt.subplot(batch_size, 3, i*3 + 2)
        plt.imshow(teacher_attention[i].cpu(), cmap='hot')
        plt.title(f'Teacher Attention {i+1}')
        plt.axis('off')
        
        # Plot student attention
        plt.subplot(batch_size, 3, i*3 + 3)
        plt.imshow(student_attention[i].cpu(), cmap='hot')
        plt.title(f'Student Attention {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.analysis_dir, 'attention', f'attention_maps{suffix}.png'))
    plt.close()
    
    # Compute metrics
    metrics = {}
    
    # Compute correlation between teacher and student attention maps
    correlations = []
    for i in range(batch_size):
        teacher_att_flat = teacher_attention[i].cpu().flatten().numpy()
        student_att_flat = student_attention[i].cpu().flatten().numpy()
        correlation = np.corrcoef(teacher_att_flat, student_att_flat)[0, 1]
        correlations.append(correlation)
    
    metrics['attention_correlations'] = correlations
    metrics['mean_correlation'] = np.mean(correlations)
    
    # Save metrics to file
    with open(os.path.join(config.analysis_dir, 'attention', f'attention_metrics{suffix}.txt'), 'w') as f:
        f.write(f"Attention map correlations: {correlations}\n")
        f.write(f"Mean correlation: {metrics['mean_correlation']}\n")
    
    return metrics

def generate_3d_model_size_visualization(all_teacher_trajectories, all_student_trajectories, size_factors, config):
    """
    Create a 3D visualization that incorporates model size as a dimension
    
    Args:
        all_teacher_trajectories: Dictionary mapping size factors to teacher trajectories
        all_student_trajectories: Dictionary mapping size factors to student trajectories
        size_factors: List of size factors to include
        config: Configuration object
    """
    os.makedirs(os.path.join(config.analysis_dir, '3d_model_size'), exist_ok=True)
    
    # We'll create several visualizations:
    # 1. Trajectory endpoints in 3D space (PCA) with model size as color
    # 2. Trajectory path length vs model size vs endpoint distance
    # 3. Wasserstein distance vs model size vs efficiency
    
    # First, let's extract endpoints and compute metrics for each size factor
    endpoints = {}
    metrics = {}
    
    for size_factor in size_factors:
        teacher_trajectories = all_teacher_trajectories[size_factor]
        student_trajectories = all_student_trajectories[size_factor]
        
        # Compute metrics for this size factor
        metrics[size_factor] = compute_trajectory_metrics(teacher_trajectories, student_trajectories, config)
        
        # Extract endpoints (last point of each trajectory)
        teacher_endpoints = [traj[-1][0].reshape(-1).numpy() for traj in teacher_trajectories]
        student_endpoints = [traj[-1][0].reshape(-1).numpy() for traj in student_trajectories]
        
        endpoints[size_factor] = {
            'teacher': teacher_endpoints,
            'student': student_endpoints
        }
    
    # 1. Visualize trajectory endpoints in 3D space with model size as color
    # Combine all endpoints for PCA
    all_endpoints = []
    for size_factor in size_factors:
        all_endpoints.extend(endpoints[size_factor]['teacher'])
        all_endpoints.extend(endpoints[size_factor]['student'])
    
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(np.vstack(all_endpoints))
    
    # Split back into teacher and student endpoints for each size factor
    pca_endpoints = {}
    idx = 0
    for size_factor in size_factors:
        n_teacher = len(endpoints[size_factor]['teacher'])
        n_student = len(endpoints[size_factor]['student'])
        
        pca_endpoints[size_factor] = {
            'teacher': pca_result[idx:idx+n_teacher],
            'student': pca_result[idx+n_teacher:idx+n_teacher+n_student]
        }
        idx += n_teacher + n_student
    
    # Create 3D visualization of endpoints with model size as color
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a colormap for size factors
    norm = plt.Normalize(min(size_factors), max(size_factors))
    cmap = plt.cm.viridis
    
    # Plot teacher endpoints (all same color since they're from the same model)
    for size_factor in size_factors:
        teacher_points = pca_endpoints[size_factor]['teacher']
        ax.scatter(
            teacher_points[:, 0], 
            teacher_points[:, 1], 
            teacher_points[:, 2],
            c='blue', 
            marker='o', 
            s=50, 
            alpha=0.7,
            label='Teacher' if size_factor == size_factors[0] else ""
        )
    
    # Plot student endpoints with color based on size factor
    for size_factor in size_factors:
        student_points = pca_endpoints[size_factor]['student']
        scatter = ax.scatter(
            student_points[:, 0], 
            student_points[:, 1], 
            student_points[:, 2],
            c=[cmap(norm(size_factor))] * len(student_points), 
            marker='^', 
            s=80, 
            alpha=0.8,
            label=f'Student (size {size_factor})'
        )
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label('Student Model Size Factor', fontsize=12)
    
    # Add labels and legend
    ax.set_xlabel('PCA Component 1', fontsize=12)
    ax.set_ylabel('PCA Component 2', fontsize=12)
    ax.set_zlabel('PCA Component 3', fontsize=12)
    ax.set_title('3D Visualization of Trajectory Endpoints by Model Size', fontsize=14)
    
    # Create a custom legend with one entry for teacher and one for each student size
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=10, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.analysis_dir, '3d_model_size', 'endpoints_by_size_3d.png'), dpi=300)
    plt.close()
    
    # 2. Create 3D plot of path length vs model size vs endpoint distance
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract metrics for the plot
    x = []  # size factors
    y = []  # avg path lengths
    z = []  # avg endpoint distances
    c = []  # colors (same as size factor for visual consistency)
    
    for size_factor in size_factors:
        x.append(size_factor)
        y.append(np.mean(metrics[size_factor]['student_path_lengths']))
        z.append(np.mean(metrics[size_factor]['endpoint_distances']))
        c.append(cmap(norm(size_factor)))
    
    # Create the 3D scatter plot
    scatter = ax.scatter(x, y, z, c=c, s=100, alpha=0.8)
    
    # Connect points with a line to show the trend
    ax.plot(x, y, z, 'k--', alpha=0.5)
    
    # Add labels
    ax.set_xlabel('Model Size Factor', fontsize=12)
    ax.set_ylabel('Average Path Length', fontsize=12)
    ax.set_zlabel('Average Endpoint Distance', fontsize=12)
    ax.set_title('Model Size vs Path Length vs Endpoint Distance', fontsize=14)
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label('Student Model Size Factor', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.analysis_dir, '3d_model_size', 'size_pathlength_distance_3d.png'), dpi=300)
    plt.close()
    
    # 3. Create 3D plot of Wasserstein distance vs model size vs efficiency
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract metrics for the plot
    x = []  # size factors
    y = []  # avg wasserstein distances
    z = []  # avg efficiency
    c = []  # colors (same as size factor for visual consistency)
    
    for size_factor in size_factors:
        x.append(size_factor)
        y.append(np.mean(metrics[size_factor]['wasserstein_distances']))
        z.append(np.mean(metrics[size_factor]['student_efficiency']))
        c.append(cmap(norm(size_factor)))
    
    # Create the 3D scatter plot
    scatter = ax.scatter(x, y, z, c=c, s=100, alpha=0.8)
    
    # Connect points with a line to show the trend
    ax.plot(x, y, z, 'k--', alpha=0.5)
    
    # Add labels
    ax.set_xlabel('Model Size Factor', fontsize=12)
    ax.set_ylabel('Average Wasserstein Distance', fontsize=12)
    ax.set_zlabel('Average Path Efficiency', fontsize=12)
    ax.set_title('Model Size vs Wasserstein Distance vs Efficiency', fontsize=14)
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label('Student Model Size Factor', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.analysis_dir, '3d_model_size', 'size_wasserstein_efficiency_3d.png'), dpi=300)
    plt.close()

def generate_latent_space_visualization(teacher_trajectories, student_trajectories, config, suffix=""):
    """Create a 3D visualization of trajectories over time"""
    os.makedirs(os.path.join(config.analysis_dir, '3d_visualization'), exist_ok=True)
    
    # Add suffix to filenames if provided
    suffix = suffix if suffix else ""
    
    # Take the first sample
    teacher_traj = teacher_trajectories[0]
    student_traj = student_trajectories[0]
    
    # Flatten trajectories
    teacher_flat = [t[0].reshape(-1).numpy() for t in teacher_traj]
    student_flat = [s[0].reshape(-1).numpy() for s in student_traj]
    
    # Combine for PCA
    combined = np.vstack([teacher_flat, student_flat])
    
    # Apply PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(combined)
    
    # Split back
    teacher_pca = pca_result[:len(teacher_flat)]
    student_pca = pca_result[len(teacher_flat):]
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot teacher trajectory
    ax.plot(teacher_pca[:, 0], teacher_pca[:, 1], teacher_pca[:, 2], 'b-', linewidth=2, label='Teacher')
    ax.scatter(teacher_pca[0, 0], teacher_pca[0, 1], teacher_pca[0, 2], c='blue', marker='o', s=100, label='Teacher start')
    ax.scatter(teacher_pca[-1, 0], teacher_pca[-1, 1], teacher_pca[-1, 2], c='darkblue', marker='x', s=100, label='Teacher end')
    
    # Plot student trajectory
    ax.plot(student_pca[:, 0], student_pca[:, 1], student_pca[:, 2], 'r-', linewidth=2, label='Student')
    ax.scatter(student_pca[0, 0], student_pca[0, 1], student_pca[0, 2], c='orange', marker='o', s=100, label='Student start')
    ax.scatter(student_pca[-1, 0], student_pca[-1, 1], student_pca[-1, 2], c='darkred', marker='x', s=100, label='Student end')
    
    # Add labels and legend
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    ax.set_title(f'3D Latent Space Visualization of Diffusion Trajectories', fontsize=14)
    ax.legend()
    
    # Save figure
    plt.savefig(os.path.join(config.analysis_dir, '3d_visualization', f'latent_space_3d{suffix}.png'), dpi=300)
    plt.close()

def main(config=None, teacher_model_path=None, student_model_paths=None, num_samples=50, 
         skip_metrics=False, skip_dimensionality=False, skip_noise=False, 
         skip_attention=False, skip_3d=False, skip_fid=False):
    """
    Main function for running the diffusion model analysis
    
    Args:
        config: Configuration object, will create a new one if None
        teacher_model_path: Path to the teacher model file, overrides default
        student_model_paths: Dictionary mapping size factors to model paths, or single path
        num_samples: Number of samples to generate for analysis
        skip_metrics: Skip trajectory metrics computation
        skip_dimensionality: Skip dimensionality reduction analysis
        skip_noise: Skip noise prediction analysis
        skip_attention: Skip attention map analysis
        skip_3d: Skip 3D visualization
        skip_fid: Skip FID calculation
    """
    # Initialize config if not provided
    if config is None:
        config = Config()
        config.create_directories()
    
    # Get diffusion parameters - cache these for reuse
    teacher_params = get_diffusion_params(config.teacher_steps)
    student_params = get_diffusion_params(config.student_steps)
    
    # Load teacher model
    print("Loading teacher model...")
    teacher_model = SimpleUNet(config).to(device)
    
    # Use provided path or default path for teacher
    if teacher_model_path is None:
        teacher_model_path = os.path.join(config.models_dir, 'model_epoch_1.pt')
    
    # Load the teacher model
    if os.path.exists(teacher_model_path):
        teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=device))
        print(f"Loaded teacher model from {teacher_model_path}")
    else:
        print(f"ERROR: Teacher model not found at {teacher_model_path}. Please run training first.")
        return
    
    # Set teacher model to evaluation mode
    teacher_model.eval()
    
    # Handle student models - either a single model or multiple models with different size factors
    student_models = {}
    
    # Function to determine architecture type based on size factor
    def get_architecture_type(size_factor):
        if float(size_factor) < 0.1:
            return 'tiny'     # Use the smallest architecture for very small models
        elif float(size_factor) < 0.3:
            return 'small'    # Use small architecture for small models
        elif float(size_factor) < 0.7:
            return 'medium'   # Use medium architecture for medium models
        else:
            return 'full'     # Use full architecture for large models
    
    if isinstance(student_model_paths, dict):
        # Multiple student models with different size factors
        for size_factor, path in student_model_paths.items():
            print(f"Loading student model with size factor {size_factor}...")
            
            # Determine architecture type based on size factor
            architecture_type = get_architecture_type(size_factor)
            
            print(f"Using architecture type: {architecture_type} for size factor {size_factor}")
            student_model = StudentUNet(config, size_factor=float(size_factor), architecture_type=architecture_type).to(device)
            
            if os.path.exists(path):
                student_model.load_state_dict(torch.load(path, map_location=device))
                print(f"Loaded student model from {path}")
                student_model.eval()
                student_models[float(size_factor)] = student_model
            else:
                print(f"WARNING: Student model not found at {path}. Skipping this size factor.")
    else:
        # Single student model (backward compatibility)
        if student_model_paths is None:
            # Try to find student models with different size factors
            size_factors = config.student_size_factors if hasattr(config, 'student_size_factors') else [0.25, 0.5, 0.75, 1.0]
            for size_factor in size_factors:
                path = os.path.join(config.models_dir, f'student_model_size_{size_factor}_epoch_1.pt')
                if os.path.exists(path):
                    print(f"Found student model with size factor {size_factor}...")
                    architecture_type = get_architecture_type(size_factor)
                    student_model = StudentUNet(config, size_factor=float(size_factor), architecture_type=architecture_type).to(device)
                    student_model.load_state_dict(torch.load(path, map_location=device))
                    student_model.eval()
                    student_models[float(size_factor)] = student_model
                else:
                    print(f"No student model found for size factor {size_factor}")
            
            # If no student models found, try the old naming convention
            if not student_models:
                path = os.path.join(config.models_dir, 'student_model_epoch_1.pt')
                if os.path.exists(path):
                    print("Loading student model with default size...")
                    student_model = SimpleUNet(config).to(device)
                    student_model.load_state_dict(torch.load(path, map_location=device))
                    student_model.eval()
                    student_models[1.0] = student_model
                else:
                    print(f"ERROR: No student models found. Please run training with distillation first.")
                    return
        else:
            # Single specified student model path
            path = student_model_paths
            if os.path.exists(path):
                print("Loading student model...")
                # Try to extract size factor from filename
                size_factor = 1.0  # Default
                if "size_" in path:
                    try:
                        size_str = path.split("size_")[1].split("_")[0]
                        size_factor = float(size_str)
                    except:
                        pass
                
                architecture_type = get_architecture_type(size_factor)
                student_model = StudentUNet(config, size_factor=size_factor, architecture_type=architecture_type).to(device)
                student_model.load_state_dict(torch.load(path, map_location=device))
                student_model.eval()
                student_models[size_factor] = student_model
            else:
                print(f"ERROR: Student model not found at {path}. Please run training with distillation first.")
                return
    
    # Print summary of loaded models
    print(f"\nLoaded {len(student_models)} student models with size factors: {list(student_models.keys())}")
    
    # Analyze each student model
    all_metrics = {}
    all_fid_results = {}
    all_teacher_trajectories = {}
    all_student_trajectories = {}
    
    for size_factor, student_model in student_models.items():
        print(f"\n{'='*80}")
        print(f"Analyzing student model with size factor {size_factor}")
        print(f"{'='*80}")
        
        # 1. Generate multiple trajectories
        print("Generating trajectories...")
        teacher_trajectories, student_trajectories = generate_trajectories(
            teacher_model, student_model, config, num_samples=num_samples
        )
        
        # Store trajectories for 3D model size visualization
        all_teacher_trajectories[size_factor] = teacher_trajectories
        all_student_trajectories[size_factor] = student_trajectories
        
        # Run only the selected analysis modules
        if not skip_metrics:
            # 2. Compute trajectory metrics
            print("Computing trajectory metrics...")
            metrics = compute_trajectory_metrics(teacher_trajectories, student_trajectories, config)
            
            # 3. Visualize metrics
            print("Visualizing metrics...")
            summary = visualize_metrics(metrics, config, suffix=f"_size_{size_factor}")
            print("Metrics summary:", summary)
            
            # Store metrics for comparative analysis
            all_metrics[size_factor] = summary
        else:
            print("Skipping trajectory metrics analysis.")
            
        # Calculate FID scores
        if not skip_fid:
            print("Calculating FID scores...")
            os.makedirs(os.path.join(config.analysis_dir, 'fid'), exist_ok=True)
            
            # Get real images from the dataset
            real_images = get_real_images(config, num_samples=100)
            
            # Generate samples from both models
            print("Generating samples from teacher model...")
            teacher_params = get_diffusion_params(config.teacher_steps)
            teacher_samples = []
            for i in tqdm(range(10)):  # Generate 10 batches of 10 samples each
                with torch.no_grad():
                    batch = p_sample_loop(
                        teacher_model,
                        shape=(10, config.channels, config.image_size, config.image_size),
                        timesteps=config.teacher_steps,
                        diffusion_params=teacher_params
                    )
                    teacher_samples.append(batch)
            teacher_samples = torch.cat(teacher_samples, dim=0)
            
            print(f"Generating samples from student model (size factor {size_factor})...")
            student_params = get_diffusion_params(config.student_steps)
            student_samples = []
            for i in tqdm(range(10)):  # Generate 10 batches of 10 samples each
                with torch.no_grad():
                    batch = p_sample_loop(
                        student_model,
                        shape=(10, config.channels, config.image_size, config.image_size),
                        timesteps=config.student_steps,
                        diffusion_params=student_params
                    )
                    student_samples.append(batch)
            student_samples = torch.cat(student_samples, dim=0)
            
            # Calculate FID scores
            print("Calculating FID between real images and teacher samples...")
            teacher_fid = calculate_fid(real_images, teacher_samples)
            
            print(f"Calculating FID between real images and student samples (size {size_factor})...")
            student_fid = calculate_fid(real_images, student_samples)
            
            print(f"Calculating FID between teacher and student samples (size {size_factor})...")
            teacher_student_fid = calculate_fid(teacher_samples, student_samples)
            
            # Save FID scores
            fid_results = {
                'teacher_fid': teacher_fid,
                'student_fid': student_fid,
                'teacher_student_fid': teacher_student_fid
            }
            
            all_fid_results[size_factor] = fid_results
            
            with open(os.path.join(config.analysis_dir, 'fid', f'fid_scores_size_{size_factor}.txt'), 'w') as f:
                for key, value in fid_results.items():
                    f.write(f"{key}: {value}\n")
            
            print(f"FID Scores for size factor {size_factor}:")
            print(f"  Teacher FID (vs real): {teacher_fid:.4f}")
            print(f"  Student FID (vs real): {student_fid:.4f}")
            print(f"  Teacher vs Student FID: {teacher_student_fid:.4f}")
            
            # Visualize some samples
            plt.figure(figsize=(15, 10))
            
            # Plot real images
            for i in range(5):
                plt.subplot(3, 5, i+1)
                img = real_images[i].cpu()
                img = (img + 1) / 2  # Convert from [-1, 1] to [0, 1]
                img = img.permute(1, 2, 0)  # Change from CxHxW to HxWxC
                plt.imshow(img)
                plt.axis('off')
                if i == 2:
                    plt.title('Real Images', fontsize=14)
            
            # Plot teacher samples
            for i in range(5):
                plt.subplot(3, 5, i+6)
                img = teacher_samples[i].cpu()
                img = (img + 1) / 2  # Convert from [-1, 1] to [0, 1]
                img = img.permute(1, 2, 0)  # Change from CxHxW to HxWxC
                plt.imshow(img)
                plt.axis('off')
                if i == 2:
                    plt.title('Teacher Samples', fontsize=14)
            
            # Plot student samples
            for i in range(5):
                plt.subplot(3, 5, i+11)
                img = student_samples[i].cpu()
                img = (img + 1) / 2  # Convert from [-1, 1] to [0, 1]
                img = img.permute(1, 2, 0)  # Change from CxHxW to HxWxC
                plt.imshow(img)
                plt.axis('off')
                if i == 2:
                    plt.title(f'Student Samples (size {size_factor})', fontsize=14)
            
            plt.tight_layout()
            plt.savefig(os.path.join(config.analysis_dir, 'fid', f'sample_comparison_size_{size_factor}.png'))
            plt.close()
        else:
            print("Skipping FID calculation.")
        
        if not skip_dimensionality:
            # 4. Dimensionality reduction analysis
            print("Performing dimensionality reduction analysis...")
            dimensionality_reduction_analysis(teacher_trajectories, student_trajectories, config, suffix=f"_size_{size_factor}")
        else:
            print("Skipping dimensionality reduction analysis.")
        
        if not skip_noise:
            # 5. Noise prediction analysis
            print("Analyzing noise prediction patterns...")
            noise_metrics = analyze_noise_prediction(teacher_model, student_model, config, suffix=f"_size_{size_factor}")
        else:
            print("Skipping noise prediction analysis.")
        
        if not skip_attention:
            # 6. Attention map analysis
            print("Analyzing attention maps...")
            attention_metrics = analyze_attention_maps(teacher_model, student_model, config, suffix=f"_size_{size_factor}")
        else:
            print("Skipping attention map analysis.")
        
        if not skip_3d:
            # 7. Generate 3D latent space visualization
            print("Creating 3D latent space visualization...")
            generate_latent_space_visualization(teacher_trajectories, student_trajectories, config, suffix=f"_size_{size_factor}")
        else:
            print("Skipping 3D latent space visualization.")
    
    # Create comparative visualizations across different student model sizes
    if len(student_models) > 1 and not skip_metrics:
        print("\nCreating comparative visualizations across student model sizes...")
        create_model_size_comparisons(all_metrics, all_fid_results, config)
        
        # Create 3D visualization that incorporates model size as a dimension
        print("Creating 3D model size visualization...")
        generate_3d_model_size_visualization(all_teacher_trajectories, all_student_trajectories, 
                                            sorted(student_models.keys()), config)
    
    print("\nAnalysis complete. Results saved in the analysis directory.")


def create_model_size_comparisons(all_metrics, all_fid_results, config):
    """
    Create comprehensive comparative visualizations and analysis across different student model sizes.
    This is the main focus of the analysis - understanding how model size affects performance.
    
    Args:
        all_metrics: Dictionary mapping size factors to metrics
        all_fid_results: Dictionary mapping size factors to FID scores
        config: Configuration object
    """
    # Create directory for model size comparison results
    comparison_dir = os.path.join(config.analysis_dir, 'model_size_comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Sort size factors
    size_factors = sorted(all_metrics.keys())
    
    # Extract metrics for comparison
    wasserstein_distances = [all_metrics[sf]['avg_wasserstein'] for sf in size_factors]
    path_lengths = [all_metrics[sf]['avg_student_path_length'] for sf in size_factors]
    endpoint_distances = [all_metrics[sf]['avg_endpoint_distance'] for sf in size_factors]
    path_efficiencies = [all_metrics[sf]['avg_student_efficiency'] for sf in size_factors]
    
    # Extract FID scores if available
    if all_fid_results:
        student_fids = [all_fid_results[sf]['student_fid'] for sf in size_factors]
        teacher_student_fids = [all_fid_results[sf]['teacher_student_fid'] for sf in size_factors]
    
    # Calculate model parameter counts (estimated based on size factor)
    # This is an approximation - actual parameter counts would be better if available
    base_params = 1.0  # Normalized to teacher model size
    param_counts = [sf**2 * base_params for sf in size_factors]  # Quadratic relationship with size factor
    
    # 1. Plot metrics vs model size with improved visualization
    plt.figure(figsize=(16, 12))
    
    plt.subplot(2, 2, 1)
    plt.plot(size_factors, wasserstein_distances, 'o-', linewidth=2, markersize=8)
    plt.title('Average Wasserstein Distance vs Model Size', fontsize=14)
    plt.xlabel('Student Model Size Factor', fontsize=12)
    plt.ylabel('Avg Wasserstein Distance', fontsize=12)
    plt.grid(True, alpha=0.3)
    # Add trend line
    z = np.polyfit(size_factors, wasserstein_distances, 2)
    p = np.poly1d(z)
    x_trend = np.linspace(min(size_factors), max(size_factors), 100)
    plt.plot(x_trend, p(x_trend), 'r--', alpha=0.7)
    
    plt.subplot(2, 2, 2)
    plt.plot(size_factors, path_lengths, 'o-', linewidth=2, markersize=8)
    plt.title('Path Length vs Model Size', fontsize=14)
    plt.xlabel('Student Model Size Factor', fontsize=12)
    plt.ylabel('Avg Path Length', fontsize=12)
    plt.grid(True, alpha=0.3)
    # Add trend line
    z = np.polyfit(size_factors, path_lengths, 2)
    p = np.poly1d(z)
    plt.plot(x_trend, p(x_trend), 'r--', alpha=0.7)
    
    plt.subplot(2, 2, 3)
    plt.plot(size_factors, endpoint_distances, 'o-', linewidth=2, markersize=8)
    plt.title('Endpoint Distance vs Model Size', fontsize=14)
    plt.xlabel('Student Model Size Factor', fontsize=12)
    plt.ylabel('Avg Endpoint Distance', fontsize=12)
    plt.grid(True, alpha=0.3)
    # Add trend line
    z = np.polyfit(size_factors, endpoint_distances, 2)
    p = np.poly1d(z)
    plt.plot(x_trend, p(x_trend), 'r--', alpha=0.7)
    
    plt.subplot(2, 2, 4)
    plt.plot(size_factors, path_efficiencies, 'o-', linewidth=2, markersize=8)
    plt.title('Path Efficiency vs Model Size', fontsize=14)
    plt.xlabel('Student Model Size Factor', fontsize=12)
    plt.ylabel('Avg Path Efficiency', fontsize=12)
    plt.grid(True, alpha=0.3)
    # Add trend line
    z = np.polyfit(size_factors, path_efficiencies, 2)
    p = np.poly1d(z)
    plt.plot(x_trend, p(x_trend), 'r--', alpha=0.7)
    
    plt.suptitle('Effect of Model Size on Trajectory Metrics', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(comparison_dir, 'metrics_vs_size.png'), dpi=300)
    plt.close()
    
    # 2. Plot FID scores vs model size if available with improved visualization
    if all_fid_results:
        plt.figure(figsize=(15, 7))
        
        plt.subplot(1, 2, 1)
        plt.plot(size_factors, student_fids, 'o-', linewidth=2, markersize=8, color='#1f77b4')
        plt.title('Student FID vs Real Images', fontsize=14)
        plt.xlabel('Student Model Size Factor', fontsize=12)
        plt.ylabel('FID Score (lower is better)', fontsize=12)
        plt.grid(True, alpha=0.3)
        # Add trend line
        z = np.polyfit(size_factors, student_fids, 2)
        p = np.poly1d(z)
        plt.plot(x_trend, p(x_trend), 'r--', alpha=0.7)
        
        plt.subplot(1, 2, 2)
        plt.plot(size_factors, teacher_student_fids, 'o-', linewidth=2, markersize=8, color='#ff7f0e')
        plt.title('Teacher vs Student FID', fontsize=14)
        plt.xlabel('Student Model Size Factor', fontsize=12)
        plt.ylabel('FID Score (lower is better)', fontsize=12)
        plt.grid(True, alpha=0.3)
        # Add trend line
        z = np.polyfit(size_factors, teacher_student_fids, 2)
        p = np.poly1d(z)
        plt.plot(x_trend, p(x_trend), 'r--', alpha=0.7)
        
        plt.suptitle('Effect of Model Size on FID Scores', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(os.path.join(comparison_dir, 'fid_vs_size.png'), dpi=300)
        plt.close()
    
    # 3. Create a combined plot showing all metrics normalized for direct comparison
    plt.figure(figsize=(12, 8))
    
    # Normalize all metrics to [0, 1] range for comparison
    def normalize(values):
        min_val = min(values)
        max_val = max(values)
        if max_val > min_val:
            return [(v - min_val) / (max_val - min_val) for v in values]
        return values
    
    norm_wasserstein = normalize(wasserstein_distances)
    norm_path_lengths = normalize(path_lengths)
    norm_endpoint = normalize(endpoint_distances)
    norm_efficiency = normalize(path_efficiencies)
    
    # For FID scores, lower is better, so invert the normalization
    if all_fid_results:
        norm_student_fid = normalize(student_fids)
        norm_student_fid = [1 - v for v in norm_student_fid]  # Invert so higher is better
        norm_teacher_student_fid = normalize(teacher_student_fids)
        norm_teacher_student_fid = [1 - v for v in norm_teacher_student_fid]  # Invert so higher is better
    
    # Plot all normalized metrics
    plt.plot(size_factors, norm_wasserstein, 'o-', label='Wasserstein Distance (lower is better)', linewidth=2)
    plt.plot(size_factors, norm_path_lengths, 's-', label='Path Length (lower is better)', linewidth=2)
    plt.plot(size_factors, norm_endpoint, '^-', label='Endpoint Distance (lower is better)', linewidth=2)
    plt.plot(size_factors, norm_efficiency, 'D-', label='Path Efficiency (higher is better)', linewidth=2)
    
    if all_fid_results:
        plt.plot(size_factors, norm_student_fid, 'X-', label='Student FID (higher is better)', linewidth=2)
        plt.plot(size_factors, norm_teacher_student_fid, '*-', label='Teacher-Student FID (higher is better)', linewidth=2)
    
    plt.title('Normalized Performance Metrics vs Model Size', fontsize=16)
    plt.xlabel('Student Model Size Factor', fontsize=14)
    plt.ylabel('Normalized Performance (higher is better)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'normalized_metrics_comparison.png'), dpi=300)
    plt.close()
    
    # 4. Create 3D surface plot for model size vs metrics
    if len(size_factors) >= 3:  # Need at least 3 points for a meaningful surface
        # Create a grid of size factors and metrics
        X, Y = np.meshgrid(size_factors, [0, 1, 2, 3])  # 4 metrics
        Z = np.array([
            wasserstein_distances,
            path_lengths,
            endpoint_distances,
            path_efficiencies
        ])
        
        # Normalize each metric to [0, 1] range for better visualization
        Z_norm = np.zeros_like(Z)
        for i in range(Z.shape[0]):
            min_val = np.min(Z[i])
            max_val = np.max(Z[i])
            if max_val > min_val:
                Z_norm[i] = (Z[i] - min_val) / (max_val - min_val)
            else:
                Z_norm[i] = Z[i]
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot surface
        surf = ax.plot_surface(X, Y, Z_norm, cmap='viridis', edgecolor='none', alpha=0.8)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Set labels
        ax.set_xlabel('Model Size Factor', fontsize=12)
        ax.set_ylabel('Metric Type', fontsize=12)
        ax.set_zlabel('Normalized Value', fontsize=12)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['Wasserstein', 'Path Length', 'Endpoint Dist', 'Efficiency'])
        ax.set_title('Model Size vs Performance Metrics', fontsize=14)
        
        plt.savefig(os.path.join(comparison_dir, '3d_metrics_surface.png'), dpi=300)
        plt.close()
        
        # If FID scores are available, create a separate 3D visualization
        if all_fid_results:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot FID scores
            ax.plot(size_factors, student_fids, zs=0, zdir='y', label='Student FID vs Real', linewidth=2)
            ax.plot(size_factors, teacher_student_fids, zs=1, zdir='y', label='Teacher vs Student FID', linewidth=2)
            
            # Set labels
            ax.set_xlabel('Model Size Factor', fontsize=12)
            ax.set_ylabel('FID Type', fontsize=12)
            ax.set_zlabel('FID Score (lower is better)', fontsize=12)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Student vs Real', 'Teacher vs Student'])
            ax.set_title('Model Size vs FID Scores', fontsize=14)
            ax.legend(fontsize=10)
            
            plt.savefig(os.path.join(comparison_dir, '3d_fid_comparison.png'), dpi=300)
            plt.close()
    
    # 5. Create a performance vs model size efficiency plot
    plt.figure(figsize=(10, 8))
    
    # Plot performance metrics vs parameter count (log scale)
    plt.subplot(2, 1, 1)
    plt.semilogx(param_counts, norm_wasserstein, 'o-', label='Wasserstein', linewidth=2)
    plt.semilogx(param_counts, norm_path_lengths, 's-', label='Path Length', linewidth=2)
    plt.semilogx(param_counts, norm_endpoint, '^-', label='Endpoint Dist', linewidth=2)
    plt.semilogx(param_counts, norm_efficiency, 'D-', label='Efficiency', linewidth=2)
    
    if all_fid_results:
        plt.semilogx(param_counts, norm_student_fid, 'X-', label='Student FID', linewidth=2)
        plt.semilogx(param_counts, norm_teacher_student_fid, '*-', label='Teacher-Student FID', linewidth=2)
    
    plt.title('Performance vs Model Parameters (Log Scale)', fontsize=14)
    plt.xlabel('Relative Parameter Count (log scale)', fontsize=12)
    plt.ylabel('Normalized Performance', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Plot performance per parameter (efficiency)
    plt.subplot(2, 1, 2)
    
    # Calculate efficiency (performance per parameter)
    efficiency_wasserstein = [n/p for n, p in zip(norm_wasserstein, param_counts)]
    efficiency_path = [n/p for n, p in zip(norm_path_lengths, param_counts)]
    efficiency_endpoint = [n/p for n, p in zip(norm_endpoint, param_counts)]
    efficiency_path_eff = [n/p for n, p in zip(norm_efficiency, param_counts)]
    
    plt.plot(size_factors, efficiency_wasserstein, 'o-', label='Wasserstein', linewidth=2)
    plt.plot(size_factors, efficiency_path, 's-', label='Path Length', linewidth=2)
    plt.plot(size_factors, efficiency_endpoint, '^-', label='Endpoint Dist', linewidth=2)
    plt.plot(size_factors, efficiency_path_eff, 'D-', label='Efficiency', linewidth=2)
    
    if all_fid_results:
        efficiency_student_fid = [n/p for n, p in zip(norm_student_fid, param_counts)]
        efficiency_teacher_student_fid = [n/p for n, p in zip(norm_teacher_student_fid, param_counts)]
        plt.plot(size_factors, efficiency_student_fid, 'X-', label='Student FID', linewidth=2)
        plt.plot(size_factors, efficiency_teacher_student_fid, '*-', label='Teacher-Student FID', linewidth=2)
    
    plt.title('Performance Efficiency vs Model Size', fontsize=14)
    plt.xlabel('Student Model Size Factor', fontsize=12)
    plt.ylabel('Performance per Parameter', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, 'performance_efficiency.png'), dpi=300)
    plt.close()
    
    # 6. Generate a comprehensive summary report
    with open(os.path.join(comparison_dir, 'model_size_analysis_summary.txt'), 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("MODEL SIZE PERFORMANCE ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("This analysis examines how model size affects various performance metrics.\n\n")
        
        f.write("Model Size Factors Analyzed: " + ", ".join([str(sf) for sf in size_factors]) + "\n\n")
        
        # Find optimal size factors for each metric
        best_wasserstein_idx = np.argmin(wasserstein_distances)
        best_path_length_idx = np.argmin(path_lengths)
        best_endpoint_idx = np.argmin(endpoint_distances)
        best_efficiency_idx = np.argmax(path_efficiencies)
        
        f.write("OPTIMAL MODEL SIZES BY METRIC:\n")
        f.write(f"- Wasserstein Distance: {size_factors[best_wasserstein_idx]} (value: {wasserstein_distances[best_wasserstein_idx]:.4f})\n")
        f.write(f"- Path Length: {size_factors[best_path_length_idx]} (value: {path_lengths[best_path_length_idx]:.4f})\n")
        f.write(f"- Endpoint Distance: {size_factors[best_endpoint_idx]} (value: {endpoint_distances[best_endpoint_idx]:.4f})\n")
        f.write(f"- Path Efficiency: {size_factors[best_efficiency_idx]} (value: {path_efficiencies[best_efficiency_idx]:.4f})\n")
        
        if all_fid_results:
            best_student_fid_idx = np.argmin(student_fids)
            best_teacher_student_fid_idx = np.argmin(teacher_student_fids)
            f.write(f"- Student FID: {size_factors[best_student_fid_idx]} (value: {student_fids[best_student_fid_idx]:.4f})\n")
            f.write(f"- Teacher-Student FID: {size_factors[best_teacher_student_fid_idx]} (value: {teacher_student_fids[best_teacher_student_fid_idx]:.4f})\n")
        
        f.write("\nPERFORMANCE TRENDS:\n")
        
        # Calculate trends (using simple linear regression)
        def get_trend(values):
            x = np.array(size_factors)
            y = np.array(values)
            slope, _ = np.polyfit(x, y, 1)
            return "decreasing" if slope < 0 else "increasing"
        
        f.write(f"- Wasserstein Distance: {get_trend(wasserstein_distances)} with model size\n")
        f.write(f"- Path Length: {get_trend(path_lengths)} with model size\n")
        f.write(f"- Endpoint Distance: {get_trend(endpoint_distances)} with model size\n")
        f.write(f"- Path Efficiency: {get_trend(path_efficiencies)} with model size\n")
        
        if all_fid_results:
            f.write(f"- Student FID: {get_trend(student_fids)} with model size\n")
            f.write(f"- Teacher-Student FID: {get_trend(teacher_student_fids)} with model size\n")
        
        # Calculate efficiency sweet spots
        efficiency_metrics = [
            ("Wasserstein", efficiency_wasserstein),
            ("Path Length", efficiency_path),
            ("Endpoint Distance", efficiency_endpoint),
            ("Path Efficiency", efficiency_path_eff)
        ]
        
        if all_fid_results:
            efficiency_metrics.extend([
                ("Student FID", efficiency_student_fid),
                ("Teacher-Student FID", efficiency_teacher_student_fid)
            ])
        
        f.write("\nEFFICIENCY SWEET SPOTS (best performance per parameter):\n")
        for name, values in efficiency_metrics:
            best_idx = np.argmax(values)
            f.write(f"- {name}: Size factor {size_factors[best_idx]}\n")
        
        # Overall recommendation
        f.write("\nOVERALL RECOMMENDATION:\n")
        
        # Simple approach: count how many times each size factor is optimal
        optimal_counts = {}
        for sf in size_factors:
            optimal_counts[sf] = 0
        
        optimal_counts[size_factors[best_wasserstein_idx]] += 1
        optimal_counts[size_factors[best_path_length_idx]] += 1
        optimal_counts[size_factors[best_endpoint_idx]] += 1
        optimal_counts[size_factors[best_efficiency_idx]] += 1
        
        if all_fid_results:
            optimal_counts[size_factors[best_student_fid_idx]] += 1
            optimal_counts[size_factors[best_teacher_student_fid_idx]] += 1
        
        # Find the size factor with the most optimal metrics
        best_overall = max(optimal_counts.items(), key=lambda x: x[1])
        
        f.write(f"Based on the analysis, a model size factor of {best_overall[0]} appears to offer the best overall performance across metrics.\n")
        
        # Add efficiency consideration
        for name, values in efficiency_metrics:
            best_idx = np.argmax(values)
            if size_factors[best_idx] < best_overall[0]:
                f.write(f"\nHowever, if parameter efficiency is a priority, a smaller model with size factor {size_factors[best_idx]} offers the best {name} efficiency.\n")
                break


if __name__ == "__main__":
    main()
