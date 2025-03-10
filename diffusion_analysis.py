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

def calculate_fid(real_images, generated_images):
    """
    Calculate Fréchet Inception Distance (FID) between real and generated images.
    
    Args:
        real_images: Tensor of real images, shape (N, C, H, W)
        generated_images: Tensor of generated images, shape (N, C, H, W)
        
    Returns:
        fid_score: The FID score (lower is better)
    """
    # Load Inception model
    inception_model = models.inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    
    # Remove the final classification layer
    inception_model = nn.Sequential(*list(inception_model.children())[:-1]).to(device)
    
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
                feat = inception_model(batch).squeeze()
                features.append(feat.cpu().numpy())
        
        features = np.concatenate(features, axis=0)
        return features
    
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

def get_real_images(config, num_samples=100):
    """Get a batch of real images from the dataset"""
    if config.dataset == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        dataset = datasets.CIFAR10(
            root='data',
            train=False,  # Use test set for evaluation
            download=True,
            transform=transform
        )
        
        # Create a DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=num_samples,
            shuffle=True
        )
        
        # Get a batch of images
        images, _ = next(iter(dataloader))
        return images
    
    elif config.dataset == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        dataset = datasets.MNIST(
            root='data',
            train=False,  # Use test set for evaluation
            download=True,
            transform=transform
        )
        
        # Create a DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=num_samples,
            shuffle=True
        )
        
        # Get a batch of images
        images, _ = next(iter(dataloader))
        return images
    
    else:
        raise ValueError(f"Dataset {config.dataset} not supported for FID calculation")

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
    
    # Get diffusion parameters
    teacher_params = get_diffusion_params(config.teacher_steps, config)
    student_params = get_diffusion_params(config.student_steps, config)
    
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
    
    if isinstance(student_model_paths, dict):
        # Multiple student models with different size factors
        for size_factor, path in student_model_paths.items():
            print(f"Loading student model with size factor {size_factor}...")
            
            # Determine architecture type based on size factor
            architecture_type = None
            if float(size_factor) < 0.1:
                architecture_type = 'tiny'     # Use the smallest architecture for very small models
            elif float(size_factor) < 0.3:
                architecture_type = 'small'    # Use small architecture for small models
            elif float(size_factor) < 0.7:
                architecture_type = 'medium'   # Use medium architecture for medium models
            else:
                architecture_type = 'full'     # Use full architecture for large models
            
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
                    student_model = StudentUNet(config, size_factor=float(size_factor)).to(device)
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
                
                student_model = StudentUNet(config, size_factor=size_factor).to(device)
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
    
    for size_factor, student_model in student_models.items():
        print(f"\n{'='*80}")
        print(f"Analyzing student model with size factor {size_factor}")
        print(f"{'='*80}")
        
        # 1. Generate multiple trajectories
        print("Generating trajectories...")
        teacher_trajectories, student_trajectories = generate_trajectories(
            teacher_model, student_model, config, num_samples=num_samples
        )
        
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
            teacher_params = get_diffusion_params(config.teacher_steps, config)
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
            student_params = get_diffusion_params(config.student_steps, config)
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
    
    print("\nAnalysis complete. Results saved in the analysis directory.")


def create_model_size_comparisons(all_metrics, all_fid_results, config):
    """Create comparative visualizations across different student model sizes"""
    os.makedirs(os.path.join(config.analysis_dir, 'model_size_comparison'), exist_ok=True)
    
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
    
    # 1. Plot metrics vs model size
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(size_factors, wasserstein_distances, 'o-')
    plt.title('Average Wasserstein Distance vs Model Size')
    plt.xlabel('Student Model Size Factor')
    plt.ylabel('Avg Wasserstein Distance')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(size_factors, path_lengths, 'o-')
    plt.title('Path Length vs Model Size')
    plt.xlabel('Student Model Size Factor')
    plt.ylabel('Avg Path Length')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(size_factors, endpoint_distances, 'o-')
    plt.title('Endpoint Distance vs Model Size')
    plt.xlabel('Student Model Size Factor')
    plt.ylabel('Avg Endpoint Distance')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(size_factors, path_efficiencies, 'o-')
    plt.title('Path Efficiency vs Model Size')
    plt.xlabel('Student Model Size Factor')
    plt.ylabel('Avg Path Efficiency')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.analysis_dir, 'model_size_comparison', 'metrics_vs_size.png'))
    plt.close()
    
    # 2. Plot FID scores vs model size if available
    if all_fid_results:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(size_factors, student_fids, 'o-')
        plt.title('Student FID vs Real Images')
        plt.xlabel('Student Model Size Factor')
        plt.ylabel('FID Score (lower is better)')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(size_factors, teacher_student_fids, 'o-')
        plt.title('Teacher vs Student FID')
        plt.xlabel('Student Model Size Factor')
        plt.ylabel('FID Score (lower is better)')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.analysis_dir, 'model_size_comparison', 'fid_vs_size.png'))
        plt.close()
    
    # 3. Create 3D surface plot for model size vs metrics
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
        ax.set_xlabel('Model Size Factor')
        ax.set_ylabel('Metric Type')
        ax.set_zlabel('Normalized Value')
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['Wasserstein', 'Path Length', 'Endpoint Dist', 'Efficiency'])
        ax.set_title('Model Size vs Performance Metrics')
        
        plt.savefig(os.path.join(config.analysis_dir, 'model_size_comparison', '3d_metrics_surface.png'))
        plt.close()
        
        # If FID scores are available, create a separate 3D visualization
        if all_fid_results:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot FID scores
            ax.plot(size_factors, student_fids, zs=0, zdir='y', label='Student FID vs Real')
            ax.plot(size_factors, teacher_student_fids, zs=1, zdir='y', label='Teacher vs Student FID')
            
            # Set labels
            ax.set_xlabel('Model Size Factor')
            ax.set_ylabel('FID Type')
            ax.set_zlabel('FID Score (lower is better)')
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Student vs Real', 'Teacher vs Student'])
            ax.set_title('Model Size vs FID Scores')
            ax.legend()
            
            plt.savefig(os.path.join(config.analysis_dir, 'model_size_comparison', '3d_fid_comparison.png'))
            plt.close()


if __name__ == "__main__":
    main()

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
    ax.plot(teacher_pca[:, 0], teacher_pca[:, 1], teacher_pca[:, 2], 'b-', label='Teacher', alpha=0.7)
    ax.scatter(teacher_pca[0, 0], teacher_pca[0, 1], teacher_pca[0, 2], c='blue', marker='o', s=100, label='Teacher start')
    ax.scatter(teacher_pca[-1, 0], teacher_pca[-1, 1], teacher_pca[-1, 2], c='darkblue', marker='x', s=100, label='Teacher end')
    
    # Plot student trajectory
    ax.plot(student_pca[:, 0], student_pca[:, 1], student_pca[:, 2], 'r-', label='Student', alpha=0.7)
    ax.scatter(student_pca[0, 0], student_pca[0, 1], student_pca[0, 2], c='orange', marker='o', s=100, label='Student start')
    ax.scatter(student_pca[-1, 0], student_pca[-1, 1], student_pca[-1, 2], c='darkred', marker='x', s=100, label='Student end')
    
    # Add timestep markers at intervals
    step_size = max(1, len(teacher_pca) // 5)
    for i in range(0, len(teacher_pca), step_size):
        ax.text(teacher_pca[i, 0], teacher_pca[i, 1], teacher_pca[i, 2], f"T{i}", color='blue')
    
    step_size_student = max(1, len(student_pca) // 5)
    for i in range(0, len(student_pca), step_size_student):
        ax.text(student_pca[i, 0], student_pca[i, 1], student_pca[i, 2], f"S{i}", color='red')
    
    ax.set_title('3D PCA of Latent Trajectories: Teacher vs Student')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
    ax.legend()
    
    # Save static image
    plt.savefig(os.path.join(config.analysis_dir, '3d_visualization', f'3d_trajectories{suffix}.png'))
    plt.close()
    
    # Save the PCA components for any further analysis
    np.save(os.path.join(config.analysis_dir, '3d_visualization', f'teacher_pca_3d{suffix}.npy'), teacher_pca)
    np.save(os.path.join(config.analysis_dir, '3d_visualization', f'student_pca_3d{suffix}.npy'), student_pca)
    
    return teacher_pca, student_pca
    
import os
