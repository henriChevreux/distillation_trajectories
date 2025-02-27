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

def main(config=None, teacher_model_path=None, student_model_path=None, num_samples=5, 
         skip_metrics=False, skip_dimensionality=False, skip_noise=False, 
         skip_attention=False, skip_3d=False):
    """
    Main function for running the diffusion model analysis
    
    Args:
        config: Configuration object, will create a new one if None
        teacher_model_path: Path to the teacher model file, overrides default
        student_model_path: Path to the student model file, overrides default
        num_samples: Number of samples to generate for analysis
        skip_metrics: Skip trajectory metrics computation
        skip_dimensionality: Skip dimensionality reduction analysis
        skip_noise: Skip noise prediction analysis
        skip_attention: Skip attention map analysis
        skip_3d: Skip 3D visualization
    """
    # Initialize config if not provided
    if config is None:
        config = Config()
        config.create_directories()
    
    # Get diffusion parameters
    teacher_params = get_diffusion_params(config.teacher_steps, config)
    student_params = get_diffusion_params(config.student_steps, config)
    
    # Load models
    print("Loading teacher and student models...")
    teacher_model = SimpleUNet(config).to(device)
    student_model = SimpleUNet(config).to(device)
    
    # Use provided paths or default paths
    if teacher_model_path is None:
        teacher_model_path = os.path.join(config.models_dir, 'model_epoch_1.pt')
    
    if student_model_path is None:
        student_model_path = os.path.join(config.models_dir, 'student_model_epoch_1.pt')
    
    # Load the models
    if os.path.exists(teacher_model_path):
        teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=device))
        print(f"Loaded teacher model from {teacher_model_path}")
    else:
        print(f"ERROR: Teacher model not found at {teacher_model_path}. Please run training first.")
        return
    
    if os.path.exists(student_model_path):
        student_model.load_state_dict(torch.load(student_model_path, map_location=device))
        print(f"Loaded student model from {student_model_path}")
    else:
        print(f"ERROR: Student model not found at {student_model_path}. Please run training with distillation first.")
        return
    
    # Set models to evaluation mode
    teacher_model.eval()
    student_model.eval()
    
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
        summary = visualize_metrics(metrics, config)
        print("Metrics summary:", summary)
    else:
        print("Skipping trajectory metrics analysis.")
    
    if not skip_dimensionality:
        # 4. Dimensionality reduction analysis
        print("Performing dimensionality reduction analysis...")
        dimensionality_reduction_analysis(teacher_trajectories, student_trajectories, config)
    else:
        print("Skipping dimensionality reduction analysis.")
    
    if not skip_noise:
        # 5. Noise prediction analysis
        print("Analyzing noise prediction patterns...")
        noise_metrics = analyze_noise_prediction(teacher_model, student_model, config)
    else:
        print("Skipping noise prediction analysis.")
    
    if not skip_attention:
        # 6. Attention map analysis
        print("Analyzing attention maps...")
        attention_metrics = analyze_attention_maps(teacher_model, student_model, config)
    else:
        print("Skipping attention map analysis.")
    
    if not skip_3d:
        # 7. Generate 3D latent space visualization
        print("Creating 3D latent space visualization...")
        generate_latent_space_visualization(teacher_trajectories, student_trajectories, config)
    else:
        print("Skipping 3D latent space visualization.")
    
    print("\nAnalysis complete. Results saved in the analysis directory.")


if __name__ == "__main__":
    main()

def generate_latent_space_visualization(teacher_trajectories, student_trajectories, config):
    """Create a 3D visualization of trajectories over time"""
    os.makedirs(os.path.join(config.analysis_dir, '3d_visualization'), exist_ok=True)
    
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
    plt.savefig(os.path.join(config.analysis_dir, '3d_visualization', '3d_trajectories.png'))
    plt.close()
    
    # Save the PCA components for any further analysis
    np.save(os.path.join(config.analysis_dir, '3d_visualization', 'teacher_pca_3d.npy'), teacher_pca)
    np.save(os.path.join(config.analysis_dir, '3d_visualization', 'student_pca_3d.npy'), student_pca)
    
    return teacher_pca, student_pca
    
def analyze_attention_maps(teacher_model, student_model, config):
    """Analyze intermediate activations and create attention-like maps"""
    os.makedirs(os.path.join(config.analysis_dir, 'attention_maps'), exist_ok=True)
    
    teacher_params = get_diffusion_params(config.teacher_steps, config)
    student_params = get_diffusion_params(config.student_steps, config)
    
    # Create a sample image
    shape = (1, config.channels, config.image_size, config.image_size)
    
    # Fix seed for reproducibility
    torch.manual_seed(42)
    
    # Generate samples with intermediate activations
    _, _, teacher_intermediates = p_sample_loop(
        teacher_model, 
        shape=shape,
        timesteps=config.teacher_steps,
        diffusion_params=teacher_params,
        track_trajectory=True,
        track_intermediates=True
    )
    
    torch.manual_seed(42)
    _, _, student_intermediates = p_sample_loop(
        student_model, 
        shape=shape,
        timesteps=config.student_steps,
        diffusion_params=student_params,
        track_trajectory=True,
        track_intermediates=True
    )
    
    # Select a few key timesteps to visualize
    teacher_steps = np.linspace(0, len(teacher_intermediates)-1, 5).astype(int)
    student_steps = np.linspace(0, len(student_intermediates)-1, 5).astype(int)
    
    # For each selected timestep, visualize the image and gradient heat map
    for i, (t_idx, s_idx) in enumerate(zip(teacher_steps, student_steps)):
        t_data = teacher_intermediates[t_idx]
        s_data = student_intermediates[s_idx]
        
        # Get the gradients (these show what parts of the image the model is focusing on)
        t_grad = t_data['grad'][0].abs().sum(dim=0).numpy()
        s_grad = s_data['grad'][0].abs().sum(dim=0).numpy()
        
        # Normalize for visualization
        t_grad = (t_grad - t_grad.min()) / (t_grad.max() - t_grad.min() + 1e-8)
        s_grad = (s_grad - s_grad.min()) / (s_grad.max() - s_grad.min() + 1e-8)
        
        # Get the current image
        t_img = t_data['img'][0, 0].numpy()
        s_img = s_data['img'][0, 0].numpy()
        
        # Normalize images for visualization
        t_img = (t_img - t_img.min()) / (t_img.max() - t_img.min() + 1e-8)
        s_img = (s_img - s_img.min()) / (s_img.max() - s_img.min() + 1e-8)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Teacher image
        axes[0, 0].imshow(t_img, cmap='gray')
        axes[0, 0].set_title(f'Teacher Image (t={t_data["step"]})')
        axes[0, 0].axis('off')
        
        # Teacher gradient heatmap
        im1 = axes[0, 1].imshow(t_grad, cmap='viridis')
        axes[0, 1].set_title('Teacher Attention Map')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Teacher overlay
        axes[0, 2].imshow(t_img, cmap='gray')
        axes[0, 2].imshow(t_grad, cmap='viridis', alpha=0.5)
        axes[0, 2].set_title('Teacher Overlay')
        axes[0, 2].axis('off')
        
        # Student image
        axes[1, 0].imshow(s_img, cmap='gray')
        axes[1, 0].set_title(f'Student Image (t={s_data["step"]})')
        axes[1, 0].axis('off')
        
        # Student gradient heatmap
        im2 = axes[1, 1].imshow(s_grad, cmap='viridis')
        axes[1, 1].set_title('Student Attention Map')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1])
        
        # Student overlay
        axes[1, 2].imshow(s_img, cmap='gray')
        axes[1, 2].imshow(s_grad, cmap='viridis', alpha=0.5)
        axes[1, 2].set_title('Student Overlay')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.analysis_dir, 'attention_maps', f'attention_t{i}.png'))
        plt.close()
    
    # Compute attention similarity across timesteps
    similarity_measures = []
    
    # Interpolate student intermediates to match teacher length
    x_student = np.linspace(0, 1, len(student_intermediates))
    x_teacher = np.linspace(0, 1, len(teacher_intermediates))
    
    for t_idx in range(len(teacher_intermediates)):
        # Find corresponding student step (interpolation)
        s_idx_float = np.interp(t_idx / len(teacher_intermediates), 
                              x_teacher, range(len(student_intermediates)))
        s_idx = int(s_idx_float)
        s_idx = min(s_idx, len(student_intermediates) - 1)
        
        t_grad = teacher_intermediates[t_idx]['grad'][0].abs().sum(dim=0).flatten().numpy()
        s_grad = student_intermediates[s_idx]['grad'][0].abs().sum(dim=0).flatten().numpy()
        
        # Normalize
        t_grad = (t_grad - t_grad.min()) / (t_grad.max() - t_grad.min() + 1e-8)
        s_grad = (s_grad - s_grad.min()) / (s_grad.max() - s_grad.min() + 1e-8)
        
        # Compute similarity metrics
        cosine = np.dot(t_grad, s_grad) / (np.linalg.norm(t_grad) * np.linalg.norm(s_grad))
        
        similarity_measures.append({
            'teacher_step': teacher_intermediates[t_idx]['step'],
            'student_step': student_intermediates[s_idx]['step'],
            'cosine_similarity': cosine
        })
    
    # Plot similarity over time
    plt.figure(figsize=(10, 6))
    plt.plot([s['teacher_step'] for s in similarity_measures], 
             [s['cosine_similarity'] for s in similarity_measures])
    plt.title('Attention Map Similarity between Teacher and Student')
    plt.xlabel('Teacher Timestep')
    plt.ylabel('Cosine Similarity')
    plt.grid(True)
    plt.savefig(os.path.join(config.analysis_dir, 'attention_maps', 'similarity_over_time.png'))
    plt.close()
    
    return similarity_measures
def analyze_noise_prediction(teacher_model, student_model, config, num_samples=5):
    """Analyze how the teacher and student models predict noise at different denoising steps"""
    os.makedirs(os.path.join(config.analysis_dir, 'noise_prediction'), exist_ok=True)
    
    teacher_params = get_diffusion_params(config.teacher_steps, config)
    student_params = get_diffusion_params(config.student_steps, config)
    
    # Create sample images
    shape = (num_samples, config.channels, config.image_size, config.image_size)
    
    # Sample some random noise
    torch.manual_seed(42)
    noises = torch.randn(shape, device=device)
    
    # Create timestep sequences for teacher and student
    teacher_timesteps = torch.linspace(0, config.teacher_steps-1, 10).long().to(device)
    student_timesteps = torch.linspace(0, config.student_steps-1, 10).long().to(device)
    
    # For each timestep, compare the noise prediction
    noise_prediction_errors = []
    
    for i, (t_teacher, t_student) in enumerate(zip(teacher_timesteps, student_timesteps)):
        teacher_preds = []
        student_preds = []
        
        for j in range(num_samples):
            noise = noises[j:j+1]
            
            # Create noisy image at this timestep
            t_teacher_batch = torch.tensor([t_teacher] * noise.shape[0], device=device)
            t_student_batch = torch.tensor([t_student] * noise.shape[0], device=device)
            
            # Get noisy image at this timestep
            x_teacher, true_noise = q_sample(noise, t_teacher_batch, teacher_params)
            x_student, _ = q_sample(noise, t_student_batch, student_params)
            
            # Predict noise
            with torch.no_grad():
                teacher_pred = teacher_model(x_teacher, t_teacher_batch)
                student_pred = student_model(x_student, t_student_batch)
            
            teacher_preds.append(teacher_pred.detach().cpu())
            student_preds.append(student_pred.detach().cpu())
        
        # Stack predictions
        teacher_preds = torch.cat(teacher_preds, dim=0)
        student_preds = torch.cat(student_preds, dim=0)
        
        # Calculate error metrics
        mse = F.mse_loss(student_preds, teacher_preds).item()
        cosine_sim = F.cosine_similarity(student_preds.flatten(), teacher_preds.flatten(), dim=0).item()
        
        noise_prediction_errors.append({
            'timestep_idx': i,
            'teacher_timestep': t_teacher.item(),
            'student_timestep': t_student.item(),
            'mse': mse,
            'cosine_similarity': cosine_sim
        })
        
        # Visualize the noise predictions for the first sample
        if i % 3 == 0:  # Only visualize a subset of timesteps
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original noise prediction by teacher
            teacher_img = teacher_preds[0].squeeze().numpy()
            im0 = axes[0].imshow(teacher_img, cmap='viridis')
            axes[0].set_title(f'Teacher t={t_teacher.item()}')
            plt.colorbar(im0, ax=axes[0])
            
            # Noise prediction by student
            student_img = student_preds[0].squeeze().numpy()
            im1 = axes[1].imshow(student_img, cmap='viridis')
            axes[1].set_title(f'Student t={t_student.item()}')
            plt.colorbar(im1, ax=axes[1])
            
            # Difference
            diff_img = teacher_img - student_img
            im2 = axes[2].imshow(diff_img, cmap='bwr')
            axes[2].set_title(f'Difference (MSE={mse:.4f})')
            plt.colorbar(im2, ax=axes[2])
            
            plt.tight_layout()
            plt.savefig(os.path.join(config.analysis_dir, 'noise_prediction', f'noise_pred_t{i}.png'))
            plt.close()
    
    # Plot error metrics across timesteps
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot([e['timestep_idx'] for e in noise_prediction_errors], 
             [e['mse'] for e in noise_prediction_errors])
    plt.title('MSE between Teacher and Student Noise Predictions')
    plt.xlabel('Timestep Index')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot([e['timestep_idx'] for e in noise_prediction_errors], 
             [e['cosine_similarity'] for e in noise_prediction_errors])
    plt.title('Cosine Similarity of Noise Predictions')
    plt.xlabel('Timestep Index')
    plt.ylabel('Cosine Similarity')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.analysis_dir, 'noise_prediction', 'error_metrics.png'))
    plt.close()
    
    return noise_prediction_errors


# Configuration class
class Config:
    def __init__(self):
        # Dataset
        self.dataset = "MNIST"
        self.image_size = 28
        self.channels = 1
        self.batch_size = 64
        
        # Model
        self.latent_dim = 32
        self.hidden_dims = [32, 64, 128]
        
        # Diffusion process
        self.timesteps = 50
        self.beta_start = 1e-4
        self.beta_end = 0.02
        
        # Training
        self.epochs = 1
        self.lr = 1e-4
        self.save_interval = 5
        
        # Directories
        self.results_dir = "results"
        self.models_dir = "models"
        self.trajectory_dir = "trajectories"
        self.analysis_dir = "analysis"
        
        # Distillation
        self.distill = True
        self.teacher_steps = 50
        self.student_steps = 50
        
    def create_directories(self):
        for dir_path in [self.results_dir, self.models_dir, self.trajectory_dir, self.analysis_dir]:
            os.makedirs(dir_path, exist_ok=True)


# Simple U-Net architecture adapted for limited computational resources
class SimpleUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initial projection
        self.conv_in = nn.Conv2d(config.channels, config.hidden_dims[0], 3, padding=1)
        
        # Downsampling path
        self.downs = nn.ModuleList([])
        for i in range(len(config.hidden_dims) - 1):
            self.downs.append(nn.Sequential(
                nn.Conv2d(config.hidden_dims[i], config.hidden_dims[i+1], 3, padding=1),
                nn.BatchNorm2d(config.hidden_dims[i+1]),
                nn.LeakyReLU(),
                nn.Conv2d(config.hidden_dims[i+1], config.hidden_dims[i+1], 3, padding=1),
                nn.BatchNorm2d(config.hidden_dims[i+1]),
                nn.LeakyReLU(),
                nn.MaxPool2d(2)
            ))
        
        # Middle
        mid_channels = config.hidden_dims[-1]
        self.mid = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels * 2, 3, padding=1),
            nn.BatchNorm2d(mid_channels * 2),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels * 2, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(),
        )
        
        # Upsampling path
        self.ups = nn.ModuleList([])
        for i in range(len(config.hidden_dims) - 1, 0, -1):
            self.ups.append(nn.Sequential(
                nn.ConvTranspose2d(config.hidden_dims[i], config.hidden_dims[i-1], 4, stride=2, padding=1),
                nn.BatchNorm2d(config.hidden_dims[i-1]),
                nn.LeakyReLU(),
                nn.Conv2d(config.hidden_dims[i-1], config.hidden_dims[i-1], 3, padding=1),
                nn.BatchNorm2d(config.hidden_dims[i-1]),
                nn.LeakyReLU()
            ))
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU()
        )
        
        # Time projection layers for each spatial resolution
        self.time_projections = nn.ModuleList([
            nn.Conv2d(32, dim, 1) for dim in config.hidden_dims
        ])
        
        # Output projection
        self.conv_out = nn.Conv2d(config.hidden_dims[0], config.channels, 3, padding=1)
        
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1).float())
        
        # Initial convolution
        x = self.conv_in(x)
        
        # Store residuals for skip connections
        residuals = []
        for down in self.downs:
            residuals.append(x)
            x = down(x)
        
        # Add time embedding - simplified to avoid dimension issues
        # Create a more straightforward time embedding approach
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)
        # Instead of adding directly, use a small projection layer
        t_projection = nn.Conv2d(32, x.shape[1], 1).to(x.device)
        x = x + t_projection(t_emb.expand(-1, -1, x.shape[2], x.shape[3]))
        
        # Middle
        x = self.mid(x)
        
        # Upsampling with skip connections
        for up, res in zip(self.ups, reversed(residuals)):
            x = up(x)
            x = x + res
            
        # Output
        x = self.conv_out(x)
        return x


# Utility functions
def extract(a, t, x_shape):
    """Extract coefficients at specified timesteps t"""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps, config):
    """Linear beta schedule as proposed in the original DDPM paper"""
    scale = 1000 / timesteps
    beta_start = config.beta_start
    beta_end = config.beta_end
    return torch.linspace(beta_start, beta_end, timesteps)


def get_diffusion_params(timesteps, config):
    # Define beta schedule
    betas = linear_beta_schedule(timesteps, config)
    
    # Define alphas
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    
    # Calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    
    # Calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    
    return {
        'betas': betas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_recip_alphas': sqrt_recip_alphas,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'posterior_variance': posterior_variance
    }


# Forward diffusion process
def q_sample(x_start, t, diffusion_params):
    """Forward diffusion process: Add noise to the input according to timestep t"""
    noise = torch.randn_like(x_start)
    
    sqrt_alphas_cumprod_t = extract(diffusion_params['sqrt_alphas_cumprod'], t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        diffusion_params['sqrt_one_minus_alphas_cumprod'], t, x_start.shape
    )
    
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise


# Sampling functions 
@torch.no_grad()
def p_sample(model, x, t, t_index, diffusion_params):
    """Sample from p(x_{t-1} | x_t) - single step denoising"""
    betas_t = extract(diffusion_params['betas'], t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        diffusion_params['sqrt_one_minus_alphas_cumprod'], t, x.shape
    )
    sqrt_recip_alphas_t = extract(diffusion_params['sqrt_recip_alphas'], t, x.shape)
    
    # Equation 11 in the DDPM paper
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(diffusion_params['posterior_variance'], t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_loop(model, shape, timesteps, diffusion_params, track_trajectory=False, track_intermediates=False):
    """Generate samples by iteratively denoising from pure noise"""
    device = next(model.parameters()).device
    b = shape[0]
    
    # Start from pure noise
    img = torch.randn(shape, device=device)
    
    trajectory = [img.detach().cpu()]
    intermediates = []
    
    # Iteratively denoise
    for i in tqdm(reversed(range(0, timesteps)), desc='Sampling time steps', total=timesteps):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        
        # Track intermediate activations if requested
        if track_intermediates:
            with torch.enable_grad():
                img.requires_grad_(True)
                noise_pred = model(img, t)
                noise_pred.mean().backward()
                grad = img.grad.detach().cpu()
                intermediates.append({
                    'step': i,
                    'img': img.detach().cpu(),
                    'grad': grad,
                    'noise_pred': noise_pred.detach().cpu()
                })
                img.requires_grad_(False)
        
        img = p_sample(model, img, t, i, diffusion_params)
        
        if track_trajectory:
            trajectory.append(img.detach().cpu())
    
    if track_intermediates:
        return img, trajectory, intermediates
    elif track_trajectory:
        return img, trajectory
    return img


# Analysis Functions
def generate_trajectories(teacher_model, student_model, config, num_samples=10):
    """Generate multiple trajectories for both teacher and student models"""
    teacher_params = get_diffusion_params(config.teacher_steps, config)
    student_params = get_diffusion_params(config.student_steps, config)
    
    teacher_trajectories = []
    student_trajectories = []
    
    for seed in tqdm(range(num_samples), desc="Generating trajectories"):
        # Use the same seed for both models
        torch.manual_seed(seed)
        shape = (1, config.channels, config.image_size, config.image_size)
        
        _, teacher_traj = p_sample_loop(
            teacher_model, 
            shape=shape,
            timesteps=config.teacher_steps,
            diffusion_params=teacher_params,
            track_trajectory=True
        )
        
        torch.manual_seed(seed)
        _, student_traj = p_sample_loop(
            student_model, 
            shape=shape,
            timesteps=config.student_steps,
            diffusion_params=student_params,
            track_trajectory=True
        )
        
        teacher_trajectories.append(teacher_traj)
        student_trajectories.append(student_traj)
    
    return teacher_trajectories, student_trajectories


def compute_trajectory_metrics(teacher_trajectories, student_trajectories, config):
    """Compute various metrics to compare trajectory characteristics"""
    num_samples = len(teacher_trajectories)
    metrics = {
        'wasserstein_distances': [],
        'normalized_path_lengths': [],
        'endpoint_distances': [],
        'velocity_profiles': [],
        'acceleration_profiles': [],
        'path_efficiencies': []
    }
    
    for i in range(num_samples):
        # Flatten trajectories
        teacher_traj = [t[0].reshape(-1).numpy() for t in teacher_trajectories[i]]
        student_traj = [s[0].reshape(-1).numpy() for s in student_trajectories[i]]
        
        # Interpolate student trajectory to match teacher length
        x_teacher = np.linspace(0, 1, len(teacher_traj))
        x_student = np.linspace(0, 1, len(student_traj))
        
        student_interp_traj = []
        for dim in range(teacher_traj[0].shape[0]):
            teacher_values = np.array([t[dim] for t in teacher_traj])
            student_values = np.array([s[dim] for s in student_traj])
            
            f = interp1d(x_student, student_values, kind='linear')
            student_interp = f(x_teacher)
            student_interp_traj.append(student_interp)
        
        student_interp_traj = np.array(student_interp_traj).T
        
        # 1. Wasserstein distance at key timesteps
        distances = []
        for t_idx in range(len(teacher_traj)):
            wd = wasserstein_distance(teacher_traj[t_idx], student_interp_traj[t_idx])
            distances.append(wd)
        metrics['wasserstein_distances'].append(distances)
        
        # 2. Normalized path length
        teacher_path_length = 0
        student_path_length = 0
        
        for t in range(1, len(teacher_traj)):
            teacher_path_length += np.linalg.norm(teacher_traj[t] - teacher_traj[t-1])
        
        for t in range(1, len(student_traj)):
            student_path_length += np.linalg.norm(student_traj[t] - student_traj[t-1])
        
        # Normalize by the number of steps
        teacher_norm_length = teacher_path_length / (len(teacher_traj) - 1)
        student_norm_length = student_path_length / (len(student_traj) - 1)
        
        metrics['normalized_path_lengths'].append((teacher_norm_length, student_norm_length))
        
        # 3. Endpoint distance
        endpoint_distance = np.linalg.norm(teacher_traj[-1] - student_traj[-1])
        metrics['endpoint_distances'].append(endpoint_distance)
        
        # 4. Velocity profiles (first derivative)
        teacher_velocities = [np.linalg.norm(teacher_traj[t+1] - teacher_traj[t]) 
                             for t in range(len(teacher_traj)-1)]
        
        student_velocities = [np.linalg.norm(student_traj[t+1] - student_traj[t]) 
                             for t in range(len(student_traj)-1)]
        
        # Interpolate student velocities to match teacher length
        x_student_vel = np.linspace(0, 1, len(student_velocities))
        x_teacher_vel = np.linspace(0, 1, len(teacher_velocities))
        f_vel = interp1d(x_student_vel, student_velocities, kind='linear')
        student_interp_velocities = f_vel(x_teacher_vel)
        
        metrics['velocity_profiles'].append((teacher_velocities, student_interp_velocities))
        
        # 5. Acceleration profiles (second derivative)
        teacher_accels = [teacher_velocities[t+1] - teacher_velocities[t] 
                         for t in range(len(teacher_velocities)-1)]
        
        student_interp_accels = [student_interp_velocities[t+1] - student_interp_velocities[t] 
                               for t in range(len(student_interp_velocities)-1)]
        
        metrics['acceleration_profiles'].append((teacher_accels, student_interp_accels))
        
        # 6. Path efficiency (straight line distance / actual path length)
        teacher_efficiency = np.linalg.norm(teacher_traj[-1] - teacher_traj[0]) / teacher_path_length
        student_efficiency = np.linalg.norm(student_traj[-1] - student_traj[0]) / student_path_length
        
        metrics['path_efficiencies'].append((teacher_efficiency, student_efficiency))
    
    return metrics


def visualize_metrics(metrics, config):
    """Create visualizations of the computed trajectory metrics"""
    # Prepare directory
    os.makedirs(os.path.join(config.analysis_dir, 'metrics'), exist_ok=True)
    
    # 1. Plot Wasserstein distances
    plt.figure(figsize=(12, 6))
    avg_distances = np.mean(metrics['wasserstein_distances'], axis=0)
    std_distances = np.std(metrics['wasserstein_distances'], axis=0)
    
    x = np.linspace(0, 1, len(avg_distances))
    plt.plot(x, avg_distances, label='Average Wasserstein Distance')
    plt.fill_between(x, avg_distances - std_distances, avg_distances + std_distances, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.title('Wasserstein Distance Between Teacher and Student Trajectories')
    plt.xlabel('Normalized Timestep')
    plt.ylabel('Wasserstein Distance')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.analysis_dir, 'metrics', 'wasserstein_distances.png'))
    plt.close()
    
    # 2. Path length comparison
    teacher_lengths = [t for t, _ in metrics['normalized_path_lengths']]
    student_lengths = [s for _, s in metrics['normalized_path_lengths']]
    
    plt.figure(figsize=(10, 6))
    positions = np.arange(len(teacher_lengths))
    width = 0.35
    
    plt.bar(positions - width/2, teacher_lengths, width, label='Teacher')
    plt.bar(positions + width/2, student_lengths, width, label='Student')
    plt.axhline(y=np.mean(teacher_lengths), color='b', linestyle='--', alpha=0.5)
    plt.axhline(y=np.mean(student_lengths), color='orange', linestyle='--', alpha=0.5)
    
    plt.title('Normalized Path Length Comparison')
    plt.xlabel('Sample Index')
    plt.ylabel('Normalized Path Length')
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(config.analysis_dir, 'metrics', 'path_lengths.png'))
    plt.close()
    
    # 3. Endpoint distances
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(metrics['endpoint_distances'])), metrics['endpoint_distances'])
    plt.axhline(y=np.mean(metrics['endpoint_distances']), color='r', linestyle='--')
    plt.title('Endpoint Distance Between Teacher and Student Trajectories')
    plt.xlabel('Sample Index')
    plt.ylabel('L2 Distance')
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(config.analysis_dir, 'metrics', 'endpoint_distances.png'))
    plt.close()
    
    # 4. Velocity profiles
    plt.figure(figsize=(12, 6))
    # Take first sample for illustration
    teacher_vels, student_vels = metrics['velocity_profiles'][0]
    x_teacher = np.linspace(0, 1, len(teacher_vels))
    
    plt.plot(x_teacher, teacher_vels, label='Teacher Velocity')
    plt.plot(x_teacher, student_vels, label='Student Velocity (Interpolated)')
    plt.title('Velocity Profile Comparison (Sample 0)')
    plt.xlabel('Normalized Timestep')
    plt.ylabel('Velocity Magnitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.analysis_dir, 'metrics', 'velocity_profile.png'))
    plt.close()
    
    # 5. Acceleration profiles
    plt.figure(figsize=(12, 6))
    # Take first sample for illustration
    teacher_accels, student_accels = metrics['acceleration_profiles'][0]
    x_teacher = np.linspace(0, 1, len(teacher_accels))
    
    plt.plot(x_teacher, teacher_accels, label='Teacher Acceleration')
    plt.plot(x_teacher, student_accels, label='Student Acceleration (Interpolated)')
    plt.title('Acceleration Profile Comparison (Sample 0)')
    plt.xlabel('Normalized Timestep')
    plt.ylabel('Acceleration Magnitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.analysis_dir, 'metrics', 'acceleration_profile.png'))
    plt.close()
    
    # 6. Path efficiency comparison
    teacher_effs = [t for t, _ in metrics['path_efficiencies']]
    student_effs = [s for _, s in metrics['path_efficiencies']]
    
    plt.figure(figsize=(10, 6))
    positions = np.arange(len(teacher_effs))
    width = 0.35
    
    plt.bar(positions - width/2, teacher_effs, width, label='Teacher')
    plt.bar(positions + width/2, student_effs, width, label='Student')
    plt.axhline(y=np.mean(teacher_effs), color='b', linestyle='--', alpha=0.5)
    plt.axhline(y=np.mean(student_effs), color='orange', linestyle='--', alpha=0.5)
    
    plt.title('Path Efficiency Comparison (Higher is More Direct)')
    plt.xlabel('Sample Index')
    plt.ylabel('Path Efficiency (Straight/Actual)')
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(config.analysis_dir, 'metrics', 'path_efficiency.png'))
    plt.close()
    
    # 7. Summary metrics
    summary = {
        'avg_wasserstein': np.mean(avg_distances),
        'avg_teacher_path_length': np.mean(teacher_lengths),
        'avg_student_path_length': np.mean(student_lengths),
        'avg_endpoint_distance': np.mean(metrics['endpoint_distances']),
        'avg_teacher_efficiency': np.mean(teacher_effs),
        'avg_student_efficiency': np.mean(student_effs),
    }
    
    # Save summary as text
    with open(os.path.join(config.analysis_dir, 'metrics', 'summary.txt'), 'w') as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    return summary


def dimensionality_reduction_analysis(teacher_trajectories, student_trajectories, config):
    """Analyze trajectories using dimensionality reduction techniques (PCA, t-SNE, UMAP)"""
    # Prepare directory
    os.makedirs(os.path.join(config.analysis_dir, 'dimensionality'), exist_ok=True)
    
    # Sample index to analyze (you can loop through all or select specific ones)
    sample_idx = 0
    
    # Flatten trajectories
    teacher_traj = [t[0].reshape(-1).numpy() for t in teacher_trajectories[sample_idx]]
    student_traj = [s[0].reshape(-1).numpy() for s in student_trajectories[sample_idx]]
    
    # 1. PCA
    pca = PCA(n_components=2)
    combined = np.vstack([teacher_traj, student_traj])
    pca_result = pca.fit_transform(combined)
    
    teacher_pca = pca_result[:len(teacher_traj)]
    student_pca = pca_result[len(teacher_traj):]
    
    # Interpolate student to match teacher length
    x_teacher = np.linspace(0, 1, len(teacher_pca))
    x_student = np.linspace(0, 1, len(student_pca))
    
    f_student_0 = interp1d(x_student, student_pca[:, 0], kind='cubic')
    f_student_1 = interp1d(x_student, student_pca[:, 1], kind='cubic')
    
    student_pca_interp = np.column_stack([
        f_student_0(x_teacher),
        f_student_1(x_teacher)
    ])
    
    plt.figure(figsize=(12, 10))
    
    # Create a custom colormap for the trajectory progression
    cmap_teacher = plt.cm.Blues(np.linspace(0.3, 1, len(teacher_pca)))
    cmap_student = plt.cm.Oranges(np.linspace(0.3, 1, len(student_pca_interp)))
    
    # Plot with color gradient to show direction
    for i in range(len(teacher_pca) - 1):
        plt.plot(teacher_pca[i:i+2, 0], teacher_pca[i:i+2, 1], 'b-', alpha=0.7,
                 color=cmap_teacher[i])
        
    for i in range(len(student_pca_interp) - 1):
        plt.plot(student_pca_interp[i:i+2, 0], student_pca_interp[i:i+2, 1], 'r-', alpha=0.7,
                 color=cmap_student[i])
    
    # Add markers for start and end points
    plt.scatter(teacher_pca[0, 0], teacher_pca[0, 1], c='blue', marker='o', s=100, label='Teacher start')
    plt.scatter(teacher_pca[-1, 0], teacher_pca[-1, 1], c='darkblue', marker='x', s=100, label='Teacher end')
    plt.scatter(student_pca_interp[0, 0], student_pca_interp[0, 1], c='orange', marker='o', s=100, label='Student start')
    plt.scatter(student_pca_interp[-1, 0], student_pca_interp[-1, 1], c='darkred', marker='x', s=100, label='Student end')
    
    # Add timestep markers
    step_size = max(1, len(teacher_pca) // 10)
    for i in range(0, len(teacher_pca), step_size):
        plt.annotate(f"{i}", (teacher_pca[i, 0], teacher_pca[i, 1]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
        
    step_size_student = max(1, len(student_pca) // 10)
    for i in range(0, len(student_pca), step_size_student):
        interp_idx = int(i * len(student_pca_interp) / len(student_pca))
        if interp_idx < len(student_pca_interp):
            plt.annotate(f"{i}", (student_pca_interp[interp_idx, 0], student_pca_interp[interp_idx, 1]), 
                        textcoords="offset points", xytext=(0,-10), ha='center')
    
    plt.title('PCA of Latent Trajectories: Teacher vs Student (Sample 0)')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.analysis_dir, 'dimensionality', 'pca_trajectories.png'))
    plt.close()
    
    # 2. t-SNE for a different perspective
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined)-1))
    tsne_result = tsne.fit_transform(combined)
    
    teacher_tsne = tsne_result[:len(teacher_traj)]
    student_tsne = tsne_result[len(teacher_traj):]
    
    plt.figure(figsize=(12, 10))
    
    # Plot with color gradient
    cmap_teacher = plt.cm.Blues(np.linspace(0.3, 1, len(teacher_tsne)))
    cmap_student = plt.cm.Oranges(np.linspace(0.3, 1, len(student_tsne)))
    
    for i in range(len(teacher_tsne) - 1):
        plt.plot(teacher_tsne[i:i+2, 0], teacher_tsne[i:i+2, 1], '-', alpha=0.7,
                 color=cmap_teacher[i])
        
    for i in range(len(student_tsne) - 1):
        plt.plot(student_tsne[i:i+2, 0], student_tsne[i:i+2, 1], '-', alpha=0.7,
                 color=cmap_student[i])
    
    # Add markers for start and end points
    plt.scatter(teacher_tsne[0, 0], teacher_tsne[0, 1], c='blue', marker='o', s=100, label='Teacher start')
    plt.scatter(teacher_tsne[-1, 0], teacher_tsne[-1, 1], c='darkblue', marker='x', s=100, label='Teacher end')
    plt.scatter(student_tsne[0, 0], student_tsne[0, 1], c='orange', marker='o', s=100, label='Student start')
    plt.scatter(student_tsne[-1, 0], student_tsne[-1, 1], c='darkred', marker='x', s=100, label='Student end')
    
    plt.title('t-SNE of Latent Trajectories: Teacher vs Student (Sample 0)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.analysis_dir, 'dimensionality', 'tsne_trajectories.png'))
    plt.close()
    
    # 3. UMAP visualization (more preservative of global structure than t-SNE)
    if umap is not None:
        reducer = umap.UMAP(random_state=42)
        umap_result = reducer.fit_transform(combined)
        
        teacher_umap = umap_result[:len(teacher_traj)]
        student_umap = umap_result[len(teacher_traj):]
        
        plt.figure(figsize=(12, 10))
        
        # Plot with color gradient
        cmap_teacher = plt.cm.Blues(np.linspace(0.3, 1, len(teacher_umap)))
        cmap_student = plt.cm.Oranges(np.linspace(0.3, 1, len(student_umap)))
        
        for i in range(len(teacher_umap) - 1):
            plt.plot(teacher_umap[i:i+2, 0], teacher_umap[i:i+2, 1], '-', alpha=0.7,
                     color=cmap_teacher[i])
            
        for i in range(len(student_umap) - 1):
            plt.plot(student_umap[i:i+2, 0], student_umap[i:i+2, 1], '-', alpha=0.7,
                     color=cmap_student[i])
        
        # Add markers for start and end points
        plt.scatter(teacher_umap[0, 0], teacher_umap[0, 1], c='blue', marker='o', s=100, label='Teacher start')
        plt.scatter(teacher_umap[-1, 0], teacher_umap[-1, 1], c='darkblue', marker='x', s=100, label='Teacher end')
        plt.scatter(student_umap[0, 0], student_umap[0, 1], c='orange', marker='o', s=100, label='Student start')
        plt.scatter(student_umap[-1, 0], student_umap[-1, 1], c='darkred', marker='x', s=100, label='Student end')
        
        plt.title('UMAP of Latent Trajectories: Teacher vs Student (Sample 0)')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(config.analysis_dir, 'dimensionality', 'umap_trajectories.png'))
        plt.close()