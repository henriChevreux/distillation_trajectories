import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Set device - use MPS if available on newer macOS, otherwise CPU
# But add a safety flag to fall back to CPU if MPS causes issues
use_mps = False  # Set this to False to force CPU usage regardless of MPS availability

if torch.backends.mps.is_available() and use_mps:
    device = torch.device("mps")
    print("Using MPS acceleration")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA acceleration")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Configuration
class Config:
    def __init__(self):
        # Dataset
        self.dataset = "MNIST"  # Simple dataset to work with on limited resources
        self.image_size = 28
        self.channels = 1
        self.batch_size = 64
        
        # Model
        self.latent_dim = 32
        self.hidden_dims = [32, 64, 128]  # Smaller network for MacBook
        
        # Diffusion process
        self.timesteps = 50  # Reduced from typical 1000 for faster computation
        self.beta_start = 1e-4
        self.beta_end = 0.02
        
        # Training
        self.epochs = 20
        self.lr = 1e-4
        self.save_interval = 5
        
        # Directories
        self.results_dir = "results"
        self.models_dir = "models"
        self.trajectory_dir = "trajectories"
        
        # Distillation
        self.distill = True
        self.teacher_steps = 50  # Original teacher model timesteps
        self.student_steps = 10  # Distilled student model timesteps
        
    def create_directories(self):
        for dir_path in [self.results_dir, self.models_dir, self.trajectory_dir]:
            os.makedirs(dir_path, exist_ok=True)

config = Config()
config.create_directories()

# Utility functions
def extract(a, t, x_shape):
    """Extract coefficients at specified timesteps t"""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    Linear beta schedule as proposed in the original DDPM paper
    """
    scale = 1000 / timesteps  # Scale to match typical DDPM if using fewer steps
    beta_start = config.beta_start
    beta_end = config.beta_end
    return torch.linspace(beta_start, beta_end, timesteps)

# Set up diffusion parameters
def get_diffusion_params(timesteps):
    # Define beta schedule
    betas = linear_beta_schedule(timesteps)
    
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

# Prepare dataset
def get_data_loader(config):
    if config.dataset == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        train_dataset = datasets.MNIST(
            root='data',
            train=True,
            download=True,
            transform=transform
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )
        
        return train_loader

# Forward diffusion process
def q_sample(x_start, t, diffusion_params):
    """
    Forward diffusion process: Add noise to the input according to timestep t
    """
    noise = torch.randn_like(x_start)
    
    sqrt_alphas_cumprod_t = extract(diffusion_params['sqrt_alphas_cumprod'], t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        diffusion_params['sqrt_one_minus_alphas_cumprod'], t, x_start.shape
    )
    
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise

# Loss function for training
def p_losses(denoise_model, x_start, t, diffusion_params, noise=None):
    """
    Calculate loss for training the denoising model
    """
    if noise is None:
        noise = torch.randn_like(x_start)
        
    x_noisy, _ = q_sample(x_start, t, diffusion_params)
    predicted_noise = denoise_model(x_noisy, t)
    
    loss = F.mse_loss(predicted_noise, noise)
    return loss

# Sampling from the model (denoising process)
@torch.no_grad()
def p_sample(model, x, t, t_index, diffusion_params):
    """
    Sample from p(x_{t-1} | x_t) - single step denoising
    """
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
def p_sample_loop(model, shape, timesteps, diffusion_params, track_trajectory=False):
    """
    Generate samples by iteratively denoising from pure noise
    """
    device = next(model.parameters()).device
    b = shape[0]
    
    # Start from pure noise
    img = torch.randn(shape, device=device)
    
    if track_trajectory:
        # Track the trajectory through latent space
        trajectory = [img.detach().cpu()]
    
    # Iteratively denoise
    for i in tqdm(reversed(range(0, timesteps)), desc='Sampling time steps', total=timesteps):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, i, diffusion_params)
        
        if track_trajectory:
            trajectory.append(img.detach().cpu())
    
    if track_trajectory:
        return img, trajectory
    return img

# Training function
def train(config, diffusion_params):
    # Initialize model
    model = SimpleUNet(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    # Get data loader
    train_loader = get_data_loader(config)
    
    # Training loop
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs}')
        for batch_idx, (images, _) in enumerate(progress_bar):
            images = images.to(device)
            optimizer.zero_grad()
            
            # Sample random timesteps
            t = torch.randint(0, config.timesteps, (images.shape[0],), device=device).long()
            
            # Calculate loss
            loss = p_losses(model, images, t, diffusion_params)
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss/(batch_idx+1))
        
        # Save model periodically
        if (epoch + 1) % config.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(config.models_dir, f'model_epoch_{epoch+1}.pt'))
            
            # Generate some samples
            model.eval()
            samples = p_sample_loop(
                model, 
                shape=(16, config.channels, config.image_size, config.image_size),
                timesteps=config.timesteps,
                diffusion_params=diffusion_params
            )
            
            # Save samples
            grid = (samples + 1) / 2
            grid = torch.clamp(grid, 0, 1)
            grid = torchvision.utils.make_grid(grid, nrow=4)
            plt.figure(figsize=(10, 10))
            plt.imshow(grid.permute(1, 2, 0).cpu())
            plt.axis('off')
            plt.savefig(os.path.join(config.results_dir, f'samples_epoch_{epoch+1}.png'))
            plt.close()
    
    return model

# Knowledge distillation for diffusion models
def distill_diffusion_model(teacher_model, config, teacher_params, student_params):
    """
    Distill a diffusion model to use fewer timesteps
    """
    # Initialize student model with the same architecture
    student_model = SimpleUNet(config).to(device)
    
    # Optionally initialize with teacher weights
    student_model.load_state_dict(teacher_model.state_dict())
    
    # Optimizer for the student
    optimizer = optim.Adam(student_model.parameters(), lr=config.lr)
    
    # Get data loader
    train_loader = get_data_loader(config)
    
    # Prepare timestep conversion from teacher to student
    convert_t = lambda t_teacher: torch.floor(t_teacher * (config.student_steps / config.teacher_steps)).long()
    
    # Training loop
    for epoch in range(config.epochs // 2):  # Fewer epochs for distillation
        student_model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Distillation Epoch {epoch+1}/{config.epochs//2}')
        for batch_idx, (images, _) in enumerate(progress_bar):
            images = images.to(device)
            optimizer.zero_grad()
            
            # Sample random timesteps for teacher model
            t_teacher = torch.randint(0, config.teacher_steps, (images.shape[0],), device=device).long()
            
            # Convert to student timesteps
            t_student = convert_t(t_teacher)
            
            # Create noisy images based on the teacher's diffusion process
            with torch.no_grad():
                x_noisy, noise = q_sample(images, t_teacher, teacher_params)
                # Get teacher's predicted noise
                teacher_pred = teacher_model(x_noisy, t_teacher)
            
            # Student tries to match teacher's prediction
            student_pred = student_model(x_noisy, t_student)
            
            # MSE loss between student and teacher predictions
            loss = F.mse_loss(student_pred, teacher_pred)
            
            # Also add some loss against the true noise to maintain diversity
            loss += 0.1 * F.mse_loss(student_pred, noise)
            
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss/(batch_idx+1))
        
        # Save model periodically
        if (epoch + 1) % config.save_interval == 0:
            torch.save(student_model.state_dict(), 
                       os.path.join(config.models_dir, f'student_model_epoch_{epoch+1}.pt'))
            
            # Generate some samples
            student_model.eval()
            samples = p_sample_loop(
                student_model, 
                shape=(16, config.channels, config.image_size, config.image_size),
                timesteps=config.student_steps,
                diffusion_params=student_params
            )
            
            # Save samples
            grid = (samples + 1) / 2
            grid = torch.clamp(grid, 0, 1)
            grid = torchvision.utils.make_grid(grid, nrow=4)
            plt.figure(figsize=(10, 10))
            plt.imshow(grid.permute(1, 2, 0).cpu())
            plt.axis('off')
            plt.savefig(os.path.join(config.results_dir, f'student_samples_epoch_{epoch+1}.png'))
            plt.close()
    
    return student_model

# Compare latent trajectories between teacher and student models
def compare_trajectories(teacher_model, student_model, config, teacher_params, student_params):
    """
    Compare the latent space trajectories of teacher and student models
    """
    teacher_model.eval()
    student_model.eval()
    
    # Generate samples with trajectory tracking
    shape = (8, config.channels, config.image_size, config.image_size)
    
    # Use the same random seed for fair comparison
    torch.manual_seed(42)
    _, teacher_trajectory = p_sample_loop(
        teacher_model, 
        shape=shape,
        timesteps=config.teacher_steps,
        diffusion_params=teacher_params,
        track_trajectory=True
    )
    
    torch.manual_seed(42)
    _, student_trajectory = p_sample_loop(
        student_model, 
        shape=shape,
        timesteps=config.student_steps,
        diffusion_params=student_params,
        track_trajectory=True
    )
    
    # Normalize the trajectories by the number of steps
    # This allows us to compare trajectories of different lengths
    teacher_milestones = np.linspace(0, len(teacher_trajectory)-1, 10).astype(int)
    student_milestones = np.linspace(0, len(student_trajectory)-1, 10).astype(int)
    
    # Visualize and compare trajectories
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    
    for i, idx in enumerate(teacher_milestones[:5]):
        img = (teacher_trajectory[idx][0] + 1) / 2
        img = torch.clamp(img, 0, 1)
        axs[0, i].imshow(img.permute(1, 2, 0).numpy())
        axs[0, i].set_title(f"Teacher t={idx}")
        axs[0, i].axis('off')
    
    for i, idx in enumerate(student_milestones[:5]):
        img = (student_trajectory[idx][0] + 1) / 2
        img = torch.clamp(img, 0, 1)
        axs[1, i].imshow(img.permute(1, 2, 0).numpy())
        axs[1, i].set_title(f"Student t={idx}")
        axs[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.trajectory_dir, 'model_comparison.png'))
    
    # Analyze trajectory differences
    teacher_flattened = [traj[0].reshape(-1).numpy() for traj in teacher_trajectory]
    student_flattened = [traj[0].reshape(-1).numpy() for traj in student_trajectory]
    
    # Interpolate student trajectory to match teacher length for comparison
    student_interp = []
    for i in range(len(teacher_flattened)):
        # Find equivalent timestep in student trajectory
        student_idx = int(i * (len(student_flattened) / len(teacher_flattened)))
        if student_idx >= len(student_flattened):
            student_idx = len(student_flattened) - 1
        student_interp.append(student_flattened[student_idx])
    
    # Calculate distances in latent space
    distances = [np.linalg.norm(t - s) for t, s in zip(teacher_flattened, student_interp)]
    
    # Plot distances
    plt.figure(figsize=(10, 5))
    plt.plot(distances)
    plt.title('L2 Distance Between Teacher and Student Trajectories')
    plt.xlabel('Timestep')
    plt.ylabel('L2 Distance')
    plt.grid(True)
    plt.savefig(os.path.join(config.trajectory_dir, 'trajectory_distances.png'))
    
    return teacher_trajectory, student_trajectory

# Main function
def main():
    # Create diffusion parameters for both teacher and student
    teacher_params = get_diffusion_params(config.teacher_steps)
    student_params = get_diffusion_params(config.student_steps)
    
    print("Training teacher model...")
    teacher_model = train(config, teacher_params)
    
    if config.distill:
        print("Distilling knowledge to student model...")
        student_model = distill_diffusion_model(teacher_model, config, teacher_params, student_params)
        
        print("Comparing latent trajectories...")
        teacher_trajectory, student_trajectory = compare_trajectories(
            teacher_model, student_model, config, teacher_params, student_params
        )
        
        # Additional analysis: PCA visualization of trajectories
        from sklearn.decomposition import PCA
        
        teacher_flat = np.array([t[0].reshape(-1).numpy() for t in teacher_trajectory])
        student_flat = np.array([s[0].reshape(-1).numpy() for s in student_trajectory])
        
        # Combine for PCA
        combined = np.vstack([teacher_flat, student_flat])
        
        # Apply PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(combined)
        
        # Split back
        teacher_pca = pca_result[:len(teacher_flat)]
        student_pca = pca_result[len(teacher_flat):]
        
        # Interpolate student to match teacher length
        from scipy.interpolate import interp1d
        
        x_teacher = np.linspace(0, 1, len(teacher_pca))
        x_student = np.linspace(0, 1, len(student_pca))
        
        f_student_0 = interp1d(x_student, student_pca[:, 0], kind='linear')
        f_student_1 = interp1d(x_student, student_pca[:, 1], kind='linear')
        
        student_pca_interp = np.column_stack([
            f_student_0(x_teacher),
            f_student_1(x_teacher)
        ])
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.plot(teacher_pca[:, 0], teacher_pca[:, 1], 'b-', label='Teacher')
        plt.plot(student_pca_interp[:, 0], student_pca_interp[:, 1], 'r-', label='Student (interpolated)')
        plt.scatter(teacher_pca[0, 0], teacher_pca[0, 1], c='b', marker='o', s=100, label='Teacher start')
        plt.scatter(teacher_pca[-1, 0], teacher_pca[-1, 1], c='b', marker='x', s=100, label='Teacher end')
        plt.scatter(student_pca_interp[0, 0], student_pca_interp[0, 1], c='r', marker='o', s=100, label='Student start')
        plt.scatter(student_pca_interp[-1, 0], student_pca_interp[-1, 1], c='r', marker='x', s=100, label='Student end')
        
        # Add arrows to show direction
        for i in range(0, len(teacher_pca), len(teacher_pca)//10):
            if i+1 < len(teacher_pca):
                plt.arrow(teacher_pca[i, 0], teacher_pca[i, 1], 
                          teacher_pca[i+1, 0]-teacher_pca[i, 0], teacher_pca[i+1, 1]-teacher_pca[i, 1],
                          head_width=0.1, head_length=0.1, fc='b', ec='b', alpha=0.5)
        
        for i in range(0, len(student_pca_interp), len(student_pca_interp)//10):
            if i+1 < len(student_pca_interp):
                plt.arrow(student_pca_interp[i, 0], student_pca_interp[i, 1], 
                          student_pca_interp[i+1, 0]-student_pca_interp[i, 0], 
                          student_pca_interp[i+1, 1]-student_pca_interp[i, 1],
                          head_width=0.1, head_length=0.1, fc='r', ec='r', alpha=0.5)
        
        plt.title('PCA of Latent Trajectories: Teacher vs Student')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(config.trajectory_dir, 'pca_trajectories.png'))
        
        print("Analysis complete. Results saved in the results and trajectories directories.")

if __name__ == "__main__":
    import torchvision
    main()