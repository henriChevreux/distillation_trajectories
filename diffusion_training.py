import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torchvision
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
        self.dataset = "CIFAR10"  # Using CIFAR10 instead of MNIST
        self.image_size = 16  # Reduced from 32x32 to 16x16 for faster training
        self.channels = 3  # CIFAR10 has RGB channels
        self.batch_size = 64
        
        # Model
        self.latent_dim = 64  # Increased for more complex dataset
        self.hidden_dims = [64, 128, 256]  # Larger network for RGB images
        
        # Diffusion process
        self.timesteps = 50  # Reduced from typical 1000 for faster computation
        self.beta_start = 1e-4
        self.beta_end = 0.02
        
        # Training
        self.epochs = 10
        self.lr = 1e-4
        self.save_interval = 1
        
        # Directories
        self.results_dir = "results"
        self.models_dir = "models"
        self.trajectory_dir = "trajectories"
        
        # Distillation
        self.distill = True
        self.teacher_steps = 50  # Original teacher model timesteps
        self.student_steps = 50  # Distilled student model timesteps
        
        # Student model size variants (scaling factors relative to teacher)
        # These define how much smaller the student models will be
        # Using fewer steps in the 0.1 to 1.0 range for exploratory analysis
        self.student_size_factors = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.25, 0.5, 0.75, 1.0]
        
        # Student model architecture variants
        # These define how many hidden dimensions to use in the student model
        # The teacher model uses [64, 128, 256], so these are simplified versions
        self.student_architectures = {
            'tiny': [32, 64],           # 2 layers instead of 3
            'small': [32, 64, 128],     # 3 layers but smaller dimensions
            'medium': [48, 96, 192],    # 3 layers with 75% of teacher dimensions
            'full': [64, 128, 256]      # Same as teacher
        }
        
    def create_directories(self):
        for dir_path in [self.results_dir, self.models_dir, self.trajectory_dir]:
            os.makedirs(dir_path, exist_ok=True)

# Utility functions
def extract(a, t, x_shape):
    """Extract coefficients at specified timesteps t"""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """
    Linear beta schedule as proposed in the original DDPM paper
    """
    scale = 1000 / timesteps  # Scale to match typical DDPM if using fewer steps
    return torch.linspace(beta_start, beta_end, timesteps)

# Set up diffusion parameters
def get_diffusion_params(timesteps, config=None):
    # Define beta schedule
    if config is not None:
        beta_start = config.beta_start
        beta_end = config.beta_end
    else:
        beta_start = 1e-4
        beta_end = 0.02
    
    betas = linear_beta_schedule(timesteps, beta_start, beta_end)
    
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
    
    # Move all tensors to the global device
    params = {
        'betas': betas.to(device),
        'alphas_cumprod': alphas_cumprod.to(device),
        'sqrt_recip_alphas': sqrt_recip_alphas.to(device),
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod.to(device),
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod.to(device),
        'posterior_variance': posterior_variance.to(device)
    }
    
    return params

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
        
        # Add a dedicated time projection layer for the middle features
        self.time_proj_mid = nn.Conv2d(32, config.hidden_dims[-1], 1)
        
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
        
        # Add time embedding - using the dedicated projection layer
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)
        x = x + self.time_proj_mid(t_emb.expand(-1, -1, x.shape[2], x.shape[3]))
        
        # Middle
        x = self.mid(x)
        
        # Upsampling with skip connections
        for up, res in zip(self.ups, reversed(residuals)):
            x = up(x)
            x = x + res
            
        # Output
        x = self.conv_out(x)
        return x

# Student U-Net with adjustable size and architecture
class StudentUNet(nn.Module):
    def __init__(self, config, size_factor=0.5, architecture_type=None):
        """
        Create a student model with a fraction of the teacher's capacity and potentially fewer layers
        
        Args:
            config: Configuration object
            size_factor: Factor to scale the hidden dimensions (0.5 = half the size)
            architecture_type: Type of architecture to use ('tiny', 'small', 'medium', 'full')
                               If None, will use the same architecture as teacher but scaled by size_factor
        """
        super().__init__()
        self.config = config
        
        # Determine the architecture to use
        if architecture_type is None:
            # Use the same architecture as teacher but scaled by size_factor
            self.hidden_dims = [max(8, int(dim * size_factor)) for dim in config.hidden_dims]
        else:
            # Use a predefined architecture with potentially fewer layers
            if architecture_type not in config.student_architectures:
                print(f"Warning: Architecture type '{architecture_type}' not found. Using 'tiny' instead.")
                architecture_type = 'tiny'
            
            # Get the base architecture and scale it by size_factor
            base_dims = config.student_architectures[architecture_type]
            self.hidden_dims = [max(8, int(dim * size_factor)) for dim in base_dims]
        
        # Print model size information
        print(f"Creating student model with size factor {size_factor}")
        print(f"Teacher hidden dims: {config.hidden_dims}")
        print(f"Student hidden dims: {self.hidden_dims}")
        
        # Initial projection
        self.conv_in = nn.Conv2d(config.channels, self.hidden_dims[0], 3, padding=1)
        
        # Downsampling path
        self.downs = nn.ModuleList([])
        for i in range(len(self.hidden_dims) - 1):
            self.downs.append(nn.Sequential(
                nn.Conv2d(self.hidden_dims[i], self.hidden_dims[i+1], 3, padding=1),
                nn.BatchNorm2d(self.hidden_dims[i+1]),
                nn.LeakyReLU(),
                nn.Conv2d(self.hidden_dims[i+1], self.hidden_dims[i+1], 3, padding=1),
                nn.BatchNorm2d(self.hidden_dims[i+1]),
                nn.LeakyReLU(),
                nn.MaxPool2d(2)
            ))
        
        # Middle
        mid_channels = self.hidden_dims[-1]
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
        for i in range(len(self.hidden_dims) - 1, 0, -1):
            self.ups.append(nn.Sequential(
                nn.ConvTranspose2d(self.hidden_dims[i], self.hidden_dims[i-1], 4, stride=2, padding=1),
                nn.BatchNorm2d(self.hidden_dims[i-1]),
                nn.LeakyReLU(),
                nn.Conv2d(self.hidden_dims[i-1], self.hidden_dims[i-1], 3, padding=1),
                nn.BatchNorm2d(self.hidden_dims[i-1]),
                nn.LeakyReLU()
            ))
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU()
        )
        
        # Add a dedicated time projection layer for the middle features
        self.time_proj_mid = nn.Conv2d(32, self.hidden_dims[-1], 1)
        
        # Output projection
        self.conv_out = nn.Conv2d(self.hidden_dims[0], config.channels, 3, padding=1)
        
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
        
        # Add time embedding - using the dedicated projection layer
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)
        x = x + self.time_proj_mid(t_emb.expand(-1, -1, x.shape[2], x.shape[3]))
        
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
    if config.dataset == "CIFAR10":
        # For CIFAR10, we need to normalize with the appropriate mean and std for RGB
        # Also resize from 32x32 to 16x16 for faster training
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((config.image_size, config.image_size)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_dataset = datasets.CIFAR10(
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
    elif config.dataset == "MNIST":
        # Also resize MNIST images to match the configured image size
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((config.image_size, config.image_size)),
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
    # Generate noisy image and get the noise that was used
    x_noisy, noise = q_sample(x_start, t, diffusion_params)
    
    # Predict the noise
    predicted_noise = denoise_model(x_noisy, t)
    
    # Calculate loss using the actual noise that was added
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
        if (epoch + 1) % config.save_interval == 0 or epoch == config.epochs - 1:
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
def distill_diffusion_model(teacher_model, config, teacher_params, student_params, size_factor=1.0):
    """
    Distill a diffusion model to use fewer timesteps with a potentially smaller architecture
    
    Args:
        teacher_model: The trained teacher model
        config: Configuration object
        teacher_params: Diffusion parameters for the teacher model
        student_params: Diffusion parameters for the student model
        size_factor: Factor to scale the student model size (0.25, 0.5, 0.75, 1.0)
    
    Returns:
        Trained student model
    """
    # Determine architecture type based on size factor
    architecture_type = None
    if size_factor < 0.1:
        architecture_type = 'tiny'     # Use the smallest architecture for very small models
    elif size_factor < 0.3:
        architecture_type = 'small'    # Use small architecture for small models
    elif size_factor < 0.7:
        architecture_type = 'medium'   # Use medium architecture for medium models
    else:
        architecture_type = 'full'     # Use full architecture for large models
    
    # Initialize student model with the specified size factor and architecture
    student_model = StudentUNet(config, size_factor=size_factor, architecture_type=architecture_type).to(device)
    
    print(f"Using architecture type: {architecture_type} for size factor {size_factor}")
    
    # Get the model size in MB
    def get_model_size(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb
    
    teacher_size = get_model_size(teacher_model)
    student_size = get_model_size(student_model)
    
    print(f"Teacher model size: {teacher_size:.2f} MB")
    print(f"Student model size: {student_size:.2f} MB ({student_size/teacher_size:.2%} of teacher)")
    
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
        if (epoch + 1) % config.save_interval == 0 or epoch == (config.epochs // 2) - 1:
            # Include size factor in the filename
            save_path = os.path.join(config.models_dir, f'student_model_size_{size_factor}_epoch_{epoch+1}.pt')
            print(f"Saving student model to: {save_path}")
            torch.save(student_model.state_dict(), save_path)
            
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
            plt.savefig(os.path.join(config.results_dir, f'student_samples_size_{size_factor}_epoch_{epoch+1}.png'))
            plt.close()
    
    return student_model

# Main function
def main(skip_teacher_training=False):
    # Create configuration
    global config
    config = Config()
    config.create_directories()
    
    # Create diffusion parameters for both teacher and student
    teacher_params = get_diffusion_params(config.teacher_steps)
    student_params = get_diffusion_params(config.student_steps)
    
    # Check if teacher model already exists
    teacher_model_path = os.path.join(config.models_dir, 'model_epoch_10.pt')
    
    if skip_teacher_training and os.path.exists(teacher_model_path):
        print(f"Loading existing teacher model from {teacher_model_path}...")
        teacher_model = SimpleUNet(config).to(device)
        teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=device))
        teacher_model.eval()
    else:
        print("Training teacher model...")
        teacher_model = train(config, teacher_params)
    
    if config.distill:
        print("Distilling knowledge to student models of different sizes...")
        
        # Train student models with different size factors
        student_models = {}
        for size_factor in config.student_size_factors:
            print(f"\nDistilling to student model with size factor {size_factor}...")
            student_model = distill_diffusion_model(
                teacher_model, 
                config, 
                teacher_params, 
                student_params,
                size_factor=size_factor
            )
            student_models[size_factor] = student_model
        
        print("Training and distillation complete. Models saved in the models directory.")

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train diffusion models with knowledge distillation')
    parser.add_argument('--train_teacher', action='store_true',
                        help='Train the teacher model even if it already exists')
    args = parser.parse_args()
    
    # Run main function with skip_teacher_training=True by default
    main(skip_teacher_training=not args.train_teacher)
