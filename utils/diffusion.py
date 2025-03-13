import torch
import torch.nn.functional as F
import math
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def extract(a, t, x_shape):
    """Extract coefficients at specified timesteps t"""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def get_diffusion_params(timesteps, config=None):
    """Set up diffusion parameters with cosine schedule"""
    device = torch.device(
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() and (config.mps_enabled if config else False) else
        "cpu"
    )
    
    # Use cosine schedule
    betas = cosine_beta_schedule(timesteps)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    
    # Move tensors to device with error handling
    params = {}
    tensors_to_move = {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_recip_alphas': sqrt_recip_alphas,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'posterior_variance': posterior_variance
    }
    
    try:
        for name, tensor in tensors_to_move.items():
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            params[name] = tensor.to(device)
    except RuntimeError as e:
        print(f"\nWarning: Failed to move tensors to {device}. Using CPU.")
        params = tensors_to_move
    
    return params

def q_sample(x_start, t, diffusion_params, noise=None):
    """Forward diffusion process with optional noise input"""
    if noise is None:
        noise = torch.randn_like(x_start)
    
    sqrt_alphas_cumprod_t = extract(diffusion_params['sqrt_alphas_cumprod'], t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        diffusion_params['sqrt_one_minus_alphas_cumprod'], t, x_start.shape
    )
    
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise

def p_losses(denoise_model, x_start, t, diffusion_params, noise=None):
    """Calculate loss with SNR weighting as in the improved DDPM paper"""
    # Generate noisy image
    x_noisy, noise = q_sample(x_start, t, diffusion_params, noise)
    
    # Predict noise
    predicted_noise = denoise_model(x_noisy, t)
    
    # Calculate SNR-weighted loss
    snr = extract(diffusion_params['alphas_cumprod'] / (1 - diffusion_params['alphas_cumprod']), t, x_start.shape)
    weight = snr / (1 + snr)  # Normalized SNR weight
    
    # MSE loss with SNR weighting
    loss = F.mse_loss(predicted_noise, noise, reduction='none')
    loss = loss * weight
    return loss.mean()

@torch.no_grad()
def p_sample(model, x, t, t_index, diffusion_params):
    """Sample from p(x_{t-1} | x_t) with improved sampling"""
    betas_t = extract(diffusion_params['betas'], t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        diffusion_params['sqrt_one_minus_alphas_cumprod'], t, x.shape
    )
    sqrt_recip_alphas_t = extract(diffusion_params['sqrt_recip_alphas'], t, x.shape)
    
    # Predict noise
    predicted_noise = model(x, t)
    
    # Calculate mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
    )
    
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(diffusion_params['posterior_variance'], t, x.shape)
        noise = torch.randn_like(x)
        # Add noise scaled by the posterior variance
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, shape, timesteps, diffusion_params, device=None, config=None, track_trajectory=False):
    """Generate samples with optional trajectory tracking"""
    if device is None:
        device = next(model.parameters()).device
    
    # Start from pure noise
    img = torch.randn(shape, device=device)
    trajectory = [img.detach().cpu()] if track_trajectory else None
    
    # Progress bar settings
    from tqdm import tqdm
    progress_bar_leave = getattr(config, 'progress_bar_leave', False) if config else False
    progress_bar_position = getattr(config, 'progress_bar_position', 0) if config else 0
    
    # Iteratively denoise
    for i in tqdm(reversed(range(0, timesteps)), 
                 desc='Sampling', 
                 total=timesteps,
                 leave=progress_bar_leave,
                 position=progress_bar_position):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, i, diffusion_params)
        
        if track_trajectory:
            trajectory.append(img.detach().cpu())
    
    return (img, trajectory) if track_trajectory else img
