import torch
import torch.nn.functional as F
import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config

def extract(a, t, x_shape):
    """Extract coefficients at specified timesteps t"""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """Linear beta schedule as proposed in the original DDPM paper"""
    return torch.linspace(beta_start, beta_end, timesteps)

def get_diffusion_params(timesteps, config=None):
    """Set up diffusion parameters"""
    # Define beta schedule
    beta_start = config.beta_start if config else 1e-4
    beta_end = config.beta_end if config else 0.02
    
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
    
    # Get device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() and (config.mps_enabled if config else False) else
        "cpu"
    )
    
    # Move all tensors to device
    return {
        'betas': betas.to(device),
        'alphas_cumprod': alphas_cumprod.to(device),
        'sqrt_recip_alphas': sqrt_recip_alphas.to(device),
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod.to(device),
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod.to(device),
        'posterior_variance': posterior_variance.to(device)
    }

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
    
    # Get model prediction
    model_output = model(x, t)
    
    # Ensure model output has the same size as input
    if model_output.shape != x.shape:
        print(f"Size mismatch: model output {model_output.shape} != input {x.shape}")
        try:
            # Try to use interpolation to resize the model output
            model_output = torch.nn.functional.interpolate(
                model_output, 
                size=(x.shape[2], x.shape[3]),
                mode='bilinear', 
                align_corners=True
            )
            print(f"Resized model output to {model_output.shape}")
        except Exception as e:
            print(f"Error resizing model output: {e}")
            # If interpolation fails, try a more direct approach
            if model_output.dim() == x.dim():
                # If dimensions match but sizes don't, try to pad or crop
                if model_output.shape[2] < x.shape[2] or model_output.shape[3] < x.shape[3]:
                    # Pad if model output is smaller
                    pad_h = max(0, x.shape[2] - model_output.shape[2])
                    pad_w = max(0, x.shape[3] - model_output.shape[3])
                    model_output = torch.nn.functional.pad(model_output, (0, pad_w, 0, pad_h))
                else:
                    # Crop if model output is larger
                    model_output = model_output[:, :, :x.shape[2], :x.shape[3]]
            else:
                # If dimensions don't match, this is a more serious issue
                raise ValueError(f"Cannot reconcile model output shape {model_output.shape} with input shape {x.shape}")
    
    # Equation 11 in the DDPM paper
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
    )
    
    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(diffusion_params['posterior_variance'], t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def p_sample_loop(model, shape, timesteps, diffusion_params, device=None, config=None, track_trajectory=False):
    """
    Generate samples by iteratively denoising from pure noise
    
    Args:
        model: The diffusion model
        shape: Shape of the samples to generate
        timesteps: Number of timesteps in the diffusion process
        diffusion_params: Parameters for the diffusion process
        device: Device to use (if None, will use the device of the model)
        config: Configuration object (optional)
        track_trajectory: Whether to track and return the trajectory
    
    Returns:
        Generated samples, and optionally the trajectory
    """
    if device is None:
        device = next(model.parameters()).device
    b = shape[0]
    
    # Start from pure noise
    img = torch.randn(shape, device=device)
    
    if track_trajectory:
        # Track the trajectory through latent space
        trajectory = [img.detach().cpu()]
    
    # Iteratively denoise
    from tqdm import tqdm
    # Get progress bar settings from config if available
    progress_bar_leave = False
    progress_bar_position = 0
    
    if config:
        progress_bar_leave = getattr(config, 'progress_bar_leave', False)
        progress_bar_position = getattr(config, 'progress_bar_position', 0)
    
    for i in tqdm(reversed(range(0, timesteps)), 
                 desc='Sampling time steps', 
                 total=timesteps, 
                 leave=progress_bar_leave, 
                 position=progress_bar_position):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, i, diffusion_params)
        
        if track_trajectory:
            trajectory.append(img.detach().cpu())
    
    if track_trajectory:
        return img, trajectory
    return img
