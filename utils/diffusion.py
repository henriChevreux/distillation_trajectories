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
    
    # Ensure t is within bounds
    t = torch.clamp(t, 0, a.shape[0] - 1)
    
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """Linear beta schedule as proposed in the original DDPM paper"""
    return torch.linspace(beta_start, beta_end, timesteps)

def get_diffusion_params(sample_steps, config=None):
    """Set up diffusion parameters
    
    Args:
        sample_steps: Number of sample steps in the diffusion process (typically 4000)
        config: Configuration object
    """
    # Define beta schedule
    beta_start = config.beta_start if config else 1e-4
    beta_end = config.beta_end if config else 0.02
    
    betas = linear_beta_schedule(sample_steps, beta_start, beta_end)
    
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
    device = torch.device("cpu") if (config and hasattr(config, 'force_cpu') and config.force_cpu) else torch.device(
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

def p_losses(denoise_model, x_start, t, diffusion_params, cond=None):
    """
    Calculate loss for training the denoising model
    
    Args:
        denoise_model: The diffusion model
        x_start: Starting images
        t: Timesteps
        diffusion_params: Diffusion parameters
        cond: Conditioning tensor (optional)
    """
    # Generate noisy image and get the noise that was used
    x_noisy, noise = q_sample(x_start, t, diffusion_params)
    
    # Predict the noise with conditioning if provided
    predicted_noise = denoise_model(x_noisy, t, cond)
    
    # Calculate loss using the actual noise that was added
    loss = F.mse_loss(predicted_noise, noise)
    return loss

@torch.no_grad()
def p_sample(model, x, t, t_index, diffusion_params, guidance_scale=1.0):
    """
    Sample from p(x_{t-1} | x_t) - single step denoising with classifier-free guidance
    
    Args:
        model: The diffusion model
        x: Current noisy image
        t: Current timestep
        t_index: Index of current timestep
        diffusion_params: Diffusion parameters
        guidance_scale: Scale factor for classifier-free guidance (1.0 means no guidance)
    """
    betas_t = extract(diffusion_params['betas'], t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        diffusion_params['sqrt_one_minus_alphas_cumprod'], t, x.shape
    )
    sqrt_recip_alphas_t = extract(diffusion_params['sqrt_recip_alphas'], t, x.shape)
    
    # Get model predictions for both conditional and unconditional
    cond_output = model(x, t, cond=torch.ones(x.shape[0], 1, device=x.device))
    uncond_output = model(x, t, cond=None)
    
    # Apply classifier-free guidance
    model_output = uncond_output + guidance_scale * (cond_output - uncond_output)
    
    # Ensure model output has the same size as input
    if model_output.shape != x.shape:
        try:
            model_output = torch.nn.functional.interpolate(
                model_output, 
                size=(x.shape[2], x.shape[3]),
                mode='bilinear', 
                align_corners=True
            )
        except Exception as e:
            if model_output.dim() == x.dim():
                if model_output.shape[2] < x.shape[2] or model_output.shape[3] < x.shape[3]:
                    pad_h = max(0, x.shape[2] - model_output.shape[2])
                    pad_w = max(0, x.shape[3] - model_output.shape[3])
                    model_output = torch.nn.functional.pad(model_output, (0, pad_w, 0, pad_h))
                else:
                    model_output = model_output[:, :, :x.shape[2], :x.shape[3]]
            else:
                raise ValueError(f"Cannot reconcile model output shape {model_output.shape} with input shape {x.shape}")
    
    # Previous mean
    pred_original = model_output
    
    # Direction pointing to x_t
    pred_original_direction = (1. - sqrt_one_minus_alphas_cumprod_t) * pred_original
    
    # Random noise
    noise = torch.randn_like(x) if t_index > 0 else 0.
    
    # Final sample
    return sqrt_recip_alphas_t * (x - pred_original_direction) + noise * betas_t

@torch.no_grad()
def p_sample_loop(model, shape, sample_steps, diffusion_params, device=None, config=None, track_trajectory=False, guidance_scale=1.0):
    """
    Generate samples by iteratively denoising from pure noise with classifier-free guidance
    
    Args:
        model: The diffusion model
        shape: Shape of the samples to generate
        sample_steps: Number of sample steps in the diffusion process
        diffusion_params: Parameters for the diffusion process
        device: Device to use
        config: Configuration object
        track_trajectory: Whether to track and return the trajectory
        guidance_scale: Scale factor for classifier-free guidance (1.0 means no guidance)
    """
    if device is None:
        device = next(model.parameters()).device
    b = shape[0]
    
    # Start from pure noise
    img = torch.randn(shape, device=device)
    
    if track_trajectory:
        trajectory = [img.detach().cpu()]
    
    # Iteratively denoise with guidance
    from tqdm import tqdm
    progress_bar_leave = False
    progress_bar_position = 0
    
    if config:
        progress_bar_leave = getattr(config, 'progress_bar_leave', False)
        progress_bar_position = getattr(config, 'progress_bar_position', 0)
    
    num_timesteps = config.timesteps if config else sample_steps
    step_size = max(1, sample_steps // num_timesteps)
    timestep_indices = [min(i * step_size, sample_steps - 1) for i in range(num_timesteps)]
    timestep_indices = sorted(list(set(timestep_indices)), reverse=True)
    
    for i in tqdm(timestep_indices, 
                 desc='Sampling time steps', 
                 total=len(timestep_indices), 
                 leave=progress_bar_leave, 
                 position=progress_bar_position):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, i, diffusion_params, guidance_scale=guidance_scale)
        
        if track_trajectory:
            trajectory.append(img.detach().cpu())
    
    if track_trajectory:
        return img, trajectory
    return img
