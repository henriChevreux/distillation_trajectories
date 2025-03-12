#!/usr/bin/env python3
"""
Script to train a Consistency Trajectory Model (CTM).
"""

import os
import sys
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# Add the project root to the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config
from models import ConsistencyTrajectoryModel, SimpleUNet
from utils.diffusion import q_sample
from data.dataset import get_data_loader

# Replace the import with our own implementation
def get_diffusion_params(timesteps, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """Initialize diffusion parameters (betas, alphas, etc.)"""
    # Linear beta schedule
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
    
    # Calculations from original DDPM implementation
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), alphas_cumprod[:-1]])
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    
    # Return all parameters as a dictionary
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance,
    }

def train_ctm(config, model, teacher_model, train_loader, optimizer, diffusion_params, device, 
             epoch, writer=None, trajectory_prob=0.5, max_time_diff=0.5):
    """
    Train a CTM model for one epoch.
    
    Args:
        config: Configuration object
        model: CTM model to train
        teacher_model: Teacher model for trajectory supervision
        train_loader: DataLoader for training data
        optimizer: Optimizer for model parameters
        diffusion_params: Diffusion parameters
        device: Device to use
        epoch: Current epoch number
        writer: TensorBoard SummaryWriter
        trajectory_prob: Probability of training with trajectory mode (vs. standard mode)
        max_time_diff: Maximum time difference for trajectory training
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    if teacher_model is not None:
        teacher_model.eval()
    
    total_loss = 0
    score_losses = 0
    consistency_losses = 0
    trajectory_steps = 0
    standard_steps = 0
    
    # Track per-component losses
    epoch_score_loss = 0
    epoch_consistency_loss = 0
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Extract images from batch
        if isinstance(batch, (list, tuple)):
            x = batch[0].to(device)
        else:
            x = batch.to(device)
        
        # Ensure values are in [-1, 1]
        if x.max() > 1.0:
            x = x / 127.5 - 1
        
        batch_size = x.shape[0]
        optimizer.zero_grad()
        
        # Sample timestep t
        t = torch.randint(0, config.timesteps, (batch_size,), device=device)
        
        # Add noise to inputs
        noise = torch.randn_like(x)
        
        # Get noise scaling factors for timestep t
        sqrt_alpha_cumprod = diffusion_params["sqrt_alphas_cumprod"][t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = diffusion_params["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1, 1)
        
        # Add noise to images
        x_t = sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * noise
        
        # Randomly decide whether to use trajectory mode or standard mode
        use_trajectory = torch.rand(1).item() < trajectory_prob
        
        if use_trajectory:
            # Sample an end timestep
            # 50% of the time, set t_end=0 (predict clean image)
            # 50% of the time, sample a random timestep between 0 and t
            if torch.rand(1).item() < 0.5:
                t_end = torch.zeros_like(t)
            else:
                # Sample t_end < t for each item in batch
                t_ratios = torch.rand(batch_size, device=device) * max_time_diff  # Limit how far ahead we predict
                t_end = (t.float() * (1 - t_ratios)).long()
            
            # If teacher model is available, use it for supervision
            if teacher_model is not None:
                with torch.no_grad():
                    # Get "ground truth" denoised samples from teacher by simulating the ODE solution
                    # In a real implementation, this would use a proper ODE solver
                    # For simplicity, we're just letting the teacher predict directly
                    teacher_out = teacher_model(x_t, t_end)
                    # In a real implementation, you might have teacher_out be a numpy array or something
                    # But here we are producing it from the PyTorch model
                    teacher_denoised = teacher_out  # This would be the "correct" next point in the trajectory
            else:
                # If no teacher, we'll just use the original x (not ideal, but makes the script runnable)
                teacher_denoised = x
            
            # Forward pass in trajectory mode
            pred = model(x_t, t, t_end)
            
            # Compute loss components
            score_loss = F.mse_loss(pred['score'], noise)
            consistency_loss = F.mse_loss(pred['sample'], teacher_denoised)
            
            # Combine losses
            lambda_score = 1.0
            lambda_consistency = 1.0
            loss = lambda_score * score_loss + lambda_consistency * consistency_loss
            
            # Track component losses
            epoch_score_loss += score_loss.item()
            epoch_consistency_loss += consistency_loss.item()
            trajectory_steps += 1
            
        else:
            # Standard diffusion training (no trajectory, just score prediction)
            pred = model(x_t, t)
            
            # Score matching loss
            loss = F.mse_loss(pred['score'], noise)
            
            # Track component losses
            epoch_score_loss += loss.item()
            standard_steps += 1
        
        # Update model
        loss.backward()
        optimizer.step()
        
        # Track loss
        total_loss += loss.item()
        
        # Update progress bar
        if batch_idx % 10 == 0:
            if use_trajectory:
                pbar.set_postfix({
                    'loss': loss.item(),
                    'score': score_loss.item(),
                    'consist': consistency_loss.item(),
                    'mode': 'trajectory'
                })
            else:
                pbar.set_postfix({
                    'loss': loss.item(),
                    'mode': 'standard'
                })
        
        # Log to TensorBoard
        if writer is not None and batch_idx % 100 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/loss', loss.item(), global_step)
            if use_trajectory:
                writer.add_scalar('train/score_loss', score_loss.item(), global_step)
                writer.add_scalar('train/consistency_loss', consistency_loss.item(), global_step)
    
    # Calculate average losses
    avg_loss = total_loss / len(train_loader)
    
    # Calculate average component losses
    avg_score_loss = epoch_score_loss / len(train_loader)
    avg_consistency_loss = epoch_consistency_loss / trajectory_steps if trajectory_steps > 0 else 0
    
    # Log epoch metrics
    if writer is not None:
        writer.add_scalar('epoch/loss', avg_loss, epoch)
        writer.add_scalar('epoch/score_loss', avg_score_loss, epoch)
        writer.add_scalar('epoch/consistency_loss', avg_consistency_loss, epoch)
        writer.add_scalar('epoch/trajectory_ratio', trajectory_steps / (trajectory_steps + standard_steps), epoch)
    
    return avg_loss

def evaluate_ctm(config, model, test_loader, diffusion_params, device, epoch, writer=None):
    """Evaluate a CTM model on a test dataset."""
    model.eval()
    
    total_score_loss = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Extract images from batch
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)
            
            # Ensure values are in [-1, 1]
            if x.max() > 1.0:
                x = x / 127.5 - 1
            
            batch_size = x.shape[0]
            
            # Sample timestep t
            t = torch.randint(0, config.timesteps, (batch_size,), device=device)
            
            # Add noise to inputs
            noise = torch.randn_like(x)
            
            # Get noise scaling factors for timestep t
            sqrt_alpha_cumprod = diffusion_params["sqrt_alphas_cumprod"][t].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_cumprod = diffusion_params["sqrt_one_minus_alphas_cumprod"][t].view(-1, 1, 1, 1)
            
            # Add noise to images
            x_t = sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * noise
            
            # Forward pass (standard mode for evaluation)
            pred = model(x_t, t)
            
            # Compute score loss
            score_loss = F.mse_loss(pred['score'], noise)
            
            # Track loss
            total_score_loss += score_loss.item()
    
    # Calculate average loss
    avg_score_loss = total_score_loss / len(test_loader)
    
    # Log to TensorBoard
    if writer is not None:
        writer.add_scalar('val/score_loss', avg_score_loss, epoch)
    
    return avg_score_loss

def main():
    parser = argparse.ArgumentParser(description="Train a Consistency Trajectory Model (CTM)")
    parser.add_argument("--size", type=float, default=0.2, help="Model size factor")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--output", type=str, default="output/ctm_models", help="Output directory")
    parser.add_argument("--teacher-path", type=str, default="output/models/teacher/model_epoch_1.pt", 
                        help="Path to teacher model (if not provided, will use simplified supervision)")
    parser.add_argument("--trajectory-prob", type=float, default=0.7,
                        help="Probability of using trajectory mode during training (higher = better trajectories)")
    parser.add_argument("--max-time-diff", type=float, default=0.5,
                        help="Maximum time difference for trajectory training")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = os.path.join(args.output, f"size_{args.size}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create TensorBoard writer
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
    # Load config
    config = Config()
    print("Using configuration:")
    print(f"- Image size: {config.teacher_image_size}")
    print(f"- Timesteps: {config.timesteps}")
    print(f"- Channels: {config.channels}")
    
    # Get device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load teacher model if provided
    teacher_model = None
    if args.teacher_path and os.path.exists(args.teacher_path):
        print(f"Loading teacher model from {args.teacher_path}")
        teacher_model = SimpleUNet(config)
        teacher_model.load_state_dict(torch.load(args.teacher_path, map_location=device))
        teacher_model.to(device)
        teacher_model.eval()
    
    # Initialize CTM model
    print(f"Creating CTM model with size factor {args.size}")
    model = ConsistencyTrajectoryModel(config, size_factor=args.size)
    model.to(device)
    
    # Get diffusion parameters
    diffusion_params = get_diffusion_params(config.timesteps, device=device)
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Load dataset
    print(f"Loading {config.dataset} dataset")
    train_loader = get_data_loader(
        config,
        image_size=config.teacher_image_size
    )
    
    test_loader = get_data_loader(
        config,
        image_size=config.teacher_image_size
    )
    
    # Train
    print(f"Starting training for {args.epochs} epochs")
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_ctm(
            config, model, teacher_model, train_loader, optimizer, 
            diffusion_params, device, epoch, writer,
            trajectory_prob=args.trajectory_prob,
            max_time_diff=args.max_time_diff
        )
        
        # Evaluate
        val_loss = evaluate_ctm(
            config, model, test_loader, diffusion_params, device, epoch, writer
        )
        
        # Save model
        checkpoint_path = os.path.join(output_dir, f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_path = os.path.join(output_dir, "model_best.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! (Loss: {best_loss:.4f})")
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, "model.pt")
    torch.save(model.state_dict(), final_model_path)
    
    print("Training complete!")
    print(f"Final model saved to {final_model_path}")
    print(f"Best model (loss: {best_loss:.4f}) saved to {os.path.join(output_dir, 'model_best.pt')}")

if __name__ == "__main__":
    main() 