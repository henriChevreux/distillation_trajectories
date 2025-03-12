#!/usr/bin/env python3
"""
Script to train a Consistency Trajectory Model (CTM).
Implements the training procedure from the paper "Consistency Trajectory Models: 
Learning Probability Flow ODE Trajectory of Diffusion"
"""

import os
import sys
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config
from models import ConsistencyTrajectoryModel, SimpleUNet
from utils.diffusion import get_diffusion_params
from data.dataset import get_data_loader
from utils.fid import calculate_fid

def train_ctm(config, model, teacher_model, train_loader, optimizer, 
              diffusion_params, device, epoch, writer=None,
              trajectory_prob=0.7, max_time_diff=0.5):
    """
    Train the CTM model for one epoch
    
    Args:
        config: Configuration object
        model: CTM model to train
        teacher_model: Teacher model for supervision
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        diffusion_params: Diffusion process parameters
        device: Device to use for training
        epoch: Current epoch number
        writer: TensorBoard writer (optional)
        trajectory_prob: Probability of using trajectory mode
        max_time_diff: Maximum time difference for trajectory jumps
    """
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    with tqdm(train_loader, desc=f"Training Epoch {epoch+1}", 
              leave=False, position=0) as pbar:
        for batch_idx, (data, _) in enumerate(pbar):
            data = data.to(device)
            batch_size = data.shape[0]
            
            # Sample timesteps
            t = torch.randint(0, config.timesteps, (batch_size,), device=device)
            
            # Add noise to data
            noise = torch.randn_like(data)
            noisy_data = diffusion_params['sqrt_alphas_cumprod'][t].view(-1, 1, 1, 1) * data + \
                        diffusion_params['sqrt_one_minus_alphas_cumprod'][t].view(-1, 1, 1, 1) * noise
            
            # Decide whether to use trajectory mode
            use_trajectory = torch.rand(1).item() < trajectory_prob
            
            if use_trajectory:
                # Sample end timestep
                t_end = torch.maximum(
                    t - torch.randint(1, int(max_time_diff * config.timesteps), (batch_size,)),
                    torch.zeros_like(t)
                )
                
                # Get teacher prediction for supervision
                with torch.no_grad():
                    teacher_pred = teacher_model(noisy_data, t_end)
                
                # Get CTM prediction
                ctm_pred = model(noisy_data, t, t_end)
                
                # Compute loss
                loss = torch.nn.functional.mse_loss(ctm_pred, teacher_pred)
            else:
                # Standard denoising mode
                pred_noise = model(noisy_data, t)
                loss = torch.nn.functional.mse_loss(pred_noise, noise)
            
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
            
            if writer is not None:
                writer.add_scalar('train/batch_loss', loss.item(), 
                                epoch * num_batches + batch_idx)
    
    avg_loss = total_loss / len(train_loader)
    if writer is not None:
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)
    
    return avg_loss

def evaluate_ctm(config, model, test_loader, diffusion_params, device, epoch, writer=None):
    """
    Evaluate the CTM model
    
    Args:
        config: Configuration object
        model: CTM model to evaluate
        test_loader: DataLoader for test data
        diffusion_params: Diffusion process parameters
        device: Device to use for evaluation
        epoch: Current epoch number
        writer: TensorBoard writer (optional)
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            batch_size = data.shape[0]
            
            # Sample timesteps
            t = torch.randint(0, config.timesteps, (batch_size,), device=device)
            
            # Add noise to data
            noise = torch.randn_like(data)
            noisy_data = diffusion_params['sqrt_alphas_cumprod'][t].view(-1, 1, 1, 1) * data + \
                        diffusion_params['sqrt_one_minus_alphas_cumprod'][t].view(-1, 1, 1, 1) * noise
            
            # Get model prediction
            pred_noise = model(noisy_data, t)
            loss = torch.nn.functional.mse_loss(pred_noise, noise)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    if writer is not None:
        writer.add_scalar('val/loss', avg_loss, epoch)
    
    return avg_loss

def calculate_metrics(config, model, real_data_loader, device, num_samples=50000):
    """
    Calculate FID score for the model
    
    Args:
        config: Configuration object
        model: Model to evaluate
        real_data_loader: DataLoader for real data
        device: Device to use
        num_samples: Number of samples to generate
    """
    model.eval()
    
    # Generate samples
    samples = []
    with torch.no_grad():
        for _ in tqdm(range(0, num_samples, config.batch_size), desc="Generating samples"):
            batch_size = min(config.batch_size, num_samples - len(samples))
            sample = model.sample(batch_size, device=device)
            samples.append(sample.cpu())
    
    samples = torch.cat(samples, dim=0)
    
    # Calculate FID
    fid_score = calculate_fid(samples, real_data_loader, device)
    
    return {'fid': fid_score}

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
                        help="Path to teacher model")
    parser.add_argument("--trajectory-prob", type=float, default=0.7,
                        help="Probability of using trajectory mode during training")
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
    best_fid = float('inf')
    
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
        
        # Calculate FID score every 5 epochs
        if (epoch + 1) % 5 == 0:
            metrics = calculate_metrics(config, model, test_loader, device)
            fid_score = metrics['fid']
            writer.add_scalar('metrics/fid', fid_score, epoch)
            
            print(f"Epoch {epoch+1} - FID: {fid_score:.2f}")
            
            # Save if best FID
            if fid_score < best_fid:
                best_fid = fid_score
                best_fid_path = os.path.join(output_dir, "model_best_fid.pt")
                torch.save(model.state_dict(), best_fid_path)
                print(f"New best FID score! Saved model to {best_fid_path}")
        
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
    print(f"Best FID model (FID: {best_fid:.2f}) saved to {os.path.join(output_dir, 'model_best_fid.pt')}")

if __name__ == "__main__":
    main()
