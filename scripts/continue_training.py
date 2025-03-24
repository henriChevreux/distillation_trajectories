#!/usr/bin/env python3
"""
Script to continue training the teacher diffusion model from a checkpoint.
"""

import os
import argparse
import torch
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config
from models import SimpleUNet
from utils.diffusion import get_diffusion_params, p_losses, p_sample_loop
from data.dataset import get_data_loader

def continue_training(config, start_epoch):
    """
    Continue training the teacher diffusion model from a checkpoint
    
    Args:
        config: Configuration object
        start_epoch: The epoch to start from (the checkpoint to load)
        
    Returns:
        Trained teacher model
    """
    # Determine device
    device = torch.device("cpu") if hasattr(config, 'force_cpu') and config.force_cpu else torch.device(
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() and config.mps_enabled else
        "cpu"
    )
    print(f"Using {device} device")
    
    # Create diffusion parameters
    diffusion_params = get_diffusion_params(config.timesteps, config)
    
    # Initialize model
    model = SimpleUNet(config).to(device)
    
    # Load the checkpoint
    checkpoint_path = os.path.join(config.teacher_models_dir, f'model_epoch_{start_epoch}.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    # Get data loader
    train_loader = get_data_loader(config)
    
    # Continue training from the next epoch
    total_epochs = start_epoch + config.epochs
    
    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{total_epochs}',
                           leave=config.progress_bar_leave, 
                           ncols=config.progress_bar_ncols, 
                           position=config.progress_bar_position)
        
        for batch_idx, (images, _) in enumerate(progress_bar):
            images = images.to(device)
            optimizer.zero_grad()
            
            # Sample random timesteps
            t = torch.randint(0, config.timesteps, (images.shape[0],), device=device).long()
            
            # Calculate loss with both conditional and unconditional samples
            # For unconditional samples, we pass None as the condition
            loss_cond = p_losses(model, images, t, diffusion_params, cond=torch.ones(images.shape[0], 1).to(device))
            loss_uncond = p_losses(model, images, t, diffusion_params, cond=None)
            
            # Total loss is the average of conditional and unconditional losses
            loss = (loss_cond + loss_uncond) / 2
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss/(batch_idx+1))
        
        # Save model periodically
        if (epoch + 1) % config.save_interval == 0 or epoch == total_epochs - 1:
            torch.save(model.state_dict(), os.path.join(config.teacher_models_dir, f'model_epoch_{epoch+1}.pt'))
            
            # Generate some samples
            model.eval()
            samples = p_sample_loop(
                model=model,
                shape=(config.num_samples_to_generate, config.channels, config.image_size, config.image_size),
                sample_steps=config.sample_steps,
                diffusion_params=diffusion_params,
                device=device,
                config=config,
                track_trajectory=False
            )
            
            # Save samples
            grid = (samples + 1) / 2
            grid = torch.clamp(grid, 0, 1)
            grid = torchvision.utils.make_grid(grid, nrow=config.samples_grid_size)
            plt.figure(figsize=config.samples_figure_size)
            plt.imshow(grid.permute(1, 2, 0).cpu())
            plt.axis('off')
            plt.savefig(os.path.join(config.results_dir, f'samples_epoch_{epoch+1}.png'))
            plt.close()
    
    return model

def main():
    """Main function to continue the teacher model training"""
    parser = argparse.ArgumentParser(
        description='Continue training a diffusion model teacher from a checkpoint',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--start_epoch', type=int, required=True,
                        help='The epoch to start from (the checkpoint to load)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of additional epochs to train')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Number of timesteps for diffusion process')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training')
    args = parser.parse_args()
    
    # Create configuration
    config = Config()
    config.create_directories()
    
    # Override config with command line arguments
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.timesteps is not None:
        config.timesteps = args.timesteps
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    
    # Print training configuration
    print("\n" + "="*80)
    print("CONTINUING DIFFUSION MODEL TEACHER TRAINING")
    print("="*80)
    print(f"\nTraining Configuration:")
    print(f"Starting from epoch: {args.start_epoch}")
    print(f"Additional epochs: {config.epochs}")
    print(f"Dataset: {config.dataset}")
    print(f"Image size: {config.image_size}x{config.image_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"Timesteps: {config.timesteps}")
    print(f"Learning rate: {config.lr}")
    print(f"Save interval: {config.save_interval}")
    print(f"Models directory: {config.models_dir}")
    print(f"Results directory: {config.results_dir}")
    
    # Continue training the teacher model
    print("\nContinuing teacher model training...")
    continue_training(config, args.start_epoch)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print("\nTeacher model has been saved.")

if __name__ == "__main__":
    main() 