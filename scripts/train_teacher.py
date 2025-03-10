#!/usr/bin/env python3
"""
Script to train the teacher diffusion model.
This is extracted from diffusion_training.py to make it more modular.
"""

import os
import argparse
import torch
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config
from models import SimpleUNet
from utils.diffusion import get_diffusion_params, p_losses, p_sample_loop
from data.dataset import get_data_loader

def train_teacher(config):
    """
    Train the teacher diffusion model
    
    Args:
        config: Configuration object
        
    Returns:
        Trained teacher model
    """
    # Determine device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() and config.mps_enabled else
        "cpu"
    )
    print(f"Using {device} device")
    
    # Create diffusion parameters
    diffusion_params = get_diffusion_params(config.timesteps, config)
    
    # Initialize model
    model = SimpleUNet(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    # Get data loader
    train_loader = get_data_loader(config)
    
    # Training loop
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs}',
                           leave=config.progress_bar_leave, 
                           ncols=config.progress_bar_ncols, 
                           position=config.progress_bar_position)
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
            torch.save(model.state_dict(), os.path.join(config.teacher_models_dir, f'model_epoch_{epoch+1}.pt'))
            
            # Generate some samples
            model.eval()
            samples = p_sample_loop(
                model=model,
                shape=(config.num_samples_to_generate, config.channels, config.image_size, config.image_size),
                timesteps=config.timesteps,
                diffusion_params=diffusion_params,
                device=device,
                config=config,  # Pass the config parameter here
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
    """Main function to run the teacher model training"""
    parser = argparse.ArgumentParser(
        description='Train a diffusion model teacher',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override the number of epochs for training')
    parser.add_argument('--dataset', type=str, default=None, choices=['MNIST', 'CIFAR10'],
                        help='Dataset to use for training')
    parser.add_argument('--image_size', type=int, default=None,
                        help='Size of images to use for training')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training')
    parser.add_argument('--timesteps', type=int, default=None,
                        help='Number of timesteps for diffusion process')
    args = parser.parse_args()
    
    # Create configuration
    config = Config()
    config.create_directories()
    
    # Override config with command line arguments if provided
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.dataset is not None:
        config.dataset = args.dataset
    if args.image_size is not None:
        config.image_size = args.image_size
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.timesteps is not None:
        config.timesteps = args.timesteps
    
    # Print training configuration
    print("\n" + "="*80)
    print("DIFFUSION MODEL TEACHER TRAINING")
    print("="*80)
    print(f"\nTraining Configuration:")
    print(f"Dataset: {config.dataset}")
    print(f"Image size: {config.image_size}x{config.image_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"Timesteps: {config.timesteps}")
    print(f"Epochs: {config.epochs}")
    print(f"Learning rate: {config.lr}")
    print(f"Save interval: {config.save_interval}")
    print(f"Models directory: {config.models_dir}")
    print(f"Results directory: {config.results_dir}")
    
    # Train the teacher model
    print("\nStarting teacher model training...")
    train_teacher(config)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print("\nTeacher model has been saved.")
    print("To train student models with various size factors:")
    print("\n    python scripts/train_students.py\n")

if __name__ == "__main__":
    main()
