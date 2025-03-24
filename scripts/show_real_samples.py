#!/usr/bin/env python3
"""
Script to show real samples from the dataset for comparison with generated images.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config
from data.dataset import get_real_images

def main():
    # Create configuration
    config = Config()
    config.create_directories()
    
    print(f"Loading real images from {config.dataset} dataset...")
    
    # Get real images
    real_images = get_real_images(config, num_samples=16)
    
    # Create a grid of images
    grid = (real_images + 1) / 2  # Unnormalize from [-1, 1] to [0, 1]
    grid = torch.clamp(grid, 0, 1)
    grid = vutils.make_grid(grid, nrow=4)
    
    # Create the output directory if it doesn't exist
    os.makedirs(config.results_dir, exist_ok=True)
    
    # Save the grid of images
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.title(f'Real Images from {config.dataset}')
    plt.savefig(os.path.join(config.results_dir, 'real_images.png'))
    plt.close()
    
    print(f"Saved real images to {os.path.join(config.results_dir, 'real_images.png')}")
    
    # Save individual images for closer inspection
    for i in range(min(5, len(real_images))):
        img = real_images[i]
        img = (img + 1) / 2  # Unnormalize
        img = torch.clamp(img, 0, 1)
        
        plt.figure(figsize=(5, 5))
        plt.imshow(img.permute(1, 2, 0).cpu())
        plt.axis('off')
        plt.title(f'Real Image {i+1}')
        plt.savefig(os.path.join(config.results_dir, f'real_image_{i+1}.png'))
        plt.close()
        
    print(f"Saved {min(5, len(real_images))} individual real images for closer inspection")

if __name__ == "__main__":
    main() 