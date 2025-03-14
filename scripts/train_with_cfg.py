#!/usr/bin/env python3
"""
Script to train both teacher and student models with Classifier-Free Guidance support.
This script combines the training of both models to ensure they are trained with CFG.
"""

import os
import argparse
import torch
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config
from scripts.train_teacher import train_teacher
from scripts.train_students import train_students

def main():
    """Main function to run the CFG-enabled training"""
    parser = argparse.ArgumentParser(
        description='Train teacher and student models with Classifier-Free Guidance',
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
    parser.add_argument('--size_factors', type=str, default='0.3,1.0',
                        help='Comma-separated list of size factors for student models')
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
    
    # Parse size factors
    size_factors = [float(sf) for sf in args.size_factors.split(',')]
    
    print("\n" + "="*80)
    print("TRAINING WITH CLASSIFIER-FREE GUIDANCE")
    print("="*80)
    print(f"\nTraining Configuration:")
    print(f"Dataset: {config.dataset}")
    print(f"Image size: {config.image_size}x{config.image_size}")
    print(f"Batch size: {config.batch_size}")
    print(f"Timesteps: {config.timesteps}")
    print(f"Epochs: {config.epochs}")
    print(f"Size factors: {size_factors}")
    
    # Train teacher model first
    print("\nTraining teacher model with CFG...")
    teacher_model = train_teacher(config)
    
    # Train student models
    print("\nTraining student models with CFG...")
    student_models = train_students(config, custom_size_factors=size_factors)
    
    print("\nTraining complete! Models are now ready for CFG trajectory comparison.")

if __name__ == '__main__':
    main() 