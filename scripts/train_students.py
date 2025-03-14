#!/usr/bin/env python3
"""
Script to train student diffusion models with various size factors.
This is extracted from train_students.py to make it more modular.
"""

import os
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
import os

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config
from models import SimpleUNet, StudentUNet
from utils.diffusion import get_diffusion_params, q_sample, p_sample_loop
from data.dataset import get_data_loader

def print_size_factor_info(config):
    """Print information about the size factors that will be trained"""
    size_factors = config.student_size_factors
    
    # Group size factors by category
    tiny = [sf for sf in size_factors if sf < 0.1]
    small = [sf for sf in size_factors if 0.1 <= sf < 0.3]
    medium = [sf for sf in size_factors if 0.3 <= sf < 0.7]
    large = [sf for sf in size_factors if sf >= 0.7]
    
    # Parameter count approximation
    param_counts = {sf: sf**2 for sf in size_factors}  # Normalized to teacher model
    
    print("\n" + "="*80)
    print("MODEL SIZE SPECTRUM TRAINING")
    print("="*80)
    
    print(f"\nTraining {len(size_factors)} student models with size factors: {min(size_factors)} to {max(size_factors)}")
    
    print("\nSize distribution:")
    print(f"  Tiny (< 0.1x): {len(tiny)} models - {tiny}")
    print(f"  Small (0.1-0.3x): {len(small)} models - {small}")
    print(f"  Medium (0.3-0.7x): {len(medium)} models - {medium}")
    print(f"  Large (0.7-1.0x): {len(large)} models - {large}")
    
    print("\nArchitecture types:")
    print("  - Tiny (< 0.1): 2 layers instead of 3")
    print("  - Small (0.1-0.3): 3 layers with smaller dimensions")
    print("  - Medium (0.3-0.7): 3 layers with 75% of teacher dimensions")
    print("  - Full (0.7-1.0): Same architecture as teacher")
    
    print("\nApproximate parameter counts (relative to teacher model):")
    for category, factors in [("Tiny", tiny), ("Small", small), ("Medium", medium), ("Large", large)]:
        if factors:
            min_factor, max_factor = min(factors), max(factors)
            print(f"  {category}: {param_counts[min_factor]:.4f}x to {param_counts[max_factor]:.4f}x parameters")

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
    # Determine device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() and config.mps_enabled else
        "cpu"
    )
    
    # Initialize student model with the specified size factor
    # The architecture type will be automatically determined based on size_factor
    student_model = StudentUNet(config, size_factor=size_factor).to(device)
    
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
    for epoch in range(config.epochs):  # Use full number of epochs for distillation
        student_model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'Distillation Epoch {epoch+1}/{config.epochs}', 
                           leave=config.progress_bar_leave, 
                           ncols=config.progress_bar_ncols, 
                           position=config.progress_bar_position)
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
            
            # Ensure student prediction has the same size as teacher prediction
            if student_pred.shape != teacher_pred.shape:
                # Resize student prediction to match teacher prediction
                student_pred = F.interpolate(
                    student_pred, 
                    size=(teacher_pred.shape[2], teacher_pred.shape[3]),
                    mode='bilinear', 
                    align_corners=True
                )
            
            # MSE loss between student and teacher predictions
            loss = F.mse_loss(student_pred, teacher_pred)
            
            # Remove the diversity loss that compares with true noise
            # We want to focus solely on matching the teacher's trajectory
            
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            total_loss += loss.item()
            progress_bar.set_postfix(loss=total_loss/(batch_idx+1))
        
        # Save model at the end of each epoch
        if (epoch + 1) % config.save_interval == 0 or epoch == (config.epochs) - 1:
            # Create size-specific directory
            size_dir = os.path.join(config.student_models_dir, f'size_{size_factor}')
            os.makedirs(size_dir, exist_ok=True)
            
            # Save model with epoch in filename
            save_path = os.path.join(size_dir, f'model_epoch_{epoch+1}.pt')
            print(f"Saving student model to: {save_path}")
            torch.save(student_model.state_dict(), save_path)
            
            # Only generate samples at the end of training to save time
            if epoch == (config.epochs) - 1:
                # Generate some samples
                student_model.eval()
                samples = p_sample_loop(
                    student_model, 
                    shape=(config.num_samples_to_generate, config.channels, config.image_size, config.image_size),
                    sample_steps=config.student_steps,
                    diffusion_params=student_params,
                    device=device,
                    config=config
                )
                
                # Save samples
                grid = (samples + 1) / 2
                grid = torch.clamp(grid, 0, 1)
                grid = torchvision.utils.make_grid(grid, nrow=config.samples_grid_size)
                plt.figure(figsize=config.samples_figure_size)
                plt.imshow(grid.permute(1, 2, 0).cpu())
                plt.axis('off')
                plt.savefig(os.path.join(config.results_dir, f'student_samples_size_{size_factor}_epoch_{epoch+1}.png'))
                plt.close()
    
    return student_model

def train_students(config, custom_size_factors=None):
    """
    Train student models with various size factors
    
    Args:
        config: Configuration object
        custom_size_factors: Optional list of custom size factors to train
    """
    # Determine device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() and config.mps_enabled else
        "cpu"
    )
    print(f"Using {device} device")
    
    # Create diffusion parameters for both teacher and student
    teacher_params = get_diffusion_params(config.teacher_steps, config)
    student_params = get_diffusion_params(config.student_steps, config)
    
    # Check if teacher model exists - try both new and old paths for backward compatibility
    teacher_model_path = os.path.join(config.teacher_models_dir, 'model_epoch_1.pt')
    old_teacher_model_path = os.path.join(config.models_dir, 'model_epoch_1.pt')
    
    # Check new path first, then fall back to old path
    if os.path.exists(teacher_model_path):
        print(f"Found teacher model at {teacher_model_path}")
    elif os.path.exists(old_teacher_model_path):
        print(f"Found teacher model at old location {old_teacher_model_path}")
        teacher_model_path = old_teacher_model_path
    else:
        print("\nERROR: Teacher model not found at", teacher_model_path)
        print("Please train the teacher model first by running:")
        print("\n    python scripts/train_teacher.py\n")
        return
    
    # Load teacher model
    print(f"Loading existing teacher model from {teacher_model_path}...")
    teacher_model = SimpleUNet(config).to(device)
    teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=device))
    teacher_model.eval()
    
    # Use custom size factors if provided
    size_factors = custom_size_factors if custom_size_factors else config.student_size_factors
    
    # Train student models with different size factors
    student_models = {}
    for size_factor in size_factors:
        print(f"\nDistilling to student model with size factor {size_factor}...")
        student_model = distill_diffusion_model(
            teacher_model, 
            config, 
            teacher_params, 
            student_params,
            size_factor=size_factor
        )
        student_models[size_factor] = student_model
    
    return student_models

def main():
    """Main function to run the student model training"""
    parser = argparse.ArgumentParser(
        description='Train student diffusion models with various size factors',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override the number of epochs for training')
    parser.add_argument('--custom_size_factors', type=str, default=None,
                        help='Custom size factors to train (comma-separated, e.g., "0.1,0.5,0.9")')
    parser.add_argument('--dataset', type=str, default=None, choices=['MNIST', 'CIFAR10'],
                        help='Dataset to use for training')
    parser.add_argument('--image_size', type=int, default=None,
                        help='Size of images to use for training')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training')
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
    
    # Parse custom size factors if provided
    custom_size_factors = None
    if args.custom_size_factors:
        try:
            custom_size_factors = [float(sf) for sf in args.custom_size_factors.split(',')]
            print(f"Using custom size factors: {custom_size_factors}")
        except:
            print(f"WARNING: Invalid custom size factors format: {args.custom_size_factors}")
            print("Using default size factors instead.")
    
    # Print information about the size factors
    print_size_factor_info(config)
    
    # Train student models
    print("\nStarting student model training with various size factors...")
    train_students(config, custom_size_factors)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print("\nStudent models with various size factors have been saved.")
    print("To run the comprehensive size impact analysis:")
    print("\n    python scripts/run_analysis.py\n")

if __name__ == "__main__":
    main()
