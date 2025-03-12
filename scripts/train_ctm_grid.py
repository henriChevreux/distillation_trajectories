#!/usr/bin/env python3
"""
Script to train multiple Consistency Trajectory Models (CTM) with different configurations.
"""

import os
import sys
import torch
import argparse
from itertools import product
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config
from models import ConsistencyTrajectoryModel, SimpleUNet
from scripts.train_ctm import train_ctm, evaluate_ctm
from utils.diffusion import get_diffusion_params
from data.dataset import get_data_loader

def train_ctm_configuration(config, size_factor, trajectory_prob, max_time_diff, device, teacher_model):
    """
    Train a CTM model with specific configuration.
    
    Args:
        config: Configuration object
        size_factor: Model size factor
        trajectory_prob: Probability of using trajectory mode
        max_time_diff: Maximum time difference for trajectory jumps
        device: Device to use
        teacher_model: Teacher model for supervision
    """
    # Create model directory name
    model_dir = os.path.join(
        config.models_dir, 
        "ctm",
        f"size_{size_factor}_tprob_{trajectory_prob}_tdiff_{max_time_diff}"
    )
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\nTraining CTM with configuration:")
    print(f"- Size factor: {size_factor}")
    print(f"- Trajectory probability: {trajectory_prob}")
    print(f"- Max time difference: {max_time_diff}")
    print(f"- Output directory: {model_dir}")
    
    # Initialize model
    model = ConsistencyTrajectoryModel(config, size_factor=size_factor)
    model.to(device)
    
    # Get diffusion parameters
    diffusion_params = get_diffusion_params(config.timesteps, device=device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    
    # Get data loader
    train_loader = get_data_loader(config)
    test_loader = get_data_loader(config)
    
    # Training loop
    best_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(config.epochs):
        # Train
        train_loss = train_ctm(
            config, model, teacher_model, train_loader, optimizer,
            diffusion_params, device, epoch,
            trajectory_prob=trajectory_prob,
            max_time_diff=max_time_diff
        )
        
        # Evaluate
        val_loss = evaluate_ctm(
            config, model, test_loader, diffusion_params, device, epoch
        )
        
        print(f"Epoch {epoch+1}/{config.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save if best model
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_without_improvement = 0
            best_model_path = os.path.join(model_dir, "model_best.pt")
            
            # Remove previous best model
            if os.path.exists(best_model_path):
                os.remove(best_model_path)
            
            # Save new best model
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! (Loss: {best_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs (best loss: {best_loss:.4f})")
            
            # Early stopping
            if epochs_without_improvement >= 10:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    return best_loss

def main():
    parser = argparse.ArgumentParser(description="Train multiple CTM models with different configurations")
    
    # Model configuration ranges
    parser.add_argument("--size-factors", type=str, default="0.2,0.4,0.6,0.8,1.0",
                        help="Comma-separated list of size factors")
    parser.add_argument("--trajectory-probs", type=str, default="0.5,0.7,0.9",
                        help="Comma-separated list of trajectory probabilities")
    parser.add_argument("--time-diffs", type=str, default="0.3,0.5,0.7",
                        help="Comma-separated list of maximum time differences")
    
    # Training settings
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--teacher-path", type=str, default="output/models/teacher/model_best.pt",
                        help="Path to teacher model")
    
    args = parser.parse_args()
    
    # Parse configuration ranges
    size_factors = [float(x) for x in args.size_factors.split(",")]
    trajectory_probs = [float(x) for x in args.trajectory_probs.split(",")]
    time_diffs = [float(x) for x in args.time_diffs.split(",")]
    
    # Create configuration
    config = Config()
    config.create_directories()
    
    # Create CTM models directory
    os.makedirs(os.path.join(config.models_dir, "ctm"), exist_ok=True)
    
    # Load teacher model
    device = torch.device(args.device)
    print(f"Loading teacher model from {args.teacher_path}")
    teacher_model = SimpleUNet(config)
    teacher_model.load_state_dict(torch.load(args.teacher_path, map_location=device))
    teacher_model.to(device)
    teacher_model.eval()
    
    # Train all configurations
    results = []
    total_configs = len(size_factors) * len(trajectory_probs) * len(time_diffs)
    current_config = 0
    
    print(f"\nTraining {total_configs} different CTM configurations:")
    print(f"Size factors: {size_factors}")
    print(f"Trajectory probabilities: {trajectory_probs}")
    print(f"Maximum time differences: {time_diffs}")
    
    for size_factor, traj_prob, time_diff in product(size_factors, trajectory_probs, time_diffs):
        current_config += 1
        print(f"\n{'='*80}")
        print(f"Training configuration {current_config}/{total_configs}")
        print(f"{'='*80}")
        
        try:
            best_loss = train_ctm_configuration(
                config, size_factor, traj_prob, time_diff, device, teacher_model
            )
            
            results.append({
                'size_factor': size_factor,
                'trajectory_prob': traj_prob,
                'max_time_diff': time_diff,
                'best_loss': best_loss
            })
            
        except Exception as e:
            print(f"Error training configuration: {e}")
            continue
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    # Sort results by loss
    results.sort(key=lambda x: x['best_loss'])
    
    print("\nTop 5 configurations:")
    for i, result in enumerate(results[:5], 1):
        print(f"{i}. Loss: {result['best_loss']:.6f}")
        print(f"   - Size factor: {result['size_factor']}")
        print(f"   - Trajectory probability: {result['trajectory_prob']}")
        print(f"   - Max time difference: {result['max_time_diff']}\n")

if __name__ == "__main__":
    main() 