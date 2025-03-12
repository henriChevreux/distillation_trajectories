#!/usr/bin/env python3
"""
Script to train multiple Consistency Trajectory Models (CTM) with different configurations.
Performs grid search over model sizes and trajectory parameters.
"""

import os
import sys
import torch
import argparse
from itertools import product
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config
from models import ConsistencyTrajectoryModel, SimpleUNet
from scripts.train_ctm import train_ctm, evaluate_ctm, calculate_metrics
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
    
    Returns:
        dict: Results including best loss and FID score
    """
    # Create model directory name
    model_dir = os.path.join(
        config.models_dir, 
        "ctm",
        f"size_{size_factor}_tprob_{trajectory_prob}_tdiff_{max_time_diff}"
    )
    os.makedirs(model_dir, exist_ok=True)
    
    # Create TensorBoard writer
    log_dir = os.path.join(model_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    
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
    
    # Get data loaders
    train_loader = get_data_loader(config)
    test_loader = get_data_loader(config)
    
    # Training loop
    best_loss = float('inf')
    best_fid = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(config.epochs):
        # Train
        train_loss = train_ctm(
            config, model, teacher_model, train_loader, optimizer,
            diffusion_params, device, epoch, writer,
            trajectory_prob=trajectory_prob,
            max_time_diff=max_time_diff
        )
        
        # Evaluate
        val_loss = evaluate_ctm(
            config, model, test_loader, diffusion_params, device, epoch, writer
        )
        
        print(f"Epoch {epoch+1}/{config.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Calculate FID score every 5 epochs
        if (epoch + 1) % 5 == 0:
            metrics = calculate_metrics(config, model, test_loader, device)
            fid_score = metrics['fid']
            writer.add_scalar('metrics/fid', fid_score, epoch)
            
            print(f"Epoch {epoch+1} - FID: {fid_score:.2f}")
            
            # Save if best FID
            if fid_score < best_fid:
                best_fid = fid_score
                best_fid_path = os.path.join(model_dir, "model_best_fid.pt")
                torch.save(model.state_dict(), best_fid_path)
                print(f"New best FID score! Saved model to {best_fid_path}")
        
        # Save if best model
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_without_improvement = 0
            best_model_path = os.path.join(model_dir, "model_best.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! (Loss: {best_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs (best loss: {best_loss:.4f})")
            
            # Early stopping
            if epochs_without_improvement >= 10:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Save final model
    final_model_path = os.path.join(model_dir, "model.pt")
    torch.save(model.state_dict(), final_model_path)
    
    writer.close()
    
    return {
        'size_factor': size_factor,
        'trajectory_prob': trajectory_prob,
        'max_time_diff': max_time_diff,
        'best_loss': best_loss,
        'best_fid': best_fid,
        'epochs_trained': epoch + 1,
        'model_dir': model_dir
    }

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
    
    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(config.models_dir, "ctm", f"grid_search_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(experiment_dir, "grid_config.json")
    with open(config_path, 'w') as f:
        json.dump({
            'size_factors': size_factors,
            'trajectory_probs': trajectory_probs,
            'time_diffs': time_diffs,
            'device': args.device,
            'teacher_path': args.teacher_path,
            'timestamp': timestamp
        }, f, indent=4)
    
    for size_factor, traj_prob, time_diff in product(size_factors, trajectory_probs, time_diffs):
        current_config += 1
        print(f"\n{'='*80}")
        print(f"Training configuration {current_config}/{total_configs}")
        print(f"{'='*80}")
        
        try:
            result = train_ctm_configuration(
                config, size_factor, traj_prob, time_diff, device, teacher_model
            )
            results.append(result)
            
            # Save intermediate results
            results_path = os.path.join(experiment_dir, "results.json")
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)
            
        except Exception as e:
            print(f"Error training configuration: {e}")
            continue
    
    # Sort results by FID score
    results.sort(key=lambda x: x['best_fid'])
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    print("\nTop 5 configurations by FID score:")
    for i, result in enumerate(results[:5], 1):
        print(f"\n{i}. FID: {result['best_fid']:.2f}, Loss: {result['best_loss']:.4f}")
        print(f"   - Size factor: {result['size_factor']}")
        print(f"   - Trajectory probability: {result['trajectory_prob']}")
        print(f"   - Max time difference: {result['max_time_diff']}")
        print(f"   - Model directory: {result['model_dir']}")
    
    # Save final results
    final_results_path = os.path.join(experiment_dir, "final_results.json")
    with open(final_results_path, 'w') as f:
        json.dump({
            'config': {
                'size_factors': size_factors,
                'trajectory_probs': trajectory_probs,
                'time_diffs': time_diffs,
                'device': args.device,
                'teacher_path': args.teacher_path,
                'timestamp': timestamp
            },
            'results': results
        }, f, indent=4)
    
    print(f"\nExperiment results saved to: {experiment_dir}")

if __name__ == "__main__":
    main() 