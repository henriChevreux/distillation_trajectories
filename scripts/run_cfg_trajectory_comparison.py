#!/usr/bin/env python3
"""
Script to run trajectory comparison with classifier-free guidance between teacher and student models.
This script loads the teacher and student models and generates visualizations comparing their
trajectories with different guidance scales from the same random starting point.
"""

import os
import sys
import argparse
import torch
from dataclasses import dataclass

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from config.config import Config
from models import SimpleUNet, StudentUNet
from analysis.cfg_trajectory_comparison import compare_cfg_trajectories

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run trajectory comparison with classifier-free guidance between teacher and student models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    parser.add_argument('--teacher_model', type=str, default='model_epoch_1.pt',
                        help='Path to teacher model relative to models directory')
    parser.add_argument('--student_models', type=str, default=None,
                        help='Comma-separated list of specific student model files (e.g., size_0.3/model_epoch_2.pt,size_1.0/model_epoch_2.pt)')
    parser.add_argument('--size_factors', type=str, default='0.3,1.0',
                        help='Comma-separated list of size factors to compare')
    parser.add_argument('--guidance_scales', type=str, default='1.0,3.0,5.0,7.0',
                        help='Comma-separated list of guidance scales to use')
    parser.add_argument('--timesteps', type=int, default=50,
                        help='Number of timesteps for the diffusion process')
    
    # Output parameters
    parser.add_argument('--analysis_dir', type=str, default='analysis',
                        help='Directory to save analysis results')
    
    return parser.parse_args()

def main():
    """Main function to run the CFG trajectory comparison"""
    args = parse_args()
    
    # Load configuration
    config = Config()
    config.timesteps = args.timesteps
    config.analysis_dir = args.analysis_dir
    config.create_directories()  # Create necessary directories
    
    # Convert size factors to list of floats
    size_factors = [float(sf) for sf in args.size_factors.split(',')]
    
    # Convert guidance scales to list of floats
    guidance_scales = [float(w) for w in args.guidance_scales.split(',')]
    
    # Load teacher model
    teacher_path = os.path.join(config.teacher_models_dir, args.teacher_model)
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(f"Teacher model not found at {teacher_path}")
    
    print(f"Loading teacher model from {teacher_path}")
    teacher_model = SimpleUNet(config)
    teacher_model.load_state_dict(torch.load(teacher_path))
    teacher_model.eval()
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    teacher_model = teacher_model.to(device)
    
    # Process each student model
    if args.student_models:
        # Use specific model files
        student_paths = args.student_models.split(',')
        for student_path in student_paths:
            full_path = os.path.join(config.student_models_dir, student_path)
            if not os.path.exists(full_path):
                print(f"Warning: Student model not found at {full_path}")
                continue
            
            # Extract size factor from path
            try:
                size_dir = os.path.basename(os.path.dirname(student_path))
                size_factor = float(size_dir.split('_')[1])
            except:
                print(f"Warning: Could not determine size factor from path {student_path}")
                continue
            
            print(f"\nProcessing student model with size factor {size_factor}")
            print(f"Loading student model from {full_path}")
            
            student_model = StudentUNet(config, size_factor=size_factor)
            student_model.load_state_dict(torch.load(full_path))
            student_model = student_model.to(device)
            student_model.eval()
            
            # Compare trajectories with CFG
            compare_cfg_trajectories(
                teacher_model, 
                student_model, 
                config, 
                guidance_scales=guidance_scales,
                size_factor=size_factor
            )
    else:
        # Use size factors to load latest models
        for size_factor in size_factors:
            size_dir = os.path.join(config.student_models_dir, f'size_{size_factor}')
            if not os.path.exists(size_dir):
                print(f"Warning: No models found for size factor {size_factor}")
                continue
            
            # Find latest model file
            model_files = [f for f in os.listdir(size_dir) if f.startswith('model_epoch_') and f.endswith('.pt')]
            if not model_files:
                print(f"Warning: No model files found in {size_dir}")
                continue
            
            latest_model = max(model_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
            student_path = os.path.join(size_dir, latest_model)
            
            print(f"\nProcessing student model with size factor {size_factor}")
            print(f"Loading student model from {student_path}")
            
            student_model = StudentUNet(config, size_factor=size_factor)
            student_model.load_state_dict(torch.load(student_path))
            student_model = student_model.to(device)
            student_model.eval()
            
            # Compare trajectories with CFG
            compare_cfg_trajectories(
                teacher_model, 
                student_model, 
                config, 
                guidance_scales=guidance_scales,
                size_factor=size_factor
            )
    
    print(f"\nResults saved in {os.path.join(config.analysis_dir, 'cfg_trajectory_comparison')}")

if __name__ == '__main__':
    main() 