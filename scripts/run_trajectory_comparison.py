#!/usr/bin/env python3
"""
Script to run trajectory comparison between teacher and student models.
This script loads the teacher and student models and generates visualizations
comparing their trajectories from the same random starting point.
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
from models import SimpleUNet, StudentUNet, DiffusionUNet
from analysis.trajectory_comparison import compare_trajectories

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run trajectory comparison between teacher and student models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    parser.add_argument('--teacher_model', type=str, default='model_epoch_1.pt',
                        help='Path to teacher model relative to models directory')
    parser.add_argument('--student_models', type=str, default=None,
                        help='Comma-separated list of specific student model files (e.g., size_0.3/model_epoch_2.pt,size_1.0/model_epoch_2.pt)')
    parser.add_argument('--size_factors', type=str, default='0.3,1.0',
                        help='Comma-separated list of size factors to compare')
    parser.add_argument('--timesteps', type=int, default=50,
                        help='Number of timesteps for the diffusion process')
    
    # Output parameters
    parser.add_argument('--analysis_dir', type=str, default='analysis',
                        help='Directory to save analysis results')
    
    return parser.parse_args()

def main():
    """Main function to run the trajectory comparison"""
    args = parse_args()
    
    # Initialize configuration
    config = Config()
    config.analysis_dir = args.analysis_dir
    config.timesteps = args.timesteps
    
    # Create necessary directories
    os.makedirs(config.analysis_dir, exist_ok=True)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Parse size factors
    size_factors = [float(sf) for sf in args.size_factors.split(',')]
    
    # Parse student models if provided
    student_models = None
    if args.student_models:
        student_models = args.student_models.split(',')
        if len(student_models) != len(size_factors):
            print(f"ERROR: Number of student models ({len(student_models)}) must match number of size factors ({len(size_factors)})")
            sys.exit(1)
    
    # Load teacher model
    teacher_model_path = os.path.join(project_root, 'output', 'models', 'teacher', args.teacher_model)
    print(f"Loading teacher model from {teacher_model_path}")
    
    if not os.path.exists(teacher_model_path):
        print(f"ERROR: Teacher model not found at {teacher_model_path}")
        sys.exit(1)
    
    # Initialize teacher model
    teacher_model = SimpleUNet(config).to(device)
    
    teacher_state_dict = torch.load(teacher_model_path, map_location=device)
    teacher_model.load_state_dict(teacher_state_dict)
    teacher_model.eval()
    
    print(f"Teacher model loaded successfully")
    print(f"Teacher model dimensions: {teacher_model.dims}")
    
    # Process each size factor
    for i, size_factor in enumerate(size_factors):
        print(f"\n{'='*80}")
        print(f"Processing size factor: {size_factor}")
        print(f"{'='*80}")
        
        # Load student model
        if student_models:
            # Use specific student model file if provided
            student_model_path = os.path.join(
                project_root, 'output', 'models', 'students', student_models[i]
            )
        else:
            # Use default path with same filename as teacher model
            student_model_path = os.path.join(
                project_root, 'output', 'models', 'students', 
                f'size_{size_factor}', args.teacher_model
            )
        
        print(f"Loading student model from {student_model_path}")
        
        if not os.path.exists(student_model_path):
            print(f"WARNING: Student model not found at {student_model_path}")
            print(f"Skipping size factor {size_factor}")
            continue
        
        # Initialize student model
        student_model = StudentUNet(config, size_factor=size_factor).to(device)
        
        student_state_dict = torch.load(student_model_path, map_location=device)
        student_model.load_state_dict(student_state_dict)
        student_model.eval()
        
        print(f"Student model loaded successfully")
        print(f"Student model dimensions: {student_model.dims}")
        
        # Run trajectory comparison
        results = compare_trajectories(
            teacher_model=teacher_model,
            student_model=student_model,
            config=config,
            size_factor=size_factor
        )
        
        print(f"Trajectory comparison results:")
        print(f"  Teacher trajectory length: {results['teacher_trajectory_length']}")
        print(f"  Student trajectory length: {results['student_trajectory_length']}")
    
    print("\nTrajectory comparison completed")
    print(f"Results saved in {os.path.join(config.analysis_dir, 'trajectory_comparison')}")

if __name__ == "__main__":
    main() 