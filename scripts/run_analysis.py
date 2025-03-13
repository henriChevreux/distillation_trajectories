#!/usr/bin/env python3
"""
Script to run comprehensive analysis on trained diffusion models.
Supports running on CPU and various analysis modes including full MSE analysis.
"""

import os
import sys
import argparse
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config
from models import SimpleUNet, StudentUNet
from analysis.mse_heatmap import create_mse_heatmap
from analysis.denoising import run_denoising_analysis
from analysis.time_analysis import run_time_analysis
from analysis.trajectory_analysis import run_trajectory_analysis

def setup_analysis(args):
    """
    Set up the analysis environment by loading models and test data
    """
    # Load configuration
    config = Config()
    
    # Handle device placement
    if args.cpu:
        print("\n" + "="*80)
        print("FORCING CPU USAGE")
        print("="*80)
        device = torch.device('cpu')
        # Ensure CUDA is not used
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Load teacher model - try best model first, then fall back to final
    teacher_best_path = os.path.join(config.models_dir, 'teacher', 'model_best.pt')
    teacher_final_path = os.path.join(config.models_dir, 'teacher', 'model_final.pt')
    
    if os.path.exists(teacher_best_path):
        teacher_path = teacher_best_path
        print("Using best teacher model")
    elif os.path.exists(teacher_final_path):
        teacher_path = teacher_final_path
        print("Using final teacher model")
    else:
        raise FileNotFoundError(f"Teacher model not found at {teacher_best_path} or {teacher_final_path}")
    
    # Create teacher model instance
    teacher_model = SimpleUNet(config).to(device)
    # Load state dict
    state_dict = torch.load(teacher_path, map_location=device)
    teacher_model.load_state_dict(state_dict)
    teacher_model.eval()
    
    # Load student models
    student_models = {}
    student_dir = os.path.join(config.models_dir, 'students')
    
    # Determine which size factors to use
    if args.full_mse:
        print("Using ALL model and image sizes for analysis")
        size_factors = config.student_size_factors
        image_size_factors = config.student_image_size_factors
    else:
        # Use limited set for quicker analysis
        size_factors = config.student_size_factors[:2]
        image_size_factors = config.student_image_size_factors[:2]
    
    for size_factor in size_factors:
        for image_size_factor in image_size_factors:
            # Check for model in the size_X_img_Y/model.pt format
            model_dir = os.path.join(student_dir, f'size_{size_factor}_img_{image_size_factor}')
            model_path = os.path.join(model_dir, 'model.pt')
            
            if os.path.exists(model_path):
                print(f"Loading student model from {model_path}")
                # Create student model instance with appropriate architecture
                student_model = StudentUNet(config, size_factor).to(device)
                # Load state dict
                state_dict = torch.load(model_path, map_location=device)
                student_model.load_state_dict(state_dict)
                student_model.eval()
                student_models[(size_factor, image_size_factor)] = student_model
    
    # Load test dataset
    test_samples = []
    test_dir = os.path.join(config.data_dir, 'test')
    if os.path.exists(test_dir):
        # Create transform for test data
        transform = transforms.Compose([
            transforms.Resize(config.teacher_image_size),
            transforms.CenterCrop(config.teacher_image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Load test dataset
        test_dataset = ImageFolder(test_dir, transform=transform)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0 if args.cpu else 2  # No workers on CPU mode
        )
        test_samples = next(iter(test_loader))[0].to(device)  # Only take the images, not labels
    else:
        raise FileNotFoundError(f"Test dataset not found at {test_dir}")
    
    return config, teacher_model, student_models, test_samples

def run_analysis(args):
    """
    Run the complete analysis pipeline
    """
    try:
        print("Setting up analysis environment...")
        config, teacher_model, student_models, test_samples = setup_analysis(args)
        
        # Create output directories
        os.makedirs(config.analysis_dir, exist_ok=True)
        
        # Print analysis configuration
        print(f"\nAnalysis Configuration:")
        print(f"- Using {config.timesteps} timesteps for teacher model")
        print(f"- Using {config.student_steps} timesteps for student models")
        print(f"- Image size: {config.teacher_image_size}x{config.teacher_image_size}")
        print(f"- Found {len(student_models)} student models")
        print(f"- Analysis samples: {args.num_samples}")
        print(f"- Running on: {'CPU' if args.cpu else 'GPU if available'}")
        
        # Configure MSE analysis settings
        if args.full_mse or args.mse:
            config.mse_size_factors_limit = False
            config.mse_image_size_factors_limit = False
        
        # 1. MSE Analysis
        if args.mse or args.full_mse:
            print("\nGenerating MSE heatmap...")
            try:
                create_mse_heatmap(config, test_samples)
                print("MSE analysis completed successfully!")
            except Exception as e:
                print(f"Error in MSE analysis: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
        
        # 2. Denoising Analysis
        if args.denoising:
            print("\nRunning denoising analysis...")
            try:
                for (size_factor, _), student_model in student_models.items():
                    print(f"\nAnalyzing denoising for student model (size factor: {size_factor})...")
                    run_denoising_analysis(teacher_model, student_model, config, test_samples.device)
                print("Denoising analysis completed successfully!")
            except Exception as e:
                print(f"Error in denoising analysis: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
        
        # 3. Time-dependent Analysis
        if args.time:
            print("\nRunning time-dependent analysis...")
            try:
                run_time_analysis(teacher_model, student_models, test_samples, config)
                print("Time-dependent analysis completed successfully!")
            except Exception as e:
                print(f"Error in time-dependent analysis: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
        
        # 4. Trajectory Analysis
        if args.trajectory:
            print("\nRunning trajectory analysis...")
            try:
                run_trajectory_analysis(teacher_model, student_models, test_samples, config)
                print("Trajectory analysis completed successfully!")
            except Exception as e:
                print(f"Error in trajectory analysis: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                    
    except Exception as e:
        print(f"\nError during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive analysis on trained diffusion models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Analysis type arguments
    parser.add_argument("--mse", action="store_true", help="Run MSE analysis")
    parser.add_argument("--denoising", action="store_true", help="Run denoising analysis")
    parser.add_argument("--time", action="store_true", help="Run time-dependent analysis")
    parser.add_argument("--trajectory", action="store_true", help="Run trajectory analysis")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    
    # Configuration arguments
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for loading test data")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to use for analysis")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--full-mse", action="store_true", help="Run comprehensive MSE analysis with all model sizes")
    parser.add_argument("--verbose", action="store_true", help="Print detailed error messages")
    
    args = parser.parse_args()
    
    # If no analysis type is specified, run everything
    if not any([args.mse, args.denoising, args.time, args.trajectory, args.full_mse]) and not args.all:
        args.all = True
    
    # If --all is specified, run everything
    if args.all:
        args.mse = True
        args.denoising = True
        args.time = True
        args.trajectory = True
    
    run_analysis(args)

if __name__ == "__main__":
    main()
