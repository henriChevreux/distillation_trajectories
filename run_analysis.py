import os
import argparse
import torch

# Import from the diffusion_analysis module - using the exact name of the main function
from diffusion_analysis import Config, main as run_analysis

def parse_args():
    parser = argparse.ArgumentParser(description='Run analysis on trained diffusion models')
    
    # Analysis parameters
    parser.add_argument('--teacher_model', type=str, default='model_epoch_1.pt',
                        help='Path to teacher model relative to models directory')
    parser.add_argument('--student_model', type=str, default='student_model_epoch_1.pt',
                        help='Path to student model relative to models directory')
    parser.add_argument('--analysis_dir', type=str, default='analysis',
                        help='Directory to save analysis results')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of trajectory samples to generate')
    
    # Diffusion process parameters
    parser.add_argument('--teacher_steps', type=int, default=50,
                        help='Number of timesteps for teacher model')
    parser.add_argument('--student_steps', type=int, default=50,
                        help='Number of timesteps for student model')
    
    # Analysis module selection
    parser.add_argument('--skip_metrics', action='store_true',
                        help='Skip trajectory metrics analysis')
    parser.add_argument('--skip_dimensionality', action='store_true',
                        help='Skip dimensionality reduction analysis')
    parser.add_argument('--skip_noise', action='store_true',
                        help='Skip noise prediction analysis')
    parser.add_argument('--skip_attention', action='store_true',
                        help='Skip attention map analysis')
    parser.add_argument('--skip_3d', action='store_true',
                        help='Skip 3D visualization')
    parser.add_argument('--skip_fid', action='store_true',
                        help='Skip FID calculation')
    
    return parser.parse_args()

def setup_config(args):
    # Create a modified config based on arguments
    config = Config()
    
    # Override config parameters
    config.teacher_steps = args.teacher_steps
    config.student_steps = args.student_steps
    config.analysis_dir = args.analysis_dir
    
    # Create necessary directories
    os.makedirs(config.analysis_dir, exist_ok=True)
    
    # Save the config for reference
    config_info = vars(args)
    with open(os.path.join(config.analysis_dir, 'analysis_config.txt'), 'w') as f:
        for key, value in config_info.items():
            f.write(f"{key}: {value}\n")
    
    return config

def main():
    args = parse_args()
    config = setup_config(args)
    
    # Set device
    if torch.cuda.is_available():
        device_name = "CUDA"
    elif torch.backends.mps.is_available():
        device_name = "MPS"
    else:
        device_name = "CPU"
    
    print(f"Running analysis using {device_name} device")
    
    # Print analysis configuration
    print("\nAnalysis Configuration:")
    print(f"Teacher model: {args.teacher_model}")
    print(f"Teacher timesteps: {args.teacher_steps}")
    print(f"Student timesteps: {args.student_steps}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Analysis directory: {args.analysis_dir}")
    
    # Check if teacher model exists
    teacher_model_path = os.path.join(config.models_dir, args.teacher_model)
    
    if not os.path.exists(teacher_model_path):
        print("\nERROR: Teacher model file not found. You need to train the teacher model first.")
        print("Please run the training script first to generate the teacher model file:")
        print("\n    python diffusion_training.py\n")
        return
    
    # Find all student models with different size factors
    student_model_paths = {}
    
    # Define the size factors we expect to find (from the Config class in diffusion_training.py)
    expected_size_factors = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    for size_factor in expected_size_factors:
        model_path = os.path.join(config.models_dir, f'student_model_size_{size_factor}_epoch_1.pt')
        if os.path.exists(model_path):
            student_model_paths[size_factor] = model_path
    
    # If no student models found, check for the single student model specified in args
    if not student_model_paths and args.student_model:
        student_model_path = os.path.join(config.models_dir, args.student_model)
        if os.path.exists(student_model_path):
            student_model_paths = student_model_path
        else:
            print(f"\nWARNING: Student model file {args.student_model} not found.")
            print("No student models found. Please run the training script with distillation first:")
            print("\n    python diffusion_training.py\n")
            return
    
    if isinstance(student_model_paths, dict) and not student_model_paths:
        print("\nWARNING: No student models found. Please run the training script with distillation first:")
        print("\n    python diffusion_training.py\n")
        return
    
    # Print found student models
    if isinstance(student_model_paths, dict):
        print(f"\nFound {len(student_model_paths)} student models with size factors: {sorted(student_model_paths.keys())}")
    else:
        print(f"\nUsing student model: {args.student_model}")
    
    # Run the analysis
    print("\nStarting analysis...\n")
    try:
        run_analysis(config=config,
                    teacher_model_path=teacher_model_path,
                    student_model_paths=student_model_paths,
                    num_samples=args.num_samples,
                    skip_metrics=args.skip_metrics,
                    skip_dimensionality=args.skip_dimensionality, 
                    skip_noise=args.skip_noise,
                    skip_attention=args.skip_attention,
                    skip_3d=args.skip_3d,
                    skip_fid=args.skip_fid)
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print("\nERROR: Model architecture mismatch.")
            print("The saved models were trained with a different architecture than the current one.")
            print("This is likely because the models were trained on MNIST but the code is now configured for CIFAR10.")
            print("Please train new models with the current architecture:")
            print("\n    python diffusion_training.py\n")
        else:
            # Re-raise if it's not the size mismatch error
            raise

if __name__ == "__main__":
    main()
