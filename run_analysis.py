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
    parser.add_argument('--num_samples', type=int, default=5,
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
    print(f"Student model: {args.student_model}")
    print(f"Teacher timesteps: {args.teacher_steps}")
    print(f"Student timesteps: {args.student_steps}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Analysis directory: {args.analysis_dir}")
    
    # Run the analysis
    print("\nStarting analysis...\n")
    run_analysis(config=config,
                 teacher_model_path=os.path.join(config.models_dir, args.teacher_model),
                 student_model_path=os.path.join(config.models_dir, args.student_model),
                 num_samples=args.num_samples,
                 skip_metrics=args.skip_metrics,
                 skip_dimensionality=args.skip_dimensionality, 
                 skip_noise=args.skip_noise,
                 skip_attention=args.skip_attention,
                 skip_3d=args.skip_3d)

if __name__ == "__main__":
    main()