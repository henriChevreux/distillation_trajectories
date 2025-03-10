import os
import argparse
import torch

# Import from the diffusion_analysis module - using the exact name of the main function
from diffusion_analysis import Config, main as run_analysis

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run analysis on diffusion models with a focus on model size impact',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
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
    analysis_group = parser.add_argument_group('Analysis Modules', 'Enable/disable specific analysis modules')
    analysis_group.add_argument('--skip_metrics', action='store_true',
                        help='Skip trajectory metrics analysis')
    analysis_group.add_argument('--skip_dimensionality', action='store_true',
                        help='Skip dimensionality reduction analysis')
    analysis_group.add_argument('--skip_noise', action='store_true',
                        help='Skip noise prediction analysis')
    analysis_group.add_argument('--skip_attention', action='store_true',
                        help='Skip attention map analysis')
    analysis_group.add_argument('--skip_3d', action='store_true',
                        help='Skip 3D visualization')
    analysis_group.add_argument('--skip_fid', action='store_true',
                        help='Skip FID calculation')
    
    # Model size focus
    size_group = parser.add_argument_group('Model Size Analysis', 'Options for model size comparison')
    size_group.add_argument('--focus_size_range', type=str, default=None,
                        help='Focus on a specific size range (e.g., "0.1-0.5")')
    size_group.add_argument('--compare_specific_sizes', type=str, default=None,
                        help='Compare specific size factors (comma-separated, e.g., "0.1,0.5,1.0")')
    
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
    print("\n" + "="*80)
    print("MODEL SIZE IMPACT ANALYSIS")
    print("="*80)
    print("\nThis analysis focuses on how model size affects diffusion model performance.")
    print("Multiple student models of different sizes will be compared against the teacher model.")
    
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
    expected_size_factors = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.25, 0.5, 0.75, 1.0]
    
    # Filter by focus range if specified
    if args.focus_size_range:
        try:
            min_size, max_size = map(float, args.focus_size_range.split('-'))
            expected_size_factors = [sf for sf in expected_size_factors if min_size <= sf <= max_size]
            print(f"\nFocusing on size range: {min_size} to {max_size}")
        except:
            print(f"\nWARNING: Invalid size range format: {args.focus_size_range}. Using all available sizes.")
    
    # Filter by specific sizes if specified
    if args.compare_specific_sizes:
        try:
            specific_sizes = [float(sf) for sf in args.compare_specific_sizes.split(',')]
            expected_size_factors = [sf for sf in expected_size_factors if sf in specific_sizes]
            print(f"\nComparing specific size factors: {', '.join(map(str, specific_sizes))}")
        except:
            print(f"\nWARNING: Invalid specific sizes format: {args.compare_specific_sizes}. Using all available sizes.")
    
    for size_factor in expected_size_factors:
        model_path = os.path.join(config.models_dir, f'student_model_size_{size_factor}_epoch_1.pt')
        if os.path.exists(model_path):
            student_model_paths[size_factor] = model_path
    
    # If no student models found, check for the single student model specified in args
    if not student_model_paths and args.student_model:
        student_model_path = os.path.join(config.models_dir, args.student_model)
        if os.path.exists(student_model_path):
            student_model_paths = student_model_path
            print("\nWARNING: No multiple size models found. Using a single student model.")
            print("For a comprehensive size analysis, train multiple student models:")
            print("\n    python train_students.py\n")
        else:
            print(f"\nERROR: Student model file {args.student_model} not found.")
            print("No student models found. Please run the training script with distillation first:")
            print("\n    python train_students.py\n")
            return
    
    if isinstance(student_model_paths, dict) and not student_model_paths:
        print("\nERROR: No student models found. Please run the training script with distillation first:")
        print("\n    python train_students.py\n")
        return
    
    # Print found student models
    if isinstance(student_model_paths, dict):
        size_factors = sorted(student_model_paths.keys())
        print(f"\nFound {len(student_model_paths)} student models with size factors: {size_factors}")
        print(f"Size range: {min(size_factors)} to {max(size_factors)}")
        
        # Print size distribution
        print("\nSize distribution:")
        size_ranges = {"Tiny (< 0.1)": 0, "Small (0.1-0.3)": 0, "Medium (0.3-0.7)": 0, "Large (0.7-1.0)": 0}
        for sf in size_factors:
            if sf < 0.1:
                size_ranges["Tiny (< 0.1)"] += 1
            elif sf < 0.3:
                size_ranges["Small (0.1-0.3)"] += 1
            elif sf < 0.7:
                size_ranges["Medium (0.3-0.7)"] += 1
            else:
                size_ranges["Large (0.7-1.0)"] += 1
        
        for range_name, count in size_ranges.items():
            print(f"  {range_name}: {count} models")
    else:
        print(f"\nUsing single student model: {args.student_model}")
        print("Note: For a comprehensive size analysis, train multiple student models.")
    
    # Run the analysis
    print("\n" + "="*80)
    print("STARTING MODEL SIZE IMPACT ANALYSIS")
    print("="*80 + "\n")
    
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
