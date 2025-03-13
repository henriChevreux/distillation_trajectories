#!/usr/bin/env python3
"""
Main script to run analysis on diffusion models with a focus on model size impact.
This script replaces the original run_analysis.py with a more modular approach.
"""

import os
import argparse
import torch
import sys
import os
from torch.utils.data import DataLoader

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config
from models import SimpleUNet, StudentUNet
from utils.diffusion import get_diffusion_params
from utils.trajectory_manager import generate_trajectories_with_disk_storage
from analysis import (
    compute_trajectory_metrics,
    visualize_metrics,
    visualize_batch_metrics,
    dimensionality_reduction_analysis,
    analyze_noise_prediction,
    analyze_attention_maps,
    generate_latent_space_visualization,
    generate_3d_model_size_visualization,
    create_model_size_comparisons,
    calculate_and_visualize_fid,
    analyze_time_dependent_distances,
    plot_time_dependent_grid,
    plot_time_dependent_combined,
    plot_mse_vs_size,
    plot_metrics_vs_size
)
from analysis.noise_fid_analysis import create_denoising_comparison_plot

def parse_args():
    """Parse command line arguments"""
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
    analysis_group = parser.add_argument_group('Analysis Modules')
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
    analysis_group.add_argument('--only_denoising', action='store_true',
                        help='Only run the denoising comparison, skip all other analyses')
    
    # Model size focus
    size_group = parser.add_argument_group('Model Size Analysis')
    size_group.add_argument('--focus_size_range', type=str, default=None,
                        help='Focus on a specific size range (e.g., "0.1-0.5")')
    size_group.add_argument('--compare_specific_sizes', type=str, default=None,
                        help='Compare specific size factors (comma-separated, e.g., "0.1,0.5,1.0")')
    
    # Add new argument for denoising comparison
    parser.add_argument('--num_denoising_samples', type=int, default=5,
                       help='Number of samples to use in denoising comparison')
    
    return parser.parse_args()

def setup_config(args):
    """Create and configure the Config object based on arguments"""
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

def find_student_models(config, args):
    """Find all available student models based on specified criteria"""
    student_model_paths = {}
    
    # Define the size factors we expect to find
    expected_size_factors = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
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
    
        # Find all student models with different size factors
        for size_factor in expected_size_factors:
            # Try different naming patterns in the new directory structure
            size_dir = os.path.join(config.student_models_dir, f'size_{size_factor}')
            if os.path.exists(size_dir):
                possible_paths = [
                    os.path.join(size_dir, f'model_epoch_1.pt'),
                    os.path.join(size_dir, f'model_epoch_0.pt'),
                    os.path.join(size_dir, f'model.pt')
                ]
                
                # Also check old naming patterns for backward compatibility
                old_paths = [
                    os.path.join(config.models_dir, f'student_model_size_{size_factor}_epoch_1.pt'),
                    os.path.join(config.models_dir, f'student_model_size_{size_factor}_epoch_0.pt'),
                    os.path.join(config.models_dir, f'student_model_{size_factor}_epoch_1.pt'),
                    os.path.join(config.models_dir, f'student_model_{size_factor}.pt')
                ]
                
                all_possible_paths = possible_paths + old_paths
                
                for path in all_possible_paths:
                    if os.path.exists(path):
                        student_model_paths[size_factor] = path
                        print(f"Found student model with size factor {size_factor} at {path}")
                        break
    
    return student_model_paths

def main():
    """Main function to run the analysis"""
    args = parse_args()
    config = setup_config(args)
    
    # Determine device
    device_name = "CUDA" if torch.cuda.is_available() else "MPS" if torch.backends.mps.is_available() else "CPU"
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running analysis using {device_name} device")
    
    # Print analysis configuration
    print("\n" + "="*80)
    print("MODEL SIZE IMPACT ANALYSIS")
    print("="*80)
    print(f"\nAnalysis Configuration:")
    print(f"Teacher model: {args.teacher_model}")
    print(f"Teacher timesteps: {args.teacher_steps}")
    print(f"Student timesteps: {args.student_steps}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Analysis directory: {args.analysis_dir}")
    
    # Check if teacher model exists
    teacher_model_path = os.path.join(config.teacher_models_dir, args.teacher_model)
    if not os.path.exists(teacher_model_path):
        print("\nERROR: Teacher model file not found. You need to train the teacher model first.")
        print("Please run the training script first to generate the teacher model file:")
        print("\n    python diffusion_training.py\n")
        return
    
    # Find all student models with different size factors
    student_model_paths = {}
    
    # Define the size factors we expect to find
    expected_size_factors = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
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
    
    # Find all student models with different size factors
    for size_factor in expected_size_factors:
        # Try different naming patterns in the new directory structure
        size_dir = os.path.join(config.student_models_dir, f'size_{size_factor}')
        if os.path.exists(size_dir):
            possible_paths = [
                os.path.join(size_dir, f'model_epoch_1.pt'),
                os.path.join(size_dir, f'model_epoch_0.pt'),
                os.path.join(size_dir, f'model.pt')
            ]
            
            # Also check old naming patterns for backward compatibility
            old_paths = [
                os.path.join(config.models_dir, f'student_model_size_{size_factor}_epoch_1.pt'),
                os.path.join(config.models_dir, f'student_model_size_{size_factor}_epoch_0.pt'),
                os.path.join(config.models_dir, f'student_model_{size_factor}_epoch_1.pt'),
                os.path.join(config.models_dir, f'student_model_{size_factor}.pt')
            ]
            
            all_possible_paths = possible_paths + old_paths
            
            for path in all_possible_paths:
                if os.path.exists(path):
                    student_model_paths[size_factor] = path
                    print(f"Found student model with size factor {size_factor} at {path}")
                    break
    
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
        # Load teacher model
        print("Loading teacher model...")
        teacher_model = SimpleUNet(config).to(device)
        
        if os.path.exists(teacher_model_path):
            teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=device))
            print(f"Loaded teacher model from {teacher_model_path}")
        else:
            print(f"ERROR: Teacher model not found at {teacher_model_path}. Please run training first.")
            return
        
        # Set teacher model to evaluation mode
        teacher_model.eval()
        
        # Handle student models - either a single model or multiple models with different size factors
        student_models = {}
        
        # Function to determine architecture type based on size factor
        def get_architecture_type(size_factor):
            if float(size_factor) < 0.1:
                return 'tiny'     # Use the smallest architecture for very small models
            elif float(size_factor) < 0.3:
                return 'small'    # Use small architecture for small models
            elif float(size_factor) < 0.7:
                return 'medium'   # Use medium architecture for medium models
            else:
                return 'full'     # Use full architecture for large models
        
        if isinstance(student_model_paths, dict):
            # Multiple student models with different size factors
            for size_factor, path in student_model_paths.items():
                print(f"Loading student model with size factor {size_factor}...")
                
                # Determine architecture type based on size factor
                architecture_type = get_architecture_type(size_factor)
                
                print(f"Using architecture type: {architecture_type} for size factor {size_factor}")
                student_model = StudentUNet(config, size_factor=float(size_factor), architecture_type=architecture_type).to(device)
                
                if os.path.exists(path):
                    student_model.load_state_dict(torch.load(path, map_location=device))
                    print(f"Loaded student model from {path}")
                    student_model.eval()
                    student_models[float(size_factor)] = student_model
                else:
                    print(f"WARNING: Student model not found at {path}. Skipping this size factor.")
        else:
            # Single student model (backward compatibility)
            if student_model_paths is None:
                # Try to find student models with different size factors
                size_factors = config.student_size_factors if hasattr(config, 'student_size_factors') else [0.25, 0.5, 0.75, 1.0]
                for size_factor in size_factors:
                    path = os.path.join(config.models_dir, f'student_model_size_{size_factor}_epoch_1.pt')
                    if os.path.exists(path):
                        print(f"Found student model with size factor {size_factor}...")
                        architecture_type = get_architecture_type(size_factor)
                        student_model = StudentUNet(config, size_factor=float(size_factor), architecture_type=architecture_type).to(device)
                        student_model.load_state_dict(torch.load(path, map_location=device))
                        student_model.eval()
                        student_models[float(size_factor)] = student_model
                    else:
                        print(f"No student model found for size factor {size_factor}")
                
                # If no student models found, try the old naming convention
                if not student_models:
                    path = os.path.join(config.models_dir, 'student_model_epoch_1.pt')
                    if os.path.exists(path):
                        print("Loading student model with default size...")
                        student_model = SimpleUNet(config).to(device)
                        student_model.load_state_dict(torch.load(path, map_location=device))
                        student_model.eval()
                        student_models[1.0] = student_model
                    else:
                        print(f"ERROR: No student models found. Please run training with distillation first.")
                        return
            else:
                # Single specified student model path
                path = student_model_paths
                if os.path.exists(path):
                    print("Loading student model...")
                    # Try to extract size factor from filename
                    size_factor = 1.0  # Default
                    if "size_" in path:
                        try:
                            size_str = path.split("size_")[1].split("_")[0]
                            size_factor = float(size_str)
                        except:
                            pass
                    
                    architecture_type = get_architecture_type(size_factor)
                    student_model = StudentUNet(config, size_factor=size_factor, architecture_type=architecture_type).to(device)
                    student_model.load_state_dict(torch.load(path, map_location=device))
                    student_model.eval()
                    student_models[size_factor] = student_model
                else:
                    print(f"ERROR: Student model not found at {path}. Please run training with distillation first.")
                    return
        
        # Print summary of loaded models
        print(f"\nLoaded {len(student_models)} student models with size factors: {list(student_models.keys())}")
        
        # If only running denoising comparison, skip other analyses
        if args.only_denoising:
            print("\nRunning only denoising comparison...")
            # Get some test samples
            test_dataset = config.get_test_dataset()  # Returns CIFAR10 or MNIST dataset
            test_loader = DataLoader(test_dataset, batch_size=args.num_denoising_samples, shuffle=True)
            test_samples, _ = next(iter(test_loader))  # Unpack directly as we know it returns (data, target)
            test_samples = test_samples.to(device)
            
            # Create the comparison plot
            create_denoising_comparison_plot(
                {**{"teacher": teacher_model}, **student_models},
                config,
                save_dir=os.path.join(config.analysis_dir, "denoising_comparison")
            )
            print("Denoising comparison saved in analysis/denoising_comparison/")
            return
        
        # Analyze each student model
        all_metrics = {}
        all_fid_results = {}
        all_time_distances = {}
        
        # Only store trajectory metrics for 3D visualization, not the full trajectories
        trajectory_metrics_for_3d = {}
            
        for size_factor, student_model in student_models.items():
            print(f"\n{'='*80}")
            print(f"Analyzing student model with size factor {size_factor}")
            print(f"{'='*80}")
            
            # 1. Generate multiple trajectories
            print("Generating trajectories and storing on disk...")
            trajectory_manager = generate_trajectories_with_disk_storage(
                teacher_model, student_model, config, size_factor, num_samples=args.num_samples
            )
            
            # Run only the selected analysis modules
            if not args.skip_metrics:
                # 2. Compute trajectory metrics in batches
                print("Computing trajectory metrics...")
                metrics = trajectory_manager.compute_trajectory_metrics_batch(size_factor=size_factor)
                
                # 3. Visualize metrics
                print("Visualizing metrics...")
                summary = visualize_batch_metrics(metrics, config, suffix=f"_size_{size_factor}")
                print("Metrics summary:", summary)
                
                # Store metrics for comparative analysis
                all_metrics[size_factor] = summary
                
                # Store only the metrics needed for 3D visualization, not the full trajectories
                trajectory_metrics_for_3d[size_factor] = {
                    'wasserstein_distances': metrics['wasserstein_distances'],
                    'wasserstein_distances_per_timestep': metrics['wasserstein_distances_per_timestep'],
                    'endpoint_distances': metrics['endpoint_distances'],
                    'teacher_path_lengths': metrics['teacher_path_lengths'],
                    'student_path_lengths': metrics['student_path_lengths'],
                    'teacher_efficiency': metrics['teacher_efficiency'],
                    'student_efficiency': metrics['student_efficiency']
                }
            else:
                print("Skipping trajectory metrics analysis.")
                
            # Time-dependent analysis
            print("Performing time-dependent analysis...")
            # Load trajectories for time-dependent analysis
            teacher_traj, student_traj = trajectory_manager.load_trajectories(
                size_factor=size_factor, 
                indices=list(range(min(5, args.num_samples)))  # Use at most 5 trajectories
            )
            time_distances = analyze_time_dependent_distances(teacher_traj, student_traj, config, size_factor=size_factor)
            
            # Store time-dependent metrics for combined visualization
            all_time_distances[size_factor] = time_distances
                
            # Calculate FID scores
            if not args.skip_fid:
                print("Calculating FID scores...")
                fid_results = calculate_and_visualize_fid(
                    teacher_model, student_model, config, size_factor=size_factor
                )
                all_fid_results[size_factor] = fid_results
            else:
                print("Skipping FID calculation.")
            
            if not args.skip_dimensionality:
                # 4. Dimensionality reduction analysis
                print("Performing dimensionality reduction analysis...")
                # Load a subset of trajectories just for this analysis
                teacher_subset, student_subset = trajectory_manager.load_trajectories(
                    size_factor=size_factor, 
                    indices=list(range(min(5, args.num_samples)))  # Use at most 5 trajectories
                    )
                dimensionality_reduction_analysis(teacher_subset, student_subset, config, size_factor=size_factor)
            else:
                print("Skipping dimensionality reduction analysis.")
            
            if not args.skip_noise:
                # 5. Noise prediction analysis
                print("Analyzing noise prediction patterns...")
                noise_metrics = analyze_noise_prediction(teacher_model, student_model, config, size_factor=size_factor)
            else:
                print("Skipping noise prediction analysis.")
            
            if not args.skip_attention:
                # 6. Attention map analysis
                print("Analyzing attention maps...")
                # Get test samples for attention analysis
                test_dataset = config.get_test_dataset()
                test_loader = DataLoader(test_dataset, batch_size=args.num_samples, shuffle=True)
                test_samples, _ = next(iter(test_loader))
                test_samples = test_samples.to(device)
                
                # Create a dictionary with just this student model for the analysis
                current_student_models = {size_factor: student_model}
                attention_metrics = analyze_attention_maps(teacher_model, student_model, config, size_factor=size_factor)
            else:
                print("Skipping attention map analysis.")
            
            if not args.skip_3d:
                # 7. Generate 3D latent space visualization
                print("Creating 3D latent space visualization...")
                # Load just one trajectory for 3D visualization
                teacher_viz, student_viz = trajectory_manager.load_trajectories(
                    size_factor=size_factor, 
                    indices=[0]  # Just use the first trajectory
                    )
                generate_latent_space_visualization(teacher_viz, student_viz, config, size_factor=size_factor)
            else:
                print("Skipping 3D latent space visualization.")
            
            # Clear trajectories after analysis to free memory
            import gc
            gc.collect()  # Force garbage collection
        
        # Create comparative visualizations across different student model sizes
        if len(student_models) > 1 and not args.skip_metrics:
            print("\nCreating comparative visualizations across student model sizes...")
            create_model_size_comparisons(all_metrics, all_fid_results, config)
            
            # Create 3D visualization that incorporates model size as a dimension
            print("Creating 3D model size visualization using trajectory metrics...")
            # We're using trajectory_metrics_for_3d instead of the full trajectories to save memory
            generate_3d_model_size_visualization(trajectory_metrics_for_3d, config)
            
            # Create time-dependent visualizations
            print("Creating time-dependent visualizations...")
            plot_time_dependent_grid(all_time_distances, config)
            plot_time_dependent_combined(all_time_distances, config)
            
            # Create size-dependent visualizations
            print("Creating size-dependent visualizations...")
            plot_mse_vs_size(all_metrics, config)
            plot_metrics_vs_size(all_metrics, config)
        
        # After loading all models and before running other analyses, add:
        if len(student_models) > 0:
            print("\nGenerating denoising comparison visualization...")
            # Get some test samples
            test_dataset = config.get_test_dataset()  # Returns CIFAR10 or MNIST dataset
            test_loader = DataLoader(test_dataset, batch_size=args.num_denoising_samples, shuffle=True)
            test_samples, _ = next(iter(test_loader))  # Unpack directly as we know it returns (data, target)
            test_samples = test_samples.to(device)
            
            # Create the comparison plot
            create_denoising_comparison_plot(
                {**{"teacher": teacher_model}, **student_models},
                config,
                save_dir=os.path.join(config.analysis_dir, "denoising_comparison")
            )
            print("Denoising comparison saved in analysis/denoising_comparison/")
        
        print("\nAnalysis complete. Results saved in the analysis directory.")
        
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
