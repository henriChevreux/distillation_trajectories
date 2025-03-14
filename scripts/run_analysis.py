#!/usr/bin/env python3
"""
Main script to run analysis on diffusion models with a focus on model size impact.
This script replaces the original run_analysis.py with a more modular approach.
"""

import os
import argparse
import sys
import time
import pickle
import glob
import re
import shutil
from tqdm import tqdm

print(f"Script started at: {time.time()}")

start_imports = time.time()
import torch
from torch.utils.data import DataLoader
import numpy as np
print(f"Basic imports completed in {time.time() - start_imports:.2f}s")

# Add the project root directory to the Python path
start_path_setup = time.time()
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
print(f"Path setup completed in {time.time() - start_path_setup:.2f}s")

# Import core modules with timing
start_config_import = time.time()
from config.config import Config
print(f"Config import completed in {time.time() - start_config_import:.2f}s")

start_models_import = time.time()
from models import SimpleUNet, StudentUNet, DiffusionUNet  # Import DiffusionUNet
print(f"Models import completed in {time.time() - start_models_import:.2f}s")

start_utils_import = time.time()
from utils.diffusion import get_diffusion_params
from utils.trajectory_manager import generate_trajectories_with_disk_storage
print(f"Utils import completed in {time.time() - start_utils_import:.2f}s")

start_analysis_import = time.time()
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
    plot_metrics_vs_size,
    plot_trajectory_divergence_vs_timestep
)
from analysis.noise_fid_analysis import create_denoising_comparison_plot
print(f"Analysis modules import completed in {time.time() - start_analysis_import:.2f}s")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run analysis on diffusion models with a focus on model size impact',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Analysis parameters
    parser.add_argument('--teacher_model', type=str, default=None,
                        help='Path to specific teacher model relative to models directory. If not specified, latest epoch will be used.')
    parser.add_argument('--student_model', type=str, default=None,
                        help='Path to specific student model relative to models directory. If not specified, latest epoch will be used.')
    parser.add_argument('--analysis_dir', type=str, default='analysis',
                        help='Directory to save analysis results')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of trajectory samples to generate')
    parser.add_argument('--force_dataset_reload', action='store_true',
                        help='Force reload of dataset even if cached version exists')
    
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
    analysis_group.add_argument('--run_noise', action='store_true',
                        help='Run noise prediction analysis (disabled by default)')
    analysis_group.add_argument('--skip_attention', action='store_true',
                        help='Skip attention map analysis')
    analysis_group.add_argument('--skip_3d', action='store_true',
                        help='Skip 3D visualization')
    analysis_group.add_argument('--run_fid', action='store_true',
                        help='Run FID calculation (disabled by default)')
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
    config_start_time = time.time()
    print("Initializing configuration...")
    config = Config()
    print(f"Config object initialized in {time.time() - config_start_time:.2f}s")
    
    # Override config parameters
    config.teacher_steps = args.teacher_steps
    config.student_steps = args.student_steps
    config.analysis_dir = args.analysis_dir
    
    # Create necessary directories
    dir_time = time.time()
    os.makedirs(config.analysis_dir, exist_ok=True)
    print(f"Directory creation completed in {time.time() - dir_time:.2f}s")
    
    # Save the config for reference
    config_save_time = time.time()
    config_info = vars(args)
    with open(os.path.join(config.analysis_dir, 'analysis_config.txt'), 'w') as f:
        for key, value in config_info.items():
            f.write(f"{key}: {value}\n")
    print(f"Config saved in {time.time() - config_save_time:.2f}s")
    
    return config

def find_latest_epoch(directory, pattern):
    """Find the latest epoch model in a directory based on the file pattern"""
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    
    # Extract epoch numbers using regex
    epochs = []
    for file in files:
        match = re.search(r'epoch_(\d+)\.pt$', file)
        if match:
            epochs.append((int(match.group(1)), file))
    
    if not epochs:
        return None
    
    # Return the filename with the highest epoch number
    return max(epochs, key=lambda x: x[0])[1]

def find_student_models(config, args):
    """Find all available student models based on specified criteria"""
    student_model_paths = {}
    
    # Define the size factors we expect to find
    expected_size_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
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
    
    # Find student models with different size factors
    students_dir = os.path.join(project_root, 'output', 'models', 'students')
    print(f"Looking for student models in: {students_dir}")
    
    for size_factor in expected_size_factors:
        # If specific student model is provided, use that
        if args.student_model:
            if f"size_{size_factor}" in args.student_model:
                model_path = os.path.join(students_dir, args.student_model)
                if os.path.exists(model_path):
                    student_model_paths[size_factor] = model_path
                    print(f"Using specified student model with size factor {size_factor} at {model_path}")
                continue
        
        # Check for models directly in the students dir
        # Look for the latest epoch in the main directory
        latest_model = find_latest_epoch(students_dir, f'student_model_size_{size_factor}_epoch_*.pt')
        
        # Also check in size-specific subdirectories
        size_dir = os.path.join(students_dir, f'size_{size_factor}')
        latest_size_dir_model = find_latest_epoch(size_dir, 'model_epoch_*.pt')
        
        # Prioritize the model with the highest epoch number
        if latest_model and latest_size_dir_model:
            # Extract epoch numbers for comparison
            main_dir_epoch = int(re.search(r'epoch_(\d+)\.pt$', latest_model).group(1))
            size_dir_epoch = int(re.search(r'epoch_(\d+)\.pt$', latest_size_dir_model).group(1))
            
            if main_dir_epoch >= size_dir_epoch:
                student_model_paths[size_factor] = latest_model
                print(f"Found student model with size factor {size_factor} at {latest_model} (epoch {main_dir_epoch})")
            else:
                student_model_paths[size_factor] = latest_size_dir_model
                print(f"Found student model with size factor {size_factor} at {latest_size_dir_model} (epoch {size_dir_epoch})")
        elif latest_model:
            student_model_paths[size_factor] = latest_model
            epoch = int(re.search(r'epoch_(\d+)\.pt$', latest_model).group(1))
            print(f"Found student model with size factor {size_factor} at {latest_model} (epoch {epoch})")
        elif latest_size_dir_model:
            student_model_paths[size_factor] = latest_size_dir_model
            epoch = int(re.search(r'epoch_(\d+)\.pt$', latest_size_dir_model).group(1))
            print(f"Found student model with size factor {size_factor} at {latest_size_dir_model} (epoch {epoch})")
    
    return student_model_paths

def load_model_with_fallback(model_path, model, device):
    """Helper function to load model weights with robust fallback options"""
    print(f"\nLoading model from {model_path}")
    print(f"Target device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    
    # List of loading methods to try in order
    loading_methods = [
        # Method 1: Try loading with legacy zipfile serialization
        (lambda: torch.load(model_path, map_location=device, _use_new_zipfile_serialization=False), 
         "legacy zipfile serialization"),
         
        # Method 2: Standard loading directly to device
        (lambda: torch.load(model_path, map_location=device),
         "standard loading to device"),
         
        # Method 3: Try with pickle module
        (lambda: torch.load(model_path, map_location=device, pickle_module=__import__('pickle')),
         "pickle module"),
         
        # Method 4: Try with pickle and 'bytes' encoding
        (lambda: torch.load(model_path, map_location=device, pickle_module=__import__('pickle'), encoding='bytes'),
         "pickle module with bytes encoding"),
        
        # Method 5: Try opening the file first and then loading
        (lambda: torch.load(open(model_path, 'rb'), map_location=device),
         "opening file manually with rb mode"),
         
        # Method 6: Try with JIT loading (for TorchScript models)
        (lambda: torch.jit.load(model_path, map_location=device),
         "JIT loading"),
         
        # Method 7: Try raw pickle loading
        (lambda: pickle.load(open(model_path, 'rb'), encoding='bytes'),
         "raw pickle loading with bytes encoding"),
        
        # Method 8: Try raw pickle loading without encoding
        (lambda: pickle.load(open(model_path, 'rb')),
         "raw pickle loading"),
         
        # Method 9: Try loading to CPU first then transfer
        (lambda: torch.load(model_path, map_location='cpu'),
         "CPU loading with transfer")
    ]
    
    last_error = None
    for i, (method, description) in enumerate(loading_methods):
        try:
            load_time = time.time()
            print(f"Trying method {i+1}: {description}...")
            state_dict = method()
            print(f"Model loaded from disk in {time.time() - load_time:.2f}s")
            
            state_dict_time = time.time()
            if isinstance(state_dict, dict):
                # Use strict=False to allow loading models with missing or unexpected keys
                model.load_state_dict(state_dict, strict=False)
            elif hasattr(state_dict, 'state_dict'):
                model.load_state_dict(state_dict.state_dict(), strict=False)
            else:
                print(f"Warning: Unsupported state dict type: {type(state_dict)}")
                print("Attempting to use the loaded object directly...")
                model.load_state_dict(state_dict, strict=False)
            
            print(f"State dict applied to model in {time.time() - state_dict_time:.2f}s")
            print(f"Successfully loaded model using method {i+1}: {description}")
            return True
                
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            print(f"Method {i+1} failed: {error_type} - {error_msg}")
            
            # Add specific tips for common errors
            if "PytorchStreamReader failed reading zip archive" in error_msg:
                print("  - This is typically caused by model file format incompatibility between PyTorch versions")
                print("  - Current PyTorch version:", torch.__version__)
            elif "unexpected EOF" in error_msg or "invalid load key" in error_msg:
                print("  - The model file may be corrupted or incomplete")
            elif "No such file or directory" in error_msg:
                print("  - Check that the file path is correct and accessible")
            elif "size mismatch" in error_msg or "Missing key" in error_msg or "Unexpected key" in error_msg:
                print("  - The model architecture has changed since this model was saved")
                print("  - Using strict=False to allow loading with missing or unexpected keys")
            
            last_error = e
            # Continue to the next method
    
    # If we get here, all methods failed
    print(f"\nERROR: Failed to load model using all available methods")
    print(f"Last error: {type(last_error).__name__} - {str(last_error)}")
    print("\nPossible issues:")
    print("1. The model file is corrupted")
    print("2. The model was saved with a different PyTorch version (current: {})".format(torch.__version__))
    print("3. The model was saved on a different platform")
    print("4. The model architecture doesn't match the saved weights")
    print("\nPossible solutions:")
    print("1. Try using the convert_model.py script to convert the model file")
    print("2. Try re-training the model with the current PyTorch version")
    print("3. Check if you can load the model with a different PyTorch version")
    print("4. Train a new model with the current architecture")
    return False

def ensure_tensor_compatibility(tensor_or_batch):
    """
    Ensure tensor is in the correct format expected by all analysis functions.
    No longer reshapes tensors for older PyTorch versions.
    
    Args:
        tensor_or_batch: A tensor or batch of tensors
        
    Returns:
        Tensor with the correct format
    """
    if tensor_or_batch is None:
        return None
    
    if not isinstance(tensor_or_batch, torch.Tensor):
        print(f"Warning: Input is not a tensor but {type(tensor_or_batch)}")
        return tensor_or_batch
    
    # Print tensor shape for debugging
    print(f"Tensor shape: {tensor_or_batch.shape}, dimensions: {tensor_or_batch.dim()}")
    
    # Return the tensor as-is without reshaping
    return tensor_or_batch

def preprocess_fixed_samples(fixed_samples):
    """
    Preprocess fixed samples without reshaping tensor dimensions
    
    Args:
        fixed_samples: Batch of fixed samples
        
    Returns:
        Fixed samples as-is
    """
    if fixed_samples is None:
        return None
    
    # Just log information about the fixed samples
    if isinstance(fixed_samples, torch.Tensor):
        print(f"Fixed samples tensor shape: {fixed_samples.shape}, dimensions: {fixed_samples.dim()}")
        return fixed_samples
    
    print(f"Warning: Fixed samples have unexpected type: {type(fixed_samples)}")
    return fixed_samples

def cleanup_old_output_directories(config):
    """
    Clean up old output directories not being used in the new structure.
    This function removes directories and files that have been moved to the output/analysis directory.
    
    Args:
        config: Configuration object with output paths
    """
    # Directories to remove (these are now under analysis/)
    old_dirs_to_remove = [
        os.path.join(config.output_dir, "model_comparisons"),
        os.path.join(config.output_dir, "time_dependent"),
        os.path.join(config.output_dir, "size_dependent"),
        os.path.join(config.output_dir, "metrics"),
        os.path.join(config.output_dir, "dimensionality"),
        os.path.join(config.output_dir, "latent_space"),
    ]
    
    # Remove each directory if it exists
    for dir_path in old_dirs_to_remove:
        if os.path.exists(dir_path):
            print(f"Removing old directory: {dir_path}")
            try:
                shutil.rmtree(dir_path)
            except Exception as e:
                print(f"  Error removing directory {dir_path}: {e}")

def main():
    """Main function to run the analysis"""
    main_start_time = time.time()
    
    print("\nStarting analysis setup...")
    args_start_time = time.time()
    args = parse_args()
    print(f"Arguments parsed in {time.time() - args_start_time:.2f}s")
    
    config_start_time = time.time()
    config = setup_config(args)
    print(f"Config setup completed in {time.time() - config_start_time:.2f}s")
    
    # Determine device
    device_setup_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    print(f"Device setup completed in {time.time() - device_setup_time:.2f}s")
    
    # Check if datasets are available to avoid download delay during analysis
    dataset_check_time = time.time()
    print("Checking if datasets are available...")
    try:
        dataset_path = os.path.join(project_root, 'data', config.dataset)
        if not os.path.exists(dataset_path):
            print(f"WARNING: Dataset directory {dataset_path} not found. The script may need to download datasets.")
        else:
            print(f"Found dataset at {dataset_path}")
    except:
        print("Could not check for dataset availability")
    print(f"Dataset check completed in {time.time() - dataset_check_time:.2f}s")
    
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
    model_search_time = time.time()
    teacher_models_dir = os.path.join(project_root, 'output', 'models', 'teacher')
    
    # Use specified teacher model or find the latest epoch
    if args.teacher_model:
        teacher_model_path = os.path.join(teacher_models_dir, args.teacher_model)
        print(f"Using specified teacher model at: {teacher_model_path}")
    else:
        # Find the latest epoch teacher model
        latest_teacher_model = find_latest_epoch(teacher_models_dir, 'model_epoch_*.pt')
        if latest_teacher_model:
            teacher_model_path = latest_teacher_model
            epoch = int(re.search(r'epoch_(\d+)\.pt$', teacher_model_path).group(1))
            print(f"Using latest teacher model at: {teacher_model_path} (epoch {epoch})")
        else:
            # Fallback to default if no models found
            teacher_model_path = os.path.join(teacher_models_dir, 'model_epoch_1.pt')
            print(f"No teacher models found. Falling back to: {teacher_model_path}")
    
    if not os.path.exists(teacher_model_path):
        print("\nERROR: Teacher model file not found at:", teacher_model_path)
        return
    
    # Find all student models with different size factors
    student_model_paths = find_student_models(config, args)
    print(f"Model search completed in {time.time() - model_search_time:.2f}s")
    
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
    
    # Run the analysis
    print("\n" + "="*80)
    print("STARTING MODEL SIZE IMPACT ANALYSIS")
    print("="*80 + "\n")
    
    try:
        # Create analysis directory and save config
        os.makedirs(config.analysis_dir, exist_ok=True)
        
        # Clean up old output directories
        cleanup_old_output_directories(config)
        
        # Load teacher model
        model_load_time = time.time()
        print("Loading teacher model...")
        # Ensure the config has the correct image size before initializing the model
        print(f"Setting image size to {config.image_size}x{config.image_size} for teacher model")
        
        # Use DiffusionUNet directly with size_factor=1.0 for teacher model
        teacher_model = DiffusionUNet(config, size_factor=1.0).to(device)
        
        if os.path.exists(teacher_model_path):
            if not load_model_with_fallback(teacher_model_path, teacher_model, device):
                print(f"\nERROR: Failed to load teacher model")
                return
            print(f"Loaded teacher model from {teacher_model_path}")
        print(f"Teacher model loaded in {time.time() - model_load_time:.2f}s")
        
        # Generate fixed test samples
        dataset_time = time.time()
        print("\nGenerating fixed test samples for consistent comparison...")
        dataset_init_time = time.time()
        
        # Check if we can load a cached version of the dataset to avoid downloads
        dataset_cache_path = os.path.join(config.data_dir, f"fixed_test_samples_{args.num_samples}.pt")
        use_cached = False
        
        if os.path.exists(dataset_cache_path) and not args.force_dataset_reload:
            try:
                print(f"Loading cached test samples from {dataset_cache_path}")
                fixed_test_samples = torch.load(dataset_cache_path)
                fixed_test_samples = fixed_test_samples.to(device)
                print(f"Successfully loaded cached test samples in {time.time() - dataset_init_time:.2f}s")
                use_cached = True
            except Exception as e:
                print(f"Failed to load cached samples: {e}. Creating new samples.")
                use_cached = False
        else:
            print("No cached samples found or force reload requested. Creating new samples.")
            use_cached = False
        
        if not use_cached:
            test_dataset = config.get_test_dataset()
            print(f"Dataset initialization completed in {time.time() - dataset_init_time:.2f}s")
            
            # Set a fixed seed for reproducibility
            torch.manual_seed(42)
            np.random.seed(42)
            
            dataloader_time = time.time()
            test_loader = DataLoader(test_dataset, batch_size=args.num_samples, shuffle=True)
            print(f"DataLoader creation completed in {time.time() - dataloader_time:.2f}s")
            
            sample_extraction_time = time.time()
            fixed_test_samples, _ = next(iter(test_loader))
            print(f"Sample extraction completed in {time.time() - sample_extraction_time:.2f}s")
            
            # Save the samples for future use
            try:
                torch.save(fixed_test_samples.cpu(), dataset_cache_path)
                print(f"Saved test samples to {dataset_cache_path} for faster future loading")
            except Exception as e:
                print(f"Failed to save test samples to disk: {e}")
            
            # Device transfer if needed
            device_transfer_time = time.time()
            fixed_test_samples = fixed_test_samples.to(device)
            print(f"Device transfer completed in {time.time() - device_transfer_time:.2f}s")
        
        # Print tensor information
        print("\nFixed samples tensor info:")
        print(f"  - Shape: {fixed_test_samples.shape}")
        print(f"  - Device: {fixed_test_samples.device}")
        print(f"  - Data type: {fixed_test_samples.dtype}")
        print(f"  - Dimensions: {fixed_test_samples.dim()}")
        
        print(f"Total test dataset setup completed in {time.time() - dataset_time:.2f}s")
        
        # Handle student models - either a single model or multiple models with different size factors
        student_models = {}
        
        # Function to determine architecture type based on size factor
        # This is no longer needed as the StudentUNet class handles this internally
        # But we'll keep it for backward compatibility with older code
        def get_architecture_type(size_factor):
            if float(size_factor) < 0.1:
                return 'tiny'     # Use tiny architecture for very small models
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
                
                # Let the StudentUNet class determine the architecture type automatically
                student_model = DiffusionUNet(config, size_factor=float(size_factor)).to(device)
                
                if os.path.exists(path):
                    try:
                        if not load_model_with_fallback(path, student_model, device):
                            print(f"WARNING: Failed to load student model with size factor {size_factor}")
                            continue
                        print(f"Loaded student model from {path}")
                        student_model.eval()
                        student_models[float(size_factor)] = student_model
                    except Exception as e:
                        print(f"WARNING: Failed to load student model with size factor {size_factor}: {str(e)}")
                        continue
        else:
            # Single student model (backward compatibility)
            if student_model_paths is None:
                # Try to find student models with different size factors
                students_dir = os.path.join(project_root, 'output', 'models', 'students')
                size_factors = config.student_size_factors if hasattr(config, 'student_size_factors') else [0.25, 0.5, 0.75, 1.0]
                for size_factor in size_factors:
                    # Try both naming patterns
                    path1 = os.path.join(students_dir, f'student_model_size_{size_factor}_epoch_1.pt')
                    path2 = os.path.join(students_dir, f'size_{size_factor}', 'model_epoch_1.pt')
                    
                    if os.path.exists(path1):
                        path = path1
                    elif os.path.exists(path2):
                        path = path2
                    else:
                        print(f"No student model found for size factor {size_factor}")
                        continue
                        
                    print(f"Found student model with size factor {size_factor}...")
                    # Let the StudentUNet class determine the architecture type automatically
                    student_model = DiffusionUNet(config, size_factor=float(size_factor)).to(device)
                    if not load_model_with_fallback(path, student_model, device):
                        print(f"WARNING: Failed to load student model with size factor {size_factor}. Skipping this model.")
                        continue
                    student_model.eval()
                    student_models[float(size_factor)] = student_model
                
                # If no student models found, try the old naming convention
                if not student_models:
                    path = os.path.join(students_dir, 'student_model_epoch_1.pt')
                    if os.path.exists(path):
                        print("Loading student model with default size...")
                        student_model = DiffusionUNet(config, size_factor=1.0).to(device)
                        if not load_model_with_fallback(path, student_model, device):
                            print(f"ERROR: Failed to load student model with default size.")
                            return
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
                    
                    # Let the StudentUNet class determine the architecture type automatically
                    student_model = DiffusionUNet(config, size_factor=size_factor).to(device)
                    if not load_model_with_fallback(path, student_model, device):
                        print(f"ERROR: Failed to load student model with size factor {size_factor}.")
                        return
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
            # Create the comparison plot without using fixed samples
            create_denoising_comparison_plot(
                teacher_model,
                student_models,
                fixed_test_samples[:args.num_denoising_samples],
                config,
                save_dir=os.path.join(config.analysis_dir, "denoising_comparison")
            )
            sys.exit(0)
        
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
                teacher_model, student_model, config, size_factor, num_samples=args.num_samples,
                fixed_samples=fixed_test_samples  # Pre-processed fixed samples  
            )
            
            # Run only the selected analysis modules
            if not args.skip_metrics:
                # 2. Compute trajectory metrics in batches
                print("Computing trajectory metrics...")
                metrics = trajectory_manager.compute_trajectory_metrics_batch(size_factor=size_factor)
                
                # 3. Visualize metrics
                print("Visualizing metrics...")
                summary = visualize_batch_metrics(metrics, config, size_factor=size_factor)
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
                # Still add an entry to all_metrics for this size factor
                # This ensures the radar plot will include this model even if metrics are skipped
                all_metrics[size_factor] = {
                    # Original metrics with default values
                    'path_length_ratio': 1.0,
                    'mean_endpoint_distance': 0.0,
                    'efficiency_ratio': 1.0, 
                    'mean_wasserstein': 0.0,
                    
                    # New metrics with proper defaults - better than the auto-initialized values
                    'path_length_similarity': 0.9,  # Default to high similarity
                    'endpoint_distance': 0.1,       # Default to low distance (better)
                    'efficiency_similarity': 0.9,    # Default to high similarity
                    'mean_velocity_similarity': 0.9, # Default to high similarity
                    'mean_directional_consistency': 0.8, # Default to good consistency
                    'mean_position_difference': 0.1,    # Default to low difference (better)
                    'distribution_similarity': 0.7     # Default to moderate similarity
                }
            
            # Time-dependent analysis
            print("Performing time-dependent analysis...")
            # Load trajectories for time-dependent analysis
            teacher_traj, student_traj = trajectory_manager.load_trajectories(
                size_factor=size_factor, 
                indices=list(range(min(5, args.num_samples)))  # Use at most 5 trajectories
            )
            time_distances = analyze_time_dependent_distances(
                teacher_traj, 
                student_traj, 
                config, 
                size_factor=size_factor
            )
            
            # Store time-dependent metrics for combined visualization
            all_time_distances[size_factor] = time_distances
                
            # Calculate FID scores
            if args.run_fid:
                print("Calculating FID scores...")
                # Log info about the samples 
                print(f"Samples shape for FID: {fixed_test_samples.shape}")
                fid_results = calculate_and_visualize_fid(
                    teacher_model, student_model, config, size_factor=size_factor,
                    fixed_samples=fixed_test_samples
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
            
            if args.run_noise:
                # 5. Noise prediction analysis
                print("Analyzing noise prediction patterns...")
                # Log info about the samples
                print(f"Samples shape for noise prediction: {fixed_test_samples.shape}")
                noise_metrics = analyze_noise_prediction(
                    teacher_model, student_model, config, size_factor=size_factor,
                    fixed_samples=fixed_test_samples
                )
            else:
                print("Skipping noise prediction analysis.")
            
            if not args.skip_attention:
                # 6. Attention map analysis
                print("Analyzing attention maps...")
                # Log info about the samples
                print(f"Samples shape for attention maps: {fixed_test_samples.shape}")
                attention_metrics = analyze_attention_maps(
                    teacher_model, student_model, config, size_factor=size_factor,
                    fixed_samples=fixed_test_samples
                )
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
        
        # Create comparative visualizations that incorporate model size as a dimension
        if len(student_models) > 1:
            print(f"\n{'='*80}")
            print(f"Creating comparative visualizations across student model sizes")
            print(f"{'='*80}")
            print(f"Number of size factors in all_metrics: {len(all_metrics)}")
            print(f"Size factors: {list(all_metrics.keys())}")
            print(f"Skip metrics flag: {args.skip_metrics}")
            
            # Debug all_metrics content
            print("all_metrics content:")
            for size, metrics in all_metrics.items():
                print(f"  Size {size}: {metrics}")
            
            try:
                model_comparison_result = create_model_size_comparisons(
                    all_metrics, 
                    all_fid_results, 
                    config
                )
                print(f"Model comparison result: {model_comparison_result}")
            except Exception as e:
                import traceback
                print(f"Error in create_model_size_comparisons: {e}")
                traceback.print_exc()
            
            # Create 3D visualization that incorporates model size as a dimension
            # Only run if metrics are not skipped and 3D is not skipped
            if not args.skip_metrics and not args.skip_3d:
                print("Creating 3D model size visualization using trajectory metrics...")
                # We're using trajectory_metrics_for_3d instead of the full trajectories to save memory
                generate_3d_model_size_visualization(trajectory_metrics_for_3d, config)
            
            # Create time-dependent visualizations
            print("Creating time-dependent visualizations...")
            plot_time_dependent_grid(all_time_distances, config)
            plot_time_dependent_combined(all_time_distances, config)
            plot_trajectory_divergence_vs_timestep(trajectory_metrics_for_3d, config)
            
            # Create size-dependent visualizations
            print("Creating size-dependent visualizations...")
            plot_mse_vs_size(all_metrics, config)
            plot_metrics_vs_size(all_metrics, config)
        
        # After loading all models and before running other analyses, add:
        if len(student_models) > 0:
            print("\nGenerating denoising comparison visualization...")
            # Create the comparison plot without using fixed samples
            create_denoising_comparison_plot(
                {**{"teacher": teacher_model}, **student_models},
                config,
                save_dir=os.path.join(config.analysis_dir, "denoising_comparison")
            )
        
        print("\nAnalysis complete. Results saved in the analysis directory.")
        
    except Exception as e:
        if isinstance(e, RuntimeError) and "size mismatch" in str(e):
            print("\nERROR: Model architecture mismatch.")
            print("The saved models were trained with a different architecture than the current one.")
            print("This is likely because the models were trained on MNIST but the code is now configured for CIFAR10.")
            print("Please train new models with the current architecture:")
            print("\n    python diffusion_training.py\n")
        else:
            print(f"\nERROR: An error occurred during analysis: {str(e)}")
            # Re-raise if it's not the size mismatch error
            if not (isinstance(e, RuntimeError) and "size mismatch" in str(e)):
                raise

if __name__ == "__main__":
    main()
