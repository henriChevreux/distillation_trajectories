#!/usr/bin/env python3
"""
Script to run comprehensive analysis on trained diffusion models.
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config
from models import SimpleUNet, StudentUNet
from utils.diffusion import get_diffusion_params

from analysis import (
    attention_patterns,
    denoising_comparison,
    dimensionality_reduction,
    mse_heatmap,
    size_analysis,
    time_analysis,
    trajectory_metrics
)   
def setup_analysis(args):
    """
    Set up the analysis environment by loading models and test data
    """
    # Load configuration
    config = Config()
    
    # Load teacher model
    teacher_path = os.path.join(config.models_dir, 'teacher', 'model_final.pt')
    if not os.path.exists(teacher_path):
        raise FileNotFoundError(f"Teacher model not found at {teacher_path}")
    
    # Create teacher model instance
    teacher_model = SimpleUNet(config)
    # Load state dict
    state_dict = torch.load(teacher_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    teacher_model.load_state_dict(state_dict)
    teacher_model.eval()
    
    # Load student models
    student_models = {}
    student_dir = os.path.join(config.models_dir, 'students')
    for size_factor in config.student_size_factors:
        for image_size_factor in config.student_image_size_factors:
            # Check for model in the size_X_img_Y/model.pt format
            model_dir = os.path.join(student_dir, f'size_{size_factor}_img_{image_size_factor}')
            model_path = os.path.join(model_dir, 'model.pt')
            
            if os.path.exists(model_path):
                print(f"Loading student model from {model_path}")
                # Create student model instance with appropriate architecture
                student_model = StudentUNet(config, size_factor)
                # Load state dict
                state_dict = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
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
            num_workers=2
        )
        test_samples = next(iter(test_loader))[0]  # Only take the images, not labels
    else:
        raise FileNotFoundError(f"Test dataset not found at {test_dir}")
    
    return config, teacher_model, student_models, test_samples

def run_analysis(args):
    """
    Run the complete analysis pipeline
    """
    print("Setting up analysis environment...")
    config, teacher_model, student_models, test_samples = setup_analysis(args)
    
    # Add aliases for compatibility with analysis modules
    config.teacher_steps = config.timesteps  # Add this alias
    config.original_image_size = config.teacher_image_size  # Add this alias
    
    # Add other potentially missing attributes
    if not hasattr(config, 'convergence_dir'):
        config.convergence_dir = os.path.join(config.comparative_dir, 'convergence')
        os.makedirs(config.convergence_dir, exist_ok=True)
    
    # Override args with config if testing mode is enabled
    if hasattr(config, 'analysis_testing_mode') and config.analysis_testing_mode:
        if hasattr(config, 'analysis_num_samples'):
            args.num_samples = config.analysis_num_samples
            print(f"Using config setting for number of samples: {args.num_samples}")
    else:
        # If no testing mode in config, update config with args
        if hasattr(config, 'analysis_num_samples'):
            config.analysis_num_samples = args.num_samples
            
    print(f"Using {config.timesteps} timesteps for teacher model, {config.student_steps} for student models")
    print(f"Image size: {config.teacher_image_size}x{config.teacher_image_size}")
    print(f"Found {len(student_models)} student models")
    print(f"Analysis samples: {args.num_samples}")
    
    # Create output directories
    os.makedirs(config.analysis_dir, exist_ok=True)
    for subdir in ['metrics', 'visualizations', 'model_analysis']:
        os.makedirs(os.path.join(config.analysis_dir, subdir), exist_ok=True)
    
    # 1. Denoising Analysis
    if args.denoising:
        print("\nRunning denoising analysis...")
        print(f"Using {args.num_samples} sample(s) for visualization")
        print(f"Processing {len(student_models)} student models across various resolutions")
        print(f"This will run a total of {len(student_models) * 4 * config.student_steps} denoising operations for students")
        print(f"Plus {4 * config.timesteps} denoising operations for the teacher model\n")
        print("Starting denoising analysis - this may take some time...")
        try:
            denoising_comparison.create_denoising_comparison_plot(
                teacher_model, 
                student_models, 
                test_samples, 
                config, 
                num_samples=args.num_samples
            )
            print("Denoising analysis completed successfully!")
        except Exception as e:
            print(f"Error in denoising analysis: {e}")
    
    # 2. Attention Analysis
    if args.attention:
        print("\nAnalyzing attention patterns...")
        try:
            # Check if the model has attention modules
            has_attention = hasattr(teacher_model, 'attention') or any(hasattr(layer, 'attention') for layer in teacher_model.modules())
            
            if has_attention:
                timesteps = list(range(0, config.timesteps, config.timesteps // 10))
                attention_patterns.plot_attention_comparison(
                    teacher_model,
                    student_models,
                    test_samples,
                    timesteps,
                    config
                )
            else:
                print("Warning: No attention modules found in the models. Skipping attention analysis.")
        except Exception as e:
            print(f"Error in attention analysis: {e}")
    
    # 3. Time-dependent Analysis
    if args.time:
        print("\nAnalyzing time-dependent metrics...")
        try:
            # Check if we have student models
            if not student_models:
                print("Warning: No student models found. Skipping time-dependent analysis.")
                return
                
            all_distances = {}
            for (size_factor, _), student_model in student_models.items():
                # Create teacher and student trajectories
                teacher_trajectories = []
                student_trajectories = []
                
                # Sample a few trajectories (ensure we have at least 1)
                sample_count = min(max(1, 10), len(test_samples))
                for i in range(sample_count):
                    with torch.no_grad():
                        # Teacher trajectory
                        x = test_samples[i:i+1]
                        teacher_traj = []
                        for t in reversed(range(config.timesteps)):
                            t_batch = torch.full((1,), t, device=x.device, dtype=torch.long)
                            teacher_traj.append((x.clone(), t))
                            # Predict next step
                            pred = teacher_model(x, t_batch)
                            # Update x for next step
                            x = pred
                        teacher_trajectories.append(teacher_traj)
                        
                        # Student trajectory
                        x = test_samples[i:i+1]
                        student_traj = []
                        for t in reversed(range(config.student_steps)):
                            t_batch = torch.full((1,), t, device=x.device, dtype=torch.long)
                            student_traj.append((x.clone(), t))
                            # Predict next step
                            pred = student_model(x, t_batch)
                            # Update x for next step
                            x = pred
                        student_trajectories.append(student_traj)
                
                distances = time_analysis.analyze_time_dependent_distances(
                    teacher_trajectories,
                    student_trajectories,
                    config,
                    size_factor
                )
                all_distances[size_factor] = distances
            
            time_analysis.plot_time_dependent_grid(all_distances, config)
            time_analysis.plot_time_dependent_combined(all_distances, config)
            time_analysis.plot_phase_analysis(all_distances, config)
            time_analysis.analyze_convergence_rates(all_distances, config)
            time_analysis.analyze_size_influence(all_distances, config)
        except Exception as e:
            print(f"Error in time-dependent analysis: {e}")
    
    # 4. MSE Analysis
    if args.mse:
        print("\nGenerating MSE heatmap...")
        try:
            # Override the size factor limits if full-mse is specified
            if args.full_mse:
                print("Using ALL model and image sizes for comprehensive MSE analysis")
                config.mse_size_factors_limit = False
                config.mse_image_size_factors_limit = False
            
            # Pass the test samples to the MSE heatmap function
            mse_heatmap.create_mse_heatmap(config, test_samples)
        except Exception as e:
            print(f"Error in MSE analysis: {e}")
            import traceback
            traceback.print_exc()
    
    # 5. Trajectory Analysis
    if args.trajectory:
        print("\nAnalyzing trajectories...")
        try:
            all_metrics = {}
            
            for (size_factor, image_size_factor), student_model in student_models.items():
                # Create teacher and student trajectories
                teacher_trajectories = []
                student_trajectories = []
                
                # Sample a few trajectories
                for i in range(min(args.num_samples, len(test_samples))):
                    with torch.no_grad():
                        # Teacher trajectory
                        x = test_samples[i:i+1]
                        teacher_traj = []
                        for t in reversed(range(config.timesteps)):
                            t_batch = torch.full((1,), t, device=x.device, dtype=torch.long)
                            pred = teacher_model(x, t_batch)
                            teacher_traj.append((x.clone(), t))
                            # Update x for next step
                            x = pred
                        teacher_trajectories.append(teacher_traj)
                        
                        # Student trajectory
                        x = test_samples[i:i+1]
                        if image_size_factor != 1.0:
                            x = F.interpolate(x, scale_factor=image_size_factor, mode='bilinear')
                        student_traj = []
                        for t in reversed(range(config.student_steps)):
                            t_batch = torch.full((1,), t, device=x.device, dtype=torch.long)
                            pred = student_model(x, t_batch)
                            student_traj.append((x.clone(), t))
                            # Update x for next step
                            x = pred
                        student_trajectories.append(student_traj)
                
                # Compute trajectory metrics
                metrics = trajectory_metrics.compute_trajectory_metrics(
                    teacher_trajectories,
                    student_trajectories,
                    config,
                    size_factor,
                    image_size_factor
                )
                
                # Store metrics by size factor for later analysis
                all_metrics[(size_factor, image_size_factor)] = metrics
                
                # Visualize metrics
                trajectory_metrics.visualize_metrics(metrics, config)
            
            # Save all metrics
            metrics_path = os.path.join(config.analysis_dir, 'metrics', 'trajectory_metrics.pt')
            torch.save(all_metrics, metrics_path)
            
            # Create size-dependent plots if we have metrics for multiple size factors
            size_factors = list(set([sf for sf, _ in all_metrics.keys()]))
            if len(size_factors) > 1:
                # Group metrics by size factor
                metrics_by_size = {}
                for (sf, _), metrics in all_metrics.items():
                    if sf not in metrics_by_size:
                        metrics_by_size[sf] = []
                    metrics_by_size[sf].append(metrics)
                
                # Average metrics for each size factor
                avg_metrics = {}
                for sf, metrics_list in metrics_by_size.items():
                    avg_metrics[sf] = {
                        'avg_wasserstein': np.mean([m['avg_wasserstein'] for m in metrics_list]),
                        'avg_endpoint_distance': np.mean([m['avg_endpoint_distance'] for m in metrics_list]),
                        'avg_teacher_path_length': np.mean([m['avg_teacher_path_length'] for m in metrics_list]),
                        'avg_student_path_length': np.mean([m['avg_student_path_length'] for m in metrics_list]),
                        'avg_teacher_efficiency': np.mean([m['avg_teacher_efficiency'] for m in metrics_list]),
                        'avg_student_efficiency': np.mean([m['avg_student_efficiency'] for m in metrics_list]),
                    }
                
                # Create size-dependent plots
                size_analysis.plot_mse_vs_size(avg_metrics, config)
                size_analysis.plot_metrics_vs_size(avg_metrics, config)
        except Exception as e:
            print(f"Error in trajectory analysis: {e}")
            import traceback
            traceback.print_exc()
    
    # 6. Model Size Analysis
    if args.size:
        print("\nAnalyzing model sizes...")
        try:
            print("Size analysis is handled as part of trajectory analysis.")
        except Exception as e:
            print(f"Error in size analysis: {e}")
    
    # 7. Time Efficiency Analysis
    if args.efficiency:
        print("\nAnalyzing time efficiency...")
        try:
            # Benchmark model performance
            times = {}
            
            print("\nBenchmarking model inference times...")
            device = next(teacher_model.parameters()).device
            
            # Benchmark teacher model
            teacher_times = []
            for _ in range(args.num_runs):
                x = test_samples[:1].to(device)  # Use a single sample for benchmarking
                t = torch.randint(0, config.timesteps, (1,), device=device)
                
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)
                
                start_time.record()
                with torch.no_grad():
                    teacher_model(x, t)
                end_time.record()
                
                torch.cuda.synchronize()
                teacher_times.append(start_time.elapsed_time(end_time))
            
            times['teacher'] = {
                'mean': np.mean(teacher_times),
                'std': np.std(teacher_times),
                'size_factor': 1.0,
                'image_size_factor': 1.0
            }
            
            # Benchmark student models
            for (size_factor, image_size_factor), student_model in student_models.items():
                student_times = []
                for _ in range(args.num_runs):
                    x = test_samples[:1].to(device)
                    if image_size_factor != 1.0:
                        x = F.interpolate(x, scale_factor=image_size_factor, mode='bilinear')
                    t = torch.randint(0, config.student_steps, (1,), device=device)
                    
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)
                    
                    start_time.record()
                    with torch.no_grad():
                        student_model(x, t)
                    end_time.record()
                    
                    torch.cuda.synchronize()
                    student_times.append(start_time.elapsed_time(end_time))
                
                times[(size_factor, image_size_factor)] = {
                    'mean': np.mean(student_times),
                    'std': np.std(student_times),
                    'size_factor': size_factor,
                    'image_size_factor': image_size_factor
                }
            
            # Create visualization
            plt.figure(figsize=(10, 6))
            
            # Extract data for plotting
            size_factors = sorted(set([k[0] if isinstance(k, tuple) else 1.0 for k in times.keys()]))
            image_size_factors = sorted(set([k[1] if isinstance(k, tuple) else 1.0 for k in times.keys()]))
            
            # Create groups by size factor
            grouped_data = {}
            for sf in size_factors:
                grouped_data[sf] = []
                for isf in image_size_factors:
                    key = (sf, isf) if sf != 1.0 else 'teacher'
                    if key in times:
                        grouped_data[sf].append((isf, times[key]['mean'], times[key]['std']))
            
            # Create bar positions
            bar_width = 0.15
            positions = np.arange(len(image_size_factors))
            
            # Create a color map
            colors = plt.cm.viridis(np.linspace(0, 1, len(size_factors)))
            
            # Plot bars for each size factor
            for i, (sf, data) in enumerate(sorted(grouped_data.items())):
                if data:  # Only plot if we have data for this size factor
                    bars = []
                    x_pos = []
                    heights = []
                    errors = []
                    
                    for isf, mean_time, std_time in sorted(data):
                        idx = image_size_factors.index(isf)
                        x_pos.append(positions[idx] + (i - len(size_factors)/2 + 0.5) * bar_width)
                        heights.append(mean_time)
                        errors.append(std_time)
                    
                    plt.bar(x_pos, heights, bar_width, color=colors[i], label=f'Size: {sf}')
                    plt.errorbar(x_pos, heights, yerr=errors, fmt='none', ecolor='black', capsize=5)
            
            # Add labels and title
            plt.xlabel('Image Size Factor')
            plt.ylabel('Inference Time (ms)')
            plt.title('Model Inference Times for Different Sizes')
            plt.xticks(positions, [f'{isf}' for isf in image_size_factors])
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Save the visualization
            os.makedirs(os.path.join(config.analysis_dir, 'efficiency'), exist_ok=True)
            plt.savefig(os.path.join(config.analysis_dir, 'efficiency', 'inference_times.png'))
            plt.close()
            
            # Save raw data
            efficiency_path = os.path.join(config.analysis_dir, 'metrics', 'efficiency_metrics.pt')
            torch.save(times, efficiency_path)
            
            print(f"Efficiency analysis complete. Results saved to {efficiency_path}")
        except Exception as e:
            print(f"Error in efficiency analysis: {e}")
            import traceback
            traceback.print_exc()
    
    # 8. Dimensionality Reduction Analysis
    if args.dimensionality:
        print("\nAnalyzing latent space trajectories...")
        try:
            for (size_factor, image_size_factor), student_model in student_models.items():
                # Get trajectories for this student model
                teacher_trajectories = []
                student_trajectories = []
                
                # Sample a few trajectories
                for _ in range(args.num_samples):
                    with torch.no_grad():
                        # Teacher trajectory
                        x = test_samples[_:_+1]
                        teacher_traj = []
                        for t in reversed(range(config.timesteps)):
                            t_batch = torch.full((1,), t, device=x.device, dtype=torch.long)
                            pred = teacher_model(x, t_batch)
                            teacher_traj.append((x.clone(), t))
                            # Update x for next step
                            x = pred
                        teacher_trajectories.append(teacher_traj)
                        
                        # Student trajectory
                        x = test_samples[_:_+1]
                        if image_size_factor != 1.0:
                            x = F.interpolate(x, scale_factor=image_size_factor, mode='bilinear')
                        student_traj = []
                        for t in reversed(range(config.student_steps)):
                            t_batch = torch.full((1,), t, device=x.device, dtype=torch.long)
                            pred = student_model(x, t_batch)
                            student_traj.append((x.clone(), t))
                            # Update x for next step
                            x = pred
                        student_trajectories.append(student_traj)
                
                # Run dimensionality reduction analysis
                dimensionality_reduction.dimensionality_reduction_analysis(
                    teacher_trajectories,
                    student_trajectories,
                    config,
                    size_factor,
                    image_size_factor
                )
        except Exception as e:
            print(f"Error in dimensionality reduction analysis: {e}")

def main():
    parser = argparse.ArgumentParser(description='Run analysis on trained diffusion models')
    parser.add_argument('--denoising', action='store_true', help='Run denoising analysis')
    parser.add_argument('--attention', action='store_true', help='Run attention pattern analysis')
    parser.add_argument('--time', action='store_true', help='Run time-dependent analysis')
    parser.add_argument('--mse', action='store_true', help='Run MSE analysis')
    parser.add_argument('--trajectory', action='store_true', help='Run trajectory analysis')
    parser.add_argument('--size', action='store_true', help='Run model size analysis')
    parser.add_argument('--efficiency', action='store_true', help='Run time efficiency analysis')
    parser.add_argument('--dimensionality', action='store_true', help='Run dimensionality reduction analysis')
    parser.add_argument('--all', action='store_true', help='Run all analyses')
    parser.add_argument('--num-samples', type=int, default=1, help='Number of samples for visualization')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for analysis')
    parser.add_argument('--num-runs', type=int, default=10, help='Number of runs for efficiency analysis')
    parser.add_argument('--full-mse', action='store_true', help='Run MSE analysis on all model and image sizes')
    
    args = parser.parse_args()
    
    # If no specific analysis is selected, run all
    if not any([args.denoising, args.attention, args.time, args.mse, 
                args.trajectory, args.size, args.efficiency, args.dimensionality]) and not args.all:
        args.all = True
    
    # If --all is specified, enable all analyses
    if args.all:
        args.denoising = True
        args.attention = True
        args.time = True
        args.mse = True
        args.trajectory = True
        args.size = True
        args.efficiency = True
        args.dimensionality = True
    
    # Create all directories first
    config = Config()
    config.create_directories()
    
    run_analysis(args)

if __name__ == '__main__':
    main()
