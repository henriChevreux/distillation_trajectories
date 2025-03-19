#!/usr/bin/env python3
"""
Script to analyze the optimal CFG scale for different model sizes.
This script generates visualizations that show how different model sizes
respond to different guidance weights, helping identify the optimal
guidance scale for each model size.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import json

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from config.config import Config
from models import DiffusionUNet
from analysis.cfg_trajectory_comparison import generate_cfg_trajectory, generate_trajectory_without_cfg
from analysis.metrics.trajectory_metrics import compute_trajectory_metrics

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Analyze optimal CFG scale for different model sizes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    parser.add_argument('--teacher_model', type=str, default='model_epoch_10.pt',
                        help='Path to teacher model relative to models directory')
    parser.add_argument('--size_factors', type=str, default='0.1,0.2,0.4,0.6,0.8,1.0',
                        help='Comma-separated list of size factors to compare')
    parser.add_argument('--guidance_scales', type=str, default='1.0,1.5,2.0,2.5,3.0,4.0,5.0,7.0',
                        help='Comma-separated list of guidance scales to use')
    parser.add_argument('--timesteps', type=int, default=50,
                        help='Number of timesteps for the diffusion process')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of noise samples to average over')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='analysis/cfg_optimal_scale',
                        help='Directory to save analysis results')
    parser.add_argument('--load_cached', action='store_true',
                        help='Load cached metrics data instead of recomputing')
    
    return parser.parse_args()

def compute_cfg_impact(teacher_model, student_model, config, guidance_scales, device, num_samples=3):
    """
    Compute the impact of CFG on teacher and student models
    
    Args:
        teacher_model: The teacher diffusion model
        student_model: The student diffusion model
        config: Configuration object
        guidance_scales: List of guidance scales to evaluate
        device: Device to run on
        num_samples: Number of noise samples to average over
        
    Returns:
        Dictionary of metrics for each guidance scale
    """
    # Initialize dictionaries to store metrics
    teacher_metrics = {gs: [] for gs in guidance_scales}
    student_metrics = {gs: [] for gs in guidance_scales}
    teacher_no_cfg_metrics = []
    student_no_cfg_metrics = []
    
    # Generate trajectories for multiple noise samples
    for sample_idx in range(num_samples):
        # Set seed for reproducibility but different for each sample
        seed = 42 + sample_idx
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Generate random noise
        noise = torch.randn(1, config.channels, config.image_size, config.image_size)
        
        # Generate trajectories without CFG
        print(f"\nGenerating trajectories without CFG (sample {sample_idx+1}/{num_samples})...")
        print("Generating teacher trajectory...")
        teacher_no_cfg = generate_trajectory_without_cfg(teacher_model, noise, config.timesteps, device, seed=seed)
        print("Generating student trajectory...")
        student_no_cfg = generate_trajectory_without_cfg(student_model, noise, config.timesteps, device, seed=seed)
        
        # Compute metrics between teacher and student without CFG
        no_cfg_metrics = compute_trajectory_metrics(teacher_no_cfg, student_no_cfg, config)
        teacher_no_cfg_metrics.append(no_cfg_metrics)
        student_no_cfg_metrics.append(no_cfg_metrics)
        
        # Generate trajectories with CFG for each guidance scale
        for gs in guidance_scales:
            print(f"\nGenerating trajectories with guidance scale {gs} (sample {sample_idx+1}/{num_samples})...")
            
            print("Generating teacher trajectory...")
            teacher_traj = generate_cfg_trajectory(teacher_model, noise, config.timesteps, gs, device, seed=seed)
            
            print("Generating student trajectory...")
            student_traj = generate_cfg_trajectory(student_model, noise, config.timesteps, gs, device, seed=seed)
            
            # Compute metrics between teacher and student with CFG
            cfg_metrics = compute_trajectory_metrics(teacher_traj, student_traj, config)
            teacher_metrics[gs].append(cfg_metrics)
            student_metrics[gs].append(cfg_metrics)
    
    # Average metrics across samples
    avg_teacher_no_cfg = {}
    avg_student_no_cfg = {}
    avg_teacher_cfg = {gs: {} for gs in guidance_scales}
    avg_student_cfg = {gs: {} for gs in guidance_scales}
    
    # Average no-CFG metrics
    for key in teacher_no_cfg_metrics[0].keys():
        if isinstance(teacher_no_cfg_metrics[0][key], (int, float)) and not isinstance(teacher_no_cfg_metrics[0][key], bool):
            avg_teacher_no_cfg[key] = sum(m[key] for m in teacher_no_cfg_metrics) / len(teacher_no_cfg_metrics)
            avg_student_no_cfg[key] = sum(m[key] for m in student_no_cfg_metrics) / len(student_no_cfg_metrics)
    
    # Average CFG metrics
    for gs in guidance_scales:
        for key in teacher_metrics[gs][0].keys():
            if isinstance(teacher_metrics[gs][0][key], (int, float)) and not isinstance(teacher_metrics[gs][0][key], bool):
                avg_teacher_cfg[gs][key] = sum(m[key] for m in teacher_metrics[gs]) / len(teacher_metrics[gs])
                avg_student_cfg[gs][key] = sum(m[key] for m in student_metrics[gs]) / len(student_metrics[gs])
    
    return {
        'teacher_no_cfg': avg_teacher_no_cfg,
        'student_no_cfg': avg_student_no_cfg,
        'teacher_cfg': avg_teacher_cfg,
        'student_cfg': avg_student_cfg
    }

def find_optimal_guidance_scale(metrics_by_size):
    """
    Find the optimal guidance scale for each model size and metric
    
    Args:
        metrics_by_size: Dictionary of metrics for each size factor
        
    Returns:
        Dictionary of optimal guidance scales for each size factor and metric
    """
    # Define metrics to analyze
    metrics_to_analyze = [
        ('path_length_similarity', True),  # True means higher is better
        ('mse', False),  # False means lower is better
        ('mean_directional_consistency', True),
        ('distribution_similarity', True)
    ]
    
    # Initialize dictionary to store optimal guidance scales
    optimal_scales = {}
    
    # Find optimal guidance scale for each size factor and metric
    for size_factor in metrics_by_size:
        optimal_scales[size_factor] = {}
        metrics = metrics_by_size[size_factor]
        
        for metric_key, higher_is_better in metrics_to_analyze:
            # Get baseline (no CFG) value
            baseline = metrics['student_no_cfg'][metric_key]
            
            # Find optimal guidance scale
            best_scale = None
            best_value = None
            
            for gs in metrics['student_cfg']:
                cfg_value = metrics['student_cfg'][gs][metric_key]
                
                # Check if this guidance scale is better
                if best_value is None:
                    best_value = cfg_value
                    best_scale = gs
                elif higher_is_better and cfg_value > best_value:
                    best_value = cfg_value
                    best_scale = gs
                elif not higher_is_better and cfg_value < best_value:
                    best_value = cfg_value
                    best_scale = gs
            
            # Store optimal guidance scale and improvement ratio
            if higher_is_better:
                improvement_ratio = best_value / baseline if baseline != 0 else float('inf')
            else:
                improvement_ratio = baseline / best_value if best_value != 0 else float('inf')
            
            optimal_scales[size_factor][metric_key] = {
                'optimal_scale': best_scale,
                'optimal_value': best_value,
                'baseline_value': baseline,
                'improvement_ratio': improvement_ratio
            }
    
    return optimal_scales

def visualize_optimal_scales(optimal_scales, output_dir):
    """
    Visualize the optimal guidance scales for each model size and metric
    
    Args:
        optimal_scales: Dictionary of optimal guidance scales for each size factor and metric
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the color scheme to match the poster colors with enhanced contrast
    poster_colors = [
        '#6b68a9',  # Purple (darkest) - for largest model (1.0)
        '#6570a4',  # Purple-blue 1
        '#5f789f',  # Purple-blue 2
        '#59809a',  # Blue-purple 1
        '#538895',  # Blue-purple 2
        '#4d9090',  # Blue
        '#47988b',  # Blue-teal 1
        '#41a086',  # Blue-teal 2
        '#35b07c'   # Teal (lightest) - for smallest model (0.1)
    ]
    
    # Define metrics to analyze
    metrics_to_analyze = [
        ('path_length_similarity', 'Path Length Similarity'),
        ('mse', 'MSE Similarity (1 - MSE)'),
        ('mean_directional_consistency', 'Directional Consistency'),
        ('distribution_similarity', 'Distribution Similarity')
    ]
    
    # Extract size factors
    size_factors = sorted([float(sf) for sf in optimal_scales.keys()])
    
    # Create a figure for optimal guidance scales
    plt.figure(figsize=(12, 8))
    
    # Set background color to light gray for better contrast
    plt.gca().set_facecolor('#f5f5f5')
    
    # Plot optimal guidance scales for each metric
    for i, (metric_key, metric_name) in enumerate(metrics_to_analyze):
        optimal_scales_for_metric = [float(optimal_scales[str(sf)][metric_key]['optimal_scale']) for sf in size_factors]
        plt.plot(size_factors, optimal_scales_for_metric, '-o', 
                 label=metric_name, color=poster_colors[i % len(poster_colors)], 
                 linewidth=3.0, markersize=10, markeredgecolor='black', markeredgewidth=0.5)
    
    # Add labels and title
    plt.title('Optimal Guidance Scale by Model Size', fontsize=16, pad=20)
    plt.xlabel('Model Size Factor', fontsize=14)
    plt.ylabel('Optimal Guidance Scale', fontsize=14)
    plt.legend(loc='best', fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks to match size factors
    plt.xticks(size_factors, fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, "optimal_guidance_scales.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved optimal guidance scales to {output_path}")
    
    # Create a figure for improvement ratios
    plt.figure(figsize=(12, 8))
    
    # Set background color to light gray for better contrast
    plt.gca().set_facecolor('#f5f5f5')
    
    # Plot improvement ratios for each metric
    for i, (metric_key, metric_name) in enumerate(metrics_to_analyze):
        improvement_ratios = [optimal_scales[str(sf)][metric_key]['improvement_ratio'] for sf in size_factors]
        plt.plot(size_factors, improvement_ratios, '-o', 
                 label=metric_name, color=poster_colors[i % len(poster_colors)], 
                 linewidth=3.0, markersize=10, markeredgecolor='black', markeredgewidth=0.5)
    
    # Add reference line at ratio = 1.0 (no improvement)
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=2)
    
    # Add labels and title
    plt.title('Improvement Ratio with Optimal CFG\nRelative to No CFG', fontsize=16, pad=20)
    plt.xlabel('Model Size Factor', fontsize=14)
    plt.ylabel('Improvement Ratio (higher is better)', fontsize=14)
    plt.legend(loc='best', fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks to match size factors
    plt.xticks(size_factors, fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, "improvement_ratios.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved improvement ratios to {output_path}")

def visualize_metric_vs_guidance(metrics_by_size, output_dir, guidance_scales):
    """
    Visualize how each metric changes with guidance scale for different model sizes
    
    Args:
        metrics_by_size: Dictionary of metrics for each size factor
        output_dir: Directory to save visualizations
        guidance_scales: List of guidance scales used
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the color scheme to match the poster colors with enhanced contrast
    poster_colors = [
        '#6b68a9',  # Purple (darkest) - for largest model (1.0)
        '#6570a4',  # Purple-blue 1
        '#5f789f',  # Purple-blue 2
        '#59809a',  # Blue-purple 1
        '#538895',  # Blue-purple 2
        '#4d9090',  # Blue
        '#47988b',  # Blue-teal 1
        '#41a086',  # Blue-teal 2
        '#35b07c'   # Teal (lightest) - for smallest model (0.1)
    ]
    
    # Reverse the color order so smallest models get lightest colors and largest get darkest
    poster_colors = poster_colors[::-1]
    
    # Create a fixed mapping of size factors to colors
    standard_size_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    # Make sure we have enough colors for all size factors
    while len(poster_colors) < len(standard_size_factors):
        poster_colors = poster_colors + poster_colors
    
    color_mapping = {sf: poster_colors[i % len(poster_colors)] for i, sf in enumerate(standard_size_factors)}
    
    # Extract size factors
    size_factors = sorted(metrics_by_size.keys())
    
    # Define metrics to analyze
    metrics_to_analyze = [
        ('path_length_similarity', 'Path Length Similarity', True),  # True means higher is better
        ('mse', 'MSE Similarity (1 - MSE)', True),  # Converted to similarity
        ('mean_directional_consistency', 'Directional Consistency', True),
        ('distribution_similarity', 'Distribution Similarity', True)
    ]
    
    # Create a figure for each metric
    for metric_key, metric_name, higher_is_better in metrics_to_analyze:
        plt.figure(figsize=(12, 8))
        
        # Set background color to light gray for better contrast
        plt.gca().set_facecolor('#f5f5f5')
        
        # Plot metric values for each size factor
        for size_factor in size_factors:
            metrics = metrics_by_size[size_factor]
            
            # Get values for each guidance scale
            values = []
            for gs in guidance_scales:
                if metric_key == 'mse':
                    # Convert MSE to similarity (1 - MSE)
                    value = 1.0 - metrics['student_cfg'][gs][metric_key]
                elif metric_key == 'mean_directional_consistency':
                    # Normalize directional consistency from [-1,1] to [0,1]
                    value = (metrics['student_cfg'][gs][metric_key] + 1) / 2
                else:
                    value = metrics['student_cfg'][gs][metric_key]
                
                values.append(value)
            
            # Plot values for this size factor with enhanced line width and markers
            color = color_mapping.get(float(size_factor), poster_colors[size_factors.index(size_factor) % len(poster_colors)])
            plt.plot(guidance_scales, values, '-o', 
                     label=f'Size {size_factor}', color=color, 
                     linewidth=3.0, markersize=10, markeredgecolor='black', markeredgewidth=0.5)
        
        # Add labels and title
        plt.title(f'{metric_name} vs. Guidance Scale', fontsize=16, pad=20)
        plt.xlabel('Guidance Scale', fontsize=14)
        plt.ylabel(f'{metric_name} Value', fontsize=14)
        plt.legend(loc='best', fontsize=12, framealpha=0.9)
        plt.grid(True, alpha=0.3)
        
        # Set axis limits for better visualization
        if metric_key == 'mse':
            plt.ylim(-1.7, 0.6)  # Adjusted for MSE Similarity range
        elif metric_key == 'mean_directional_consistency':
            plt.ylim(0.4, 1.0)  # Adjusted for Directional Consistency range
        elif metric_key == 'path_length_similarity':
            plt.ylim(0.75, 1.0)  # Adjusted for Path Length Similarity range
        elif metric_key == 'distribution_similarity':
            plt.ylim(0.6, 0.95)  # Adjusted for Distribution Similarity range
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(output_dir, f"metric_vs_guidance_{metric_key}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {metric_name} vs. guidance scale to {output_path}")
    
    # Create a combined figure with all metrics
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()
    
    # Set figure background color
    fig.patch.set_facecolor('#f5f5f5')
    
    for i, (metric_key, metric_name, higher_is_better) in enumerate(metrics_to_analyze):
        ax = axs[i]
        
        # Set subplot background color
        ax.set_facecolor('#f5f5f5')
        
        # Plot metric values for each size factor
        for size_factor in size_factors:
            metrics = metrics_by_size[size_factor]
            
            # Get values for each guidance scale
            values = []
            for gs in guidance_scales:
                if metric_key == 'mse':
                    # Convert MSE to similarity (1 - MSE)
                    value = 1.0 - metrics['student_cfg'][gs][metric_key]
                elif metric_key == 'mean_directional_consistency':
                    # Normalize directional consistency from [-1,1] to [0,1]
                    value = (metrics['student_cfg'][gs][metric_key] + 1) / 2
                else:
                    value = metrics['student_cfg'][gs][metric_key]
                
                values.append(value)
            
            # Plot values for this size factor with enhanced line width and markers
            color = color_mapping.get(float(size_factor), poster_colors[size_factors.index(size_factor) % len(poster_colors)])
            ax.plot(guidance_scales, values, '-o', 
                    label=f'Size {size_factor}', color=color, 
                    linewidth=3.0, markersize=10, markeredgecolor='black', markeredgewidth=0.5)
        
        # Add labels and title
        ax.set_title(f'{metric_name}', fontsize=16)
        ax.set_xlabel('Guidance Scale', fontsize=14)
        ax.set_ylabel(f'Value', fontsize=14)
        
        # Set axis limits for better visualization
        if metric_key == 'mse':
            ax.set_ylim(-1.7, 0.6)  # Adjusted for MSE Similarity range
        elif metric_key == 'mean_directional_consistency':
            ax.set_ylim(0.4, 1.0)  # Adjusted for Directional Consistency range
        elif metric_key == 'path_length_similarity':
            ax.set_ylim(0.75, 1.0)  # Adjusted for Path Length Similarity range
        elif metric_key == 'distribution_similarity':
            ax.set_ylim(0.6, 0.95)  # Adjusted for Distribution Similarity range
        
        ax.grid(True, alpha=0.3)
    
    # Add a common legend with enhanced visibility
    handles, labels = axs[0].get_legend_handles_labels()
    legend = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.06), 
                ncol=len(size_factors), fontsize=14, framealpha=0.9)
    
    # Add a common title
    fig.suptitle('Metrics vs. Guidance Scale Across Different Model Sizes', 
                 fontsize=18, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    # Save the figure
    output_path = os.path.join(output_dir, "metrics_vs_guidance_combined.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined metrics vs. guidance scale to {output_path}")

def main():
    """Main function to run the optimal CFG scale analysis"""
    args = parse_args()
    
    # Load configuration
    config = Config()
    config.timesteps = args.timesteps
    
    # Set up output directory
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert size factors to list of floats
    size_factors = [float(sf) for sf in args.size_factors.split(',')]
    
    # Convert guidance scales to list of floats
    guidance_scales = [float(gs) for gs in args.guidance_scales.split(',')]
    
    # Check if we should load cached metrics
    metrics_cache_path = os.path.join(output_dir, "metrics_cache.json")
    if args.load_cached and os.path.exists(metrics_cache_path):
        print(f"Loading cached metrics from {metrics_cache_path}")
        with open(metrics_cache_path, 'r') as f:
            metrics_by_size = json.load(f)
        
        # Convert string keys back to floats for guidance scales
        for size_factor in metrics_by_size:
            metrics_by_size[size_factor]['teacher_cfg'] = {
                float(gs): metrics_by_size[size_factor]['teacher_cfg'][gs] 
                for gs in metrics_by_size[size_factor]['teacher_cfg']
            }
            metrics_by_size[size_factor]['student_cfg'] = {
                float(gs): metrics_by_size[size_factor]['student_cfg'][gs] 
                for gs in metrics_by_size[size_factor]['student_cfg']
            }
    else:
        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load teacher model
        teacher_path = os.path.join(config.teacher_models_dir, args.teacher_model)
        if not os.path.exists(teacher_path):
            raise FileNotFoundError(f"Teacher model not found at {teacher_path}")
        
        print(f"Loading teacher model from {teacher_path}")
        teacher_model = DiffusionUNet(config, size_factor=1.0)
        teacher_model.load_state_dict(torch.load(teacher_path, map_location=device))
        teacher_model.eval()
        teacher_model = teacher_model.to(device)
        
        # Dictionary to store metrics for each size factor
        metrics_by_size = {}
        
        # Process each student model
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
            
            student_model = DiffusionUNet(config, size_factor=size_factor)
            student_model.load_state_dict(torch.load(student_path, map_location=device))
            student_model = student_model.to(device)
            student_model.eval()
            
            # Compute CFG impact for this size factor
            print(f"Computing CFG impact for size factor {size_factor}...")
            metrics = compute_cfg_impact(
                teacher_model, 
                student_model, 
                config, 
                guidance_scales,
                device,
                num_samples=args.num_samples
            )
            
            # Store metrics for this size factor
            metrics_by_size[str(size_factor)] = metrics
        
        # Cache metrics for future use
        print(f"Caching metrics to {metrics_cache_path}")
        with open(metrics_cache_path, 'w') as f:
            # Convert float keys to strings for JSON serialization
            serializable_metrics = {}
            for size_factor in metrics_by_size:
                serializable_metrics[size_factor] = {
                    'teacher_no_cfg': metrics_by_size[size_factor]['teacher_no_cfg'],
                    'student_no_cfg': metrics_by_size[size_factor]['student_no_cfg'],
                    'teacher_cfg': {
                        str(gs): metrics_by_size[size_factor]['teacher_cfg'][gs] 
                        for gs in metrics_by_size[size_factor]['teacher_cfg']
                    },
                    'student_cfg': {
                        str(gs): metrics_by_size[size_factor]['student_cfg'][gs] 
                        for gs in metrics_by_size[size_factor]['student_cfg']
                    }
                }
            json.dump(serializable_metrics, f, indent=2)
    
    # Find optimal guidance scales
    print("\nFinding optimal guidance scales...")
    optimal_scales = find_optimal_guidance_scale(metrics_by_size)
    
    # Save optimal scales to file
    optimal_scales_path = os.path.join(output_dir, "optimal_scales.json")
    with open(optimal_scales_path, 'w') as f:
        json.dump(optimal_scales, f, indent=2)
    
    # Visualize optimal guidance scales
    print("\nVisualizing optimal guidance scales...")
    visualize_optimal_scales(optimal_scales, output_dir)
    
    # Visualize metrics vs. guidance scale
    print("\nVisualizing metrics vs. guidance scale...")
    visualize_metric_vs_guidance(metrics_by_size, output_dir, guidance_scales)
    
    print(f"\nOptimal CFG scale analysis completed")
    print(f"Results saved in {output_dir}")

if __name__ == "__main__":
    main() 