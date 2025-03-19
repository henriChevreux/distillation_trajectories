#!/usr/bin/env python3
"""
Script to analyze the impact of Classifier-Free Guidance (CFG) across different model sizes.
This script generates visualizations that show how CFG affects teacher and student models
of different sizes, focusing on the relative impact of different guidance weights.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from config.config import Config
from models import DiffusionUNet
from analysis.cfg_trajectory_comparison import generate_cfg_trajectory, generate_trajectory_without_cfg
from analysis.metrics.trajectory_metrics import compute_trajectory_metrics
from analysis.trajectory_comparison import generate_trajectory

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Analyze CFG impact across different model sizes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    parser.add_argument('--teacher_model', type=str, default='model_epoch_10.pt',
                        help='Path to teacher model relative to models directory')
    parser.add_argument('--size_factors', type=str, default='0.1,0.2,0.4,0.6,0.8,1.0',
                        help='Comma-separated list of size factors to compare')
    parser.add_argument('--guidance_scales', type=str, default='1.0,2.0,3.0,5.0,7.0',
                        help='Comma-separated list of guidance scales to use')
    parser.add_argument('--timesteps', type=int, default=50,
                        help='Number of timesteps for the diffusion process')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of noise samples to average over')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='analysis/cfg_impact',
                        help='Directory to save analysis results')
    
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
        
        # Generate trajectories without CFG - using the same function as radar plots
        print(f"\nGenerating trajectories without CFG (sample {sample_idx+1}/{num_samples})...")
        print("Generating teacher trajectory...")
        teacher_no_cfg = generate_trajectory(teacher_model, noise, config.timesteps, device, seed=seed)
        print("Generating student trajectory...")
        student_no_cfg = generate_trajectory(student_model, noise, config.timesteps, device, seed=seed)
        
        # Compute metrics between teacher and student without CFG
        no_cfg_metrics = compute_trajectory_metrics(teacher_no_cfg, student_no_cfg, config)
        teacher_no_cfg_metrics.append(no_cfg_metrics)
        student_no_cfg_metrics.append(no_cfg_metrics)
        
        # Generate trajectories with CFG for each guidance scale
        for gs in guidance_scales:
            print(f"\nGenerating trajectories with guidance scale {gs} (sample {sample_idx+1}/{num_samples})...")
            
            # For CFG, we still need to use the CFG-specific trajectory generation
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

def visualize_cfg_impact_ratio(metrics_by_size, output_dir, guidance_scales):
    """
    Visualize the impact ratio of CFG across different model sizes
    
    Args:
        metrics_by_size: Dictionary of metrics for each size factor
        output_dir: Directory to save visualizations
        guidance_scales: List of guidance scales used
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the color scheme to match the radar plots
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
        ('mse', 'MSE', False),  # False means lower is better
        ('mean_directional_consistency', 'Directional Consistency', True),
        ('distribution_similarity', 'Distribution Similarity', True)
    ]
    
    # Create a figure for each metric
    for metric_key, metric_name, higher_is_better in metrics_to_analyze:
        plt.figure(figsize=(12, 8))
        
        # Calculate impact ratio for each size factor and guidance scale
        for size_factor in size_factors:
            metrics = metrics_by_size[size_factor]
            
            # Get baseline (no CFG) value
            baseline = metrics['student_no_cfg'][metric_key]
            
            # Calculate ratio for each guidance scale
            ratios = []
            for gs in guidance_scales:
                cfg_value = metrics['student_cfg'][gs][metric_key]
                
                # Calculate ratio based on whether higher or lower is better
                if higher_is_better:
                    ratio = cfg_value / baseline if baseline != 0 else 1.0
                else:
                    ratio = baseline / cfg_value if cfg_value != 0 else 1.0
                
                ratios.append(ratio)
            
            # Plot ratio for this size factor
            color = color_mapping.get(size_factor, poster_colors[size_factors.index(size_factor) % len(poster_colors)])
            plt.plot(guidance_scales, ratios, '-o', 
                     label=f'Size {size_factor}', color=color, linewidth=2.5, markersize=8)
        
        # Add reference line at ratio = 1.0 (no change)
        plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        
        # Add labels and title
        plt.title(f'Impact of CFG on {metric_name}\nRelative to No CFG (Ratio)', fontsize=14, pad=20)
        plt.xlabel('Guidance Scale', fontsize=12)
        plt.ylabel('Impact Ratio (higher is better)', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Adjust y-axis limits to focus on the relevant range
        if higher_is_better:
            plt.ylim(0.9, max(2.0, plt.ylim()[1]))
        else:
            plt.ylim(0.5, max(2.0, plt.ylim()[1]))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(output_dir, f"cfg_impact_ratio_{metric_key}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {metric_name} impact ratio to {output_path}")
    
    # Create a combined figure with all metrics
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()
    
    for i, (metric_key, metric_name, higher_is_better) in enumerate(metrics_to_analyze):
        ax = axs[i]
        
        # Calculate impact ratio for each size factor and guidance scale
        for size_factor in size_factors:
            metrics = metrics_by_size[size_factor]
            
            # Get baseline (no CFG) value
            baseline = metrics['student_no_cfg'][metric_key]
            
            # Calculate ratio for each guidance scale
            ratios = []
            for gs in guidance_scales:
                cfg_value = metrics['student_cfg'][gs][metric_key]
                
                # Calculate ratio based on whether higher or lower is better
                if higher_is_better:
                    ratio = cfg_value / baseline if baseline != 0 else 1.0
                else:
                    ratio = baseline / cfg_value if cfg_value != 0 else 1.0
                
                ratios.append(ratio)
            
            # Plot ratio for this size factor
            color = color_mapping.get(size_factor, poster_colors[size_factors.index(size_factor) % len(poster_colors)])
            ax.plot(guidance_scales, ratios, '-o', 
                    label=f'Size {size_factor}', color=color, linewidth=2.5, markersize=8)
        
        # Add reference line at ratio = 1.0 (no change)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        
        # Add labels and title
        ax.set_title(f'{metric_name}', fontsize=14)
        ax.set_xlabel('Guidance Scale', fontsize=12)
        ax.set_ylabel('Impact Ratio', fontsize=12)
        
        # Adjust y-axis limits to focus on the relevant range
        if higher_is_better:
            ax.set_ylim(0.9, max(2.0, ax.get_ylim()[1]))
        else:
            ax.set_ylim(0.5, max(2.0, ax.get_ylim()[1]))
        
        ax.grid(True, alpha=0.3)
    
    # Add a common legend
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), 
               ncol=len(size_factors), fontsize=12)
    
    # Add a common title
    fig.suptitle('Impact of CFG Across Different Model Sizes\nRelative to No CFG (Ratio)', 
                 fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    
    # Save the figure
    output_path = os.path.join(output_dir, "cfg_impact_ratio_combined.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined impact ratio to {output_path}")

def visualize_cfg_heatmap(metrics_by_size, output_dir, guidance_scales):
    """
    Create heatmaps showing the impact of CFG across model sizes and guidance scales
    
    Args:
        metrics_by_size: Dictionary of metrics for each size factor
        output_dir: Directory to save visualizations
        guidance_scales: List of guidance scales used
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract size factors
    size_factors = sorted(metrics_by_size.keys())
    
    # Define metrics to analyze - using the same metrics and transformations as radar plots
    metrics_to_analyze = [
        ('path_length_similarity', 'Path Length Similarity', True, 'viridis'),  # True means higher is better
        ('mse', 'MSE Similarity (1 - MSE)', True, 'viridis'),  # Converted to similarity
        ('mean_directional_consistency', 'Directional Consistency', True, 'viridis'),
        ('distribution_similarity', 'Distribution Similarity', True, 'viridis')
    ]
    
    # Create a figure for each metric
    for metric_key, metric_name, higher_is_better, cmap_name in metrics_to_analyze:
        plt.figure(figsize=(12, 8))
        
        # Create data matrix for heatmap
        data = np.zeros((len(size_factors), len(guidance_scales)))
        
        for i, size_factor in enumerate(size_factors):
            metrics = metrics_by_size[size_factor]
            
            for j, gs in enumerate(guidance_scales):
                if metric_key == 'mse':
                    # Convert MSE to similarity (1 - MSE) - same as in radar plots
                    value = 1.0 - metrics['student_cfg'][gs][metric_key]
                elif metric_key == 'mean_directional_consistency':
                    # For directional consistency, use raw values without normalization
                    # to match radar plots (which use values in range [-1, 1])
                    value = metrics['student_cfg'][gs][metric_key]
                else:
                    value = metrics['student_cfg'][gs][metric_key]
                
                data[i, j] = value
        
        # Create custom colormap to match poster colors with enhanced contrast
        # From lighter teal/green to dark purple (matching the poster's gradient)
        poster_cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
            'poster_colors',
            [
                '#65d0a0',  # Lighter teal/green (lightest)
                '#50c090',
                '#41a086',
                '#47988b',
                '#4d9090',
                '#538895',
                '#59809a',
                '#5f789f',
                '#6570a4',
                '#6b68a9'   # Purple (darkest)
            ],
            N=256
        )
        
        # Create heatmap with the custom colormap
        plt.imshow(data, cmap=poster_cmap, aspect='auto', interpolation='nearest')
        
        # Add colorbar
        cbar = plt.colorbar()
        cbar.set_label(f'{metric_name} Value', fontsize=12)
        
        # Add labels and title
        plt.title(f'Impact of CFG on {metric_name}\nAcross Model Sizes and Guidance Scales', fontsize=14, pad=20)
        plt.xlabel('Guidance Scale', fontsize=12)
        plt.ylabel('Model Size Factor', fontsize=12)
        
        # Set tick labels
        plt.xticks(np.arange(len(guidance_scales)), [str(gs) for gs in guidance_scales])
        plt.yticks(np.arange(len(size_factors)), [str(sf) for sf in size_factors])
        
        # Add value annotations with consistent white text for better readability
        for i in range(len(size_factors)):
            for j in range(len(guidance_scales)):
                plt.text(j, i, f'{data[i, j]:.3f}', 
                         ha='center', va='center', color='white',
                         fontsize=10, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(output_dir, f"cfg_heatmap_{metric_key}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {metric_name} heatmap to {output_path}")
    
    # Create a combined figure with all metrics
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()
    
    # Create custom colormap to match poster colors with enhanced contrast
    poster_cmap = plt.cm.colors.LinearSegmentedColormap.from_list(
        'poster_colors',
        [
            '#65d0a0',  # Lighter teal/green (lightest)
            '#50c090',
            '#41a086',
            '#47988b',
            '#4d9090',
            '#538895',
            '#59809a',
            '#5f789f',
            '#6570a4',
            '#6b68a9'   # Purple (darkest)
        ],
        N=256
    )
    
    for i, (metric_key, metric_name, higher_is_better, cmap_name) in enumerate(metrics_to_analyze):
        ax = axs[i]
        
        # Create data matrix for heatmap
        data = np.zeros((len(size_factors), len(guidance_scales)))
        
        for i_sf, size_factor in enumerate(size_factors):
            metrics = metrics_by_size[size_factor]
            
            for j_gs, gs in enumerate(guidance_scales):
                if metric_key == 'mse':
                    # Convert MSE to similarity (1 - MSE) - same as in radar plots
                    value = 1.0 - metrics['student_cfg'][gs][metric_key]
                elif metric_key == 'mean_directional_consistency':
                    # For directional consistency, use raw values without normalization
                    # to match radar plots (which use values in range [-1, 1])
                    value = metrics['student_cfg'][gs][metric_key]
                else:
                    value = metrics['student_cfg'][gs][metric_key]
                
                data[i_sf, j_gs] = value
        
        # Create heatmap with the custom colormap
        im = ax.imshow(data, cmap=poster_cmap, aspect='auto', interpolation='nearest')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(f'{metric_name} Value', fontsize=10)
        
        # Add labels and title
        ax.set_title(f'{metric_name}', fontsize=14)
        ax.set_xlabel('Guidance Scale', fontsize=12)
        ax.set_ylabel('Model Size Factor', fontsize=12)
        
        # Set tick labels
        ax.set_xticks(np.arange(len(guidance_scales)))
        ax.set_xticklabels([str(gs) for gs in guidance_scales])
        ax.set_yticks(np.arange(len(size_factors)))
        ax.set_yticklabels([str(sf) for sf in size_factors])
        
        # Add value annotations with consistent white text for better readability
        for i_sf in range(len(size_factors)):
            for j_gs in range(len(guidance_scales)):
                ax.text(j_gs, i_sf, f'{data[i_sf, j_gs]:.3f}', 
                        ha='center', va='center', color='white',
                        fontsize=8, fontweight='bold')
    
    # Add a common title
    fig.suptitle('Impact of CFG Across Different Model Sizes and Guidance Scales', 
                 fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    output_path = os.path.join(output_dir, "cfg_heatmap_combined.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved combined heatmap to {output_path}")

def main():
    """Main function to run the CFG impact analysis"""
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
        metrics_by_size[size_factor] = metrics
    
    # Visualize CFG impact
    print("\nVisualizing CFG impact...")
    visualize_cfg_impact_ratio(metrics_by_size, output_dir, guidance_scales)
    visualize_cfg_heatmap(metrics_by_size, output_dir, guidance_scales)
    
    print(f"\nCFG impact analysis completed")
    print(f"Results saved in {output_dir}")

if __name__ == "__main__":
    main() 