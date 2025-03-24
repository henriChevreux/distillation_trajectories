#!/usr/bin/env python3
"""
Script to analyze the impact of Classifier-Free Guidance (CFG) across different model sizes.
This script uses the trajectory generation function to ensure consistency with radar plots.
"""

import os
import sys

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec

from config.config import Config
from models import DiffusionUNet
from analysis.trajectory_engine import compare_trajectories
from analysis.metrics.trajectory_metrics import compute_trajectory_metrics
from utils.metric_transformations import transform_metrics

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Analyze CFG impact across different model sizes using trajectory generation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    parser.add_argument('--teacher_model', type=str, default='model_epoch_1.pt',
                        help='Path to teacher model relative to models directory')
    parser.add_argument('--size_factors', type=str, default='0.05,0.75,0.1,0.2,0.4,0.6,0.8,1.0',
                        help='Comma-separated list of size factors to compare')
    parser.add_argument('--guidance_scales', type=str, 
                        default='1.0,2.0,3.0,5.0,7.5,10.0,15.0,20.0,30.0,50.0',
                        help='Comma-separated list of guidance scales to use')
    parser.add_argument('--timesteps', type=int, default=100,
                        help='Number of timesteps for the diffusion process')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of noise samples to average over')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='analysis/heatmaps',
                        help='Directory to save analysis results')
    
    return parser.parse_args()

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
        ('trajectory_mse', 'Trajectory MSE Similarity', True, 'viridis'),  # True means higher is better
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
                # Get raw metric values
                if metric_key == 'mean_directional_consistency':
                    value = metrics['student_metrics'][gs][metric_key]
                else:
                    value = metrics['student_metrics'][gs][metric_key]
                
                # Apply transformations using shared utility
                transformed_metrics = transform_metrics(
                    metrics['student_metrics'][gs]['path_length_similarity'],
                    metrics['student_metrics'][gs]['trajectory_mse'],
                    metrics['student_metrics'][gs]['mean_directional_consistency'],
                    metrics['student_metrics'][gs]['distribution_similarity']
                )
                
                data[i, j] = transformed_metrics[metric_key]
        
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
                # Get raw metric values
                if metric_key == 'mean_directional_consistency':
                    value = metrics['student_metrics'][gs][metric_key]
                else:
                    value = metrics['student_metrics'][gs][metric_key]
                
                # Apply transformations using shared utility
                transformed_metrics = transform_metrics(
                    metrics['student_metrics'][gs]['path_length_similarity'],
                    metrics['student_metrics'][gs]['trajectory_mse'],
                    metrics['student_metrics'][gs]['mean_directional_consistency'],
                    metrics['student_metrics'][gs]['distribution_similarity']
                )
                
                data[i_sf, j_gs] = transformed_metrics[metric_key]
        
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

def create_radar_plot_grid(metrics_by_size, output_dir, guidance_scales):
    """
    Create radar plots for each model size and guidance scale.
    Uses consistent metric transformations with heatmap visualization.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract size factors
    size_factors = sorted(metrics_by_size.keys())
    
    # Define metrics to analyze - using the same metrics and transformations as heatmap
    metrics_to_analyze = [
        ('path_length_similarity', 'Path Length Similarity'),
        ('trajectory_mse', 'Trajectory MSE Similarity'),
        ('mean_directional_consistency', 'Directional Consistency'),
        ('distribution_similarity', 'Distribution Similarity')
    ]
    
    # Create a figure for each guidance scale
    for gs in guidance_scales:
        plt.figure(figsize=(15, 10))
        
        # Calculate grid dimensions
        n_sizes = len(size_factors)
        n_cols = min(3, n_sizes)
        n_rows = (n_sizes + n_cols - 1) // n_cols
        
        # Create radar plots for each size
        for idx, size_factor in enumerate(size_factors):
            metrics = metrics_by_size[size_factor]
            
            # Get raw metric values
            path_length_similarity = metrics['student_metrics'][gs]['path_length_similarity']
            trajectory_mse = metrics['student_metrics'][gs]['trajectory_mse']
            directional_consistency = metrics['student_metrics'][gs]['mean_directional_consistency']
            distribution_similarity = metrics['student_metrics'][gs]['distribution_similarity']
            
            # Apply transformations using shared utility
            transformed_metrics = transform_metrics(
                path_length_similarity,
                trajectory_mse,
                directional_consistency,
                distribution_similarity
            )
            
            # Create radar plot
            ax = plt.subplot(n_rows, n_cols, idx + 1, projection='polar')
            
            # Plot metrics
            angles = np.linspace(0, 2*np.pi, len(metrics_to_analyze), endpoint=False)
            values = []
            for metric_key, _ in metrics_to_analyze:
                if metric_key == 'mean_directional_consistency':
                    values.append(transformed_metrics['mean_directional_consistency'])
                else:
                    values.append(transformed_metrics[metric_key])
            
            # Close the plot by repeating first value
            values = np.concatenate((values, [values[0]]))
            angles = np.concatenate((angles, [angles[0]]))
            
            ax.plot(angles, values, 'o-', linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            
            # Set labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([label for _, label in metrics_to_analyze])
            
            # Set title
            ax.set_title(f'Size Factor: {size_factor:.2f}\nGuidance Scale: {gs:.1f}')
            
            # Set y-axis limits
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'radar_plots_gs_{gs:.1f}.png'))
        plt.close()

def create_composite_radar_plot(metrics_by_size, output_dir, guidance_scales):
    """
    Create a composite radar plot comparing all model sizes at a specific guidance scale.
    Uses consistent metric transformations with heatmap visualization.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract size factors
    size_factors = sorted(metrics_by_size.keys())
    
    # Define metrics to analyze - using the same metrics and transformations as heatmap
    metrics_to_analyze = [
        ('path_length_similarity', 'Path Length Similarity'),
        ('trajectory_mse', 'Trajectory MSE Similarity'),
        ('mean_directional_consistency', 'Directional Consistency'),
        ('distribution_similarity', 'Distribution Similarity')
    ]
    
    # Create a figure for each guidance scale
    for gs in guidance_scales:
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, projection='polar')
        
        # Plot metrics for each size
        angles = np.linspace(0, 2*np.pi, len(metrics_to_analyze), endpoint=False)
        
        for size_factor in size_factors:
            metrics = metrics_by_size[size_factor]
            
            # Get raw metric values
            path_length_similarity = metrics['student_metrics'][gs]['path_length_similarity']
            trajectory_mse = metrics['student_metrics'][gs]['trajectory_mse']
            directional_consistency = metrics['student_metrics'][gs]['mean_directional_consistency']
            distribution_similarity = metrics['student_metrics'][gs]['distribution_similarity']
            
            # Apply transformations using shared utility
            transformed_metrics = transform_metrics(
                path_length_similarity,
                trajectory_mse,
                directional_consistency,
                distribution_similarity
            )
            
            # Get values for this size
            values = []
            for metric_key, _ in metrics_to_analyze:
                if metric_key == 'mean_directional_consistency':
                    values.append(transformed_metrics['mean_directional_consistency'])
                else:
                    values.append(transformed_metrics[metric_key])
            
            # Make sure angles and values have the same length before closing the plot
            plot_angles = angles.copy()
            plot_values = values.copy()
            
            # Close the plot by repeating first value
            plot_values = np.concatenate((plot_values, [plot_values[0]]))
            plot_angles = np.concatenate((plot_angles, [plot_angles[0]]))
            
            # Plot with label
            ax.plot(plot_angles, plot_values, 'o-', linewidth=2, label=f'Size: {size_factor:.2f}')
            ax.fill(plot_angles, plot_values, alpha=0.1)
        
        # Set labels using original angles (without the closing point)
        ax.set_xticks(angles)
        ax.set_xticklabels([label for _, label in metrics_to_analyze])
        
        # Set title and legend
        ax.set_title(f'Composite Radar Plot\nGuidance Scale: {gs:.1f}')
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'composite_radar_plot_gs_{gs:.1f}.png'))
        plt.close()

def create_radar_plot(metrics, title, output_path):
    """Create a radar plot for a single model size"""
    # Get raw metric values
    path_length_similarity = metrics['path_length_similarity']
    trajectory_mse = metrics['trajectory_mse']
    directional_consistency = metrics['directional_consistency']
    distribution_similarity = metrics['distribution_similarity']
    
    # Transform metrics using shared utility
    path_length_score, mse_similarity, directional_score, distribution_score = transform_metrics(
        path_length_similarity, trajectory_mse, directional_consistency, distribution_similarity
    )
    
    # Combine the metrics into the order we want for the radar plot
    values = [path_length_score, mse_similarity, directional_score, distribution_score]
    
    # Create radar plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Plot values
    angles = np.linspace(0, 2*np.pi, len(values), endpoint=False)
    values = np.concatenate((values, [values[0]]))  # Close the plot
    angles = np.concatenate((angles, [angles[0]]))  # Close the plot
    
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Path Length', 'MSE Similarity', 'Directional', 'Distribution'])
    
    # Set title
    plt.title(title)
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

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
        
        # Compute CFG impact for this size factor using the trajectory engine
        print(f"Computing CFG impact for size factor {size_factor}...")
        metrics = compare_trajectories(
            teacher_model, 
            student_model, 
            config, 
            guidance_scales=guidance_scales,
            size_factor=size_factor,
            num_samples=args.num_samples
        )
        
        # Store metrics for this size factor
        metrics_by_size[size_factor] = metrics
    
    # Visualize CFG impact
    print("\nVisualizing CFG heatmaps...")
    visualize_cfg_heatmap(metrics_by_size, output_dir, guidance_scales)
    
    # Create radar plots
    print("\nCreating radar plots...")
    create_radar_plot_grid(metrics_by_size, output_dir, guidance_scales)
    create_composite_radar_plot(metrics_by_size, output_dir, guidance_scales)
    
    print(f"\nCFG analysis completed")
    print(f"Results saved in {output_dir}")

if __name__ == "__main__":
    main() 