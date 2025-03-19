#!/usr/bin/env python3
"""
Script to run enhanced trajectory comparison for different model sizes.
This script generates trajectory comparisons with enhanced metrics and radar plots.
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from config.config import Config
from models import DiffusionUNet
from analysis.trajectory_comparison import compare_trajectories

def create_metrics_comparison_plot(metrics_by_size, output_dir):
    """
    Create a line plot comparing metrics across different model sizes
    
    Args:
        metrics_by_size: Dictionary mapping size factors to metric dictionaries
        output_dir: Directory to save the plot
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract size factors and sort them
    size_factors = sorted(list(metrics_by_size.keys()))
    
    # Define metrics to plot - both enhanced and traditional
    enhanced_metrics = [
        ('point_by_point_similarity', 'Point-by-Point Similarity'),
        ('log_mse_similarity', 'Log MSE Similarity'),
        ('weighted_directional_consistency', 'Weighted Directional Consistency'),
        ('path_alignment', 'Path Alignment')
    ]
    
    traditional_metrics = [
        ('path_length_similarity', 'Path Length Similarity'),
        ('mean_directional_consistency', 'Mean Directional Consistency'),
        ('distribution_similarity', 'Distribution Similarity')
    ]
    
    # Create figure for enhanced metrics
    plt.figure(figsize=(12, 8))
    
    # Plot each enhanced metric
    for key, label in enhanced_metrics:
        values = []
        for size in size_factors:
            if key in metrics_by_size[size]:
                values.append(metrics_by_size[size][key])
            else:
                values.append(0)  # Default if metric not available
        
        plt.plot(size_factors, values, marker='o', linewidth=2, label=label)
    
    plt.xlabel('Model Size Factor')
    plt.ylabel('Metric Value')
    plt.title('Enhanced Metrics Comparison Across Model Sizes')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save enhanced metrics plot
    enhanced_plot_path = os.path.join(output_dir, 'enhanced_metrics_comparison.png')
    plt.savefig(enhanced_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Enhanced metrics comparison plot saved to: {enhanced_plot_path}")
    
    # Create figure for traditional metrics
    plt.figure(figsize=(12, 8))
    
    # Plot each traditional metric
    for key, label in traditional_metrics:
        values = []
        for size in size_factors:
            if key in metrics_by_size[size]:
                values.append(metrics_by_size[size][key])
            else:
                values.append(0)  # Default if metric not available
        
        plt.plot(size_factors, values, marker='o', linewidth=2, label=label)
    
    # Also plot MSE similarity (1-MSE) for comparison
    mse_sim_values = []
    for size in size_factors:
        if 'mse' in metrics_by_size[size]:
            mse_sim_values.append(1.0 - metrics_by_size[size]['mse'])
        else:
            mse_sim_values.append(0)
    
    plt.plot(size_factors, mse_sim_values, marker='o', linewidth=2, label='MSE Similarity (1-MSE)')
    
    plt.xlabel('Model Size Factor')
    plt.ylabel('Metric Value')
    plt.title('Traditional Metrics Comparison Across Model Sizes')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save traditional metrics plot
    traditional_plot_path = os.path.join(output_dir, 'traditional_metrics_comparison.png')
    plt.savefig(traditional_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Traditional metrics comparison plot saved to: {traditional_plot_path}")
    
    # Create a combined plot with all metrics
    plt.figure(figsize=(14, 10))
    
    # Plot all metrics with different line styles to distinguish them
    all_metrics = enhanced_metrics + traditional_metrics + [('mse', 'MSE Similarity (1-MSE)')]
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', 'x', '+', '*', 'v']
    
    for i, (key, label) in enumerate(all_metrics):
        values = []
        for size in size_factors:
            if key == 'mse' and key in metrics_by_size[size]:
                # For MSE, convert to similarity
                values.append(1.0 - metrics_by_size[size][key])
            elif key in metrics_by_size[size]:
                values.append(metrics_by_size[size][key])
            else:
                values.append(0)  # Default if metric not available
        
        plt.plot(size_factors, values, 
                 linestyle=line_styles[i % len(line_styles)],
                 marker=markers[i % len(markers)], 
                 linewidth=2, 
                 label=label)
    
    plt.xlabel('Model Size Factor')
    plt.ylabel('Metric Value')
    plt.title('All Metrics Comparison Across Model Sizes')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
    # Save combined metrics plot
    combined_plot_path = os.path.join(output_dir, 'all_metrics_comparison.png')
    plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined metrics comparison plot saved to: {combined_plot_path}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run enhanced trajectory comparison for different model sizes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    parser.add_argument('--teacher_model', type=str, default='model_epoch_10.pt',
                        help='Path to teacher model relative to models directory')
    parser.add_argument('--size_factors', type=str, default='0.1,0.3,0.5,0.7,0.9,1.0',
                        help='Comma-separated list of size factors to compare')
    parser.add_argument('--student_epoch', type=int, default=5,
                        help='Epoch of student models to use')
    
    return parser.parse_args()

def main():
    """Main function to run enhanced trajectory comparison"""
    args = parse_args()
    
    # Initialize configuration
    config = Config()
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Parse size factors
    size_factors = [float(sf) for sf in args.size_factors.split(',')]
    
    # Load teacher model
    teacher_model_path = os.path.join(project_root, 'output', 'models', 'teacher', args.teacher_model)
    print(f"Loading teacher model from {teacher_model_path}")
    
    if not os.path.exists(teacher_model_path):
        print(f"ERROR: Teacher model not found at {teacher_model_path}")
        sys.exit(1)
    
    # Initialize teacher model
    teacher_model = DiffusionUNet(config, size_factor=1.0).to(device)
    
    teacher_state_dict = torch.load(teacher_model_path, map_location=device)
    teacher_model.load_state_dict(teacher_state_dict)
    teacher_model.eval()
    
    print(f"Teacher model loaded successfully")
    
    # Dictionary to store metrics for each size factor
    metrics_by_size = {}
    
    # Process each size factor
    for size_factor in size_factors:
        print(f"\n{'='*80}")
        print(f"Processing size factor: {size_factor}")
        print(f"{'='*80}")
        
        # Load student model
        student_model_path = os.path.join(
            project_root, 'output', 'models', 'students', 
            f'size_{size_factor}', f'model_epoch_{args.student_epoch}.pt'
        )
        
        print(f"Loading student model from {student_model_path}")
        
        if not os.path.exists(student_model_path):
            print(f"WARNING: Student model not found at {student_model_path}")
            print(f"Skipping size factor {size_factor}")
            continue
        
        # Initialize student model
        student_model = DiffusionUNet(config, size_factor=size_factor).to(device)
        
        student_state_dict = torch.load(student_model_path, map_location=device)
        student_model.load_state_dict(student_state_dict)
        student_model.eval()
        
        print(f"Student model loaded successfully")
        
        # Run trajectory comparison
        compare_trajectories(teacher_model, student_model, config, size_factor)
        
        # Extract metrics from the comparison
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'analysis', 'trajectory_comparison')
        metrics_file = os.path.join(output_dir, f'metrics_summary_size_{size_factor}.txt')
        
        # Read metrics from file
        metrics = {}
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip()
                        try:
                            value = float(value.strip())
                            # Convert key to snake_case for consistency
                            key = key.lower().replace(' ', '_')
                            metrics[key] = value
                        except ValueError:
                            # Skip lines that don't have numeric values
                            pass
        
        # Store metrics for this size factor
        metrics_by_size[size_factor] = metrics
    
    # Create metrics comparison plot
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'analysis', 'trajectory_comparison')
    create_metrics_comparison_plot(metrics_by_size, output_dir)
    
    print("\nEnhanced trajectory comparison completed")
    print(f"Results saved in {os.path.join(project_root, 'analysis', 'trajectory_comparison')}")

if __name__ == "__main__":
    main() 