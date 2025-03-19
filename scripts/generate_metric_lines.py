#!/usr/bin/env python3
"""
Script to generate line graphs for trajectory metrics across different model sizes.
Uses the same color scheme as the radar plots for consistency.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from analysis.metrics.model_comparisons import create_radar_plot_grid, create_composite_radar_plot
from config.config import Config

def generate_metric_line_graphs(metrics_by_size, output_dir):
    """
    Generate line graphs for each metric across different model sizes.
    
    Args:
        metrics_by_size: Dictionary of metrics keyed by size factor
        output_dir: Directory to save visualizations
    
    Returns:
        Path to the saved line graph image
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating metric line graphs...")
    
    # Convert all keys to floats and sort
    size_factors = []
    for key in metrics_by_size.keys():
        try:
            size_factors.append(float(key))
        except ValueError:
            print(f"  Warning: Could not convert size factor '{key}' to float, skipping.")
    
    size_factors.sort()
    
    if not size_factors:
        print("  Warning: No valid size factors found. Skipping line graphs.")
        return None
    
    # Extract metrics for each size factor
    path_length_similarities = []
    mse_similarities = []
    directional_consistencies = []
    distribution_similarities = []
    
    for size_factor in size_factors:
        metrics = metrics_by_size[size_factor]
        
        # Extract raw metric values
        path_length_similarity = metrics.get('path_length_similarity', 1.0)
        mse = metrics.get('mse', 0.0)
        mse_similarity = 1.0 - mse  # Convert MSE to MSE Similarity
        directional_consistency = metrics.get('mean_directional_consistency', 0.0)
        # Normalize directional consistency from [-1,1] to [0,1]
        directional_consistency = (directional_consistency + 1) / 2 if directional_consistency != float('nan') else 0.5
        distribution_similarity = metrics.get('distribution_similarity', 0.0)
        
        path_length_similarities.append(path_length_similarity)
        mse_similarities.append(mse_similarity)
        directional_consistencies.append(directional_consistency)
        distribution_similarities.append(distribution_similarity)
    
    # Use the same color scheme as the radar plots
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
    
    # Set up the figure for all metrics in one plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define metric names and data
    metrics = [
        ("Path Length Similarity", path_length_similarities),
        ("MSE Similarity", mse_similarities),
        ("Directional Consistency", directional_consistencies),
        ("Distribution Similarity", distribution_similarities)
    ]
    
    # Use the poster colors for the metrics instead of default matplotlib colors
    # We'll use 4 distinct colors from our poster color palette
    metric_colors = [
        '#6b68a9',  # Purple for Path Length Similarity
        '#59809a',  # Blue-purple for MSE Similarity
        '#4d9090',  # Blue for Directional Consistency
        '#35b07c'   # Teal for Distribution Similarity
    ]
    
    # Define line styles for each metric to differentiate them
    line_styles = ['-', '-', '-', '-']  # Use solid lines for all
    marker_styles = ['o', 's', '^', 'D']
    
    # Plot each metric
    for i, (metric_name, metric_values) in enumerate(metrics):
        plt.plot(
            size_factors, 
            metric_values, 
            label=metric_name,
            linestyle=line_styles[i],
            marker=marker_styles[i],
            linewidth=3.0,
            markersize=10,
            color=metric_colors[i]  # Use our poster colors
        )
    
    # Set the y-axis to start from 0.7 to match the radar plots
    plt.ylim(0.7, 1.01)
    
    # Add grid lines
    plt.grid(True, alpha=0.3, linestyle='-')
    
    # Add labels and title
    plt.xlabel('Model Size Factor', fontsize=14)
    plt.ylabel('Metric Value', fontsize=14)
    plt.title('Trajectory Metrics Across Model Sizes', fontsize=16, pad=20)
    
    # Add legend
    plt.legend(loc='lower right', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, "metric_line_graph.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved metric line graph to {output_path}")
    
    # Now create individual line graphs for each metric with colored lines by size factor
    
    # Set up a 2x2 grid for the four metrics
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs = axs.flatten()
    
    # Define metric names, data, and y-axis labels
    metrics_info = [
        ("Path Length Similarity", path_length_similarities, "Path Length\nSimilarity"),
        ("MSE Similarity", mse_similarities, "MSE\nSimilarity"),
        ("Directional Consistency", directional_consistencies, "Directional\nConsistency"),
        ("Distribution Similarity", distribution_similarities, "Distribution\nSimilarity")
    ]
    
    # Use the poster colors for the metrics
    metric_colors = [
        '#6b68a9',  # Purple for Path Length Similarity
        '#59809a',  # Blue-purple for MSE Similarity
        '#4d9090',  # Blue for Directional Consistency
        '#35b07c'   # Teal for Distribution Similarity
    ]
    
    # Plot each metric in its own subplot with colored lines by size factor
    for i, (metric_name, metric_values, y_label) in enumerate(metrics_info):
        ax = axs[i]
        
        # Plot the metric line with the corresponding poster color
        ax.plot(
            size_factors, 
            metric_values, 
            linestyle='-',
            marker='o',
            linewidth=3.0,
            markersize=10,
            color=metric_colors[i]  # Use our poster colors
        )
        
        # Add colored markers for each size factor
        for j, size_factor in enumerate(size_factors):
            color = color_mapping.get(size_factor, poster_colors[j % len(poster_colors)])
            ax.plot(
                size_factor, 
                metric_values[j], 
                marker='o',
                markersize=12,
                color=color,
                markeredgecolor='black',
                markeredgewidth=0.5
            )
        
        # Set the y-axis to start from 0.7 to match the radar plots
        ax.set_ylim(0.7, 1.01)
        
        # Add grid lines
        ax.grid(True, alpha=0.3, linestyle='-')
        
        # Add labels and title
        ax.set_xlabel('Model Size Factor', fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(metric_name, fontsize=14, pad=10)
        
        # Add size factor annotations
        for j, size_factor in enumerate(size_factors):
            ax.annotate(
                f"{size_factor}",
                (size_factor, metric_values[j]),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=9,
                fontweight='bold'
            )
    
    # Add a common title
    plt.suptitle('Trajectory Metrics Across Model Sizes', fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle
    
    # Save the figure
    output_path_grid = os.path.join(output_dir, "metric_line_grid.png")
    plt.savefig(output_path_grid, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved metric line grid to {output_path_grid}")
    
    return output_path, output_path_grid

def main():
    """Main function to run the metric line graph generation"""
    parser = argparse.ArgumentParser(description='Generate line graphs for trajectory metrics')
    parser.add_argument('--size_factors', type=str, default="0.1,0.2,0.4,0.6,0.8,1.0",
                        help='Comma-separated list of size factors to include')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save the line graphs')
    args = parser.parse_args()
    
    # Load configuration
    config = Config()
    
    # Parse size factors
    size_factors = [float(sf) for sf in args.size_factors.split(',')]
    
    # Set output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(config.analysis_dir, "trajectory_radar", "model_comparisons")
    
    # Create a dictionary of metrics for each size factor
    # This is a simplified version for demonstration purposes
    # In a real scenario, you would load these metrics from saved files
    metrics_by_size = {}
    
    # Check if we can load metrics from the trajectory_radar directory
    metrics_file = os.path.join(config.analysis_dir, "trajectory_radar", "metrics_by_size.npy")
    if os.path.exists(metrics_file):
        print(f"Loading metrics from {metrics_file}")
        metrics_by_size = np.load(metrics_file, allow_pickle=True).item()
    else:
        print(f"Metrics file not found at {metrics_file}")
        print("Please run the trajectory radar analysis first to generate metrics.")
        print("Example: python scripts/run_trajectory_radar.py --size_factors '0.1,0.2,0.4,0.6,0.8,1.0'")
        return
    
    # Generate the line graphs
    generate_metric_line_graphs(metrics_by_size, output_dir)
    
    print("\nMetric line graphs generation completed")
    print(f"Results saved in {output_dir}")

if __name__ == "__main__":
    main() 