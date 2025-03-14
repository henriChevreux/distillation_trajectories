"""
Model comparison metrics for diffusion models
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.gridspec as gridspec

def create_radar_plot_grid(metrics_by_size, config, output_dir=None):
    """
    Create a grid of radar plots to compare trajectory metrics across different model sizes.
    
    Each radar plot represents a specific model size and displays four key metrics:
    - Path Length (top): How close the student's trajectory length is to the teacher's. 
      Higher score = more similar length.
    - Endpoint Distance (left): How close the student's final image is to the teacher's.
      Higher score = closer final result.
    - Path Efficiency (bottom): How directly the model moves through latent space.
      Higher score = more efficient trajectory.
    - Wasserstein Distance (right): Overall distribution similarity between trajectories.
      Higher score = more similar distribution.
    
    All metrics are normalized using min-max scaling to [0,1] where 1 is the best performance. 
    The filled area of each radar plot gives a visual indication of overall model performance - 
    larger area means better overall performance across all metrics.
    
    Args:
        metrics_by_size: Dictionary of metrics keyed by size factor
        config: Configuration object
        output_dir: Directory to save visualizations
        
    Returns:
        Path to the saved radar plot grid image
    """
    if output_dir is None:
        output_dir = os.path.join(config.output_dir, "model_comparisons")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating radar plot grid for model size comparisons...")
    
    # Check if we have any valid metrics
    if not metrics_by_size:
        print("  Warning: No metrics available for comparison. Skipping radar plot grid.")
        return None
    
    # Convert all keys to floats to ensure consistent handling of size factors
    # This is important as size factors might be stored as strings in some places
    metrics_by_size_normalized = {}
    for key, value in metrics_by_size.items():
        try:
            float_key = float(key)
            metrics_by_size_normalized[float_key] = value
        except (ValueError, TypeError) as e:
            print(f"  Warning: Could not convert size factor '{key}' to float: {e}")
            # Keep the original key if conversion fails
            metrics_by_size_normalized[key] = value
    
    # Replace the original dictionary with the normalized one
    metrics_by_size = metrics_by_size_normalized
    
    # Extract size factors and sort them
    size_factors = sorted(metrics_by_size.keys())
    
    # Print size factors for debugging
    print(f"  Size factors after normalization: {size_factors}")
    
    # First, collect all metrics to find min and max values for min-max scaling
    all_path_length_ratios = []
    all_endpoint_distances = []
    all_efficiency_ratios = []
    all_wasserstein_distances = []
    
    for size_factor in size_factors:
        metrics = metrics_by_size[size_factor]
        
        # Extract raw metric values
        path_length_ratio = metrics.get('path_length_ratio', 1.0)
        endpoint_distance = metrics.get('mean_endpoint_distance', 0.0)
        efficiency_ratio = metrics.get('efficiency_ratio', 1.0)
        wasserstein_distance = metrics.get('mean_wasserstein', 0.0)
        
        all_path_length_ratios.append(path_length_ratio)
        all_endpoint_distances.append(endpoint_distance)
        all_efficiency_ratios.append(efficiency_ratio)
        all_wasserstein_distances.append(wasserstein_distance)
    
    # Calculate min and max for each metric
    min_path_ratio = min(all_path_length_ratios)
    max_path_ratio = max(all_path_length_ratios)
    min_endpoint = min(all_endpoint_distances)
    max_endpoint = max(all_endpoint_distances)
    min_efficiency = min(all_efficiency_ratios)
    max_efficiency = max(all_efficiency_ratios)
    min_wasserstein = min(all_wasserstein_distances)
    max_wasserstein = max(all_wasserstein_distances)
    
    # Print ranges for debugging
    print(f"  Value ranges for scaling:")
    print(f"    Path Length Ratio: {min_path_ratio:.4f} to {max_path_ratio:.4f}")
    print(f"    Endpoint Distance: {min_endpoint:.4f} to {max_endpoint:.4f}")
    print(f"    Efficiency Ratio: {min_efficiency:.4f} to {max_efficiency:.4f}")
    print(f"    Wasserstein Distance: {min_wasserstein:.4f} to {max_wasserstein:.4f}")
    
    # Get number of plots
    n_plots = len(size_factors)
    
    # Define grid layout
    n_cols = 3  # 3 plots per row
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure for the grid
    fig = plt.figure(figsize=(5*n_cols, 5*n_rows + 0.8), facecolor='white')  # Increased space for title
    plt.rcParams['axes.facecolor'] = 'white'
    
    # Add a global title to the entire figure
    fig.suptitle("Model Size Comparison - Trajectory Metrics", 
                 fontsize=18, fontweight='bold', y=0.99)  # Moved higher
    
    # Add some padding between subplots
    plt.subplots_adjust(wspace=0.4, hspace=0.6, top=0.92)  # Added top margin
    
    # Define the colormap for different size factors
    # Use a blue to green to yellow color map similar to the reference
    colors = {
        0.1: (0.0, 0.0, 0.5),  # Dark blue
        0.2: (0.0, 0.0, 0.8),  # Blue
        0.3: (0.0, 0.2, 0.8),  # Blue-medium
        0.4: (0.0, 0.5, 0.8),  # Teal blue
        0.5: (0.0, 0.6, 0.6),  # Teal
        0.6: (0.0, 0.7, 0.5),  # Teal-green
        0.7: (0.0, 0.8, 0.3),  # Green
        0.8: (0.4, 0.8, 0.2),  # Light green
        0.9: (0.7, 0.8, 0.1),  # Yellow-green
        1.0: (1.0, 0.9, 0.0),  # Yellow
    }
    
    # Convert string color keys to floats for compatibility
    colors_normalized = {}
    for key, value in colors.items():
        colors_normalized[float(key)] = value
    colors = colors_normalized
    
    # Setup axes for each radar plot
    for i, size_factor in enumerate(size_factors):
        metrics = metrics_by_size[size_factor]
        
        # Get color based on size factor or use a default from the colormap
        if size_factor in colors:
            color = colors[size_factor]
        else:
            # Fallback to viridis colormap if size_factor not in predefined colors
            cmap = plt.cm.viridis
            norm = plt.Normalize(min(size_factors), max(size_factors))
            color = cmap(norm(size_factor))
        
        # Create subplot with white background
        row = i // n_cols
        col = i % n_cols
        ax = plt.subplot(n_rows, n_cols, i+1, polar=True)
        ax.set_facecolor('white')
        
        # Define the metrics to plot and their labels
        # Match the label placement from the reference image
        labels = ['Path Length Ratio', 'Endpoint Dist', 'Path Efficiency', 'Wasserstein']
        
        # Extract metrics and normalize them using proper scaling
        path_length_ratio = metrics.get('path_length_ratio', 1.0)
        endpoint_distance = metrics.get('mean_endpoint_distance', 0.0)
        efficiency_ratio = metrics.get('efficiency_ratio', 1.0)
        wasserstein_distance = metrics.get('mean_wasserstein', 0.0)
        
        # For path length ratio:
        # Ideal value is 1.0. Score based on deviation from 1.0
        # Use max_path_deviation = max(max_ratio - 1, 1 - min_ratio) as the worst possible deviation
        max_path_deviation = max(max_path_ratio - 1.0, 1.0 - min_path_ratio) if min_path_ratio < 1.0 else max_path_ratio - 1.0
        path_deviation = abs(path_length_ratio - 1.0)
        # Ensure we don't divide by zero
        max_path_deviation = max(0.001, max_path_deviation)
        path_length_score = max(0.1, 1.0 - (path_deviation / max_path_deviation))
        
        # For endpoint distance:
        # Ideal value is 0. Scale from 0 to max, with 0 being best (score=1.0)
        # Ensure we don't divide by zero
        max_endpoint = max(0.001, max_endpoint)
        endpoint_distance_score = max(0.1, 1.0 - (endpoint_distance / max_endpoint))
        
        # For efficiency ratio:
        # Ideal value is 1.0. Scale from 0 to 1, with 1 being best (score=1.0)
        # Ensure we don't divide by zero
        efficiency_score = max(0.1, efficiency_ratio / 1.0)
        
        # For Wasserstein distance:
        # Ideal value is 0. Scale from 0 to max, with 0 being best (score=1.0)
        # Ensure we don't divide by zero
        max_wasserstein = max(0.001, max_wasserstein)
        wasserstein_score = max(0.1, 1.0 - (wasserstein_distance / max_wasserstein))
        
        # Combine the metrics into the order we want for the radar plot
        # Order: Path Length (top), Endpoint Dist (left), Path Efficiency (bottom), Wasserstein (right)
        values = [path_length_score, endpoint_distance_score, efficiency_score, wasserstein_score]
        
        # Print diagnostic information to verify metric values
        print(f"  Size {size_factor} metrics:")
        print(f"    Path Length Ratio: {path_length_ratio:.4f} → Score: {path_length_score:.4f}")
        print(f"    Endpoint Distance: {endpoint_distance:.4f} → Score: {endpoint_distance_score:.4f}")
        print(f"    Efficiency Ratio: {efficiency_ratio:.4f} → Score: {efficiency_score:.4f}")
        print(f"    Wasserstein Distance: {wasserstein_distance:.4f} → Score: {wasserstein_score:.4f}")
        
        # Number of metrics/axes
        N = len(labels)
        
        # Set the angle for each axis (divide the plot into equal parts)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Add the values for each axis, also closing the loop
        values += values[:1]
        
        # Draw the axes
        ax.set_theta_offset(np.pi / 2)  # Start from the top
        ax.set_theta_direction(-1)  # Go clockwise
        
        # Draw the labels for each axis
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        
        # Draw radial lines and circles
        ax.grid(True, alpha=0.3, linestyle='-')
        
        # Set y-ticks (concentric circles)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=8, alpha=0.7)
        ax.set_rlim(0, 1)
        
        # Plot the metrics with thicker line
        ax.plot(angles, values, linewidth=1.5, linestyle='solid', color=color)
        
        # Fill the polygon with slightly higher alpha for better visibility
        ax.fill(angles, values, alpha=0.35, color=color)
        
        # Add a thin black outline to the polygon for definition
        ax.plot(angles, values, linewidth=0.5, linestyle='solid', color='black', alpha=0.3)
        
        # Set the title with increased padding to avoid overlapping with Path Length Ratio label
        ax.set_title(f"Model Size {size_factor}", size=12, weight='bold', pad=20)  # Increased pad from 15 to 20
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, "radar_plot_grid.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved radar plot grid to {output_path}")
    
    return output_path

def create_composite_radar_plot(metrics_by_size, config, output_dir=None):
    """
    Create a single radar plot showing all model sizes together for direct comparison.
    
    This provides a complementary view to the grid of individual radar plots, allowing
    for direct comparison between model sizes on the same axes.
    
    Args:
        metrics_by_size: Dictionary of metrics keyed by size factor
        config: Configuration object
        output_dir: Directory to save visualization
        
    Returns:
        Path to the saved composite radar plot image
    """
    if output_dir is None:
        output_dir = os.path.join(config.output_dir, "model_comparisons")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating composite radar plot for all model sizes...")
    
    # Check if we have any valid metrics
    if not metrics_by_size:
        print("  Warning: No metrics available for comparison. Skipping composite radar plot.")
        return None
    
    # Convert all keys to floats to ensure consistent handling of size factors
    metrics_by_size_normalized = {}
    for key, value in metrics_by_size.items():
        try:
            float_key = float(key)
            metrics_by_size_normalized[float_key] = value
        except (ValueError, TypeError) as e:
            print(f"  Warning: Could not convert size factor '{key}' to float: {e}")
            metrics_by_size_normalized[key] = value
    
    # Replace the original dictionary with the normalized one
    metrics_by_size = metrics_by_size_normalized
    
    # Extract size factors and sort them
    size_factors = sorted(metrics_by_size.keys())
    
    # First, collect all metrics to find min and max values for scaling
    all_path_length_ratios = []
    all_endpoint_distances = []
    all_efficiency_ratios = []
    all_wasserstein_distances = []
    
    for size_factor in size_factors:
        metrics = metrics_by_size[size_factor]
        
        # Extract raw metric values
        path_length_ratio = metrics.get('path_length_ratio', 1.0)
        endpoint_distance = metrics.get('mean_endpoint_distance', 0.0)
        efficiency_ratio = metrics.get('efficiency_ratio', 1.0)
        wasserstein_distance = metrics.get('mean_wasserstein', 0.0)
        
        all_path_length_ratios.append(path_length_ratio)
        all_endpoint_distances.append(endpoint_distance)
        all_efficiency_ratios.append(efficiency_ratio)
        all_wasserstein_distances.append(wasserstein_distance)
    
    # Calculate min and max for each metric
    min_path_ratio = min(all_path_length_ratios)
    max_path_ratio = max(all_path_length_ratios)
    min_endpoint = min(all_endpoint_distances)
    max_endpoint = max(all_endpoint_distances)
    min_efficiency = min(all_efficiency_ratios)
    max_efficiency = max(all_efficiency_ratios)
    min_wasserstein = min(all_wasserstein_distances)
    max_wasserstein = max(all_wasserstein_distances)
    
    # Print ranges for debugging
    print(f"  Value ranges for scaling (composite plot):")
    print(f"    Path Length Ratio: {min_path_ratio:.4f} to {max_path_ratio:.4f}")
    print(f"    Endpoint Distance: {min_endpoint:.4f} to {max_endpoint:.4f}")
    print(f"    Efficiency Ratio: {min_efficiency:.4f} to {max_efficiency:.4f}")
    print(f"    Wasserstein Distance: {min_wasserstein:.4f} to {max_wasserstein:.4f}")
    
    # Define the colormap for different size factors
    # Use a blue to green to yellow color map similar to the reference
    colors = {
        0.1: (0.0, 0.0, 0.5),  # Dark blue
        0.2: (0.0, 0.0, 0.8),  # Blue
        0.3: (0.0, 0.2, 0.8),  # Blue-medium
        0.4: (0.0, 0.5, 0.8),  # Teal blue
        0.5: (0.0, 0.6, 0.6),  # Teal
        0.6: (0.0, 0.7, 0.5),  # Teal-green
        0.7: (0.0, 0.8, 0.3),  # Green
        0.8: (0.4, 0.8, 0.2),  # Light green
        0.9: (0.7, 0.8, 0.1),  # Yellow-green
        1.0: (1.0, 0.9, 0.0),  # Yellow
    }
    
    # Convert string color keys to floats for compatibility
    colors_normalized = {}
    for key, value in colors.items():
        colors_normalized[float(key)] = value
    colors = colors_normalized
    
    # Create figure
    fig = plt.figure(figsize=(10, 10.5), facecolor='white')  # Increased vertical size
    ax = plt.subplot(111, polar=True)
    ax.set_facecolor('white')
    
    # Define the metrics to plot and their labels
    labels = ['Path Length Ratio', 'Endpoint Dist', 'Path Efficiency', 'Wasserstein']
    
    # Number of metrics/axes
    N = len(labels)
    
    # Set the angle for each axis (divide the plot into equal parts)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Draw the axes
    ax.set_theta_offset(np.pi / 2)  # Start from the top
    ax.set_theta_direction(-1)  # Go clockwise
    
    # Draw the labels for each axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=12)
    
    # Draw radial lines and circles
    ax.grid(True, alpha=0.3, linestyle='-')
    
    # Set y-ticks (concentric circles)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=10, alpha=0.7)
    ax.set_rlim(0, 1)
    
    # Plot data for each size factor
    for size_factor in size_factors:
        metrics = metrics_by_size[size_factor]
        
        # Get color based on size factor or use a default
        if size_factor in colors:
            color = colors[size_factor]
        else:
            # Fallback to viridis colormap if size_factor not in predefined colors
            cmap = plt.cm.viridis
            norm = plt.Normalize(min(size_factors), max(size_factors))
            color = cmap(norm(size_factor))
        
        # Extract metrics and normalize them
        path_length_ratio = metrics.get('path_length_ratio', 1.0)
        endpoint_distance = metrics.get('mean_endpoint_distance', 0.0)
        efficiency_ratio = metrics.get('efficiency_ratio', 1.0)
        wasserstein_distance = metrics.get('mean_wasserstein', 0.0)
        
        # For path length ratio:
        # Ideal value is 1.0. Score based on deviation from 1.0
        max_path_deviation = max(max_path_ratio - 1.0, 1.0 - min_path_ratio) if min_path_ratio < 1.0 else max_path_ratio - 1.0
        path_deviation = abs(path_length_ratio - 1.0)
        max_path_deviation = max(0.001, max_path_deviation)
        path_length_score = max(0.1, 1.0 - (path_deviation / max_path_deviation))
        
        # For endpoint distance:
        # Ideal value is 0. Scale from 0 to max, with 0 being best (score=1.0)
        max_endpoint = max(0.001, max_endpoint)
        endpoint_distance_score = max(0.1, 1.0 - (endpoint_distance / max_endpoint))
        
        # For efficiency ratio:
        # Ideal value is 1.0. Scale from 0 to 1, with 1 being best (score=1.0)
        efficiency_score = max(0.1, efficiency_ratio / 1.0)
        
        # For Wasserstein distance:
        # Ideal value is 0. Scale from 0 to max, with 0 being best (score=1.0)
        max_wasserstein = max(0.001, max_wasserstein)
        wasserstein_score = max(0.1, 1.0 - (wasserstein_distance / max_wasserstein))
        
        # Combine the metrics into the order we want for the radar plot
        values = [path_length_score, endpoint_distance_score, efficiency_score, wasserstein_score]
        values += values[:1]  # Close the loop
        
        # Plot the metrics with a line and filled polygon
        line = ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"Size {size_factor}")
        line_color = line[0].get_color() if size_factor not in colors else color
        ax.fill(angles, values, alpha=0.1, color=line_color)
    
    # Add legend outside the plot
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
    
    # Add title
    plt.title("Comparison of Trajectory Metrics Across Model Sizes", size=16, weight='bold', pad=40)  # Increased padding
    
    # Adjust layout with more margin at top
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Keep space at top for title
    
    # Save the figure
    output_path = os.path.join(output_dir, "composite_radar_plot.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved composite radar plot to {output_path}")
    
    return output_path

def create_model_size_comparisons(metrics_by_size, fid_results, config, output_dir=None):
    """
    Create comparisons of metrics across different model sizes
    
    Args:
        metrics_by_size: Dictionary of metrics keyed by size factor
        fid_results: Dictionary of FID results keyed by size factor
        config: Configuration object
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary of comparison results
    """
    if output_dir is None:
        output_dir = os.path.join(config.output_dir, "model_comparisons")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n===== MODEL SIZE COMPARISONS =====")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    
    # Convert all keys to floats to ensure consistent handling of size factors
    # This is important as size factors might be stored as strings in some places
    metrics_by_size_normalized = {}
    for key, value in metrics_by_size.items():
        try:
            float_key = float(key)
            metrics_by_size_normalized[float_key] = value
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not convert size factor '{key}' to float: {e}")
            # Keep the original key if conversion fails
            metrics_by_size_normalized[key] = value
    
    # Replace the original dictionary with the normalized one
    metrics_by_size = metrics_by_size_normalized
    
    print(f"Available metrics for size factors: {sorted(metrics_by_size.keys())}")
    
    # Print structure of the first size factor metrics to help debug
    if metrics_by_size:
        first_size = sorted(metrics_by_size.keys())[0]
        print(f"Sample metrics for size {first_size}:")
        for key in ['path_length_ratio', 'mean_endpoint_distance', 'efficiency_ratio', 'mean_wasserstein']:
            print(f"  - {key}: {metrics_by_size[first_size].get(key, 'NOT FOUND')}")
    
    # Check if we have any valid metrics
    if not metrics_by_size:
        print("WARNING: No metrics available for comparison. Skipping model size comparisons.")
        return {"status": "no_data"}
    
    results = {"status": "success"}
    
    # Create radar plot grid comparing metrics across model sizes
    print("Generating radar plot grid...")
    try:
        # Create grid of individual radar plots
        radar_plot_path = create_radar_plot_grid(metrics_by_size, config, output_dir)
        print(f"Radar plot grid saved to: {os.path.abspath(radar_plot_path)}")
        results["radar_plot_grid_path"] = radar_plot_path
        
        # Create a single composite radar plot with all models
        composite_path = create_composite_radar_plot(metrics_by_size, config, output_dir)
        print(f"Composite radar plot saved to: {os.path.abspath(composite_path)}")
        results["composite_radar_plot_path"] = composite_path
        
        # Verify the files were actually created
        for path in [radar_plot_path, composite_path]:
            if path and os.path.exists(path):
                file_size = os.path.getsize(path)
                print(f"File {os.path.basename(path)} size: {file_size} bytes")
            elif path:
                print(f"WARNING: File was not created at {path}")
        
        return results
    
    except Exception as e:
        import traceback
        print(f"ERROR generating radar plots: {str(e)}")
        print(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e)
        } 