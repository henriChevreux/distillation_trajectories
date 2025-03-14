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
    - Path Length Similarity (top): How similar the student's trajectory length is to the teacher's. 
      Higher score = more similar length.
    - Endpoint Alignment (left): How close the student's final image is to the teacher's.
      Higher score = closer final result.
    - Directional Consistency (bottom): How consistently the student follows the teacher's direction.
      Higher score = more consistent directional alignment.
    - Distribution Similarity (right): Overall similarity between student and teacher trajectories.
      Higher score = more similar distributions throughout the trajectory.
    
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
        output_dir = config.model_comparisons_dir
    
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
    all_path_length_similarities = []
    all_endpoint_distances = []
    all_directional_consistencies = []
    all_distribution_similarities = []
    
    for size_factor in size_factors:
        metrics = metrics_by_size[size_factor]
        
        # Extract raw metric values
        path_length_similarity = metrics.get('path_length_similarity', 1.0)
        endpoint_distance = metrics.get('endpoint_distance', 0.0)
        directional_consistency = metrics.get('mean_directional_consistency', 0.0)
        distribution_similarity = metrics.get('distribution_similarity', 0.0)
        
        all_path_length_similarities.append(path_length_similarity)
        all_endpoint_distances.append(endpoint_distance)
        all_directional_consistencies.append(directional_consistency)
        all_distribution_similarities.append(distribution_similarity)
    
    # Calculate min and max for each metric
    min_path_similarity = min(all_path_length_similarities) if all_path_length_similarities else 0.0
    max_path_similarity = max(all_path_length_similarities) if all_path_length_similarities else 1.0
    min_endpoint = min(all_endpoint_distances) if all_endpoint_distances else 0.0
    max_endpoint = max(all_endpoint_distances) if all_endpoint_distances else 1.0
    min_directional = min(all_directional_consistencies) if all_directional_consistencies else 0.0
    max_directional = max(all_directional_consistencies) if all_directional_consistencies else 1.0
    min_distribution = min(all_distribution_similarities) if all_distribution_similarities else 0.0
    max_distribution = max(all_distribution_similarities) if all_distribution_similarities else 1.0
    
    # Setup figure for the grid
    n_cols = min(3, len(size_factors))
    n_rows = int(np.ceil(len(size_factors) / n_cols))
    
    fig = plt.figure(figsize=(n_cols * 5, n_rows * 5))
    gs = gridspec.GridSpec(n_rows, n_cols)
    
    # Use a consistent set of colors for radar plots
    # These colors should be distinguishable and visually pleasing
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Set labels for radar chart axes
    labels = ['Path Length\nSimilarity', 'Endpoint\nAlignment', 'Directional\nConsistency', 'Distribution\nSimilarity']
    
    # Setup axes for each radar plot
    for i, size_factor in enumerate(size_factors):
        metrics = metrics_by_size[size_factor]
        
        # Get the raw metric values
        path_length_similarity = metrics.get('path_length_similarity', 1.0)
        endpoint_distance = metrics.get('endpoint_distance', 0.0)
        directional_consistency = metrics.get('mean_directional_consistency', 0.0)
        distribution_similarity = metrics.get('distribution_similarity', 0.0)
        
        # Normalize: path_length_similarity (higher is better)
        # Already in [0,1], so we can use it directly
        path_length_score = path_length_similarity
        
        # Normalize: endpoint_distance (lower is better)
        # Convert to a score where 1.0 is best (lowest distance)
        # Prevent division by zero
        endpoint_range = max_endpoint - min_endpoint
        if endpoint_range > 0:
            endpoint_distance_score = 1.0 - ((endpoint_distance - min_endpoint) / endpoint_range)
        else:
            endpoint_distance_score = 1.0
        
        # Normalize: directional_consistency (higher is better)
        # Already in [-1,1], scale to [0,1] where 1 is best
        directional_score = (directional_consistency + 1) / 2 if directional_consistency != float('nan') else 0.5
        
        # Normalize: distribution_similarity (higher is better)
        # Already in [0,1], so we can use it directly
        distribution_score = distribution_similarity
        
        # Combine the metrics into the order we want for the radar plot
        # Order: Path Length (top), Endpoint Dist (left), Directional Consistency (bottom), Distribution Similarity (right)
        values = [path_length_score, endpoint_distance_score, directional_score, distribution_score]
        
        # Print diagnostic information to verify metric values
        print(f"  Size {size_factor} metrics:")
        print(f"    Path Length Similarity: {path_length_similarity:.4f} → Score: {path_length_score:.4f}")
        print(f"    Endpoint Distance: {endpoint_distance:.4f} → Score: {endpoint_distance_score:.4f}")
        print(f"    Directional Consistency: {directional_consistency:.4f} → Score: {directional_score:.4f}")
        print(f"    Distribution Similarity: {distribution_similarity:.4f} → Score: {distribution_score:.4f}")
        
        # Number of metrics/axes
        N = len(labels)
        
        # Set the angle for each axis (divide the plot into equal parts)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Add the values for each axis, also closing the loop
        values += values[:1]
        
        # Select color for this radar plot
        color = colors[i % len(colors)]
        
        # Create the radar chart
        ax = plt.subplot(gs[i], polar=True)
        
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
        
        # Set the title with increased padding to avoid overlapping with Path Length Similarity label
        ax.set_title(f"Model Size {size_factor}", size=12, weight='bold', pad=20)  # Increased pad from 15 to 20
    
    # Adjust layout to prevent overlap
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
    direct comparison of all model sizes on the same axes. The plot uses the same four
    metrics as the grid:
    - Path Length Similarity (top)
    - Endpoint Alignment (left)
    - Directional Consistency (bottom)
    - Distribution Similarity (right)
    
    Args:
        metrics_by_size: Dictionary of metrics keyed by size factor
        config: Configuration object
        output_dir: Directory to save visualizations
        
    Returns:
        Path to the saved composite radar plot image
    """
    if output_dir is None:
        output_dir = config.model_comparisons_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating composite radar plot for all model sizes...")
    
    # Check if we have any valid metrics
    if not metrics_by_size:
        print("  Warning: No metrics available for comparison. Skipping composite radar plot.")
        return None
    
    # Convert all keys to floats and sort
    size_factors = []
    for key in metrics_by_size.keys():
        try:
            size_factors.append(float(key))
        except ValueError:
            print(f"  Warning: Could not convert size factor '{key}' to float, skipping.")
    
    size_factors.sort()
    
    if not size_factors:
        print("  Warning: No valid size factors found. Skipping composite radar plot.")
        return None
    
    # First, collect all metrics to find min and max values for min-max scaling
    all_path_length_similarities = []
    all_endpoint_distances = []
    all_directional_consistencies = []
    all_distribution_similarities = []
    
    for size_factor in size_factors:
        metrics = metrics_by_size[size_factor]
        
        path_length_similarity = metrics.get('path_length_similarity', 1.0)
        endpoint_distance = metrics.get('endpoint_distance', 0.0)
        directional_consistency = metrics.get('mean_directional_consistency', 0.0)
        distribution_similarity = metrics.get('distribution_similarity', 0.0)
        
        all_path_length_similarities.append(path_length_similarity)
        all_endpoint_distances.append(endpoint_distance)
        all_directional_consistencies.append(directional_consistency)
        all_distribution_similarities.append(distribution_similarity)
    
    # Calculate min and max for each metric
    min_path_similarity = min(all_path_length_similarities) if all_path_length_similarities else 0.0
    max_path_similarity = max(all_path_length_similarities) if all_path_length_similarities else 1.0
    min_endpoint = min(all_endpoint_distances) if all_endpoint_distances else 0.0
    max_endpoint = max(all_endpoint_distances) if all_endpoint_distances else 1.0
    min_directional = min(all_directional_consistencies) if all_directional_consistencies else 0.0
    max_directional = max(all_directional_consistencies) if all_directional_consistencies else 1.0
    min_distribution = min(all_distribution_similarities) if all_distribution_similarities else 0.0
    max_distribution = max(all_distribution_similarities) if all_distribution_similarities else 1.0
    
    # Set up the figure
    fig = plt.figure(figsize=(10, 8))
    
    # Set labels for radar chart axes
    labels = ['Path Length\nSimilarity', 'Endpoint\nAlignment', 'Directional\nConsistency', 'Distribution\nSimilarity']
    
    # Number of metrics/axes
    N = len(labels)
    
    # Set the angle for each axis (divide the plot into equal parts)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Use a color map that clearly differentiates the different models
    # viridis is a good choice for sequential data like model sizes
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 0.9, len(size_factors)))
    
    # Create the axis for the radar chart
    ax = plt.subplot(111, polar=True)
    
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
    
    # Create a legend
    legend_elements = []
    
    # Plot each model size
    for i, size_factor in enumerate(size_factors):
        metrics = metrics_by_size[size_factor]
        
        # Get the raw metric values
        path_length_similarity = metrics.get('path_length_similarity', 1.0)
        endpoint_distance = metrics.get('endpoint_distance', 0.0)
        directional_consistency = metrics.get('mean_directional_consistency', 0.0)
        distribution_similarity = metrics.get('distribution_similarity', 0.0)
        
        # Normalize the metrics for radar plot
        # path_length_similarity (higher is better)
        # Already in [0,1], so we can use it directly
        path_length_score = path_length_similarity
        
        # endpoint_distance (lower is better)
        # Convert to a score where 1.0 is best (lowest distance)
        # Prevent division by zero
        endpoint_range = max_endpoint - min_endpoint
        if endpoint_range > 0:
            endpoint_distance_score = 1.0 - ((endpoint_distance - min_endpoint) / endpoint_range)
        else:
            endpoint_distance_score = 1.0
        
        # directional_consistency (higher is better)
        # Already in [-1,1], scale to [0,1] where 1 is best
        directional_score = (directional_consistency + 1) / 2 if directional_consistency != float('nan') else 0.5
        
        # distribution_similarity (higher is better)
        # Already in [0,1], so we can use it directly
        distribution_score = distribution_similarity
        
        # Combine the metrics into the order we want for the radar plot
        # Order: Path Length (top), Endpoint Dist (left), Directional Consistency (bottom), Distribution Similarity (right)
        values = [path_length_score, endpoint_distance_score, directional_score, distribution_score]
        
        # Add the values for each axis, also closing the loop
        values += values[:1]
        
        # Select color for this radar plot
        color = colors[i]
        
        # Plot the metrics with thicker line
        ax.plot(angles, values, linewidth=2, linestyle='solid', color=color, label=f"Size {size_factor}")
        
        # Fill the polygon with lower alpha for better visibility when overlapping
        ax.fill(angles, values, alpha=0.1, color=color)
        
    # Add legend outside the radar plot
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # Set title
    plt.title('Trajectory Metrics Comparison Across Model Sizes', size=14, weight='bold', pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
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
        output_dir = config.model_comparisons_dir
    
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
        for key in ['path_length_similarity', 'endpoint_distance', 'mean_directional_consistency', 'distribution_similarity']:
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