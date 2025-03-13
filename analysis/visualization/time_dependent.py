"""
Time-dependent visualization functions for diffusion model analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid

def plot_time_dependent_grid(time_distances_dict, config, save_dir=None):
    """
    Plot a grid of time-dependent metrics for different size factors
    
    Args:
        time_distances_dict: Dictionary of time distances for different size factors
        config: Configuration object
        save_dir: Directory to save results
        
    Returns:
        None
    """
    print("Plotting time-dependent grid...")
    
    if save_dir is None:
        save_dir = os.path.join(config.output_dir, "time_dependent")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if we have any valid data
    if not time_distances_dict:
        print("  Warning: No time-dependent data available. Skipping grid plot.")
        return
    
    # Filter out size factors with empty data
    valid_size_factors = []
    for size_factor, time_distances in time_distances_dict.items():
        if ('teacher_avg_per_timestep' in time_distances and 
            'student_avg_per_timestep' in time_distances and
            time_distances['teacher_avg_per_timestep'] and 
            time_distances['student_avg_per_timestep']):
            valid_size_factors.append(size_factor)
    
    if not valid_size_factors:
        print("  Warning: No valid time-dependent data available. Skipping grid plot.")
        return
    
    # Sort size factors
    valid_size_factors.sort()
    
    # Determine grid dimensions
    n_plots = len(valid_size_factors)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    
    # Flatten axes if needed
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    
    # Plot each size factor
    for i, size_factor in enumerate(valid_size_factors):
        # Get row and column indices
        row = i // n_cols
        col = i % n_cols
        
        # Get the axis
        if n_rows == 1 and n_cols == 1:
            ax = axes[0]
        elif n_rows == 1:
            ax = axes[col]
        elif n_cols == 1:
            ax = axes[row]
        else:
            ax = axes[row, col]
        
        # Get time distances for this size factor
        time_distances = time_distances_dict[size_factor]
        
        # Plot teacher and student distances
        if ('teacher_avg_per_timestep' in time_distances and 
            'student_avg_per_timestep' in time_distances and
            time_distances['teacher_avg_per_timestep'] and 
            time_distances['student_avg_per_timestep']):
            ax.plot(time_distances['teacher_avg_per_timestep'], label='Teacher', color='blue')
            ax.plot(time_distances['student_avg_per_timestep'], label='Student', color='orange')
            
            # Add title and labels
            ax.set_title(f"Size Factor: {size_factor}")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Average Distance")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
    
    # Remove empty subplots
    for i in range(len(valid_size_factors), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows == 1:
            fig.delaxes(axes[col])
        elif n_cols == 1:
            fig.delaxes(axes[row])
        else:
            fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "time_dependent_grid.png"), dpi=300, bbox_inches='tight')
    plt.close()

def plot_time_dependent_combined(time_distances_dict, config, save_dir=None):
    """
    Plot a combined visualization of time-dependent metrics for different size factors
    
    Args:
        time_distances_dict: Dictionary of time distances for different size factors
        config: Configuration object
        save_dir: Directory to save results
        
    Returns:
        None
    """
    print("Plotting combined time-dependent visualization...")
    
    if save_dir is None:
        save_dir = os.path.join(config.output_dir, "time_dependent")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if we have any valid data
    if not time_distances_dict:
        print("  Warning: No time-dependent data available. Skipping combined plot.")
        return
    
    # Filter out size factors with empty data
    valid_size_factors = []
    for size_factor, time_distances in time_distances_dict.items():
        if ('student_avg_per_timestep' in time_distances and 
            time_distances['student_avg_per_timestep']):
            valid_size_factors.append(size_factor)
    
    if not valid_size_factors:
        print("  Warning: No valid time-dependent data available. Skipping combined plot.")
        return
    
    # Sort size factors
    valid_size_factors.sort()
    
    # Create a single plot with multiple lines for different size factors
    plt.figure(figsize=(12, 8))
    
    # Define a colormap for size factors
    cmap = plt.cm.viridis
    colors = [cmap(i / len(valid_size_factors)) for i in range(len(valid_size_factors))]
    
    # Plot student distances for each size factor
    for i, size_factor in enumerate(valid_size_factors):
        time_distances = time_distances_dict[size_factor]
        
        if ('student_avg_per_timestep' in time_distances and 
            time_distances['student_avg_per_timestep']):
            plt.plot(
                time_distances['student_avg_per_timestep'], 
                label=f"Size Factor: {size_factor}", 
                color=colors[i]
            )
    
    # Plot teacher distances (same for all size factors, so just use the first one)
    teacher_plotted = False
    for size_factor in valid_size_factors:
        if ('teacher_avg_per_timestep' in time_distances_dict[size_factor] and 
            time_distances_dict[size_factor]['teacher_avg_per_timestep']):
            plt.plot(
                time_distances_dict[size_factor]['teacher_avg_per_timestep'],
                label='Teacher',
                color='black',
                linestyle='--',
                linewidth=2
            )
            teacher_plotted = True
            break
    
    # Add title and labels
    plt.title("Time-Dependent Distances Across Size Factors")
    plt.xlabel("Timestep")
    plt.ylabel("Average Distance")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, "time_dependent_combined.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a ratio plot (student/teacher) if we have teacher data
    if teacher_plotted:
        plt.figure(figsize=(12, 8))
        
        for i, size_factor in enumerate(valid_size_factors):
            time_distances = time_distances_dict[size_factor]
            
            if ('student_avg_per_timestep' in time_distances and 
                'teacher_avg_per_timestep' in time_distances and
                time_distances['student_avg_per_timestep'] and
                time_distances['teacher_avg_per_timestep']):
                
                # Calculate ratio at each timestep
                student = time_distances['student_avg_per_timestep']
                teacher = time_distances['teacher_avg_per_timestep']
                
                # Ensure same length
                min_len = min(len(student), len(teacher))
                student = student[:min_len]
                teacher = teacher[:min_len]
                
                # Calculate ratio (avoid division by zero)
                ratio = []
                for s, t in zip(student, teacher):
                    if t > 0:
                        ratio.append(s / t)
                    else:
                        ratio.append(1.0)  # Default to 1 if teacher is 0
                
                plt.plot(
                    ratio,
                    label=f"Size Factor: {size_factor}",
                    color=colors[i]
                )
        
        # Add title and labels
        plt.title("Student/Teacher Distance Ratio Across Size Factors")
        plt.xlabel("Timestep")
        plt.ylabel("Ratio (Student/Teacher)")
        plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the plot
        plt.savefig(os.path.join(save_dir, "time_dependent_ratio.png"), dpi=300, bbox_inches='tight')
        plt.close() 