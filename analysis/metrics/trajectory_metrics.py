"""
Functions for computing and visualizing trajectory metrics
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from tqdm import tqdm

def compute_trajectory_metrics(teacher_trajectory, student_trajectory, config=None):
    """
    Compute metrics between teacher and student trajectories
    
    Args:
        teacher_trajectory: List of (image, timestep) pairs for teacher
        student_trajectory: List of (image, timestep) pairs for student
        config: Configuration object
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Extract images from trajectories
    # Each item in the trajectory is a tuple of (image, timestep)
    # Handle both cases: when items are tuples or when they are already images
    if isinstance(teacher_trajectory[0], tuple):
        teacher_images = [item[0] for item in teacher_trajectory]
    else:
        teacher_images = teacher_trajectory
        
    if isinstance(student_trajectory[0], tuple):
        student_images = [item[0] for item in student_trajectory]
    else:
        student_images = student_trajectory
    
    # Ensure the images have the same shape for comparison
    if teacher_images[-1].shape != student_images[-1].shape:
        # Resize student images to match teacher images if needed
        if teacher_images[-1].shape[2:] != student_images[-1].shape[2:]:
            resized_student_images = []
            for img in student_images:
                resized_img = torch.nn.functional.interpolate(
                    img, 
                    size=teacher_images[0].shape[2:],
                    mode='bilinear', 
                    align_corners=True
                )
                resized_student_images.append(resized_img)
            student_images = resized_student_images
    
    # Compute endpoint distance
    endpoint_distance = torch.norm(teacher_images[-1] - student_images[-1]).item()
    metrics['endpoint_distance'] = endpoint_distance
    
    # Compute path length
    teacher_path_length = 0
    student_path_length = 0
    
    for i in range(1, len(teacher_images)):
        teacher_path_length += torch.norm(teacher_images[i] - teacher_images[i-1]).item()
    
    for i in range(1, len(student_images)):
        student_path_length += torch.norm(student_images[i] - student_images[i-1]).item()
    
    metrics['teacher_path_length'] = teacher_path_length
    metrics['student_path_length'] = student_path_length
    metrics['path_length_ratio'] = student_path_length / teacher_path_length if teacher_path_length > 0 else float('inf')
    
    # Compute path efficiency (endpoint distance / path length)
    teacher_efficiency = endpoint_distance / teacher_path_length if teacher_path_length > 0 else 0
    student_efficiency = endpoint_distance / student_path_length if student_path_length > 0 else 0
    
    metrics['teacher_efficiency'] = teacher_efficiency
    metrics['student_efficiency'] = student_efficiency
    metrics['efficiency_ratio'] = student_efficiency / teacher_efficiency if teacher_efficiency > 0 else float('inf')
    
    # Compute velocity profile
    teacher_velocities = []
    student_velocities = []
    
    for i in range(1, len(teacher_images)):
        teacher_velocities.append(torch.norm(teacher_images[i] - teacher_images[i-1]).item())
    
    for i in range(1, len(student_images)):
        student_velocities.append(torch.norm(student_images[i] - student_images[i-1]).item())
    
    metrics['teacher_velocities'] = teacher_velocities
    metrics['student_velocities'] = student_velocities
    
    # Compute acceleration profile
    teacher_accelerations = []
    student_accelerations = []
    
    for i in range(1, len(teacher_velocities)):
        teacher_accelerations.append(teacher_velocities[i] - teacher_velocities[i-1])
    
    for i in range(1, len(student_velocities)):
        student_accelerations.append(student_velocities[i] - student_velocities[i-1])
    
    metrics['teacher_accelerations'] = teacher_accelerations
    metrics['student_accelerations'] = student_accelerations
    
    # Compute Wasserstein distance between distributions
    # Flatten images to compute distribution
    teacher_flat = [img.flatten().cpu().numpy() for img in teacher_images]
    student_flat = [img.flatten().cpu().numpy() for img in student_images]
    
    # Sample points for Wasserstein calculation (can be slow for full distributions)
    sample_size = 1000
    wasserstein_distances = []
    
    for t_img, s_img in zip(teacher_flat, student_flat):
        # Sample random indices
        indices = np.random.choice(len(t_img), min(sample_size, len(t_img)), replace=False)
        t_sample = t_img[indices]
        s_sample = s_img[indices]
        
        # Compute Wasserstein distance
        w_dist = wasserstein_distance(t_sample, s_sample)
        wasserstein_distances.append(w_dist)
    
    metrics['wasserstein_distances'] = wasserstein_distances
    metrics['mean_wasserstein'] = np.mean(wasserstein_distances)
    
    return metrics

def visualize_metrics(metrics_dict, output_dir=None, size_factor=None, suffix=""):
    """
    Visualize trajectory metrics
    
    Args:
        metrics_dict: Dictionary of metrics from compute_trajectory_metrics
        output_dir: Directory to save visualizations
        size_factor: Size factor of the student model for labeling
        suffix: Optional suffix to add to output filenames
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set up figure size and style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot path lengths
    plt.figure(figsize=(10, 6))
    plt.bar(['Teacher', 'Student'], 
            [metrics_dict['teacher_path_length'], metrics_dict['student_path_length']], 
            color=['blue', 'orange'])
    plt.title(f'Path Lengths (Size Factor: {size_factor})' if size_factor else 'Path Lengths')
    plt.ylabel('Path Length')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'path_lengths{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot endpoint distances
    plt.figure(figsize=(8, 5))
    plt.bar(['Endpoint Distance'], [metrics_dict['endpoint_distance']], color='green')
    plt.title(f'Endpoint Distance (Size Factor: {size_factor})' if size_factor else 'Endpoint Distance')
    plt.ylabel('Distance')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'endpoint_distances{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot velocity profiles
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_dict['teacher_velocities'], label='Teacher', color='blue')
    plt.plot(metrics_dict['student_velocities'], label='Student', color='orange')
    plt.title(f'Velocity Profile (Size Factor: {size_factor})' if size_factor else 'Velocity Profile')
    plt.xlabel('Step')
    plt.ylabel('Velocity')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'velocity_profile{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot acceleration profiles
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_dict['teacher_accelerations'], label='Teacher', color='blue')
    plt.plot(metrics_dict['student_accelerations'], label='Student', color='orange')
    plt.title(f'Acceleration Profile (Size Factor: {size_factor})' if size_factor else 'Acceleration Profile')
    plt.xlabel('Step')
    plt.ylabel('Acceleration')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'acceleration_profile{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot path efficiency
    plt.figure(figsize=(10, 6))
    plt.bar(['Teacher', 'Student'], 
            [metrics_dict['teacher_efficiency'], metrics_dict['student_efficiency']], 
            color=['blue', 'orange'])
    plt.title(f'Path Efficiency (Size Factor: {size_factor})' if size_factor else 'Path Efficiency')
    plt.ylabel('Efficiency (Endpoint Distance / Path Length)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'path_efficiency{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Wasserstein distances
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_dict['wasserstein_distances'], color='purple')
    plt.axhline(y=metrics_dict['mean_wasserstein'], color='red', linestyle='--', 
                label=f'Mean: {metrics_dict["mean_wasserstein"]:.4f}')
    plt.title(f'Wasserstein Distances (Size Factor: {size_factor})' if size_factor else 'Wasserstein Distances')
    plt.xlabel('Step')
    plt.ylabel('Wasserstein Distance')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'wasserstein_distances{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a summary text file
    if output_dir:
        with open(os.path.join(output_dir, f'summary{suffix}.txt'), 'w') as f:
            f.write(f"Size Factor: {size_factor if size_factor else 'N/A'}\n")
            f.write(f"Endpoint Distance: {metrics_dict['endpoint_distance']:.4f}\n")
            f.write(f"Path Length Ratio (Student/Teacher): {metrics_dict['path_length_ratio']:.4f}\n")
            f.write(f"Efficiency Ratio (Student/Teacher): {metrics_dict['efficiency_ratio']:.4f}\n")
            f.write(f"Mean Wasserstein Distance: {metrics_dict['mean_wasserstein']:.4f}\n")
            f.write(f"Teacher Path Length: {metrics_dict['teacher_path_length']:.4f}\n")
            f.write(f"Student Path Length: {metrics_dict['student_path_length']:.4f}\n")

def visualize_batch_metrics(metrics_dict, config, suffix=""):
    """
    Visualize trajectory metrics from batch processing
    
    Args:
        metrics_dict: Dictionary of metrics from compute_trajectory_metrics_batch
        config: Configuration object with output_dir
        suffix: Optional suffix to add to output filenames
        
    Returns:
        summary: Dictionary of summary metrics
    """
    output_dir = os.path.join(config.output_dir, "metrics")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up figure size and style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Extract size factor from suffix if available
    size_factor = None
    if suffix.startswith("_size_"):
        size_factor = suffix.split("_size_")[1]
    
    # Create summary dictionary
    summary = {}
    
    # Plot wasserstein distances histogram
    if 'wasserstein_distances' in metrics_dict and metrics_dict['wasserstein_distances']:
        plt.figure(figsize=(10, 6))
        plt.hist(metrics_dict['wasserstein_distances'], bins=20, alpha=0.7, color='purple')
        mean_wasserstein = sum(metrics_dict['wasserstein_distances']) / len(metrics_dict['wasserstein_distances'])
        plt.axvline(x=mean_wasserstein, color='red', linestyle='--', 
                    label=f'Mean: {mean_wasserstein:.4f}')
        plt.title(f'Wasserstein Distances Distribution (Size Factor: {size_factor})' if size_factor else 'Wasserstein Distances Distribution')
        plt.xlabel('Wasserstein Distance')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, f'wasserstein_distances_hist{suffix}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        summary['mean_wasserstein'] = mean_wasserstein
    
    # Plot endpoint distances histogram
    if 'endpoint_distances' in metrics_dict and metrics_dict['endpoint_distances']:
        plt.figure(figsize=(10, 6))
        plt.hist(metrics_dict['endpoint_distances'], bins=20, alpha=0.7, color='green')
        mean_endpoint = sum(metrics_dict['endpoint_distances']) / len(metrics_dict['endpoint_distances'])
        plt.axvline(x=mean_endpoint, color='red', linestyle='--', 
                    label=f'Mean: {mean_endpoint:.4f}')
        plt.title(f'Endpoint Distances Distribution (Size Factor: {size_factor})' if size_factor else 'Endpoint Distances Distribution')
        plt.xlabel('Endpoint Distance')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, f'endpoint_distances_hist{suffix}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        summary['mean_endpoint_distance'] = mean_endpoint
    
    # Plot path lengths comparison
    if 'teacher_path_lengths' in metrics_dict and 'student_path_lengths' in metrics_dict:
        if metrics_dict['teacher_path_lengths'] and metrics_dict['student_path_lengths']:
            # Compute means
            mean_teacher_path = sum(metrics_dict['teacher_path_lengths']) / len(metrics_dict['teacher_path_lengths'])
            mean_student_path = sum(metrics_dict['student_path_lengths']) / len(metrics_dict['student_path_lengths'])
            
            # Plot means
            plt.figure(figsize=(10, 6))
            plt.bar(['Teacher', 'Student'], [mean_teacher_path, mean_student_path], color=['blue', 'orange'])
            plt.title(f'Average Path Lengths (Size Factor: {size_factor})' if size_factor else 'Average Path Lengths')
            plt.ylabel('Path Length')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(output_dir, f'avg_path_lengths{suffix}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot distributions
            plt.figure(figsize=(12, 6))
            plt.hist(metrics_dict['teacher_path_lengths'], bins=20, alpha=0.5, label='Teacher', color='blue')
            plt.hist(metrics_dict['student_path_lengths'], bins=20, alpha=0.5, label='Student', color='orange')
            plt.axvline(x=mean_teacher_path, color='blue', linestyle='--', 
                        label=f'Teacher Mean: {mean_teacher_path:.4f}')
            plt.axvline(x=mean_student_path, color='orange', linestyle='--', 
                        label=f'Student Mean: {mean_student_path:.4f}')
            plt.title(f'Path Lengths Distribution (Size Factor: {size_factor})' if size_factor else 'Path Lengths Distribution')
            plt.xlabel('Path Length')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(output_dir, f'path_lengths_hist{suffix}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Compute path length ratio
            path_length_ratio = mean_student_path / mean_teacher_path if mean_teacher_path > 0 else float('inf')
            summary['mean_teacher_path_length'] = mean_teacher_path
            summary['mean_student_path_length'] = mean_student_path
            summary['path_length_ratio'] = path_length_ratio
    
    # Plot efficiency comparison
    if 'teacher_efficiency' in metrics_dict and 'student_efficiency' in metrics_dict:
        if metrics_dict['teacher_efficiency'] and metrics_dict['student_efficiency']:
            # Compute means
            mean_teacher_efficiency = sum(metrics_dict['teacher_efficiency']) / len(metrics_dict['teacher_efficiency'])
            mean_student_efficiency = sum(metrics_dict['student_efficiency']) / len(metrics_dict['student_efficiency'])
            
            # Plot means
            plt.figure(figsize=(10, 6))
            plt.bar(['Teacher', 'Student'], [mean_teacher_efficiency, mean_student_efficiency], color=['blue', 'orange'])
            plt.title(f'Average Path Efficiency (Size Factor: {size_factor})' if size_factor else 'Average Path Efficiency')
            plt.ylabel('Efficiency (Endpoint Distance / Path Length)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(output_dir, f'avg_path_efficiency{suffix}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot distributions
            plt.figure(figsize=(12, 6))
            plt.hist(metrics_dict['teacher_efficiency'], bins=20, alpha=0.5, label='Teacher', color='blue')
            plt.hist(metrics_dict['student_efficiency'], bins=20, alpha=0.5, label='Student', color='orange')
            plt.axvline(x=mean_teacher_efficiency, color='blue', linestyle='--', 
                        label=f'Teacher Mean: {mean_teacher_efficiency:.4f}')
            plt.axvline(x=mean_student_efficiency, color='orange', linestyle='--', 
                        label=f'Student Mean: {mean_student_efficiency:.4f}')
            plt.title(f'Path Efficiency Distribution (Size Factor: {size_factor})' if size_factor else 'Path Efficiency Distribution')
            plt.xlabel('Efficiency')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(output_dir, f'path_efficiency_hist{suffix}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Compute efficiency ratio
            efficiency_ratio = mean_student_efficiency / mean_teacher_efficiency if mean_teacher_efficiency > 0 else float('inf')
            summary['mean_teacher_efficiency'] = mean_teacher_efficiency
            summary['mean_student_efficiency'] = mean_student_efficiency
            summary['efficiency_ratio'] = efficiency_ratio
    
    # Plot wasserstein distances per timestep if available
    if 'wasserstein_distances_per_timestep' in metrics_dict and metrics_dict['wasserstein_distances_per_timestep']:
        # Compute average wasserstein distance at each timestep
        num_timesteps = len(metrics_dict['wasserstein_distances_per_timestep'][0])
        avg_wasserstein_per_timestep = [0] * num_timesteps
        
        for distances in metrics_dict['wasserstein_distances_per_timestep']:
            for t in range(min(len(distances), num_timesteps)):
                avg_wasserstein_per_timestep[t] += distances[t]
        
        # Divide by number of trajectories
        num_trajectories = len(metrics_dict['wasserstein_distances_per_timestep'])
        avg_wasserstein_per_timestep = [d / num_trajectories for d in avg_wasserstein_per_timestep]
        
        # Plot average wasserstein distance per timestep
        plt.figure(figsize=(12, 6))
        plt.plot(avg_wasserstein_per_timestep, color='purple')
        plt.title(f'Average Wasserstein Distance per Timestep (Size Factor: {size_factor})' if size_factor else 'Average Wasserstein Distance per Timestep')
        plt.xlabel('Timestep')
        plt.ylabel('Wasserstein Distance')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, f'wasserstein_per_timestep{suffix}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        summary['wasserstein_per_timestep'] = avg_wasserstein_per_timestep
    
    # Create a summary text file
    with open(os.path.join(output_dir, f'summary{suffix}.txt'), 'w') as f:
        f.write(f"Size Factor: {size_factor if size_factor else 'N/A'}\n\n")
        
        if 'mean_endpoint_distance' in summary:
            f.write(f"Mean Endpoint Distance: {summary['mean_endpoint_distance']:.4f}\n")
        
        if 'path_length_ratio' in summary:
            f.write(f"Mean Path Length Ratio (Student/Teacher): {summary['path_length_ratio']:.4f}\n")
            f.write(f"Mean Teacher Path Length: {summary['mean_teacher_path_length']:.4f}\n")
            f.write(f"Mean Student Path Length: {summary['mean_student_path_length']:.4f}\n")
        
        if 'efficiency_ratio' in summary:
            f.write(f"Mean Efficiency Ratio (Student/Teacher): {summary['efficiency_ratio']:.4f}\n")
            f.write(f"Mean Teacher Efficiency: {summary['mean_teacher_efficiency']:.4f}\n")
            f.write(f"Mean Student Efficiency: {summary['mean_student_efficiency']:.4f}\n")
        
        if 'mean_wasserstein' in summary:
            f.write(f"Mean Wasserstein Distance: {summary['mean_wasserstein']:.4f}\n")
    
    return summary 