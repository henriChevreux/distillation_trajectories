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
    
    # Compute endpoint distance (distance between final images)
    endpoint_distance = torch.norm(teacher_images[-1] - student_images[-1]).item()
    metrics['endpoint_distance'] = endpoint_distance
    
    # Compute path length for teacher and student
    teacher_path_length = 0
    student_path_length = 0
    
    for i in range(1, len(teacher_images)):
        teacher_path_length += torch.norm(teacher_images[i] - teacher_images[i-1]).item()
    
    for i in range(1, len(student_images)):
        student_path_length += torch.norm(student_images[i] - student_images[i-1]).item()
    
    metrics['teacher_path_length'] = teacher_path_length
    metrics['student_path_length'] = student_path_length
    
    # Compute path length similarity (ratio closer to 1.0 is better)
    path_length_ratio = student_path_length / teacher_path_length if teacher_path_length > 0 else float('inf')
    metrics['path_length_ratio'] = path_length_ratio
    
    # Calculate path length similarity (1.0 means identical lengths)
    path_length_similarity = min(teacher_path_length, student_path_length) / max(teacher_path_length, student_path_length) if max(teacher_path_length, student_path_length) > 0 else 1.0
    metrics['path_length_similarity'] = path_length_similarity
    
    # Compute proper path efficiency with start-to-end distance for each model
    teacher_endpoint_to_start = torch.norm(teacher_images[-1] - teacher_images[0]).item()
    student_endpoint_to_start = torch.norm(student_images[-1] - student_images[0]).item()
    
    teacher_efficiency = teacher_endpoint_to_start / teacher_path_length if teacher_path_length > 0 else 0
    student_efficiency = student_endpoint_to_start / student_path_length if student_path_length > 0 else 0
    
    metrics['teacher_efficiency'] = teacher_efficiency
    metrics['student_efficiency'] = student_efficiency
    
    # Calculate efficiency similarity (1.0 means identical efficiency)
    efficiency_similarity = min(teacher_efficiency, student_efficiency) / max(teacher_efficiency, student_efficiency) if max(teacher_efficiency, student_efficiency) > 0 else 1.0
    metrics['efficiency_similarity'] = efficiency_similarity
    
    # Compute velocity profile
    teacher_velocities = []
    student_velocities = []
    
    for i in range(1, len(teacher_images)):
        teacher_velocities.append(torch.norm(teacher_images[i] - teacher_images[i-1]).item())
    
    for i in range(1, len(student_images)):
        student_velocities.append(torch.norm(student_images[i] - student_images[i-1]).item())
    
    metrics['teacher_velocities'] = teacher_velocities
    metrics['student_velocities'] = student_velocities
    
    # Compute velocity similarity at each timestep
    velocity_similarities = []
    for i in range(min(len(teacher_velocities), len(student_velocities))):
        t_vel = teacher_velocities[i]
        s_vel = student_velocities[i]
        similarity = min(t_vel, s_vel) / max(t_vel, s_vel) if max(t_vel, s_vel) > 0 else 1.0
        velocity_similarities.append(similarity)
    
    metrics['velocity_similarities'] = velocity_similarities
    metrics['mean_velocity_similarity'] = np.mean(velocity_similarities) if velocity_similarities else 0.0
    
    # Compute acceleration profile
    teacher_accelerations = []
    student_accelerations = []
    
    for i in range(1, len(teacher_velocities)):
        teacher_accelerations.append(teacher_velocities[i] - teacher_velocities[i-1])
    
    for i in range(1, len(student_velocities)):
        student_accelerations.append(student_velocities[i] - student_velocities[i-1])
    
    metrics['teacher_accelerations'] = teacher_accelerations
    metrics['student_accelerations'] = student_accelerations
    
    # Add timestep-by-timestep position difference metric
    position_differences = []
    for i in range(min(len(teacher_images), len(student_images))):
        position_differences.append(torch.norm(teacher_images[i] - student_images[i]).item())
    
    metrics['position_differences'] = position_differences
    metrics['mean_position_difference'] = np.mean(position_differences) if position_differences else 0.0
    metrics['max_position_difference'] = np.max(position_differences) if position_differences else 0.0
    
    # Measure directional consistency (how consistently student moves in same direction as teacher)
    directional_consistency = []
    for i in range(min(len(teacher_images), len(student_images))-1):
        # Teacher direction vector
        t_dir = teacher_images[i+1] - teacher_images[i]
        # Student direction vector
        s_dir = student_images[i+1] - student_images[i]
        
        # Compute cosine similarity between direction vectors
        t_dir_norm = torch.norm(t_dir)
        s_dir_norm = torch.norm(s_dir)
        
        if t_dir_norm > 0 and s_dir_norm > 0:
            # Flatten the tensors for dot product
            t_dir_flat = t_dir.flatten()
            s_dir_flat = s_dir.flatten()
            cos_sim = torch.sum(t_dir_flat * s_dir_flat) / (torch.norm(t_dir_flat) * torch.norm(s_dir_flat))
            directional_consistency.append(cos_sim.item())
    
    metrics['directional_consistency'] = directional_consistency
    metrics['mean_directional_consistency'] = np.mean(directional_consistency) if directional_consistency else 0.0
    
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
    
    # Calculate distribution similarity score (inverse of mean Wasserstein, normalized to [0, 1])
    # Lower Wasserstein distance means higher similarity
    # We'll use an exponential decay function to map Wasserstein to a similarity score
    distribution_similarity = np.exp(-metrics['mean_wasserstein'])
    metrics['distribution_similarity'] = distribution_similarity
    
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
    plt.title(f'Path Lengths (Size Factor: {size_factor}, Similarity: {metrics_dict["path_length_similarity"]:.3f})' 
              if size_factor else f'Path Lengths (Similarity: {metrics_dict["path_length_similarity"]:.3f})')
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
    plt.title(f'Velocity Profile (Size Factor: {size_factor}, Mean Similarity: {metrics_dict["mean_velocity_similarity"]:.3f})' 
              if size_factor else f'Velocity Profile (Mean Similarity: {metrics_dict["mean_velocity_similarity"]:.3f})')
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
    plt.title(f'Path Efficiency (Size Factor: {size_factor}, Similarity: {metrics_dict["efficiency_similarity"]:.3f})' 
              if size_factor else f'Path Efficiency (Similarity: {metrics_dict["efficiency_similarity"]:.3f})')
    plt.ylabel('Efficiency (Endpoint-to-Start Distance / Path Length)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'path_efficiency{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Wasserstein distances
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_dict['wasserstein_distances'], color='purple')
    plt.axhline(y=metrics_dict['mean_wasserstein'], color='red', linestyle='--', 
                label=f'Mean: {metrics_dict["mean_wasserstein"]:.4f}')
    plt.title(f'Wasserstein Distances (Size Factor: {size_factor}, Distribution Similarity: {metrics_dict["distribution_similarity"]:.3f})' 
              if size_factor else f'Wasserstein Distances (Distribution Similarity: {metrics_dict["distribution_similarity"]:.3f})')
    plt.xlabel('Step')
    plt.ylabel('Wasserstein Distance')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'wasserstein_distances{suffix}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot directional consistency
    if 'directional_consistency' in metrics_dict and metrics_dict['directional_consistency']:
        plt.figure(figsize=(12, 6))
        plt.plot(metrics_dict['directional_consistency'], color='green')
        plt.axhline(y=metrics_dict['mean_directional_consistency'], color='red', linestyle='--', 
                    label=f'Mean: {metrics_dict["mean_directional_consistency"]:.4f}')
        plt.title(f'Directional Consistency (Size Factor: {size_factor})' if size_factor else 'Directional Consistency')
        plt.xlabel('Step')
        plt.ylabel('Cosine Similarity')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'directional_consistency{suffix}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot position differences
    if 'position_differences' in metrics_dict and metrics_dict['position_differences']:
        plt.figure(figsize=(12, 6))
        plt.plot(metrics_dict['position_differences'], color='orange')
        plt.axhline(y=metrics_dict['mean_position_difference'], color='red', linestyle='--', 
                    label=f'Mean: {metrics_dict["mean_position_difference"]:.4f}')
        plt.title(f'Position Differences (Size Factor: {size_factor})' if size_factor else 'Position Differences')
        plt.xlabel('Step')
        plt.ylabel('Distance')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'position_differences{suffix}.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create a summary text file
    if output_dir:
        with open(os.path.join(output_dir, f'summary{suffix}.txt'), 'w') as f:
            f.write(f"Size Factor: {size_factor if size_factor else 'N/A'}\n")
            f.write(f"Endpoint Distance: {metrics_dict['endpoint_distance']:.4f}\n")
            f.write(f"Path Length Similarity: {metrics_dict['path_length_similarity']:.4f}\n")
            f.write(f"Path Length Ratio (Student/Teacher): {metrics_dict['path_length_ratio']:.4f}\n")
            f.write(f"Efficiency Similarity: {metrics_dict['efficiency_similarity']:.4f}\n")
            f.write(f"Mean Velocity Similarity: {metrics_dict['mean_velocity_similarity']:.4f}\n")
            f.write(f"Mean Directional Consistency: {metrics_dict['mean_directional_consistency']:.4f}\n")
            f.write(f"Mean Position Difference: {metrics_dict['mean_position_difference']:.4f}\n")
            f.write(f"Distribution Similarity: {metrics_dict['distribution_similarity']:.4f}\n")
            f.write(f"Mean Wasserstein Distance: {metrics_dict['mean_wasserstein']:.4f}\n")
            f.write(f"Teacher Path Length: {metrics_dict['teacher_path_length']:.4f}\n")
            f.write(f"Student Path Length: {metrics_dict['student_path_length']:.4f}\n")

def visualize_batch_metrics(metrics_batch, config, size_factor=None, output_dir=None):
    """
    Visualize metrics computed for a batch of trajectories
    
    Args:
        metrics_batch: Batch of metrics
        config: Configuration object
        size_factor: Size factor of the student model
        output_dir: Directory to save the visualizations
        
    Returns:
        Dictionary of paths to the saved plots
    """
    # Setup output directory
    if output_dir is None:
        output_dir = config.metrics_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up figure size and style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Extract size factor from suffix if available
    if size_factor:
        size_factor_str = f"_size_{size_factor}"
    else:
        size_factor_str = ""
    
    # Create summary dictionary
    summary = {}
    
    # Plot wasserstein distances histogram
    if 'wasserstein_distances' in metrics_batch and metrics_batch['wasserstein_distances']:
        plt.figure(figsize=(10, 6))
        plt.hist(metrics_batch['wasserstein_distances'], bins=20, alpha=0.7, color='purple')
        mean_wasserstein = sum(metrics_batch['wasserstein_distances']) / len(metrics_batch['wasserstein_distances'])
        plt.axvline(x=mean_wasserstein, color='red', linestyle='--', 
                    label=f'Mean: {mean_wasserstein:.4f}')
        plt.title(f'Wasserstein Distances Distribution (Size Factor: {size_factor_str})' if size_factor_str else 'Wasserstein Distances Distribution')
        plt.xlabel('Wasserstein Distance')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, f'wasserstein_distances_hist{size_factor_str}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        summary['mean_wasserstein'] = mean_wasserstein
    
    # Plot endpoint distances histogram
    if 'endpoint_distances' in metrics_batch and metrics_batch['endpoint_distances']:
        plt.figure(figsize=(10, 6))
        plt.hist(metrics_batch['endpoint_distances'], bins=20, alpha=0.7, color='green')
        mean_endpoint = sum(metrics_batch['endpoint_distances']) / len(metrics_batch['endpoint_distances'])
        plt.axvline(x=mean_endpoint, color='red', linestyle='--', 
                    label=f'Mean: {mean_endpoint:.4f}')
        plt.title(f'Endpoint Distances Distribution (Size Factor: {size_factor_str})' if size_factor_str else 'Endpoint Distances Distribution')
        plt.xlabel('Endpoint Distance')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, f'endpoint_distances_hist{size_factor_str}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        summary['mean_endpoint_distance'] = mean_endpoint
        # Also include the endpoint_distance for radar plot compatibility
        summary['endpoint_distance'] = mean_endpoint
    
    # Plot path lengths comparison
    if 'teacher_path_lengths' in metrics_batch and 'student_path_lengths' in metrics_batch:
        if metrics_batch['teacher_path_lengths'] and metrics_batch['student_path_lengths']:
            # Compute means
            mean_teacher_path = sum(metrics_batch['teacher_path_lengths']) / len(metrics_batch['teacher_path_lengths'])
            mean_student_path = sum(metrics_batch['student_path_lengths']) / len(metrics_batch['student_path_lengths'])
            
            # Plot means
            plt.figure(figsize=(10, 6))
            plt.bar(['Teacher', 'Student'], [mean_teacher_path, mean_student_path], color=['blue', 'orange'])
            plt.title(f'Average Path Lengths (Size Factor: {size_factor_str})' if size_factor_str else 'Average Path Lengths')
            plt.ylabel('Path Length')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(output_dir, f'avg_path_lengths{size_factor_str}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot distributions
            plt.figure(figsize=(12, 6))
            plt.hist(metrics_batch['teacher_path_lengths'], bins=20, alpha=0.5, label='Teacher', color='blue')
            plt.hist(metrics_batch['student_path_lengths'], bins=20, alpha=0.5, label='Student', color='orange')
            plt.axvline(x=mean_teacher_path, color='blue', linestyle='--', 
                        label=f'Teacher Mean: {mean_teacher_path:.4f}')
            plt.axvline(x=mean_student_path, color='orange', linestyle='--', 
                        label=f'Student Mean: {mean_student_path:.4f}')
            plt.title(f'Path Lengths Distribution (Size Factor: {size_factor_str})' if size_factor_str else 'Path Lengths Distribution')
            plt.xlabel('Path Length')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(output_dir, f'path_lengths_hist{size_factor_str}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Compute path length ratio
            path_length_ratio = mean_student_path / mean_teacher_path if mean_teacher_path > 0 else float('inf')
            summary['mean_teacher_path_length'] = mean_teacher_path
            summary['mean_student_path_length'] = mean_student_path
            summary['path_length_ratio'] = path_length_ratio
            
            # Add path_length_similarity if available in metrics_batch
            if 'path_length_similarity_avg' in metrics_batch:
                summary['path_length_similarity'] = metrics_batch['path_length_similarity_avg']
            elif 'path_length_similarity' in metrics_batch:
                summary['path_length_similarity'] = metrics_batch['path_length_similarity']
            else:
                # Calculate it if not available
                path_length_similarity = min(mean_teacher_path, mean_student_path) / max(mean_teacher_path, mean_student_path)
                summary['path_length_similarity'] = path_length_similarity
    
    # Plot efficiency comparison
    if 'teacher_efficiency' in metrics_batch and 'student_efficiency' in metrics_batch:
        if metrics_batch['teacher_efficiency'] and metrics_batch['student_efficiency']:
            # Compute means
            mean_teacher_efficiency = sum(metrics_batch['teacher_efficiency']) / len(metrics_batch['teacher_efficiency'])
            mean_student_efficiency = sum(metrics_batch['student_efficiency']) / len(metrics_batch['student_efficiency'])
            
            # Plot means
            plt.figure(figsize=(10, 6))
            plt.bar(['Teacher', 'Student'], [mean_teacher_efficiency, mean_student_efficiency], color=['blue', 'orange'])
            plt.title(f'Average Path Efficiency (Size Factor: {size_factor_str})' if size_factor_str else 'Average Path Efficiency')
            plt.ylabel('Efficiency (Endpoint Distance / Path Length)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(output_dir, f'avg_path_efficiency{size_factor_str}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot distributions
            plt.figure(figsize=(12, 6))
            plt.hist(metrics_batch['teacher_efficiency'], bins=20, alpha=0.5, label='Teacher', color='blue')
            plt.hist(metrics_batch['student_efficiency'], bins=20, alpha=0.5, label='Student', color='orange')
            plt.axvline(x=mean_teacher_efficiency, color='blue', linestyle='--', 
                        label=f'Teacher Mean: {mean_teacher_efficiency:.4f}')
            plt.axvline(x=mean_student_efficiency, color='orange', linestyle='--', 
                        label=f'Student Mean: {mean_student_efficiency:.4f}')
            plt.title(f'Path Efficiency Distribution (Size Factor: {size_factor_str})' if size_factor_str else 'Path Efficiency Distribution')
            plt.xlabel('Efficiency')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(os.path.join(output_dir, f'path_efficiency_hist{size_factor_str}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Compute efficiency ratio
            efficiency_ratio = mean_student_efficiency / mean_teacher_efficiency if mean_teacher_efficiency > 0 else float('inf')
            summary['mean_teacher_efficiency'] = mean_teacher_efficiency
            summary['mean_student_efficiency'] = mean_student_efficiency
            summary['efficiency_ratio'] = efficiency_ratio
            
            # Add efficiency_similarity if available in metrics_batch
            if 'efficiency_similarity_avg' in metrics_batch:
                summary['efficiency_similarity'] = metrics_batch['efficiency_similarity_avg']
            elif 'efficiency_similarity' in metrics_batch:
                summary['efficiency_similarity'] = metrics_batch['efficiency_similarity']
            else:
                # Calculate it if not available
                efficiency_similarity = min(mean_teacher_efficiency, mean_student_efficiency) / max(mean_teacher_efficiency, mean_student_efficiency)
                summary['efficiency_similarity'] = efficiency_similarity
    
    # Plot wasserstein distances per timestep if available
    if 'wasserstein_distances_per_timestep' in metrics_batch and metrics_batch['wasserstein_distances_per_timestep']:
        # Compute average wasserstein distance at each timestep
        num_timesteps = len(metrics_batch['wasserstein_distances_per_timestep'][0])
        avg_wasserstein_per_timestep = [0] * num_timesteps
        
        for distances in metrics_batch['wasserstein_distances_per_timestep']:
            for t in range(min(len(distances), num_timesteps)):
                avg_wasserstein_per_timestep[t] += distances[t]
        
        # Divide by number of trajectories
        num_trajectories = len(metrics_batch['wasserstein_distances_per_timestep'])
        avg_wasserstein_per_timestep = [d / num_trajectories for d in avg_wasserstein_per_timestep]
        
        # Plot average wasserstein distance per timestep
        plt.figure(figsize=(12, 6))
        plt.plot(avg_wasserstein_per_timestep, color='purple')
        plt.title(f'Average Wasserstein Distance per Timestep (Size Factor: {size_factor_str})' if size_factor_str else 'Average Wasserstein Distance per Timestep')
        plt.xlabel('Timestep')
        plt.ylabel('Wasserstein Distance')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, f'wasserstein_per_timestep{size_factor_str}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        summary['wasserstein_per_timestep'] = avg_wasserstein_per_timestep
    
    # Add the new metrics to the summary if available
    for metric_name in [
        'mean_velocity_similarity', 
        'mean_directional_consistency', 
        'mean_position_difference', 
        'distribution_similarity'
    ]:
        if f'{metric_name}_avg' in metrics_batch:
            # Use the average value if available
            summary[metric_name] = metrics_batch[f'{metric_name}_avg']
        elif metric_name in metrics_batch:
            # Otherwise use the direct value if available
            summary[metric_name] = metrics_batch[metric_name]
    
    # Create a summary text file
    with open(os.path.join(output_dir, f'summary{size_factor_str}.txt'), 'w') as f:
        f.write(f"Size Factor: {size_factor_str if size_factor_str else 'N/A'}\n\n")
        
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
        
        # Add new metrics to the summary file
        for metric_name, display_name in [
            ('path_length_similarity', 'Path Length Similarity'),
            ('efficiency_similarity', 'Efficiency Similarity'),
            ('mean_velocity_similarity', 'Mean Velocity Similarity'),
            ('mean_directional_consistency', 'Mean Directional Consistency'),
            ('mean_position_difference', 'Mean Position Difference'),
            ('distribution_similarity', 'Distribution Similarity')
        ]:
            if metric_name in summary:
                f.write(f"{display_name}: {summary[metric_name]:.4f}\n")
    
    # Print summary for debugging
    print(f"Metrics summary for size factor {size_factor_str}:")
    for key, value in summary.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            print(f"  {key}: {value:.4f}")
    
    return summary 