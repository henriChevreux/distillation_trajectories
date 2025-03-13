"""
Time-dependent metrics for diffusion model analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def analyze_time_dependent_distances(teacher_trajectories, student_trajectories, config, size_factor=None, save_dir=None):
    """
    Analyze the distances between consecutive timesteps in trajectories
    
    Args:
        teacher_trajectories: List of teacher trajectories
        student_trajectories: List of student trajectories
        config: Configuration object
        size_factor: Size factor of the student model (optional)
        save_dir: Directory to save results (optional)
        
    Returns:
        Dictionary of distance metrics
    """
    print("Analyzing time-dependent distances...")
    
    # Create placeholder results
    results = {
        "teacher_distances": [],
        "student_distances": [],
        "teacher_avg_distance": 0,
        "student_avg_distance": 0,
        "teacher_std_distance": 0,
        "student_std_distance": 0,
        "size_factor": size_factor
    }
    
    # Check if trajectories are empty
    if not teacher_trajectories or not student_trajectories:
        print("  Warning: Empty trajectories provided. Returning empty results.")
        return results
    
    # Process teacher trajectories
    print("  Processing teacher trajectories...")
    teacher_all_distances = []
    
    for trajectory in teacher_trajectories:
        # Extract images from trajectory
        if isinstance(trajectory[0], tuple):
            images = [item[0] for item in trajectory]
        else:
            images = trajectory
            
        # Calculate Euclidean distances between consecutive timesteps
        traj_distances = []
        for i in range(1, len(images)):
            # Calculate distance between consecutive images
            dist = torch.norm(images[i] - images[i-1]).item()
            traj_distances.append(dist)
        
        if traj_distances:  # Only add if not empty
            teacher_all_distances.append(traj_distances)
    
    # Process student trajectories
    print("  Processing student trajectories...")
    student_all_distances = []
    
    for trajectory in student_trajectories:
        # Extract images from trajectory
        if isinstance(trajectory[0], tuple):
            images = [item[0] for item in trajectory]
        else:
            images = trajectory
            
        # Calculate Euclidean distances between consecutive timesteps
        traj_distances = []
        for i in range(1, len(images)):
            # Calculate distance between consecutive images
            dist = torch.norm(images[i] - images[i-1]).item()
            traj_distances.append(dist)
        
        if traj_distances:  # Only add if not empty
            student_all_distances.append(traj_distances)
    
    # Compute average distances per timestep
    teacher_avg_per_timestep = []
    student_avg_per_timestep = []
    
    # Check if we have any valid trajectories
    if teacher_all_distances and student_all_distances:
        # Find the minimum length of all trajectories
        min_teacher_length = min(len(distances) for distances in teacher_all_distances)
        min_student_length = min(len(distances) for distances in student_all_distances)
        
        # Compute average distance at each timestep
        for t in range(min_teacher_length):
            teacher_avg_per_timestep.append(
                sum(distances[t] for distances in teacher_all_distances) / len(teacher_all_distances)
            )
        
        for t in range(min_student_length):
            student_avg_per_timestep.append(
                sum(distances[t] for distances in student_all_distances) / len(student_all_distances)
            )
    
    # Store results
    results["teacher_distances"] = teacher_all_distances
    results["student_distances"] = student_all_distances
    results["teacher_avg_per_timestep"] = teacher_avg_per_timestep
    results["student_avg_per_timestep"] = student_avg_per_timestep
    results["teacher_avg_distance"] = sum(teacher_avg_per_timestep) / len(teacher_avg_per_timestep) if teacher_avg_per_timestep else 0
    results["student_avg_distance"] = sum(student_avg_per_timestep) / len(student_avg_per_timestep) if student_avg_per_timestep else 0
    
    # Calculate standard deviations
    if teacher_avg_per_timestep:
        teacher_squared_diffs = [(d - results["teacher_avg_distance"])**2 for d in teacher_avg_per_timestep]
        results["teacher_std_distance"] = (sum(teacher_squared_diffs) / len(teacher_squared_diffs))**0.5
    
    if student_avg_per_timestep:
        student_squared_diffs = [(d - results["student_avg_distance"])**2 for d in student_avg_per_timestep]
        results["student_std_distance"] = (sum(student_squared_diffs) / len(student_squared_diffs))**0.5
    
    # Save results if directory is provided and we have data to plot
    if save_dir and teacher_avg_per_timestep and student_avg_per_timestep:
        os.makedirs(save_dir, exist_ok=True)
        
        # Create a plot of average distances per timestep
        plt.figure(figsize=(12, 6))
        
        # Plot teacher and student distances
        plt.plot(teacher_avg_per_timestep, label='Teacher', color='blue')
        plt.plot(student_avg_per_timestep, label='Student', color='orange')
        
        # Add title and labels
        title = "Average Distance Between Consecutive Timesteps"
        if size_factor is not None:
            title += f" (Size Factor: {size_factor})"
        
        plt.title(title)
        plt.xlabel("Timestep")
        plt.ylabel("Average Distance")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the plot
        filename = "time_dependent_distances.png"
        if size_factor is not None:
            filename = f"time_dependent_distances_size_{size_factor}.png"
        
        plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    return results 