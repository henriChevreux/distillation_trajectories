import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import seaborn as sns
from tqdm import tqdm
import pandas as pd

def analyze_time_dependent_distances(teacher_trajectories, student_trajectories, config, size_factor):
    """
    Analyze distances between teacher and student trajectories over time.
    
    Args:
        teacher_trajectories: List of teacher model trajectories
        student_trajectories: List of student model trajectories
        config: Configuration object
        size_factor: Size factor of the student model
        
    Returns:
        Dictionary containing various distance metrics over time
    """
    distances = {
        'mse': [],
        'cosine': [],
        'timesteps': [],
        'size_factor': size_factor
    }
    
    # Analyze each trajectory pair
    for teacher_traj, student_traj in zip(teacher_trajectories, student_trajectories):
        for (teacher_x, t_teacher), (student_x, t_student) in zip(teacher_traj, student_traj):
            # Calculate MSE
            mse = torch.mean((teacher_x - student_x) ** 2).item()
            
            # Calculate cosine similarity
            teacher_flat = teacher_x.view(teacher_x.size(0), -1)
            student_flat = student_x.view(student_x.size(0), -1)
            cosine = F.cosine_similarity(teacher_flat, student_flat).item()
            
            distances['mse'].append(mse)
            distances['cosine'].append(cosine)
            distances['timesteps'].append(t_teacher)
    
    return distances

def plot_time_dependent_grid(all_distances, config):
    """Plot grid of time-dependent metrics for each model size"""
    metrics = ['mse', 'cosine']
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for size_factor, distances in all_distances.items():
            df = pd.DataFrame({
                'timestep': distances['timesteps'],
                metric: distances[metric]
            })
            sns.lineplot(data=df, x='timestep', y=metric, label=f'Size {size_factor}', ax=ax)
        
        ax.set_title(f'{metric.upper()} over Time')
        ax.set_xlabel('Timestep')
        ax.set_ylabel(metric.upper())
    
    plt.tight_layout()
    save_path = os.path.join(config.time_analysis_dir, 'time_metrics_grid.png')
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_time_dependent_combined(all_distances, config):
    """Plot combined view of all time-dependent metrics"""
    df_list = []
    for size_factor, distances in all_distances.items():
        df = pd.DataFrame({
            'timestep': distances['timesteps'],
            'mse': distances['mse'],
            'cosine': distances['cosine'],
            'size_factor': size_factor
        })
        df_list.append(df)
    
    combined_df = pd.concat(df_list)
    
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=combined_df,
        x='timestep',
        y='mse',
        hue='size_factor',
        style='size_factor'
    )
    
    plt.title('Combined Time-dependent MSE Analysis')
    plt.xlabel('Timestep')
    plt.ylabel('MSE')
    
    save_path = os.path.join(config.time_analysis_dir, 'time_metrics_combined.png')
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_convergence_rates(all_distances, config):
    """Analyze how quickly different sized models converge to teacher behavior"""
    convergence_metrics = {}
    
    for size_factor, distances in all_distances.items():
        df = pd.DataFrame({
            'timestep': distances['timesteps'],
            'mse': distances['mse']
        })
        
        # Calculate convergence metrics
        convergence_metrics[size_factor] = {
            'mean_mse': df['mse'].mean(),
            'final_mse': df['mse'].iloc[-10:].mean(),  # Average of last 10 timesteps
            'convergence_rate': (df['mse'].iloc[0] - df['mse'].iloc[-1]) / len(df)
        }
    
    # Plot convergence metrics
    metrics_df = pd.DataFrame(convergence_metrics).T
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, metric in enumerate(['mean_mse', 'final_mse', 'convergence_rate']):
        sns.barplot(data=metrics_df, y=metric, ax=axes[idx])
        axes[idx].set_title(f'{metric}')
    
    plt.tight_layout()
    save_path = os.path.join(config.convergence_dir, 'convergence_metrics.png')
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save raw data
    metrics_df.to_csv(os.path.join(config.convergence_dir, 'convergence_metrics.csv'))

def run_time_analysis(teacher_model, student_models, test_samples, config):
    """
    Run comprehensive time-dependent analysis on teacher and student models.
    
    Args:
        teacher_model: The teacher diffusion model
        student_models: Dictionary of student models
        test_samples: Test dataset samples
        config: Configuration object
    """
    print("\nRunning time-dependent analysis...")
    
    all_distances = {}
    for (size_factor, _), student_model in student_models.items():
        # Create teacher and student trajectories
        teacher_trajectories = []
        student_trajectories = []
        
        # Sample a few trajectories
        sample_count = config.time_analysis_sample_count
        for i in range(sample_count):
            with torch.no_grad():
                # Teacher trajectory
                x = test_samples[i:i+1]
                teacher_traj = []
                for t in reversed(range(config.timesteps)):
                    t_batch = torch.full((1,), t, device=x.device, dtype=torch.long)
                    teacher_traj.append((x.clone(), t))
                    pred = teacher_model(x, t_batch)
                    x = pred
                teacher_trajectories.append(teacher_traj)
                
                # Student trajectory
                x = test_samples[i:i+1]
                student_traj = []
                for t in reversed(range(config.student_steps)):
                    t_batch = torch.full((1,), t, device=x.device, dtype=torch.long)
                    student_traj.append((x.clone(), t))
                    pred = student_model(x, t_batch)
                    x = pred
                student_trajectories.append(student_traj)
        
        distances = analyze_time_dependent_distances(
            teacher_trajectories,
            student_trajectories,
            config,
            size_factor
        )
        all_distances[size_factor] = distances
    
    # Generate all plots
    plot_time_dependent_grid(all_distances, config)
    plot_time_dependent_combined(all_distances, config)
    analyze_convergence_rates(all_distances, config)
    
    print("Time-dependent analysis completed successfully!") 