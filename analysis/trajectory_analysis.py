import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from sklearn.manifold import TSNE
import umap

def compute_trajectory_embeddings(trajectories, method='tsne'):
    """
    Compute low-dimensional embeddings of trajectories using t-SNE or UMAP.
    
    Args:
        trajectories: List of trajectories (each trajectory is a list of tensors)
        method: 'tsne' or 'umap'
    
    Returns:
        Array of shape (n_trajectories * n_timesteps, 2) containing embeddings
    """
    # Flatten trajectories into a matrix
    flattened = []
    for traj in trajectories:
        for x, _ in traj:
            flat = x.view(x.size(0), -1).cpu().numpy()
            flattened.append(flat)
    
    X = np.vstack(flattened)
    
    # Compute embeddings
    if method == 'tsne':
        embedder = TSNE(n_components=2, random_state=42)
    else:  # umap
        embedder = umap.UMAP(n_components=2, random_state=42)
    
    embeddings = embedder.fit_transform(X)
    return embeddings

def plot_trajectory_embeddings(teacher_trajectories, student_trajectories, config, save_dir=None):
    """
    Plot trajectory embeddings using both t-SNE and UMAP.
    
    Args:
        teacher_trajectories: List of teacher model trajectories
        student_trajectories: List of student model trajectories
        config: Configuration object
        save_dir: Directory to save visualizations
    """
    # Combine trajectories
    all_trajectories = teacher_trajectories + student_trajectories
    n_teacher = len(teacher_trajectories)
    n_student = len(student_trajectories)
    
    # Compute embeddings using both methods
    for method in ['tsne', 'umap']:
        embeddings = compute_trajectory_embeddings(all_trajectories, method)
        
        # Split embeddings back into teacher and student
        teacher_points = embeddings[:n_teacher * len(teacher_trajectories[0])]
        student_points = embeddings[n_teacher * len(teacher_trajectories[0]):]
        
        # Plot
        plt.figure(figsize=(10, 10))
        plt.scatter(teacher_points[:, 0], teacher_points[:, 1], 
                   alpha=0.5, label='Teacher', c='blue')
        plt.scatter(student_points[:, 0], student_points[:, 1], 
                   alpha=0.5, label='Student', c='red')
        
        plt.title(f'Trajectory Embeddings ({method.upper()})')
        plt.legend()
        
        if save_dir:
            save_path = os.path.join(save_dir, f'trajectory_embeddings_{method}.png')
            Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

def analyze_trajectory_statistics(teacher_trajectories, student_trajectories, config):
    """
    Compute and analyze statistical properties of trajectories.
    
    Args:
        teacher_trajectories: List of teacher model trajectories
        student_trajectories: List of student model trajectories
        config: Configuration object
        
    Returns:
        Dictionary containing trajectory statistics
    """
    stats = {
        'teacher': {
            'mean_values': [],
            'std_values': [],
            'entropy': []
        },
        'student': {
            'mean_values': [],
            'std_values': [],
            'entropy': []
        }
    }
    
    # Analyze teacher trajectories
    for traj in teacher_trajectories:
        values = torch.stack([x for x, _ in traj])
        stats['teacher']['mean_values'].append(values.mean().item())
        stats['teacher']['std_values'].append(values.std().item())
        
        # Compute approximate entropy
        flat_values = values.view(-1).cpu().numpy()
        hist, _ = np.histogram(flat_values, bins=50, density=True)
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        stats['teacher']['entropy'].append(entropy)
    
    # Analyze student trajectories
    for traj in student_trajectories:
        values = torch.stack([x for x, _ in traj])
        stats['student']['mean_values'].append(values.mean().item())
        stats['student']['std_values'].append(values.std().item())
        
        # Compute approximate entropy
        flat_values = values.view(-1).cpu().numpy()
        hist, _ = np.histogram(flat_values, bins=50, density=True)
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        stats['student']['entropy'].append(entropy)
    
    return stats

def plot_trajectory_statistics(stats, config, save_dir=None):
    """Plot statistical analysis of trajectories"""
    metrics = ['mean_values', 'std_values', 'entropy']
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    
    for idx, metric in enumerate(metrics):
        teacher_values = stats['teacher'][metric]
        student_values = stats['student'][metric]
        
        ax = axes[idx]
        ax.boxplot([teacher_values, student_values], labels=['Teacher', 'Student'])
        ax.set_title(f'Distribution of {metric.replace("_", " ").title()}')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, 'trajectory_statistics.png')
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def run_trajectory_analysis(teacher_model, student_models, test_samples, config):
    """
    Run comprehensive trajectory analysis on teacher and student models.
    
    Args:
        teacher_model: The teacher diffusion model
        student_models: Dictionary of student models
        test_samples: Test dataset samples
        config: Configuration object
    """
    print("\nRunning trajectory analysis...")
    
    for (size_factor, _), student_model in student_models.items():
        print(f"\nAnalyzing trajectories for student model (size factor: {size_factor})...")
        
        # Generate trajectories
        teacher_trajectories = []
        student_trajectories = []
        
        sample_count = min(5, len(test_samples))
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
        
        # Create save directory for this model size
        save_dir = os.path.join(config.trajectory_viz_dir, f'size_{size_factor}')
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate visualizations
        plot_trajectory_embeddings(teacher_trajectories, student_trajectories, config, save_dir)
        
        # Compute and plot statistics
        stats = analyze_trajectory_statistics(teacher_trajectories, student_trajectories, config)
        plot_trajectory_statistics(stats, config, save_dir)
        
        # Save raw statistics
        stats_df = pd.DataFrame({
            'model': ['teacher'] * len(stats['teacher']['mean_values']) + 
                    ['student'] * len(stats['student']['mean_values']),
            'mean': stats['teacher']['mean_values'] + stats['student']['mean_values'],
            'std': stats['teacher']['std_values'] + stats['student']['std_values'],
            'entropy': stats['teacher']['entropy'] + stats['student']['entropy']
        })
        stats_df.to_csv(os.path.join(save_dir, 'trajectory_statistics.csv'), index=False)
    
    print("Trajectory analysis completed successfully!")