"""
Dimensionality reduction analysis for diffusion model trajectories
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from tqdm import tqdm

def dimensionality_reduction_analysis(teacher_trajectories, student_trajectories, config, size_factor=None):
    """
    Perform dimensionality reduction analysis on teacher and student trajectories
    
    Args:
        teacher_trajectories: List of teacher trajectories
        student_trajectories: List of student trajectories  
        config: Configuration object
        size_factor: Size factor of the student model
    """
    # Create output directory
    output_dir = config.dimensionality_dir
    if size_factor is not None:
        output_dir = os.path.join(output_dir, f"size_{size_factor}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Performing dimensionality reduction analysis for size factor {size_factor}...")
    
    # Process all trajectories
    for traj_idx, (teacher_traj, student_traj) in enumerate(zip(teacher_trajectories, student_trajectories)):
        # Skip if too many trajectories to avoid excessive computation
        if traj_idx >= 3:  # Process max 3 trajectories
            break
            
        # Create subdirectory for this trajectory
        traj_dir = os.path.join(output_dir, f"trajectory_{traj_idx}")
        os.makedirs(traj_dir, exist_ok=True)
        
        # Extract images from trajectories
        teacher_images = [item[0] for item in teacher_traj]
        student_images = [item[0] for item in student_traj]
        
        # Flatten images for dimensionality reduction
        teacher_flat = [img.flatten().cpu().numpy() for img in teacher_images]
        student_flat = [img.flatten().cpu().numpy() for img in student_images]
        
        # Combine teacher and student data for joint embedding
        combined_data = np.vstack([teacher_flat, student_flat])
        
        # Perform PCA
        print(f"  Performing PCA for trajectory {traj_idx}...")
        pca = PCA(n_components=2)
        try:
            pca_result = pca.fit_transform(combined_data)
            
            # Split results back into teacher and student
            teacher_pca = pca_result[:len(teacher_flat)]
            student_pca = pca_result[len(teacher_flat):]
            
            # Plot PCA results
            plt.figure(figsize=(10, 8))
            
            # Create a colormap for the trajectories
            teacher_colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(teacher_pca)))
            student_colors = plt.cm.Oranges(np.linspace(0.3, 1.0, len(student_pca)))
            
            # Plot teacher points with arrows to show direction
            for i in range(len(teacher_pca) - 1):
                plt.scatter(teacher_pca[i, 0], teacher_pca[i, 1], color=teacher_colors[i], 
                           marker='o', s=50, alpha=0.7)
                plt.arrow(teacher_pca[i, 0], teacher_pca[i, 1], 
                         teacher_pca[i+1, 0] - teacher_pca[i, 0], 
                         teacher_pca[i+1, 1] - teacher_pca[i, 1], 
                         color=teacher_colors[i], width=0.01, head_width=0.1, alpha=0.5)
            
            # Plot final teacher point
            plt.scatter(teacher_pca[-1, 0], teacher_pca[-1, 1], color=teacher_colors[-1], 
                       marker='*', s=200, alpha=0.7, label='Teacher End')
            
            # Plot student points with arrows to show direction
            for i in range(len(student_pca) - 1):
                plt.scatter(student_pca[i, 0], student_pca[i, 1], color=student_colors[i], 
                           marker='o', s=50, alpha=0.7)
                plt.arrow(student_pca[i, 0], student_pca[i, 1], 
                         student_pca[i+1, 0] - student_pca[i, 0], 
                         student_pca[i+1, 1] - student_pca[i, 1], 
                         color=student_colors[i], width=0.01, head_width=0.1, alpha=0.5)
            
            # Plot final student point
            plt.scatter(student_pca[-1, 0], student_pca[-1, 1], color=student_colors[-1], 
                       marker='*', s=200, alpha=0.7, label='Student End')
            
            # Add starting points with different markers
            plt.scatter(teacher_pca[0, 0], teacher_pca[0, 1], color='blue', 
                       marker='D', s=100, alpha=1.0, label='Teacher Start')
            plt.scatter(student_pca[0, 0], student_pca[0, 1], color='orange', 
                       marker='D', s=100, alpha=1.0, label='Student Start')
            
            plt.title(f'PCA - Trajectory {traj_idx} (Size Factor: {size_factor})')
            plt.xlabel(f'Principal Component 1 (Variance: {pca.explained_variance_ratio_[0]:.2%})')
            plt.ylabel(f'Principal Component 2 (Variance: {pca.explained_variance_ratio_[1]:.2%})')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            plt.savefig(os.path.join(traj_dir, 'pca_trajectory.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"  Error performing PCA: {e}")
        
        # Perform t-SNE if data not too large
        if len(combined_data) <= 500:  # t-SNE can be slow for large datasets
            print(f"  Performing t-SNE for trajectory {traj_idx}...")
            try:
                tsne = TSNE(n_components=2, perplexity=min(30, len(combined_data)//5), random_state=42)
                tsne_result = tsne.fit_transform(combined_data)
                
                # Split results back into teacher and student
                teacher_tsne = tsne_result[:len(teacher_flat)]
                student_tsne = tsne_result[len(teacher_flat):]
                
                # Plot t-SNE results (similar to PCA plot)
                plt.figure(figsize=(10, 8))
                
                # Create a colormap for the trajectories
                teacher_colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(teacher_tsne)))
                student_colors = plt.cm.Oranges(np.linspace(0.3, 1.0, len(student_tsne)))
                
                # Plot teacher points with arrows
                for i in range(len(teacher_tsne) - 1):
                    plt.scatter(teacher_tsne[i, 0], teacher_tsne[i, 1], color=teacher_colors[i], 
                               marker='o', s=50, alpha=0.7)
                    plt.arrow(teacher_tsne[i, 0], teacher_tsne[i, 1], 
                             teacher_tsne[i+1, 0] - teacher_tsne[i, 0], 
                             teacher_tsne[i+1, 1] - teacher_tsne[i, 1], 
                             color=teacher_colors[i], width=0.01, head_width=0.1, alpha=0.5)
                
                # Plot final teacher point
                plt.scatter(teacher_tsne[-1, 0], teacher_tsne[-1, 1], color=teacher_colors[-1], 
                           marker='*', s=200, alpha=0.7, label='Teacher End')
                
                # Plot student points with arrows
                for i in range(len(student_tsne) - 1):
                    plt.scatter(student_tsne[i, 0], student_tsne[i, 1], color=student_colors[i], 
                               marker='o', s=50, alpha=0.7)
                    plt.arrow(student_tsne[i, 0], student_tsne[i, 1], 
                             student_tsne[i+1, 0] - student_tsne[i, 0], 
                             student_tsne[i+1, 1] - student_tsne[i, 1], 
                             color=student_colors[i], width=0.01, head_width=0.1, alpha=0.5)
                
                # Plot final student point
                plt.scatter(student_tsne[-1, 0], student_tsne[-1, 1], color=student_colors[-1], 
                           marker='*', s=200, alpha=0.7, label='Student End')
                
                # Add starting points with different markers
                plt.scatter(teacher_tsne[0, 0], teacher_tsne[0, 1], color='blue', 
                           marker='D', s=100, alpha=1.0, label='Teacher Start')
                plt.scatter(student_tsne[0, 0], student_tsne[0, 1], color='orange', 
                           marker='D', s=100, alpha=1.0, label='Student Start')
                
                plt.title(f't-SNE - Trajectory {traj_idx} (Size Factor: {size_factor})')
                plt.xlabel('t-SNE Dimension 1')
                plt.ylabel('t-SNE Dimension 2')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                
                plt.savefig(os.path.join(traj_dir, 'tsne_trajectory.png'), dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"  Error performing t-SNE: {e}")
        else:
            print(f"  Skipping t-SNE for trajectory {traj_idx} (too many points)")
        
        # Create UMAP visualization 
        print(f"  Performing UMAP for trajectory {traj_idx}...")
        try:
            # Configure UMAP embedding
            reducer = umap.UMAP(n_components=2, random_state=42, 
                               n_neighbors=min(15, len(combined_data)//3),
                               min_dist=0.1)
            umap_result = reducer.fit_transform(combined_data)
            
            # Split results back into teacher and student
            teacher_umap = umap_result[:len(teacher_flat)]
            student_umap = umap_result[len(teacher_flat):]
            
            # Plot UMAP results (similar to previous plots)
            plt.figure(figsize=(10, 8))
            
            # Create a colormap for the trajectories
            teacher_colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(teacher_umap)))
            student_colors = plt.cm.Oranges(np.linspace(0.3, 1.0, len(student_umap)))
            
            # Plot teacher points with arrows
            for i in range(len(teacher_umap) - 1):
                plt.scatter(teacher_umap[i, 0], teacher_umap[i, 1], color=teacher_colors[i], 
                           marker='o', s=50, alpha=0.7)
                plt.arrow(teacher_umap[i, 0], teacher_umap[i, 1], 
                         teacher_umap[i+1, 0] - teacher_umap[i, 0], 
                         teacher_umap[i+1, 1] - teacher_umap[i, 1], 
                         color=teacher_colors[i], width=0.01, head_width=0.1, alpha=0.5)
            
            # Plot final teacher point
            plt.scatter(teacher_umap[-1, 0], teacher_umap[-1, 1], color=teacher_colors[-1], 
                       marker='*', s=200, alpha=0.7, label='Teacher End')
            
            # Plot student points with arrows
            for i in range(len(student_umap) - 1):
                plt.scatter(student_umap[i, 0], student_umap[i, 1], color=student_colors[i], 
                           marker='o', s=50, alpha=0.7)
                plt.arrow(student_umap[i, 0], student_umap[i, 1], 
                         student_umap[i+1, 0] - student_umap[i, 0], 
                         student_umap[i+1, 1] - student_umap[i, 1], 
                         color=student_colors[i], width=0.01, head_width=0.1, alpha=0.5)
            
            # Plot final student point
            plt.scatter(student_umap[-1, 0], student_umap[-1, 1], color=student_colors[-1], 
                       marker='*', s=200, alpha=0.7, label='Student End')
            
            # Add starting points with different markers
            plt.scatter(teacher_umap[0, 0], teacher_umap[0, 1], color='blue', 
                       marker='D', s=100, alpha=1.0, label='Teacher Start')
            plt.scatter(student_umap[0, 0], student_umap[0, 1], color='orange', 
                       marker='D', s=100, alpha=1.0, label='Student Start')
            
            plt.title(f'UMAP - Trajectory {traj_idx} (Size Factor: {size_factor})')
            plt.xlabel('UMAP Dimension 1')
            plt.ylabel('UMAP Dimension 2')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            
            plt.savefig(os.path.join(traj_dir, 'umap_trajectory.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"  Error performing UMAP: {e}")
    
    print(f"Dimensionality reduction analysis completed for size factor {size_factor}")
    return os.path.abspath(output_dir) 