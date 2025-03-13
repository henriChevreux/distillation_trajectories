#!/usr/bin/env python3
"""
Script to analyze the impact of editing on distilled and teacher diffusion models.
"""

import os
import argparse
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from PIL import Image
import lpips
from sklearn.decomposition import PCA

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config
from models import SimpleUNet, StudentUNet
from utils.diffusion import get_diffusion_params
from utils.trajectory_manager import TrajectoryManager

# Import editing modules
from editing.prompt_editing import apply_prompt_editing, visualize_prompt_editing
from editing.masked_inpainting import apply_masked_inpainting, visualize_inpainting, create_random_mask, generate_image
from editing.latent_manipulation import apply_latent_manipulation, visualize_latent_manipulation

# Import evaluation metrics
from evaluation.metrics import compute_lpips, compute_fid, compute_trajectory_divergence, visualize_metrics

def generate_image_with_trajectory(model, diffusion_params, config, device):
    """
    Generate an image and record the trajectory
    
    Args:
        model: Diffusion model
        diffusion_params: Diffusion parameters
        config: Configuration object
        device: Device to run on
        
    Returns:
        Tuple of (generated_image, trajectory)
    """
    # Initialize from random noise
    x = torch.randn(1, config.channels, config.image_size, config.image_size).to(device)
    
    # Record trajectory
    trajectory = []
    
    # Denoise step by step
    with torch.no_grad():
        for t in range(diffusion_params["timesteps"] - 1, -1, -1):
            t_tensor = torch.tensor([t], device=device)
            
            # Record current state
            trajectory.append((x.clone(), t))
            
            # Predict noise
            noise_pred = model(x, t_tensor)
            
            # Ensure noise_pred has the same shape as x
            if noise_pred.shape != x.shape:
                print(f"Resizing noise prediction from {noise_pred.shape} to {x.shape}")
                noise_pred = torch.nn.functional.interpolate(
                    noise_pred, 
                    size=(x.shape[2], x.shape[3]),
                    mode='bilinear', 
                    align_corners=True
                )
            
            # Update x
            if t > 0:
                # Sample noise for the next step
                noise = torch.randn_like(x)
                
                # Apply diffusion update
                alpha_t = diffusion_params["alphas"][t]
                alpha_t_prev = diffusion_params["alphas"][t-1] if t > 0 else torch.tensor(1.0)
                
                # Compute coefficients
                c1 = torch.sqrt(alpha_t_prev) / torch.sqrt(alpha_t)
                c2 = torch.sqrt(1 - alpha_t_prev) - torch.sqrt(alpha_t_prev / alpha_t) * torch.sqrt(1 - alpha_t)
                
                # Update x
                x = c1 * x - c2 * noise_pred
                
                # Add noise for the next step
                sigma_t = torch.sqrt(1 - alpha_t_prev) * torch.sqrt(1 - alpha_t / alpha_t_prev)
                x = x + sigma_t * noise
    
    # Normalize to [0, 1] range for visualization
    image = (x + 1) / 2
    image = torch.clamp(image, 0, 1)
    
    return image, trajectory

def visualize_trajectory_comparison(original_teacher_trajectory, edited_teacher_trajectory, 
                                   original_student_trajectory, edited_student_trajectory,
                                   edit_type, output_dir, size_factor=None, direction_name=None, strength=None):
    """
    Create visualization comparing trajectories with and without editing for both teacher and student models
    
    Args:
        original_teacher_trajectory: Trajectory of teacher model without editing
        edited_teacher_trajectory: Trajectory of teacher model with editing
        original_student_trajectory: Trajectory of student model without editing
        edited_student_trajectory: Trajectory of student model with editing
        edit_type: Type of editing being performed (e.g., 'latent', 'prompt', 'inpainting')
        output_dir: Directory to save visualizations
        size_factor: Size factor of the student model for labeling
        direction_name: Name of the direction used for latent manipulation (if applicable)
        strength: Strength of the manipulation (if applicable)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract images from trajectories
    teacher_orig_images = []
    teacher_orig_timesteps = []
    teacher_edit_images = []
    teacher_edit_timesteps = []
    
    student_orig_images = []
    student_orig_timesteps = []
    student_edit_images = []
    student_edit_timesteps = []
    
    # Process teacher trajectories
    for img, timestep in original_teacher_trajectory:
        if isinstance(img, torch.Tensor):
            img_np = img.cpu().detach().numpy()
            img_flat = img_np.reshape(1, -1)
            teacher_orig_images.append(img_flat)
            teacher_orig_timesteps.append(timestep)
    
    for img, timestep in edited_teacher_trajectory:
        if isinstance(img, torch.Tensor):
            img_np = img.cpu().detach().numpy()
            img_flat = img_np.reshape(1, -1)
            teacher_edit_images.append(img_flat)
            teacher_edit_timesteps.append(timestep)
    
    # Process student trajectories
    for img, timestep in original_student_trajectory:
        if isinstance(img, torch.Tensor):
            img_np = img.cpu().detach().numpy()
            img_flat = img_np.reshape(1, -1)
            student_orig_images.append(img_flat)
            student_orig_timesteps.append(timestep)
    
    for img, timestep in edited_student_trajectory:
        if isinstance(img, torch.Tensor):
            img_np = img.cpu().detach().numpy()
            img_flat = img_np.reshape(1, -1)
            student_edit_images.append(img_flat)
            student_edit_timesteps.append(timestep)
    
    # Stack all flattened images
    if (teacher_orig_images and teacher_edit_images and 
        student_orig_images and student_edit_images):
        
        teacher_orig_features = np.vstack(teacher_orig_images)
        teacher_edit_features = np.vstack(teacher_edit_images)
        student_orig_features = np.vstack(student_orig_images)
        student_edit_features = np.vstack(student_edit_images)
        
        # Combine features for PCA fitting
        combined_features = np.vstack([
            teacher_orig_features, 
            teacher_edit_features,
            student_orig_features,
            student_edit_features
        ])
        
        # Fit PCA
        pca = PCA(n_components=2)
        combined_pca = pca.fit_transform(combined_features)
        
        # Split back into separate trajectories
        teacher_orig_pca = combined_pca[:len(teacher_orig_features)]
        teacher_edit_pca = combined_pca[len(teacher_orig_features):len(teacher_orig_features)+len(teacher_edit_features)]
        student_orig_pca = combined_pca[len(teacher_orig_features)+len(teacher_edit_features):len(teacher_orig_features)+len(teacher_edit_features)+len(student_orig_features)]
        student_edit_pca = combined_pca[len(teacher_orig_features)+len(teacher_edit_features)+len(student_orig_features):]
        
        # Use the first point of the teacher original trajectory as the common starting point
        common_start_point = teacher_orig_pca[0].copy()
        
        # Instead of just aligning the starting points, we'll set them all to exactly the same value
        # Save the original first points for each trajectory
        teacher_orig_first = teacher_orig_pca[0].copy()
        teacher_edit_first = teacher_edit_pca[0].copy()
        student_orig_first = student_orig_pca[0].copy()
        student_edit_first = student_edit_pca[0].copy()
        
        # Set the first point of each trajectory to the common starting point
        teacher_orig_pca[0] = common_start_point
        teacher_edit_pca[0] = common_start_point
        student_orig_pca[0] = common_start_point
        student_edit_pca[0] = common_start_point
        
        # Calculate the translation vectors for the rest of the points in each trajectory
        teacher_orig_translation = common_start_point - teacher_orig_first
        teacher_edit_translation = common_start_point - teacher_edit_first
        student_orig_translation = common_start_point - student_orig_first
        student_edit_translation = common_start_point - student_edit_first
        
        # Apply the translation to the rest of the points in each trajectory
        for i in range(1, len(teacher_orig_pca)):
            teacher_orig_pca[i] += teacher_orig_translation
        
        for i in range(1, len(teacher_edit_pca)):
            teacher_edit_pca[i] += teacher_edit_translation
        
        for i in range(1, len(student_orig_pca)):
            student_orig_pca[i] += student_orig_translation
        
        for i in range(1, len(student_edit_pca)):
            student_edit_pca[i] += student_edit_translation
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Create colormaps
        teacher_orig_cmap = plt.cm.viridis
        teacher_edit_cmap = plt.cm.plasma
        student_orig_cmap = plt.cm.cool
        student_edit_cmap = plt.cm.hot
        
        # Normalize timesteps to [0, 1] for colormap
        max_timestep = max(
            np.max(teacher_orig_timesteps) if teacher_orig_timesteps else 0,
            np.max(teacher_edit_timesteps) if teacher_edit_timesteps else 0,
            np.max(student_orig_timesteps) if student_orig_timesteps else 0,
            np.max(student_edit_timesteps) if student_edit_timesteps else 0
        )
        
        norm_teacher_orig_timesteps = np.array(teacher_orig_timesteps) / max_timestep if teacher_orig_timesteps else []
        norm_teacher_edit_timesteps = np.array(teacher_edit_timesteps) / max_timestep if teacher_edit_timesteps else []
        norm_student_orig_timesteps = np.array(student_orig_timesteps) / max_timestep if student_orig_timesteps else []
        norm_student_edit_timesteps = np.array(student_edit_timesteps) / max_timestep if student_edit_timesteps else []
        
        # Plot teacher original trajectory
        sc1 = plt.scatter(
            teacher_orig_pca[:, 0], teacher_orig_pca[:, 1],
            c=norm_teacher_orig_timesteps, cmap=teacher_orig_cmap, marker='o', s=50, alpha=0.7, 
            label='Teacher (Original)'
        )
        
        # Plot teacher edited trajectory
        sc2 = plt.scatter(
            teacher_edit_pca[:, 0], teacher_edit_pca[:, 1],
            c=norm_teacher_edit_timesteps, cmap=teacher_edit_cmap, marker='*', s=50, alpha=0.7, 
            label='Teacher (Edited)'
        )
        
        # Plot student original trajectory
        sc3 = plt.scatter(
            student_orig_pca[:, 0], student_orig_pca[:, 1],
            c=norm_student_orig_timesteps, cmap=student_orig_cmap, marker='x', s=50, alpha=0.7, 
            label='Student (Original)'
        )
        
        # Plot student edited trajectory
        sc4 = plt.scatter(
            student_edit_pca[:, 0], student_edit_pca[:, 1],
            c=norm_student_edit_timesteps, cmap=student_edit_cmap, marker='+', s=50, alpha=0.7, 
            label='Student (Edited)'
        )
        
        # Connect points in sequence for teacher original with arrows to show direction
        for i in range(len(teacher_orig_pca) - 1):
            plt.arrow(
                teacher_orig_pca[i, 0], teacher_orig_pca[i, 1],
                teacher_orig_pca[i+1, 0] - teacher_orig_pca[i, 0],
                teacher_orig_pca[i+1, 1] - teacher_orig_pca[i, 1],
                head_width=0.2, head_length=0.3, fc='blue', ec='blue', alpha=0.3
            )
        
        # Connect points in sequence for teacher edited with arrows
        for i in range(len(teacher_edit_pca) - 1):
            plt.arrow(
                teacher_edit_pca[i, 0], teacher_edit_pca[i, 1],
                teacher_edit_pca[i+1, 0] - teacher_edit_pca[i, 0],
                teacher_edit_pca[i+1, 1] - teacher_edit_pca[i, 1],
                head_width=0.2, head_length=0.3, fc='red', ec='red', alpha=0.3
            )
        
        # Connect points in sequence for student original with arrows
        for i in range(len(student_orig_pca) - 1):
            plt.arrow(
                student_orig_pca[i, 0], student_orig_pca[i, 1],
                student_orig_pca[i+1, 0] - student_orig_pca[i, 0],
                student_orig_pca[i+1, 1] - student_orig_pca[i, 1],
                head_width=0.2, head_length=0.3, fc='green', ec='green', alpha=0.3
            )
        
        # Connect points in sequence for student edited with arrows
        for i in range(len(student_edit_pca) - 1):
            plt.arrow(
                student_edit_pca[i, 0], student_edit_pca[i, 1],
                student_edit_pca[i+1, 0] - student_edit_pca[i, 0],
                student_edit_pca[i+1, 1] - student_edit_pca[i, 1],
                head_width=0.2, head_length=0.3, fc='yellow', ec='yellow', alpha=0.3
            )
        
        # Mark common starting point with a large marker
        plt.scatter(
            common_start_point[0], common_start_point[1],
            marker='D', s=300, color='black', edgecolor='white', zorder=11,
            label='Common Starting Point'
        )
        
        # Add colorbar
        cbar = plt.colorbar(sc1)
        cbar.set_label('Normalized Timestep')
        
        # Add labels and title
        plt.xlabel(f'Principal Component 1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2f})')
        plt.ylabel(f'Principal Component 2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2f})')
        
        # Create title with editing information
        title = f'PCA of {edit_type.capitalize()} Editing Comparison'
        if direction_name and strength:
            title += f' (direction_{direction_name}, Strength: {strength})'
        if size_factor:
            title += f' (Size Factor: {size_factor})'
        
        plt.title(title)
        
        # Add legend with a smaller font size and more columns
        plt.legend(loc='best', fontsize='small', ncol=2)
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure
        filename = f'pca_comparison_{edit_type}'
        if direction_name:
            filename += f'_{direction_name}'
        if strength:
            filename += f'_strength_{strength}'
        
        plt.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create 3D PCA visualization if we have enough dimensions
        if combined_features.shape[1] >= 3:
            pca = PCA(n_components=3)
            combined_pca = pca.fit_transform(combined_features)
            
            # Split back into separate trajectories
            teacher_orig_pca = combined_pca[:len(teacher_orig_features)]
            teacher_edit_pca = combined_pca[len(teacher_orig_features):len(teacher_orig_features)+len(teacher_edit_features)]
            student_orig_pca = combined_pca[len(teacher_orig_features)+len(teacher_edit_features):len(teacher_orig_features)+len(teacher_edit_features)+len(student_orig_features)]
            student_edit_pca = combined_pca[len(teacher_orig_features)+len(teacher_edit_features)+len(student_orig_features):]
            
            # Use the first point of the teacher original trajectory as the common starting point
            common_start_point_3d = teacher_orig_pca[0].copy()
            
            # Save the original first points for each trajectory
            teacher_orig_first_3d = teacher_orig_pca[0].copy()
            teacher_edit_first_3d = teacher_edit_pca[0].copy()
            student_orig_first_3d = student_orig_pca[0].copy()
            student_edit_first_3d = student_edit_pca[0].copy()
            
            # Set the first point of each trajectory to the common starting point
            teacher_orig_pca[0] = common_start_point_3d
            teacher_edit_pca[0] = common_start_point_3d
            student_orig_pca[0] = common_start_point_3d
            student_edit_pca[0] = common_start_point_3d
            
            # Calculate the translation vectors for the rest of the points in each trajectory
            teacher_orig_translation_3d = common_start_point_3d - teacher_orig_first_3d
            teacher_edit_translation_3d = common_start_point_3d - teacher_edit_first_3d
            student_orig_translation_3d = common_start_point_3d - student_orig_first_3d
            student_edit_translation_3d = common_start_point_3d - student_edit_first_3d
            
            # Apply the translation to the rest of the points in each trajectory
            for i in range(1, len(teacher_orig_pca)):
                teacher_orig_pca[i] += teacher_orig_translation_3d
            
            for i in range(1, len(teacher_edit_pca)):
                teacher_edit_pca[i] += teacher_edit_translation_3d
            
            for i in range(1, len(student_orig_pca)):
                student_orig_pca[i] += student_orig_translation_3d
            
            for i in range(1, len(student_edit_pca)):
                student_edit_pca[i] += student_edit_translation_3d
            
            # Create 3D figure
            fig = plt.figure(figsize=(14, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot teacher original trajectory
            sc1 = ax.scatter(
                teacher_orig_pca[:, 0], teacher_orig_pca[:, 1], teacher_orig_pca[:, 2],
                c=norm_teacher_orig_timesteps, cmap=teacher_orig_cmap, marker='o', s=50, alpha=0.7, 
                label='Teacher (Original)'
            )
            
            # Plot teacher edited trajectory
            sc2 = ax.scatter(
                teacher_edit_pca[:, 0], teacher_edit_pca[:, 1], teacher_edit_pca[:, 2],
                c=norm_teacher_edit_timesteps, cmap=teacher_edit_cmap, marker='*', s=50, alpha=0.7, 
                label='Teacher (Edited)'
            )
            
            # Plot student original trajectory
            sc3 = ax.scatter(
                student_orig_pca[:, 0], student_orig_pca[:, 1], student_orig_pca[:, 2],
                c=norm_student_orig_timesteps, cmap=student_orig_cmap, marker='x', s=50, alpha=0.7, 
                label='Student (Original)'
            )
            
            # Plot student edited trajectory
            sc4 = ax.scatter(
                student_edit_pca[:, 0], student_edit_pca[:, 1], student_edit_pca[:, 2],
                c=norm_student_edit_timesteps, cmap=student_edit_cmap, marker='+', s=50, alpha=0.7, 
                label='Student (Edited)'
            )
            
            # Mark common starting point
            ax.scatter(
                common_start_point_3d[0], common_start_point_3d[1], common_start_point_3d[2],
                marker='D', s=300, color='black', edgecolor='white', zorder=11,
                label='Common Starting Point'
            )
            
            # Connect points with lines
            ax.plot(
                teacher_orig_pca[:, 0], teacher_orig_pca[:, 1], teacher_orig_pca[:, 2],
                'b-', alpha=0.3
            )
            
            ax.plot(
                teacher_edit_pca[:, 0], teacher_edit_pca[:, 1], teacher_edit_pca[:, 2],
                'r-', alpha=0.3
            )
            
            ax.plot(
                student_orig_pca[:, 0], student_orig_pca[:, 1], student_orig_pca[:, 2],
                'g-', alpha=0.3
            )
            
            ax.plot(
                student_edit_pca[:, 0], student_edit_pca[:, 1], student_edit_pca[:, 2],
                'y-', alpha=0.3
            )
            
            # Add colorbar
            cbar = plt.colorbar(sc1, ax=ax, shrink=0.7)
            cbar.set_label('Normalized Timestep')
            
            # Add labels and title
            ax.set_xlabel(f'PC1 (Var: {pca.explained_variance_ratio_[0]:.2f})')
            ax.set_ylabel(f'PC2 (Var: {pca.explained_variance_ratio_[1]:.2f})')
            ax.set_zlabel(f'PC3 (Var: {pca.explained_variance_ratio_[2]:.2f})')
            
            # Create title with editing information
            title = f'3D PCA of {edit_type.capitalize()} Editing Comparison'
            if direction_name and strength:
                title += f' (direction_{direction_name}, Strength: {strength})'
            if size_factor:
                title += f' (Size Factor: {size_factor})'
            
            ax.set_title(title)
            
            # Add legend with smaller font size
            ax.legend(fontsize='small', ncol=2)
            
            # Save figure
            filename = f'pca_3d_comparison_{edit_type}'
            if direction_name:
                filename += f'_{direction_name}'
            if strength:
                filename += f'_strength_{strength}'
            
            plt.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=300, bbox_inches='tight')
            plt.close()

def visualize_trajectory_pca(teacher_trajectories, student_trajectories, edit_type, edit_points, output_dir, size_factor=None, direction_name=None, strength=None):
    """
    Create PCA visualization of teacher and student model trajectories
    
    Args:
        teacher_trajectories: List of teacher trajectories
        student_trajectories: List of student trajectories
        edit_type: Type of editing being performed (e.g., 'latent', 'prompt', 'inpainting')
        edit_points: Dictionary with 'teacher' and 'student' keys, each containing the timestep indices where editing is applied
        output_dir: Directory to save visualizations
        size_factor: Size factor of the student model for labeling
        direction_name: Name of the direction used for latent manipulation (if applicable)
        strength: Strength of the manipulation (if applicable)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract images from trajectories
    teacher_images = []
    teacher_timesteps = []
    student_images = []
    student_timesteps = []
    
    for trajectory in teacher_trajectories:
        for img, timestep in trajectory:
            # Convert to numpy and flatten
            if isinstance(img, torch.Tensor):
                img_np = img.cpu().detach().numpy()
                img_flat = img_np.reshape(1, -1)
                teacher_images.append(img_flat)
                teacher_timesteps.append(timestep)
    
    for trajectory in student_trajectories:
        for img, timestep in trajectory:
            # Convert to numpy and flatten
            if isinstance(img, torch.Tensor):
                img_np = img.cpu().detach().numpy()
                img_flat = img_np.reshape(1, -1)
                student_images.append(img_flat)
                student_timesteps.append(timestep)
    
    # Stack all flattened images
    if teacher_images and student_images:
        teacher_features = np.vstack(teacher_images)
        student_features = np.vstack(student_images)
        
        # Combine features for PCA fitting
        combined_features = np.vstack([teacher_features, student_features])
        
        # Fit PCA
        pca = PCA(n_components=2)
        combined_pca = pca.fit_transform(combined_features)
        
        # Split back into teacher and student
        teacher_pca = combined_pca[:len(teacher_features)]
        student_pca = combined_pca[len(teacher_features):]
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # Create colormaps
        teacher_cmap = plt.cm.viridis
        student_cmap = plt.cm.plasma
        
        # Normalize timesteps to [0, 1] for colormap
        max_timestep = max(np.max(teacher_timesteps), np.max(student_timesteps))
        norm_teacher_timesteps = np.array(teacher_timesteps) / max_timestep
        norm_student_timesteps = np.array(student_timesteps) / max_timestep
        
        # Plot teacher trajectory
        sc1 = plt.scatter(
            teacher_pca[:, 0], teacher_pca[:, 1],
            c=norm_teacher_timesteps, cmap=teacher_cmap, marker='o', s=50, alpha=0.7, label='Teacher'
        )
        
        # Plot student trajectory
        sc2 = plt.scatter(
            student_pca[:, 0], student_pca[:, 1],
            c=norm_student_timesteps, cmap=student_cmap, marker='x', s=50, alpha=0.7, label='Student'
        )
        
        # Connect points in sequence for teacher
        for i in range(len(teacher_pca) - 1):
            plt.plot(
                [teacher_pca[i, 0], teacher_pca[i+1, 0]],
                [teacher_pca[i, 1], teacher_pca[i+1, 1]],
                'b-', alpha=0.3
            )
        
        # Connect points in sequence for student
        for i in range(len(student_pca) - 1):
            plt.plot(
                [student_pca[i, 0], student_pca[i+1, 0]],
                [student_pca[i, 1], student_pca[i+1, 1]],
                'r-', alpha=0.3
            )
        
        # Mark edit points with stars
        if 'teacher' in edit_points and edit_points['teacher'] is not None:
            for idx in edit_points['teacher']:
                if 0 <= idx < len(teacher_pca):
                    plt.scatter(
                        teacher_pca[idx, 0], teacher_pca[idx, 1],
                        marker='*', s=200, color='blue', edgecolor='black', zorder=10,
                        label='Teacher Edit Point' if idx == edit_points['teacher'][0] else ""
                    )
        
        if 'student' in edit_points and edit_points['student'] is not None:
            for idx in edit_points['student']:
                if 0 <= idx < len(student_pca):
                    plt.scatter(
                        student_pca[idx, 0], student_pca[idx, 1],
                        marker='*', s=200, color='red', edgecolor='black', zorder=10,
                        label='Student Edit Point' if idx == edit_points['student'][0] else ""
                    )
        
        # Add colorbar
        cbar = plt.colorbar(sc1)
        cbar.set_label('Normalized Timestep')
        
        # Add labels and title
        plt.xlabel(f'Principal Component 1 (Explained Variance: {pca.explained_variance_ratio_[0]:.2f})')
        plt.ylabel(f'Principal Component 2 (Explained Variance: {pca.explained_variance_ratio_[1]:.2f})')
        
        # Create title with editing information
        title = f'PCA of {edit_type.capitalize()} Editing Trajectories'
        if direction_name and strength:
            title += f' ({direction_name}, Strength: {strength})'
        if size_factor:
            title += f' (Size Factor: {size_factor})'
        
        plt.title(title)
        
        # Add legend
        plt.legend(loc='best')
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure
        filename = f'pca_trajectories_{edit_type}'
        if direction_name:
            filename += f'_{direction_name}'
        if strength:
            filename += f'_strength_{strength}'
        
        plt.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create 3D PCA visualization if we have enough dimensions
        if combined_features.shape[1] >= 3:
            pca = PCA(n_components=3)
            combined_pca = pca.fit_transform(combined_features)
            
            # Split back into teacher and student
            teacher_pca = combined_pca[:len(teacher_features)]
            student_pca = combined_pca[len(teacher_features):]
            
            # Create 3D figure
            fig = plt.figure(figsize=(14, 12))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot teacher trajectory
            sc1 = ax.scatter(
                teacher_pca[:, 0], teacher_pca[:, 1], teacher_pca[:, 2],
                c=norm_teacher_timesteps, cmap=teacher_cmap, marker='o', s=50, alpha=0.7, label='Teacher'
            )
            
            # Plot student trajectory
            sc2 = ax.scatter(
                student_pca[:, 0], student_pca[:, 1], student_pca[:, 2],
                c=norm_student_timesteps, cmap=student_cmap, marker='x', s=50, alpha=0.7, label='Student'
            )
            
            # Mark edit points with stars
            if 'teacher' in edit_points and edit_points['teacher'] is not None:
                for idx in edit_points['teacher']:
                    if 0 <= idx < len(teacher_pca):
                        ax.scatter(
                            teacher_pca[idx, 0], teacher_pca[idx, 1], teacher_pca[idx, 2],
                            marker='*', s=200, color='blue', edgecolor='black', zorder=10,
                            label='Teacher Edit Point' if idx == edit_points['teacher'][0] else ""
                        )
            
            if 'student' in edit_points and edit_points['student'] is not None:
                for idx in edit_points['student']:
                    if 0 <= idx < len(student_pca):
                        ax.scatter(
                            student_pca[idx, 0], student_pca[idx, 1], student_pca[idx, 2],
                            marker='*', s=200, color='red', edgecolor='black', zorder=10,
                            label='Student Edit Point' if idx == edit_points['student'][0] else ""
                        )
            
            # Add colorbar
            cbar = plt.colorbar(sc1, ax=ax, shrink=0.7)
            cbar.set_label('Normalized Timestep')
            
            # Add labels and title
            ax.set_xlabel(f'PC1 (Var: {pca.explained_variance_ratio_[0]:.2f})')
            ax.set_ylabel(f'PC2 (Var: {pca.explained_variance_ratio_[1]:.2f})')
            ax.set_zlabel(f'PC3 (Var: {pca.explained_variance_ratio_[2]:.2f})')
            
            # Create title with editing information
            title = f'3D PCA of {edit_type.capitalize()} Editing Trajectories'
            if direction_name and strength:
                title += f' ({direction_name}, Strength: {strength})'
            if size_factor:
                title += f' (Size Factor: {size_factor})'
            
            ax.set_title(title)
            
            # Add legend
            ax.legend()
            
            # Save figure
            filename = f'pca_3d_trajectories_{edit_type}'
            if direction_name:
                filename += f'_{direction_name}'
            if strength:
                filename += f'_strength_{strength}'
            
            plt.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=300, bbox_inches='tight')
            plt.close()

def visualize_latent_metrics(metrics, output_dir, size_factor=None):
    """
    Visualize metrics for latent manipulation analysis
    
    Args:
        metrics: Dictionary of metrics
        output_dir: Directory to save visualizations
        size_factor: Size factor of the student model for labeling
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot LPIPS distances
    if "lpips" in metrics:
        plt.figure(figsize=(10, 6))
        plt.bar(['LPIPS Distance'], [np.mean(metrics["lpips"])], color='purple')
        plt.title(f'LPIPS Distance (Size Factor: {size_factor})' if size_factor else 'LPIPS Distance')
        plt.ylabel('Distance')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'lpips_distance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot FID score
    if "fid" in metrics:
        plt.figure(figsize=(10, 6))
        plt.bar(['FID Score'], [metrics["fid"]], color='green')
        plt.title(f'FID Score (Size Factor: {size_factor})' if size_factor else 'FID Score')
        plt.ylabel('Score')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'fid_score.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot trajectory divergence metrics
    if "trajectory_divergence" in metrics:
        div = metrics["trajectory_divergence"]
        
        # Plot average distance
        plt.figure(figsize=(10, 6))
        plt.bar(['Average Distance'], [div["avg_distance"]], color='blue')
        plt.title(f'Average Trajectory Distance (Size Factor: {size_factor})' if size_factor else 'Average Trajectory Distance')
        plt.ylabel('Distance')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'avg_distance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot max distance
        plt.figure(figsize=(10, 6))
        plt.bar(['Max Distance'], [div["max_distance"]], color='red')
        plt.title(f'Max Trajectory Distance (Size Factor: {size_factor})' if size_factor else 'Max Trajectory Distance')
        plt.ylabel('Distance')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'max_distance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot average similarity
        plt.figure(figsize=(10, 6))
        plt.bar(['Average Similarity'], [div["avg_similarity"]], color='green')
        plt.title(f'Average Trajectory Similarity (Size Factor: {size_factor})' if size_factor else 'Average Trajectory Similarity')
        plt.ylabel('Similarity')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'avg_similarity.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot min similarity
        plt.figure(figsize=(10, 6))
        plt.bar(['Min Similarity'], [div["min_similarity"]], color='orange')
        plt.title(f'Min Trajectory Similarity (Size Factor: {size_factor})' if size_factor else 'Min Trajectory Similarity')
        plt.ylabel('Similarity')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'min_similarity.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot length ratio
        plt.figure(figsize=(10, 6))
        plt.bar(['Length Ratio'], [div["length_ratio"]], color='purple')
        plt.title(f'Trajectory Length Ratio (Size Factor: {size_factor})' if size_factor else 'Trajectory Length Ratio')
        plt.ylabel('Ratio (Student/Teacher)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'length_ratio.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot distance over time
        plt.figure(figsize=(12, 6))
        plt.plot(div["distances"], color='blue')
        plt.title(f'Trajectory Distance Over Time (Size Factor: {size_factor})' if size_factor else 'Trajectory Distance Over Time')
        plt.xlabel('Step')
        plt.ylabel('Distance')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'distance_over_time.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot similarity over time
        plt.figure(figsize=(12, 6))
        plt.plot(div["similarities"], color='green')
        plt.title(f'Trajectory Similarity Over Time (Size Factor: {size_factor})' if size_factor else 'Trajectory Similarity Over Time')
        plt.xlabel('Step')
        plt.ylabel('Similarity')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'similarity_over_time.png'), dpi=300, bbox_inches='tight')
        plt.close()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Analyze editing capabilities of diffusion models")
    parser.add_argument("--teacher_model", type=str, default="output/models/teacher/model_epoch_10.pt", help="Path to teacher model")
    parser.add_argument("--student_model", type=str, default=None, help="Path to student model")
    parser.add_argument("--size_factor", type=float, default=0.5, help="Size factor of student model")
    parser.add_argument("--output_dir", type=str, default="results/editing", help="Output directory")
    parser.add_argument("--edit_mode", type=str, choices=["prompt", "inpainting", "latent", "all"], 
                        default="all", help="Editing mode to analyze")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--enable_fid", action="store_true", help="Enable FID calculations (disabled by default)")
    return parser.parse_args()

def main():
    """Main function to run the editing analysis"""
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load configuration
    config = Config()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() and config.mps_enabled else
        "cpu"
    )
    print(f"Using {device} device")
    
    # Load teacher model
    teacher_model_path = args.teacher_model or os.path.join('output/models/teacher/', 'model_epoch_10.pt')
    print(f"Loading teacher model from {teacher_model_path}...")
    teacher_model = SimpleUNet(config).to(device)
    teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=device))
    teacher_model.eval()
    
    # Load student model
    student_model_path = args.student_model or os.path.join(
        'output/models/students/', f'size_{args.size_factor}/model_epoch_1.pt')
    print(f"Loading student model from {student_model_path}...")
    
    # Determine architecture type based on size factor
    architecture_type = 'full'
    if args.size_factor < 0.1:
        architecture_type = 'tiny'
    elif args.size_factor < 0.3:
        architecture_type = 'small'
    elif args.size_factor < 0.7:
        architecture_type = 'medium'
    
    student_model = StudentUNet(config, size_factor=args.size_factor, 
                               architecture_type=architecture_type).to(device)
    student_model.load_state_dict(torch.load(student_model_path, map_location=device))
    student_model.eval()
    
    # Create diffusion parameters
    teacher_params = get_diffusion_params(config.teacher_steps, config)
    student_params = get_diffusion_params(config.student_steps, config)
    
    # Add timesteps to diffusion parameters
    teacher_params['timesteps'] = config.teacher_steps
    student_params['timesteps'] = config.student_steps
    
    # Add alphas to diffusion parameters
    teacher_betas = teacher_params['betas']
    student_betas = student_params['betas']
    
    teacher_alphas = 1.0 - teacher_betas
    student_alphas = 1.0 - student_betas
    
    teacher_params['alphas'] = teacher_alphas
    student_params['alphas'] = student_alphas
    
    # Run editing analysis based on selected mode
    results = {}
    
    if args.edit_mode in ["prompt", "all"]:
        print("\nRunning prompt-based editing analysis...")
        prompt_results = run_prompt_editing_analysis(
            teacher_model, student_model, 
            teacher_params, student_params,
            config, args, device
        )
        results["prompt"] = prompt_results
    
    if args.edit_mode in ["inpainting", "all"]:
        print("\nRunning masked inpainting analysis...")
        inpainting_results = run_inpainting_analysis(
            teacher_model, student_model, 
            teacher_params, student_params,
            config, args, device
        )
        results["inpainting"] = inpainting_results
    
    if args.edit_mode in ["latent", "all"]:
        print("\nRunning latent-space manipulation analysis...")
        latent_results = run_latent_manipulation_analysis(
            teacher_model, student_model, 
            teacher_params, student_params,
            config, args, device
        )
        results["latent"] = latent_results
    
    # Generate summary report
    generate_summary_report(results, args.output_dir, args.size_factor)
    
    print(f"\nEditing analysis complete. Results saved to {args.output_dir}")

def run_prompt_editing_analysis(teacher_model, student_model, 
                               teacher_params, student_params,
                               config, args, device):
    """Run prompt-based editing analysis"""
    print("\nRunning prompt-based editing analysis...")
    
    # Create output directory for prompt editing
    prompt_dir = os.path.join(args.output_dir, "prompt_editing")
    os.makedirs(prompt_dir, exist_ok=True)
    
    # Define original and edited prompts
    # In a real implementation, these would be used with a text-conditioned model
    prompt_pairs = [
        ("A dog", "A dog with a hat"),
        ("A cat", "A cat with sunglasses"),
        ("A house", "A house with a red roof"),
        ("A car", "A sports car"),
        ("A person", "A person smiling")
    ]
    
    # Run prompt editing for teacher model
    print("Running prompt editing for teacher model...")
    teacher_results = []
    
    for original_prompt, edited_prompt in prompt_pairs:
        result = apply_prompt_editing(
            teacher_model, 
            teacher_params, 
            original_prompt, 
            edited_prompt, 
            config, 
            device
        )
        teacher_results.append(result)
        
        # Visualize results
        teacher_prompt_dir = os.path.join(prompt_dir, "teacher", f"{original_prompt}_to_{edited_prompt}")
        os.makedirs(teacher_prompt_dir, exist_ok=True)
        visualize_prompt_editing(result, teacher_prompt_dir)
    
    # Run prompt editing for student model
    print("Running prompt editing for student model...")
    student_results = []
    
    for original_prompt, edited_prompt in prompt_pairs:
        result = apply_prompt_editing(
            student_model, 
            student_params, 
            original_prompt, 
            edited_prompt, 
            config, 
            device
        )
        student_results.append(result)
        
        # Visualize results
        student_prompt_dir = os.path.join(prompt_dir, "student", f"{original_prompt}_to_{edited_prompt}")
        os.makedirs(student_prompt_dir, exist_ok=True)
        visualize_prompt_editing(result, student_prompt_dir, args.size_factor)
    
    # Compute metrics
    metrics = {}
    
    # LPIPS
    lpips_distances = []
    for teacher_result, student_result in zip(teacher_results, student_results):
        # Compare edited images
        lpips_distance = compute_lpips(
            teacher_result["edited_image"], 
            student_result["edited_image"], 
            device
        )
        lpips_distances.append(lpips_distance)
    
    metrics["lpips"] = lpips_distances
    
    # FID
    if args.enable_fid:
        teacher_edited_images = [result["edited_image"] for result in teacher_results]
        student_edited_images = [result["edited_image"] for result in student_results]
        
        fid_score = compute_fid(teacher_edited_images, student_edited_images, device)
        metrics["fid"] = fid_score
    
    # Trajectory divergence
    trajectory_divergences = []
    for teacher_result, student_result in zip(teacher_results, student_results):
        if "edited_trajectory" in teacher_result and "edited_trajectory" in student_result:
            divergence = compute_trajectory_divergence(
                teacher_result["edited_trajectory"],
                student_result["edited_trajectory"]
            )
            trajectory_divergences.append(divergence)
    
    if trajectory_divergences:
        # Compute average metrics across all trajectories
        avg_divergence = {
            "distances": np.mean([d["distances"] for d in trajectory_divergences], axis=0).tolist(),
            "similarities": np.mean([d["similarities"] for d in trajectory_divergences], axis=0).tolist(),
            "avg_distance": np.mean([d["avg_distance"] for d in trajectory_divergences]),
            "max_distance": np.mean([d["max_distance"] for d in trajectory_divergences]),
            "avg_similarity": np.mean([d["avg_similarity"] for d in trajectory_divergences]),
            "min_similarity": np.mean([d["min_similarity"] for d in trajectory_divergences]),
            "length_ratio": np.mean([d["length_ratio"] for d in trajectory_divergences])
        }
        metrics["trajectory_divergence"] = avg_divergence
    
    # Visualize metrics
    metrics_dir = os.path.join(prompt_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    visualize_metrics(metrics, metrics_dir, args.size_factor)
    
    # Add PCA visualization of trajectories
    if trajectory_divergences:
        # Create PCA visualization directory
        pca_dir = os.path.join(prompt_dir, "pca_visualizations")
        os.makedirs(pca_dir, exist_ok=True)
        
        # Create comparison directory
        comparison_dir = os.path.join(prompt_dir, "trajectory_comparisons")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # For each prompt pair, create a PCA visualization
        for i, (teacher_result, student_result) in enumerate(zip(teacher_results, student_results)):
            if "edited_trajectory" in teacher_result and "edited_trajectory" in student_result:
                # Get trajectories
                teacher_edited_trajectory = teacher_result["edited_trajectory"]
                student_edited_trajectory = student_result["edited_trajectory"]
                
                # For prompt editing, we need to generate original trajectories for comparison
                # We'll use the original trajectories if available, otherwise we'll skip
                if "original_trajectory" in teacher_result and "original_trajectory" in student_result:
                    teacher_original_trajectory = teacher_result["original_trajectory"]
                    student_original_trajectory = student_result["original_trajectory"]
                    
                    # For prompt editing, the edit point is at the beginning
                    edit_points = {
                        "teacher": [0],  # First point in the trajectory
                        "student": [0]   # First point in the trajectory
                    }
                    
                    # Get prompt information for labeling
                    original_prompt, edited_prompt = prompt_pairs[i]
                    prompt_label = f"{original_prompt}_to_{edited_prompt}"
                    
                    # Create PCA visualization
                    visualize_trajectory_pca(
                        [teacher_edited_trajectory],
                        [student_edited_trajectory],
                        "prompt",
                        edit_points,
                        os.path.join(pca_dir, prompt_label),
                        args.size_factor,
                        prompt_label,
                        None
                    )
                    
                    # Create comparison visualization
                    visualize_trajectory_comparison(
                        teacher_original_trajectory,
                        teacher_edited_trajectory,
                        student_original_trajectory,
                        student_edited_trajectory,
                        "prompt",
                        os.path.join(comparison_dir, prompt_label),
                        args.size_factor,
                        prompt_label,
                        None
                    )
        
        # Also create a combined visualization with all prompt pairs
        all_teacher_edited_trajectories = [result["edited_trajectory"] for result in teacher_results if "edited_trajectory" in result]
        all_student_edited_trajectories = [result["edited_trajectory"] for result in student_results if "edited_trajectory" in result]
        
        if all_teacher_edited_trajectories and all_student_edited_trajectories:
            # Create combined PCA visualization
            visualize_trajectory_pca(
                all_teacher_edited_trajectories,
                all_student_edited_trajectories,
                "prompt",
                {"teacher": [0], "student": [0]},
                os.path.join(pca_dir, "combined"),
                args.size_factor
            )
            
            # For combined comparison, use the first trajectory from each set if original trajectories are available
            all_teacher_originals = [result["original_trajectory"] for result in teacher_results if "original_trajectory" in result]
            all_student_originals = [result["original_trajectory"] for result in student_results if "original_trajectory" in result]
            
            if all_teacher_originals and all_student_originals and all_teacher_edited_trajectories and all_student_edited_trajectories:
                visualize_trajectory_comparison(
                    all_teacher_originals[0],
                    all_teacher_edited_trajectories[0],
                    all_student_originals[0],
                    all_student_edited_trajectories[0],
                    "prompt",
                    os.path.join(comparison_dir, "combined"),
                    args.size_factor
                )
    
    return {
        "teacher_results": teacher_results,
        "student_results": student_results,
        "metrics": metrics,
        "status": "completed"
    }

def run_inpainting_analysis(teacher_model, student_model, 
                           teacher_params, student_params,
                           config, args, device):
    """Run masked inpainting analysis"""
    print("\nRunning masked inpainting analysis...")
    
    # Create output directory for inpainting
    inpainting_dir = os.path.join(args.output_dir, "inpainting")
    os.makedirs(inpainting_dir, exist_ok=True)
    
    # Generate original images
    print("Generating original images...")
    original_images = []
    
    for i in range(args.num_samples):
        # Set seed for reproducibility
        torch.manual_seed(args.seed + i)
        
        # Generate image
        image, _ = generate_image(teacher_model, teacher_params, config, device)
        original_images.append(image)
    
    # Create masks
    masks = []
    for i in range(args.num_samples):
        mask = create_random_mask(config.image_size, config.image_size)
        masks.append(mask)
    
    # Run inpainting for teacher model
    print("Running inpainting for teacher model...")
    teacher_results = []
    
    for i, (image, mask) in enumerate(zip(original_images, masks)):
        result = apply_masked_inpainting(
            teacher_model, 
            teacher_params, 
            image, 
            mask, 
            config, 
            device
        )
        teacher_results.append(result)
        
        # Visualize results
        teacher_inpainting_dir = os.path.join(inpainting_dir, "teacher", f"sample_{i+1}")
        os.makedirs(teacher_inpainting_dir, exist_ok=True)
        visualize_inpainting(result, teacher_inpainting_dir)
    
    # Run inpainting for student model
    print("Running inpainting for student model...")
    student_results = []
    
    for i, (image, mask) in enumerate(zip(original_images, masks)):
        result = apply_masked_inpainting(
            student_model, 
            student_params, 
            image, 
            mask, 
            config, 
            device
        )
        student_results.append(result)
        
        # Visualize results
        student_inpainting_dir = os.path.join(inpainting_dir, "student", f"sample_{i+1}")
        os.makedirs(student_inpainting_dir, exist_ok=True)
        visualize_inpainting(result, student_inpainting_dir, args.size_factor)
    
    # Compute metrics
    metrics = {}
    
    # LPIPS
    lpips_distances = []
    for teacher_result, student_result in zip(teacher_results, student_results):
        # Compare inpainted images
        lpips_distance = compute_lpips(
            teacher_result["inpainted_image"], 
            student_result["inpainted_image"], 
            device
        )
        lpips_distances.append(lpips_distance)
    
    metrics["lpips"] = lpips_distances
    
    # FID
    if args.enable_fid:
        teacher_inpainted_images = [result["inpainted_image"] for result in teacher_results]
        student_inpainted_images = [result["inpainted_image"] for result in student_results]
        
        fid_score = compute_fid(teacher_inpainted_images, student_inpainted_images, device)
        metrics["fid"] = fid_score
    
    # Trajectory divergence
    trajectory_divergences = []
    for teacher_result, student_result in zip(teacher_results, student_results):
        if "trajectory" in teacher_result and "trajectory" in student_result:
            divergence = compute_trajectory_divergence(
                teacher_result["trajectory"],
                student_result["trajectory"]
            )
            trajectory_divergences.append(divergence)
    
    if trajectory_divergences:
        # Compute average metrics across all trajectories
        avg_divergence = {
            "distances": np.mean([d["distances"] for d in trajectory_divergences], axis=0).tolist(),
            "similarities": np.mean([d["similarities"] for d in trajectory_divergences], axis=0).tolist(),
            "avg_distance": np.mean([d["avg_distance"] for d in trajectory_divergences]),
            "max_distance": np.mean([d["max_distance"] for d in trajectory_divergences]),
            "avg_similarity": np.mean([d["avg_similarity"] for d in trajectory_divergences]),
            "min_similarity": np.mean([d["min_similarity"] for d in trajectory_divergences]),
            "length_ratio": np.mean([d["length_ratio"] for d in trajectory_divergences])
        }
        metrics["trajectory_divergence"] = avg_divergence
    
    # Visualize metrics
    metrics_dir = os.path.join(inpainting_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    visualize_metrics(metrics, metrics_dir, args.size_factor)
    
    # Add PCA visualization of trajectories
    if trajectory_divergences:
        # Create PCA visualization directory
        pca_dir = os.path.join(inpainting_dir, "pca_visualizations")
        os.makedirs(pca_dir, exist_ok=True)
        
        # Create comparison directory
        comparison_dir = os.path.join(inpainting_dir, "trajectory_comparisons")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # For each inpainting sample, create a PCA visualization
        for i, (teacher_result, student_result) in enumerate(zip(teacher_results, student_results)):
            if "trajectory" in teacher_result and "trajectory" in student_result:
                # Get trajectories
                teacher_trajectory = teacher_result["trajectory"]
                student_trajectory = student_result["trajectory"]
                
                # For inpainting, we need to generate original trajectories for comparison
                # Since inpainting doesn't have original trajectories, we'll generate them
                
                # Set seed for reproducibility
                torch.manual_seed(args.seed + i)
                
                # Generate original teacher trajectory
                original_teacher_image, original_teacher_trajectory = generate_image_with_trajectory(
                    teacher_model, teacher_params, config, device
                )
                
                # Generate original student trajectory
                torch.manual_seed(args.seed + i)  # Same seed for fair comparison
                original_student_image, original_student_trajectory = generate_image_with_trajectory(
                    student_model, student_params, config, device
                )
                
                # For inpainting, the edit point is at the beginning (mask is applied at start)
                edit_points = {
                    "teacher": [0],  # First point in the trajectory
                    "student": [0]   # First point in the trajectory
                }
                
                # Create PCA visualization
                visualize_trajectory_pca(
                    [teacher_trajectory],
                    [student_trajectory],
                    "inpainting",
                    edit_points,
                    os.path.join(pca_dir, f"sample_{i+1}"),
                    args.size_factor,
                    f"mask_{i+1}",
                    None
                )
                
                # Create comparison visualization
                visualize_trajectory_comparison(
                    original_teacher_trajectory,
                    teacher_trajectory,
                    original_student_trajectory,
                    student_trajectory,
                    "inpainting",
                    os.path.join(comparison_dir, f"sample_{i+1}"),
                    args.size_factor,
                    f"mask_{i+1}",
                    None
                )
        
        # Also create a combined visualization with all inpainting samples
        all_teacher_trajectories = [result["trajectory"] for result in teacher_results if "trajectory" in result]
        all_student_trajectories = [result["trajectory"] for result in student_results if "trajectory" in result]
        
        if all_teacher_trajectories and all_student_trajectories:
            # Create combined PCA visualization
            visualize_trajectory_pca(
                all_teacher_trajectories,
                all_student_trajectories,
                "inpainting",
                {"teacher": [0], "student": [0]},
                os.path.join(pca_dir, "combined"),
                args.size_factor
            )
            
            # For combined comparison, use the first sample's original and edited trajectories
            if len(teacher_results) > 0 and len(student_results) > 0:
                # Set seed for reproducibility
                torch.manual_seed(args.seed)
                
                # Generate original teacher trajectory
                original_teacher_image, original_teacher_trajectory = generate_image_with_trajectory(
                    teacher_model, teacher_params, config, device
                )
                
                # Generate original student trajectory
                torch.manual_seed(args.seed)  # Same seed for fair comparison
                original_student_image, original_student_trajectory = generate_image_with_trajectory(
                    student_model, student_params, config, device
                )
                
                visualize_trajectory_comparison(
                    original_teacher_trajectory,
                    all_teacher_trajectories[0],
                    original_student_trajectory,
                    all_student_trajectories[0],
                    "inpainting",
                    os.path.join(comparison_dir, "combined"),
                    args.size_factor
                )
    
    return {
        "teacher_results": teacher_results,
        "student_results": student_results,
        "metrics": metrics,
        "status": "completed"
    }

def run_latent_manipulation_analysis(teacher_model, student_model, 
                                    teacher_params, student_params,
                                    config, args, device):
    """Run latent-space manipulation analysis"""
    print("\nRunning latent-space manipulation analysis...")
    
    # Create output directory for latent manipulation
    latent_dir = os.path.join(args.output_dir, "latent_manipulation")
    os.makedirs(latent_dir, exist_ok=True)
    
    # Define manipulation strengths
    strengths = [0.5, 1.0, 1.5]
    
    # Find semantic directions (optional)
    # In a real implementation, this would find meaningful directions
    # For simplicity, we'll use random directions
    print("Generating semantic directions...")
    directions = {}
    
    # Generate a few random directions
    for i in range(3):
        latent_dim = config.channels * config.image_size * config.image_size
        direction = torch.randn(latent_dim, device=device)
        # Normalize the direction
        direction = direction / torch.norm(direction)
        directions[f"direction_{i+1}"] = direction
    
    # Run latent manipulation for teacher model
    print("Running latent manipulation for teacher model...")
    teacher_results = {}
    
    for direction_name, direction in directions.items():
        direction_results = {}
        
        for strength in strengths:
            result = apply_latent_manipulation(
                teacher_model, 
                teacher_params, 
                direction, 
                strength, 
                config, 
                device,
                num_samples=args.num_samples
            )
            direction_results[strength] = result
            
            # Visualize results
            teacher_latent_dir = os.path.join(latent_dir, "teacher", direction_name, f"strength_{strength}")
            os.makedirs(teacher_latent_dir, exist_ok=True)
            visualize_latent_manipulation(result, teacher_latent_dir)
        
        teacher_results[direction_name] = direction_results
    
    # Run latent manipulation for student model
    print("Running latent manipulation for student model...")
    student_results = {}
    
    for direction_name, direction in directions.items():
        direction_results = {}
        
        for strength in strengths:
            result = apply_latent_manipulation(
                student_model, 
                student_params, 
                direction, 
                strength, 
                config, 
                device,
                num_samples=args.num_samples
            )
            direction_results[strength] = result
            
            # Visualize results
            student_latent_dir = os.path.join(latent_dir, "student", direction_name, f"strength_{strength}")
            os.makedirs(student_latent_dir, exist_ok=True)
            visualize_latent_manipulation(result, student_latent_dir, args.size_factor)
        
        student_results[direction_name] = direction_results
    
    # Compute metrics
    all_metrics = {}
    
    for direction_name in directions:
        direction_metrics = {}
        
        for strength in strengths:
            teacher_result = teacher_results[direction_name][strength]
            student_result = student_results[direction_name][strength]
            
            metrics = {}
            
            # LPIPS
            lpips_distances = []
            for t_img, s_img in zip(teacher_result["manipulated_images"], student_result["manipulated_images"]):
                lpips_distance = compute_lpips(t_img, s_img, device)
                lpips_distances.append(lpips_distance)
            
            metrics["lpips"] = lpips_distances
            
            # FID
            if args.enable_fid:
                fid_score = compute_fid(
                    teacher_result["manipulated_images"], 
                    student_result["manipulated_images"], 
                    device
                )
                metrics["fid"] = fid_score
            
            # Trajectory divergence
            if "trajectories" in teacher_result and "trajectories" in student_result:
                trajectory_divergences = []
                
                for t_traj, s_traj in zip(teacher_result["trajectories"], student_result["trajectories"]):
                    divergence = compute_trajectory_divergence(
                        t_traj["manipulated"],
                        s_traj["manipulated"]
                    )
                    trajectory_divergences.append(divergence)
                
                if trajectory_divergences:
                    # Compute average metrics across all trajectories
                    avg_divergence = {
                        "distances": np.mean([d["distances"] for d in trajectory_divergences], axis=0).tolist(),
                        "similarities": np.mean([d["similarities"] for d in trajectory_divergences], axis=0).tolist(),
                        "avg_distance": np.mean([d["avg_distance"] for d in trajectory_divergences]),
                        "max_distance": np.mean([d["max_distance"] for d in trajectory_divergences]),
                        "avg_similarity": np.mean([d["avg_similarity"] for d in trajectory_divergences]),
                        "min_similarity": np.mean([d["min_similarity"] for d in trajectory_divergences]),
                        "length_ratio": np.mean([d["length_ratio"] for d in trajectory_divergences])
                    }
                    metrics["trajectory_divergence"] = avg_divergence
                    
                    # Add path lengths for visualization
                    teacher_path_lengths = [np.sum(np.sqrt(np.sum(np.diff([img[0].cpu().flatten().numpy() for img in t_traj["manipulated"]], axis=0)**2, axis=1))) for t_traj in teacher_result["trajectories"]]
                    student_path_lengths = [np.sum(np.sqrt(np.sum(np.diff([img[0].cpu().flatten().numpy() for img in s_traj["manipulated"]], axis=0)**2, axis=1))) for s_traj in student_result["trajectories"]]
                    
                    metrics["teacher_path_length"] = np.mean(teacher_path_lengths)
                    metrics["student_path_length"] = np.mean(student_path_lengths)
            
            # Visualize metrics
            metrics_dir = os.path.join(latent_dir, "metrics", direction_name, f"strength_{strength}")
            os.makedirs(metrics_dir, exist_ok=True)
            visualize_latent_metrics(metrics, metrics_dir, args.size_factor)
            
            # Add PCA visualization of trajectories
            if "trajectories" in teacher_result and "trajectories" in student_result:
                # For each sample, create a PCA visualization
                for sample_idx in range(len(teacher_result["trajectories"])):
                    # Get trajectories for this sample
                    teacher_trajectory = teacher_result["trajectories"][sample_idx]["manipulated"]
                    student_trajectory = student_result["trajectories"][sample_idx]["manipulated"]
                    
                    # Determine edit points (where manipulation is applied)
                    # For latent manipulation, the edit point is at the beginning of the trajectory
                    edit_points = {
                        "teacher": [0],  # First point in the trajectory
                        "student": [0]   # First point in the trajectory
                    }
                    
                    # Create PCA visualization directory
                    pca_dir = os.path.join(latent_dir, "pca_visualizations", direction_name, f"strength_{strength}")
                    os.makedirs(pca_dir, exist_ok=True)
                    
                    # Create PCA visualization
                    visualize_trajectory_pca(
                        [teacher_trajectory],
                        [student_trajectory],
                        "latent",
                        edit_points,
                        os.path.join(pca_dir, f"sample_{sample_idx+1}"),
                        args.size_factor,
                        direction_name,
                        strength
                    )
                    
                    # Also create a comparison visualization with original and edited trajectories
                    teacher_original = teacher_result["trajectories"][sample_idx]["original"]
                    student_original = student_result["trajectories"][sample_idx]["original"]
                    
                    # Create comparison directory
                    comparison_dir = os.path.join(latent_dir, "trajectory_comparisons", direction_name, f"strength_{strength}")
                    os.makedirs(comparison_dir, exist_ok=True)
                    
                    # Create comparison visualization
                    visualize_trajectory_comparison(
                        teacher_original,
                        teacher_trajectory,
                        student_original,
                        student_trajectory,
                        "latent",
                        os.path.join(comparison_dir, f"sample_{sample_idx+1}"),
                        args.size_factor,
                        direction_name,
                        strength
                    )
                
                # Also create a combined visualization with all samples
                all_teacher_trajectories = [traj["manipulated"] for traj in teacher_result["trajectories"]]
                all_student_trajectories = [traj["manipulated"] for traj in student_result["trajectories"]]
                
                # Create combined PCA visualization
                visualize_trajectory_pca(
                    all_teacher_trajectories,
                    all_student_trajectories,
                    "latent",
                    {"teacher": [0], "student": [0]},
                    os.path.join(latent_dir, "pca_visualizations", direction_name, f"strength_{strength}", "combined"),
                    args.size_factor,
                    direction_name,
                    strength
                )
                
                # Also create a combined comparison visualization
                all_teacher_originals = [traj["original"] for traj in teacher_result["trajectories"]]
                all_student_originals = [traj["original"] for traj in student_result["trajectories"]]
                
                # For combined comparison, use the first trajectory from each set if original trajectories are available
                if all_teacher_originals and all_student_originals and all_teacher_trajectories and all_student_trajectories:
                    visualize_trajectory_comparison(
                        all_teacher_originals[0],
                        all_teacher_trajectories[0],
                        all_student_originals[0],
                        all_student_trajectories[0],
                        "latent",
                        os.path.join(comparison_dir, "combined"),
                        args.size_factor,
                        direction_name,
                        strength
                    )
            
            direction_metrics[strength] = metrics
        
        all_metrics[direction_name] = direction_metrics
    
    return {
        "teacher_results": teacher_results,
        "student_results": student_results,
        "metrics": all_metrics,
        "status": "completed"
    }

def generate_summary_report(results, output_dir, size_factor):
    """Generate a summary report of editing analysis results"""
    print("Generating summary report...")
    
    # Create summary directory
    summary_dir = os.path.join(output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Write summary text file
    with open(os.path.join(summary_dir, "summary.txt"), "w") as f:
        f.write(f"Editing Analysis Summary (Size Factor: {size_factor})\n")
        f.write("=" * 50 + "\n\n")
        
        for edit_mode, mode_results in results.items():
            f.write(f"{edit_mode.upper()} EDITING MODE\n")
            f.write("-" * 30 + "\n")
            f.write(f"Status: {mode_results.get('status', 'Unknown')}\n\n")
    
    print(f"Summary report saved to {summary_dir}")

if __name__ == "__main__":
    main() 