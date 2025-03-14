import os
import torch
import numpy as np
from tqdm import tqdm
import pickle

from analysis.metrics.trajectory_metrics import compute_trajectory_metrics, visualize_metrics

class TrajectoryManager:
    """
    Class to manage diffusion trajectories for analysis
    """
    def __init__(self, teacher_model, student_model, config, size_factor=1.0, fixed_samples=None):
        """
        Initialize the trajectory manager
        
        Args:
            teacher_model: The teacher diffusion model
            student_model: The student diffusion model
            config: Configuration object
            size_factor: Size factor of the student model
            fixed_samples: Fixed samples to use for trajectory generation (for consistent comparison)
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.config = config
        self.size_factor = size_factor
        
        # Ensure fixed samples have the correct format
        if fixed_samples is not None:
            self.fixed_samples = self._ensure_tensor_compatibility(fixed_samples)
        else:
            self.fixed_samples = None
        
        # Create trajectory directory if it doesn't exist
        os.makedirs(config.trajectory_dir, exist_ok=True)
        
        # Set device
        self.device = next(teacher_model.parameters()).device
    
    def _ensure_tensor_compatibility(self, tensor_or_batch):
        """
        Ensure tensor is in the correct format expected by diffusion models.
        No longer reshapes tensors for older PyTorch versions.
        
        Args:
            tensor_or_batch: A tensor or batch of tensors
            
        Returns:
            Tensor with the correct format
        """
        if tensor_or_batch is None:
            return None
        
        if not isinstance(tensor_or_batch, torch.Tensor):
            print(f"Warning: Input is not a tensor but {type(tensor_or_batch)}")
            return tensor_or_batch
        
        # Print shape for debugging
        print(f"Tensor shape: {tensor_or_batch.shape}")
        
        # Return the tensor as-is without reshaping
        return tensor_or_batch
    
    def generate_trajectory(self, seed=None):
        """
        Generate a single trajectory pair (teacher and student)
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            teacher_trajectory: List of (image, timestep) pairs for teacher
            student_trajectory: List of (image, timestep) pairs for student
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Set models to eval mode
        self.teacher_model.eval()
        self.student_model.eval()
        
        # Generate random noise for both teacher and student
        x_teacher = torch.randn(1, self.config.channels, self.config.image_size, self.config.image_size).to(self.device)
        
        # Calculate step size to evenly distribute the timesteps across sample steps
        teacher_step_size = self.config.sample_steps // self.config.teacher_steps
        
        # Generate teacher trajectory - standard denoising process (noise to clean)
        teacher_trajectory = []
        with torch.no_grad():
            # Select the timesteps to use (evenly spaced)
            teacher_timestep_indices = [i * teacher_step_size for i in range(self.config.teacher_steps)]
            if teacher_timestep_indices[-1] != self.config.sample_steps - 1:
                teacher_timestep_indices.append(self.config.sample_steps - 1)  # Ensure we include the last step
            
            for t in reversed(teacher_timestep_indices):
                t_tensor = torch.tensor([t], device=self.device)
                
                # Store current state
                teacher_trajectory.append((x_teacher.clone(), t))
                
                # Predict noise
                noise_pred = self.teacher_model(x_teacher, t_tensor)
                
                # Update x
                if t > 0:
                    # Sample noise for the next step
                    noise = torch.randn_like(x_teacher)
                    x_teacher = self._update_x(x_teacher, noise_pred, t, noise)
        
        # Reset to the same starting noise for student
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Generate random noise for student (may have different size)
        # Check if student model has a different image size
        student_image_size = self.config.image_size
        if hasattr(self.student_model, 'image_size'):
            student_image_size = self.student_model.image_size
        
        x_student = torch.randn(1, self.config.channels, student_image_size, student_image_size).to(self.device)
        
        # Calculate step size to evenly distribute the timesteps across sample steps
        student_step_size = self.config.sample_steps // self.config.student_steps
        
        # Generate student trajectory - standard denoising process (noise to clean)
        student_trajectory = []
        with torch.no_grad():
            # Select the timesteps to use (evenly spaced)
            student_timestep_indices = [i * student_step_size for i in range(self.config.student_steps)]
            if student_timestep_indices[-1] != self.config.sample_steps - 1:
                student_timestep_indices.append(self.config.sample_steps - 1)  # Ensure we include the last step
            
            for t in reversed(student_timestep_indices):
                t_tensor = torch.tensor([t], device=self.device)
                
                # Store current state
                student_trajectory.append((x_student.clone(), t))
                
                # Predict noise
                noise_pred = self.student_model(x_student, t_tensor)
                
                # Update x
                if t > 0:
                    # Sample noise for the next step
                    noise = torch.randn_like(x_student)
                    x_student = self._update_x(x_student, noise_pred, t, noise)
        
        # Resize student trajectory images to match teacher size if needed
        if student_image_size != self.config.image_size:
            resized_student_trajectory = []
            for img, t in student_trajectory:
                resized_img = torch.nn.functional.interpolate(
                    img, 
                    size=(self.config.image_size, self.config.image_size),
                    mode='bilinear', 
                    align_corners=True
                )
                resized_student_trajectory.append((resized_img, t))
            student_trajectory = resized_student_trajectory
        
        return teacher_trajectory, student_trajectory
    
    def _update_x(self, x, noise_pred, t, noise):
        """
        Update x for the next step in the diffusion process
        
        Args:
            x: Current image
            noise_pred: Predicted noise
            t: Current timestep
            noise: Random noise for the next step
            
        Returns:
            Updated x
        """
        # Simple implementation - can be replaced with more sophisticated methods
        alpha = 0.9  # Placeholder - should be calculated based on beta schedule
        beta = 1 - alpha
        
        # Ensure noise_pred has the same shape as x
        if noise_pred.shape != x.shape:
            # Resize noise_pred to match x
            noise_pred = torch.nn.functional.interpolate(
                noise_pred, 
                size=x.shape[2:],
                mode='bilinear', 
                align_corners=True
            )
        
        # Update x - convert alpha to tensor to avoid TypeError
        alpha_tensor = torch.tensor(alpha, device=x.device)
        x = (x - beta * noise_pred) / torch.sqrt(alpha_tensor)
        
        # Add noise scaled by the timestep
        # Convert t to float to avoid division issues
        t_float = float(t)
        teacher_steps_float = float(self.config.teacher_steps)
        noise_scale = 0.1 * (t_float / teacher_steps_float)
        x = x + noise_scale * noise
        
        return x
    
    def generate_and_save_trajectories(self, num_samples=10):
        """
        Generate and save multiple trajectory pairs
        
        Args:
            num_samples: Number of trajectory pairs to generate
            
        Returns:
            List of file paths where trajectories are saved
        """
        file_paths = []
        
        # If fixed samples are provided, use them instead of generating random ones
        if self.fixed_samples is not None and num_samples <= len(self.fixed_samples):
            print(f"Using {num_samples} fixed samples for consistent comparison")
            samples_to_use = self.fixed_samples[:num_samples]
            
            for i, sample in enumerate(tqdm(samples_to_use, desc="Generating trajectories from fixed samples")):
                # Generate trajectory using the fixed sample
                try:
                    teacher_traj, student_traj = self.generate_trajectory_from_sample(sample, i)
                except Exception as e:
                    print(f"Error generating trajectory {i} from fixed sample: {e}")
                    continue
                
                # Save trajectory
                file_path = os.path.join(
                    self.config.trajectory_dir, 
                    f"trajectory_size_{self.size_factor}_sample_{i}.pkl"
                )
                
                with open(file_path, 'wb') as f:
                    pickle.dump((teacher_traj, student_traj), f)
                
                file_paths.append(file_path)
        else:
            # Use random seeds if no fixed samples or not enough fixed samples
            for i in tqdm(range(num_samples), desc="Generating trajectories"):
                # Generate trajectory
                try:
                    teacher_traj, student_traj = self.generate_trajectory(seed=i)
                except Exception as e:
                    print(f"Error generating trajectory {i}: {e}")
                    continue
                
                # Save trajectory
                file_path = os.path.join(
                    self.config.trajectory_dir, 
                    f"trajectory_size_{self.size_factor}_sample_{i}.pkl"
                )
                
                with open(file_path, 'wb') as f:
                    pickle.dump((teacher_traj, student_traj), f)
                
                file_paths.append(file_path)
        
        return file_paths
    
    def generate_trajectory_from_sample(self, sample, seed=None):
        """
        Generate a trajectory pair starting from a fixed sample
        
        Args:
            sample: Fixed sample to start from (should be noise)
            seed: Random seed for reproducibility
            
        Returns:
            teacher_trajectory: List of (image, timestep) pairs for teacher
            student_trajectory: List of (image, timestep) pairs for student
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Set models to eval mode
        self.teacher_model.eval()
        self.student_model.eval()
        
        # Ensure the sample has the correct dimensions (4D: batch, channels, height, width)
        sample = self._ensure_tensor_compatibility(sample)
        
        # Use the provided sample as the starting point (should be noise)
        x_teacher = sample.clone().to(self.device)
        
        # Log information about the sample
        print(f"Sample shape for teacher: {x_teacher.shape}")
        
        # Calculate step size to evenly distribute the timesteps across sample steps
        teacher_step_size = self.config.sample_steps // self.config.teacher_steps
        
        # Generate teacher trajectory - standard denoising process (noise to clean)
        teacher_trajectory = []
        try:
            with torch.no_grad():
                # Select the timesteps to use (evenly spaced)
                teacher_timestep_indices = [i * teacher_step_size for i in range(self.config.teacher_steps)]
                if teacher_timestep_indices[-1] != self.config.sample_steps - 1:
                    teacher_timestep_indices.append(self.config.sample_steps - 1)  # Ensure we include the last step
                
                for t in reversed(teacher_timestep_indices):
                    t_tensor = torch.tensor([t], device=self.device)
                    
                    # Store current state
                    teacher_trajectory.append((x_teacher.clone(), t))
                    
                    # Predict noise
                    noise_pred = self.teacher_model(x_teacher, t_tensor)
                    
                    # Update x
                    if t > 0:
                        # Sample noise for the next step
                        noise = torch.randn_like(x_teacher)
                        x_teacher = self._update_x(x_teacher, noise_pred, t, noise)
        except Exception as e:
            print(f"Error generating teacher trajectory: {e}")
            return [], []
        
        # Use the provided sample as the starting point for student
        # Check if student model has a different image size
        student_image_size = self.config.image_size
        if hasattr(self.student_model, 'image_size'):
            student_image_size = self.student_model.image_size
        
        # Resize sample if needed
        if sample.shape[2] != student_image_size or sample.shape[3] != student_image_size:
            x_student = torch.nn.functional.interpolate(
                sample.clone(), 
                size=(student_image_size, student_image_size),
                mode='bilinear', 
                align_corners=True
            ).to(self.device)
        else:
            x_student = sample.clone().to(self.device)
        
        # Log information about the sample for the student
        print(f"Sample shape for student: {x_student.shape}")
        
        # Calculate step size to evenly distribute the timesteps across sample steps
        student_step_size = self.config.sample_steps // self.config.student_steps
        
        # Generate student trajectory - standard denoising process (noise to clean)
        student_trajectory = []
        try:
            with torch.no_grad():
                # Select the timesteps to use (evenly spaced)
                student_timestep_indices = [i * student_step_size for i in range(self.config.student_steps)]
                if student_timestep_indices[-1] != self.config.sample_steps - 1:
                    student_timestep_indices.append(self.config.sample_steps - 1)  # Ensure we include the last step
                
                for t in reversed(student_timestep_indices):
                    t_tensor = torch.tensor([t], device=self.device)
                    
                    # Store current state
                    student_trajectory.append((x_student.clone(), t))
                    
                    # Predict noise
                    noise_pred = self.student_model(x_student, t_tensor)
                    
                    # Update x
                    if t > 0:
                        # Sample noise for the next step
                        noise = torch.randn_like(x_student)
                        x_student = self._update_x(x_student, noise_pred, t, noise)
        except Exception as e:
            print(f"Error generating student trajectory: {e}")
            return teacher_trajectory, []
        
        # Resize student trajectory images to match teacher size if needed
        if student_image_size != self.config.image_size:
            resized_student_trajectory = []
            for img, t in student_trajectory:
                resized_img = torch.nn.functional.interpolate(
                    img, 
                    size=(self.config.image_size, self.config.image_size),
                    mode='bilinear', 
                    align_corners=True
                )
                resized_student_trajectory.append((resized_img, t))
            student_trajectory = resized_student_trajectory
        
        return teacher_trajectory, student_trajectory
    
    def load_trajectories(self, size_factor=None, indices=None):
        """
        Load saved trajectories
        
        Args:
            size_factor: Size factor to load (if None, use self.size_factor)
            indices: List of indices to load (if None, load all)
            
        Returns:
            teacher_trajectories: List of teacher trajectories
            student_trajectories: List of student trajectories
        """
        if size_factor is None:
            size_factor = self.size_factor
        
        # Find all trajectory files for this size factor
        trajectory_files = [
            f for f in os.listdir(self.config.trajectory_dir)
            if f.startswith(f"trajectory_size_{size_factor}_sample_") and f.endswith(".pkl")
        ]
        
        # Sort by sample index
        trajectory_files.sort(key=lambda x: int(x.split("_sample_")[1].split(".")[0]))
        
        # Filter by indices if provided
        if indices is not None:
            trajectory_files = [
                f for f in trajectory_files
                if int(f.split("_sample_")[1].split(".")[0]) in indices
            ]
        
        # Load trajectories
        teacher_trajectories = []
        student_trajectories = []
        
        for file_name in trajectory_files:
            file_path = os.path.join(self.config.trajectory_dir, file_name)
            
            with open(file_path, 'rb') as f:
                teacher_traj, student_traj = pickle.load(f)
                teacher_trajectories.append(teacher_traj)
                student_trajectories.append(student_traj)
        
        return teacher_trajectories, student_trajectories
    
    def compute_trajectory_metrics_batch(self, size_factor=None, batch_size=10):
        """
        Compute metrics for trajectories in batches to save memory
        
        Args:
            size_factor: Size factor to load (if None, use self.size_factor)
            batch_size: Number of trajectories to process at once
            
        Returns:
            metrics: Dictionary of metrics
        """
        if size_factor is None:
            size_factor = self.size_factor
        
        # Find all trajectory files for this size factor
        trajectory_files = [
            f for f in os.listdir(self.config.trajectory_dir)
            if f.startswith(f"trajectory_size_{size_factor}_sample_") and f.endswith(".pkl")
        ]
        
        # Sort by sample index
        trajectory_files.sort(key=lambda x: int(x.split("_sample_")[1].split(".")[0]))
        
        # Initialize metrics with our new metrics
        all_metrics = {
            # Original metrics
            'wasserstein_distances': [],
            'wasserstein_distances_per_timestep': [],
            'endpoint_distances': [],
            'teacher_path_lengths': [],
            'student_path_lengths': [],
            'teacher_efficiency': [],
            'student_efficiency': [],
            
            # New metrics
            'path_length_similarity': [],
            'efficiency_similarity': [],
            'mean_velocity_similarity': [],
            'mean_directional_consistency': [],
            'mean_position_difference': [],
            'distribution_similarity': [],
            
            'architecture_type': []
        }
        
        # Process in batches
        for i in range(0, len(trajectory_files), batch_size):
            batch_files = trajectory_files[i:i+batch_size]
            
            # Load batch
            teacher_trajectories = []
            student_trajectories = []
            
            for file_name in batch_files:
                file_path = os.path.join(self.config.trajectory_dir, file_name)
                
                with open(file_path, 'rb') as f:
                    teacher_traj, student_traj = pickle.load(f)
                    teacher_trajectories.append(teacher_traj)
                    student_trajectories.append(student_traj)
            
            # Process each trajectory pair separately
            for t_traj, s_traj in zip(teacher_trajectories, student_trajectories):
                # Compute metrics for this trajectory pair
                metrics = compute_trajectory_metrics(t_traj, s_traj, self.config)
                
                # Add original metrics
                all_metrics['wasserstein_distances'].append(metrics['mean_wasserstein'])
                all_metrics['wasserstein_distances_per_timestep'].append(metrics['wasserstein_distances'])
                all_metrics['endpoint_distances'].append(metrics['endpoint_distance'])
                all_metrics['teacher_path_lengths'].append(metrics['teacher_path_length'])
                all_metrics['student_path_lengths'].append(metrics['student_path_length'])
                all_metrics['teacher_efficiency'].append(metrics['teacher_efficiency'])
                all_metrics['student_efficiency'].append(metrics['student_efficiency'])
                
                # Add new metrics
                if 'path_length_similarity' in metrics:
                    all_metrics['path_length_similarity'].append(metrics['path_length_similarity'])
                
                if 'efficiency_similarity' in metrics:
                    all_metrics['efficiency_similarity'].append(metrics['efficiency_similarity'])
                
                if 'mean_velocity_similarity' in metrics:
                    all_metrics['mean_velocity_similarity'].append(metrics['mean_velocity_similarity'])
                
                if 'mean_directional_consistency' in metrics:
                    all_metrics['mean_directional_consistency'].append(metrics['mean_directional_consistency'])
                
                if 'mean_position_difference' in metrics:
                    all_metrics['mean_position_difference'].append(metrics['mean_position_difference'])
                
                if 'distribution_similarity' in metrics:
                    all_metrics['distribution_similarity'].append(metrics['distribution_similarity'])
                
                # Add architecture type if available
                if hasattr(self, 'architecture_type'):
                    all_metrics['architecture_type'].append(self.architecture_type)
        
        # Compute averages for scalar metrics
        # Include all original metrics for backward compatibility
        for key in ['endpoint_distances', 'teacher_path_lengths', 'student_path_lengths', 
                   'teacher_efficiency', 'student_efficiency', 'wasserstein_distances']:
            if key in all_metrics and all_metrics[key]:
                all_metrics[key + '_avg'] = sum(all_metrics[key]) / len(all_metrics[key])
        
        # Compute averages for new metrics
        for key in ['path_length_similarity', 'efficiency_similarity', 'mean_velocity_similarity', 
                   'mean_directional_consistency', 'mean_position_difference', 'distribution_similarity']:
            if key in all_metrics and all_metrics[key]:
                all_metrics[key + '_avg'] = sum(all_metrics[key]) / len(all_metrics[key])
                # Also save with simpler names for compatibility with radar plots
                if key + '_avg' not in all_metrics:
                    all_metrics[key] = all_metrics[key + '_avg']
        
        return all_metrics

def generate_trajectories_with_disk_storage(teacher_model, student_model, config, size_factor=1.0, num_samples=10, fixed_samples=None):
    """
    Generate trajectories and store them on disk
    
    Args:
        teacher_model: The teacher diffusion model
        student_model: The student diffusion model
        config: Configuration object
        size_factor: Size factor of the student model
        num_samples: Number of trajectory pairs to generate
        fixed_samples: Fixed samples to use for trajectory generation (for consistent comparison)
        
    Returns:
        trajectory_manager: TrajectoryManager object
    """
    # Create trajectory manager without reshaping tensors
    trajectory_manager = TrajectoryManager(teacher_model, student_model, config, size_factor, fixed_samples)
    
    # Check if trajectories already exist
    existing_files = [
        f for f in os.listdir(config.trajectory_dir)
        if f.startswith(f"trajectory_size_{size_factor}_sample_") and f.endswith(".pkl")
    ]
    
    # Generate trajectories if needed
    if len(existing_files) < num_samples:
        print(f"Generating {num_samples - len(existing_files)} new trajectories...")
        trajectory_manager.generate_and_save_trajectories(num_samples - len(existing_files))
    else:
        print(f"Using {num_samples} existing trajectories...")
    
    return trajectory_manager
