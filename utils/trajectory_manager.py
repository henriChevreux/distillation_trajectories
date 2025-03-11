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
    def __init__(self, teacher_model, student_model, config, size_factor=1.0):
        """
        Initialize the trajectory manager
        
        Args:
            teacher_model: The teacher diffusion model
            student_model: The student diffusion model
            config: Configuration object
            size_factor: Size factor of the student model
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.config = config
        self.size_factor = size_factor
        
        # Create trajectory directory if it doesn't exist
        os.makedirs(config.trajectory_dir, exist_ok=True)
        
        # Set device
        self.device = next(teacher_model.parameters()).device
    
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
        
        # Generate random noise
        x = torch.randn(1, self.config.channels, self.config.image_size, self.config.image_size).to(self.device)
        
        # Generate teacher trajectory
        teacher_trajectory = []
        with torch.no_grad():
            for t in range(self.config.teacher_steps - 1, -1, -1):
                t_tensor = torch.tensor([t], device=self.device)
                
                # Store current state
                teacher_trajectory.append((x.clone(), t))
                
                # Predict noise
                noise_pred = self.teacher_model(x, t_tensor)
                
                # Update x
                if t > 0:
                    # Sample noise for the next step
                    noise = torch.randn_like(x)
                    x = self._update_x(x, noise_pred, t, noise)
        
        # Reset to the same starting noise
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        x = torch.randn(1, self.config.channels, self.config.image_size, self.config.image_size).to(self.device)
        
        # Generate student trajectory
        student_trajectory = []
        with torch.no_grad():
            for t in range(self.config.student_steps - 1, -1, -1):
                t_tensor = torch.tensor([t], device=self.device)
                
                # Store current state
                student_trajectory.append((x.clone(), t))
                
                # Predict noise
                noise_pred = self.student_model(x, t_tensor)
                
                # Update x
                if t > 0:
                    # Sample noise for the next step
                    noise = torch.randn_like(x)
                    x = self._update_x(x, noise_pred, t, noise)
        
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
        
        # Initialize metrics
        all_metrics = {
            'wasserstein_distances': [],
            'wasserstein_distances_per_timestep': [],
            'endpoint_distances': [],
            'teacher_path_lengths': [],
            'student_path_lengths': [],
            'teacher_efficiency': [],
            'student_efficiency': [],
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
            
            # Compute metrics for this batch
            batch_metrics = compute_trajectory_metrics(teacher_trajectories, student_trajectories, self.config)
            
            # Append to all metrics
            for key in all_metrics:
                all_metrics[key].extend(batch_metrics[key])
        
        return all_metrics

def generate_trajectories_with_disk_storage(teacher_model, student_model, config, size_factor=1.0, num_samples=10):
    """
    Generate trajectories and store them on disk
    
    Args:
        teacher_model: The teacher diffusion model
        student_model: The student diffusion model
        config: Configuration object
        size_factor: Size factor of the student model
        num_samples: Number of trajectory pairs to generate
        
    Returns:
        trajectory_manager: TrajectoryManager object
    """
    # Create trajectory manager
    trajectory_manager = TrajectoryManager(teacher_model, student_model, config, size_factor)
    
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
