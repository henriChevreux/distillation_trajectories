import os
import h5py
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

class TrajectoryManager:
    """
    Handles efficient storage and retrieval of diffusion model trajectories.
    This class manages trajectories on disk to minimize memory usage during analysis.
    """
    def __init__(self, config):
        """
        Initialize the trajectory manager.
        
        Args:
            config: Configuration object with trajectory_dir attribute
        """
        self.trajectory_dir = config.trajectory_dir
        os.makedirs(self.trajectory_dir, exist_ok=True)
        self.h5_file_path = os.path.join(self.trajectory_dir, 'trajectories.h5')
        self.config = config
        
    def store_trajectories(self, teacher_trajectories, student_trajectories, size_factor=None):
        """
        Store trajectories on disk using HDF5 format.
        
        Args:
            teacher_trajectories: List of teacher model trajectories
            student_trajectories: List of student model trajectories
            size_factor: Size factor of the student model (for organizing data)
        """
        # Append to existing file if it exists, create new otherwise
        mode = 'a' if os.path.exists(self.h5_file_path) else 'w'
        
        # Define group name based on size factor
        group_name = f"size_{size_factor}" if size_factor is not None else "default"
        
        with h5py.File(self.h5_file_path, mode) as f:
            # Create a group for this set of trajectories
            if group_name in f:
                del f[group_name]  # Remove if exists to avoid conflicts
            group = f.create_group(group_name)
            
            # Store metadata
            group.attrs['teacher_steps'] = self.config.teacher_steps
            group.attrs['student_steps'] = self.config.student_steps
            group.attrs['num_samples'] = len(teacher_trajectories)
            if size_factor is not None:
                group.attrs['size_factor'] = size_factor
            
            # Store trajectories
            teacher_group = group.create_group('teacher')
            student_group = group.create_group('student')
            
            # Process each trajectory sample
            for i, (teacher_traj, student_traj) in enumerate(zip(teacher_trajectories, student_trajectories)):
                # Convert teacher trajectory to numpy array and store
                teacher_data = np.stack([t[0].cpu().numpy() for t in teacher_traj])
                teacher_group.create_dataset(f'sample_{i}', data=teacher_data, compression='gzip')
                
                # Convert student trajectory to numpy array and store
                student_data = np.stack([s[0].cpu().numpy() for s in student_traj])
                student_group.create_dataset(f'sample_{i}', data=student_data, compression='gzip')
            
            print(f"Stored {len(teacher_trajectories)} trajectories in {self.h5_file_path} under group '{group_name}'")
    
    def load_trajectories(self, size_factor=None, indices=None):
        """
        Load trajectories from disk.
        
        Args:
            size_factor: Size factor to load (if None, loads 'default' group)
            indices: List of specific trajectory indices to load (if None, loads all)
            
        Returns:
            teacher_trajectories, student_trajectories
        """
        group_name = f"size_{size_factor}" if size_factor is not None else "default"
        
        with h5py.File(self.h5_file_path, 'r') as f:
            if group_name not in f:
                raise KeyError(f"No trajectories found for size factor {size_factor}")
            
            group = f[group_name]
            teacher_group = group['teacher']
            student_group = group['student']
            
            # Determine which samples to load
            if indices is None:
                sample_keys = [key for key in teacher_group.keys()]
            else:
                sample_keys = [f'sample_{i}' for i in indices]
            
            # Load the selected trajectories
            teacher_trajectories = []
            student_trajectories = []
            
            for key in sample_keys:
                teacher_data = teacher_group[key][()]
                student_data = student_group[key][()]
                
                # Convert numpy arrays to PyTorch tensors (list of tensors)
                teacher_traj = [torch.from_numpy(img).unsqueeze(0) for img in teacher_data]
                student_traj = [torch.from_numpy(img).unsqueeze(0) for img in student_data]
                
                teacher_trajectories.append(teacher_traj)
                student_trajectories.append(student_traj)
            
            print(f"Loaded {len(teacher_trajectories)} trajectories from {group_name}")
            return teacher_trajectories, student_trajectories
    
    def compute_trajectory_metrics_batch(self, size_factor=None, batch_size=5):
        """
        Compute trajectory metrics in batches to minimize memory usage.
        
        Args:
            size_factor: Size factor to analyze
            batch_size: Number of trajectories to process at once
            
        Returns:
            metrics: Dictionary of metrics
        """
        from scipy.stats import wasserstein_distance
        
        group_name = f"size_{size_factor}" if size_factor is not None else "default"
        
        # Initialize metrics dictionary
        metrics = {
            'wasserstein_distances': [],
            'wasserstein_distances_per_timestep': [],
            'endpoint_distances': [],
            'teacher_path_lengths': [],
            'student_path_lengths': [],
            'teacher_efficiency': [],
            'student_efficiency': [],
            'architecture_type': []
        }
        
        with h5py.File(self.h5_file_path, 'r') as f:
            if group_name not in f:
                raise KeyError(f"No trajectories found for size factor {size_factor}")
            
            group = f[group_name]
            teacher_group = group['teacher']
            student_group = group['student']
            
            num_samples = group.attrs['num_samples']
            
            # Process in batches
            num_batches = (num_samples + batch_size - 1) // batch_size
            
            # Define a batch processor function
            def process_batch(batch_idx):
                batch_metrics = {
                    'wasserstein_distances': [],
                    'wasserstein_distances_per_timestep': [],
                    'endpoint_distances': [],
                    'teacher_path_lengths': [],
                    'student_path_lengths': [],
                    'teacher_efficiency': [],
                    'student_efficiency': [],
                    'architecture_type': []
                }
                
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                
                for i in range(start_idx, end_idx):
                    sample_key = f'sample_{i}'
                    
                    # Load individual trajectory pair
                    teacher_data = teacher_group[sample_key][()]
                    student_data = student_group[sample_key][()]
                    
                    # Flatten trajectories
                    teacher_flat = [t.reshape(-1) for t in teacher_data]
                    student_flat = [s.reshape(-1) for s in student_data]
                    
                    # Compute metrics for this pair
                    batch_metrics = self._compute_metrics_for_pair(
                        teacher_flat, student_flat, batch_metrics, size_factor
                    )
                
                return batch_metrics
            
            # Process batches in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                batch_results = list(tqdm(
                    executor.map(process_batch, range(num_batches)),
                    total=num_batches,
                    desc="Processing trajectory batches"
                ))
            
            # Combine batch results
            for batch_result in batch_results:
                for key in metrics:
                    metrics[key].extend(batch_result[key])
        
        return metrics
    
    def _compute_metrics_for_pair(self, teacher_flat, student_flat, metrics, size_factor):
        """Helper method to compute metrics for a single trajectory pair"""
        from scipy.stats import wasserstein_distance
        from scipy.interpolate import interp1d
        import numpy as np
        
        # Determine architecture type based on size factor
        if float(size_factor) < 0.1:
            architecture_type = 'tiny'
        elif float(size_factor) < 0.3:
            architecture_type = 'small'
        elif float(size_factor) < 0.7:
            architecture_type = 'medium'
        else:
            architecture_type = 'full'
        
        metrics['architecture_type'].append(architecture_type)
        
        # IMPROVED: Compute Wasserstein distance between trajectories properly
        # We need to interpolate the student trajectory to match the teacher's timesteps
        if len(teacher_flat) != len(student_flat):
            # Create interpolation function for student trajectory
            student_timesteps = np.linspace(0, 1, len(student_flat))
            teacher_timesteps = np.linspace(0, 1, len(teacher_flat))
            
            # Vectorized interpolation with cubic spline for better accuracy
            student_values = np.array([s for s in student_flat])
            student_interp = np.zeros((len(teacher_timesteps), student_values.shape[1]))
            
            # Create interpolation function once - use cubic spline for smoother interpolation
            for dim in range(0, student_values.shape[1], 1000):  # Process in chunks to avoid memory issues
                end_dim = min(dim + 1000, student_values.shape[1])
                chunk_values = student_values[:, dim:end_dim]
                # Use cubic interpolation for smoother results when possible
                if len(student_timesteps) > 3:  # Cubic requires at least 4 points
                    interp_func = interp1d(student_timesteps, chunk_values, axis=0, kind='cubic')
                else:
                    interp_func = interp1d(student_timesteps, chunk_values, axis=0, kind='linear')
                student_interp[:, dim:end_dim] = interp_func(teacher_timesteps)
            
            # Compute Wasserstein distance per timestep and average
            w_dists = []
            for t in range(len(teacher_timesteps)):
                w_dist_t = wasserstein_distance(teacher_flat[t], student_interp[t])
                w_dists.append(w_dist_t)
            
            # Average Wasserstein distance across timesteps
            w_dist = np.mean(w_dists)
            w_dists_per_timestep = w_dists
        else:
            # If same number of timesteps, compute Wasserstein distance per timestep
            w_dists = []
            for t in range(len(teacher_flat)):
                w_dist_t = wasserstein_distance(teacher_flat[t], student_flat[t])
                w_dists.append(w_dist_t)
            
            # Average Wasserstein distance across timesteps
            w_dist = np.mean(w_dists)
            w_dists_per_timestep = w_dists
        
        # Add Wasserstein metrics
        metrics['wasserstein_distances'].append(w_dist)
        metrics['wasserstein_distances_per_timestep'].append(w_dists_per_timestep)
        
        # Compute endpoint distance (L2 norm between final images)
        endpoint_dist = np.linalg.norm(teacher_flat[-1] - student_flat[-1])
        metrics['endpoint_distances'].append(endpoint_dist)
        
        # Compute path length using vectorized operations
        teacher_diffs = np.array(teacher_flat[1:]) - np.array(teacher_flat[:-1])
        student_diffs = np.array(student_flat[1:]) - np.array(student_flat[:-1])
        
        teacher_path_length = np.sum(np.sqrt(np.sum(teacher_diffs**2, axis=1)))
        student_path_length = np.sum(np.sqrt(np.sum(student_diffs**2, axis=1)))
        
        metrics['teacher_path_lengths'].append(teacher_path_length)
        metrics['student_path_lengths'].append(student_path_length)
        
        # Compute path efficiency (endpoint distance / path length)
        # This measures how direct the path is (higher is more efficient)
        start_end_dist_teacher = np.linalg.norm(teacher_flat[-1] - teacher_flat[0])
        start_end_dist_student = np.linalg.norm(student_flat[-1] - student_flat[0])
        
        teacher_efficiency = start_end_dist_teacher / teacher_path_length if teacher_path_length > 0 else 0
        student_efficiency = start_end_dist_student / student_path_length if student_path_length > 0 else 0
        
        metrics['teacher_efficiency'].append(teacher_efficiency)
        metrics['student_efficiency'].append(student_efficiency)
        
        return metrics


# Define function to generate trajectories with disk storage (to be imported in diffusion_analysis.py)
def generate_trajectories_with_disk_storage(teacher_model, student_model, config, size_factor=None, num_samples=10):
    """
    Generate trajectories for both teacher and student models and store them on disk.
    
    Args:
        teacher_model: The teacher diffusion model
        student_model: The student diffusion model
        config: Configuration object
        size_factor: Size factor of the student model (for organizing data)
        num_samples: Number of samples to generate
    
    Returns:
        trajectory_manager: TrajectoryManager instance with stored trajectories
    """
    # Import required functions from diffusion_analysis or diffusion_training
    from diffusion_training import get_diffusion_params, p_sample_loop
    
    # Initialize TrajectoryManager
    trajectory_manager = TrajectoryManager(config)
    
    # Get diffusion parameters
    teacher_params = get_diffusion_params(config.teacher_steps)
    student_params = get_diffusion_params(config.student_steps)
    
    # Prepare shape tuple for reuse
    shape = (1, config.channels, config.image_size, config.image_size)
    
    # Generate trajectories in small batches to avoid memory issues
    batch_size = 5  # Process 5 trajectories at a time
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        batch_size_actual = end_idx - start_idx
        
        print(f"Generating batch {batch_idx+1}/{num_batches} with {batch_size_actual} trajectories...")
        
        # Generate trajectories for this batch
        teacher_trajectories = []
        student_trajectories = []
        
        for i in tqdm(range(start_idx, end_idx)):
            # Use the same random seed for both models to start from the same noise
            torch.manual_seed(i)
            
            # Generate teacher trajectory
            _, teacher_traj = p_sample_loop(
                teacher_model,
                shape=shape,
                timesteps=config.teacher_steps,
                diffusion_params=teacher_params,
                track_trajectory=True
            )
            
            # Reset seed to use the same initial noise
            torch.manual_seed(i)
            
            # Generate student trajectory
            _, student_traj = p_sample_loop(
                student_model,
                shape=shape,
                timesteps=config.student_steps,
                diffusion_params=student_params,
                track_trajectory=True
            )
            
            teacher_trajectories.append(teacher_traj)
            student_trajectories.append(student_traj)
        
        # Store this batch of trajectories to disk
        trajectory_manager.store_trajectories(teacher_trajectories, student_trajectories, size_factor)
        
        # Clear memory
        del teacher_trajectories
        del student_trajectories
        import gc
        gc.collect()  # Force garbage collection
    
    return trajectory_manager