import os
import torchvision
from torchvision import transforms

class Config:
    """Configuration for diffusion model training and analysis"""
    def __init__(self):
        # Dataset
        self.dataset = "CIFAR10"
        self.image_size = 32  # Updated to 32x32 for CIFAR10
        self.channels = 3
        self.batch_size = 128  # Updated to 128 as per specifications
        
        # Model
        self.latent_dim = 128  # Updated to 128 base channels
        self.hidden_dims = [128, 256, 256, 256]  # Updated to reflect [1, 2, 2, 2] channel multipliers
        self.dropout = 0.3  # Added dropout parameter
        self.num_res_blocks = 3  # Added number of residual blocks
        self.learn_sigma = True  # Added learn sigma parameter
        
        # Diffusion process
        self.sample_steps = 4000  # Total number of sample steps in the diffusion process
        self.timesteps = 50  # Number of discrete timesteps used in the diffusion process
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.noise_schedule = "cosine"  # Added noise schedule parameter
        
        # Training
        self.epochs = 10
        self.lr = 1e-4
        self.save_interval = 1
        self.adam_beta1 = 0.8  # Added Adam beta1 parameter
        self.adam_beta2 = 0.999  # Added Adam beta2 parameter
        self.ema_rate = 0.9999  # Added EMA rate parameter
        
        # Directories - organized by purpose
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Output directories
        self.output_dir = os.path.join(self.base_dir, "output")
        self.results_dir = os.path.join(self.output_dir, "results")
        
        # Model directories - organized by model type
        self.models_dir = os.path.join(self.output_dir, "models")
        self.teacher_models_dir = os.path.join(self.models_dir, "teacher")
        self.student_models_dir = os.path.join(self.models_dir, "students")
        
        # Data directories
        self.data_dir = os.path.join(self.base_dir, "data")
        self.trajectory_dir = os.path.join(self.data_dir, "trajectories")
        
        # Analysis directories - consolidated under analysis directory
        self.analysis_dir = os.path.join(self.output_dir, "analysis")
        
        # Analysis subdirectories - reorganized for clarity
        self.metrics_dir = os.path.join(self.analysis_dir, "metrics")
        self.model_comparisons_dir = os.path.join(self.analysis_dir, "model_comparisons")
        self.time_dependent_dir = os.path.join(self.analysis_dir, "time_dependent")
        self.size_dependent_dir = os.path.join(self.analysis_dir, "size_dependent")
        self.dimensionality_dir = os.path.join(self.analysis_dir, "dimensionality")
        self.latent_space_dir = os.path.join(self.analysis_dir, "latent_space")
        self.attention_dir = os.path.join(self.analysis_dir, "attention")
        self.noise_prediction_dir = os.path.join(self.analysis_dir, "noise_prediction")
        self.denoising_dir = os.path.join(self.analysis_dir, "denoising")
        self.fid_dir = os.path.join(self.analysis_dir, "fid")
        
        # Distillation
        self.distill = True
        self.teacher_steps = self.timesteps  # Using the same timesteps for both teacher and student
        self.student_steps = self.timesteps  # Using the same timesteps for both teacher and student
        
        # Student model size factors
        self.student_size_factors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        # Student model architecture variants - updated to match proper channel dimensions 
        self.student_architectures = {
            'tiny': [32, 64],           # 2 layers instead of 3
            'small': [32, 64, 128],     # 3 layers but smaller dimensions
            'medium': [48, 96, 192],    # 3 layers with 75% of teacher dimensions
            'full': [128, 256, 256, 256]      # Same as teacher - updated to match base channels of 128
        }
        
        # Progress bar configuration
        self.progress_bar_leave = False
        self.progress_bar_position = 0
        self.progress_bar_ncols = 100
        
        # Sampling configuration
        self.num_samples_to_generate = 16
        self.samples_grid_size = 4  # nrow parameter for torchvision.utils.make_grid
        self.samples_figure_size = (10, 10)  # figsize parameter for plt.figure
        
        # Training parameters
        self.noise_diversity_weight = 0.1  # Weight for noise diversity loss in distillation
        self.mps_enabled = False  # Whether to use MPS (Apple Silicon GPU) if available
    
    def create_directories(self):
        """Create necessary directories for saving results"""
        # Create main directories
        directories = [
            self.output_dir,
            self.results_dir,
            self.models_dir,
            self.teacher_models_dir,
            self.student_models_dir,
            self.data_dir,
            self.trajectory_dir,
        ]
        
        # Create analysis directory and subdirectories
        analysis_dirs = [
            self.analysis_dir,
            self.metrics_dir,
            self.model_comparisons_dir,
            self.time_dependent_dir,
            self.size_dependent_dir,
            self.dimensionality_dir,
            self.latent_space_dir,
            self.attention_dir,
            self.noise_prediction_dir,
            self.denoising_dir,
            self.fid_dir,
        ]
        
        # Create student model size directories
        student_size_dirs = []
        for size_factor in self.student_size_factors:
            size_dir = os.path.join(self.student_models_dir, f"size_{size_factor}")
            student_size_dirs.append(size_dir)
        
        # Create all directories
        all_dirs = directories + analysis_dirs + student_size_dirs
        for dir_path in all_dirs:
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"Created directory: {dir_path}")
            except Exception as e:
                print(f"Error creating directory {dir_path}: {e}")
        
        return self

    def get_test_dataset(self):
        """Get the test dataset for analysis"""
        if self.dataset.lower() == 'cifar10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            return torchvision.datasets.CIFAR10(
                root=self.data_dir,
                train=False,
                download=True,
                transform=transform
            )
        elif self.dataset.lower() == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            return torchvision.datasets.MNIST(
                root=self.data_dir,
                train=False,
                download=True,
                transform=transform
            )
        else:
            raise ValueError(f"Dataset {self.dataset} not supported")
