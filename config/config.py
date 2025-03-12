import os
import torchvision
from torchvision import transforms

class Config:
    """Configuration for diffusion model training and analysis"""
    def __init__(self):
        # ======= Teacher Training Parameters =======
        # Dataset Configuration
        self.dataset = "AFHQ"  # Using AFHQ dataset
        self.afhq_category = "wild"  # Using wildlife section
        self.high_res_category = "bedroom_train"
        self.teacher_image_size = 256  # Memory-efficient size
        self.image_size = self.teacher_image_size  # Alias for compatibility
        self.channels = 3
        self.batch_size = 8  # Increased from 2 to 8 with reduced model capacity

        # Training Hyperparameters
        self.epochs = 10
        self.lr = 1e-4
        self.timesteps = 50  # Number of diffusion steps
        self.save_interval = 1  # Save model every N epochs
        
        # Model Architecture (Used by SimpleUNet)
        # Reduced capacity configuration for 256x256:
        # - Reduced initial features (64)
        # - 4 downsampling layers
        # - Progressive channel growth
        self.latent_dim = 64  # Reduced from 128
        self.hidden_dims = [64, 128, 256, 512]  # More gradual channel growth
        
        # Diffusion Process Parameters
        self.beta_start = 1e-4
        self.beta_end = 0.02

        # Progress Bar Configuration
        self.progress_bar_leave = False
        self.progress_bar_position = 0
        self.progress_bar_ncols = 100
        
        # Sampling & Visualization
        self.num_samples_to_generate = 16
        self.samples_grid_size = 4
        self.samples_figure_size = (10, 10)
        
        # Hardware
        self.mps_enabled = False  # Apple Silicon GPU support

        # ======= Student Training Parameters =======
        # These are not used in train_teacher.py
        self.student_image_size = 128
        self.student_image_size_factors = [1.0, 0.5, 0.25, 0.125, 0.0625]  # 256->32, 128->16, 64->8
        self.student_size_factors = [0.01, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        
        # Base channel configurations for different model sizes
        self.student_base_channels = {
            'tiny': 32,    # 0.25x of full
            'small': 48,   # 0.375x of full
            'medium': 64,  # 0.5x of full
            'large': 96,   # 0.75x of full
            'full': 128    # 1x - matches teacher
        }
        
        # Maximum number of downsampling steps for each image size
        # Each downsampling reduces size by 2x
        self.max_downsample_steps = {
            256: 5,  # 256->128->64->32->16->8
            128: 4,  # 128->64->32->16->8
            64: 3,   # 64->32->16->8
            32: 2,   # 32->16->8
            16: 1,   # 16->8
            8: 1     # 8->4 (one downsampling step possible)
        }
        
        # Minimum number of layers for each image size
        self.min_layers = {
            256: 4,
            128: 3,
            64: 3,
            32: 2,
            16: 2,
            8: 2  # Input layer + 1 downsampling layer (8x8 -> 4x4)
        }
        
        self.noise_diversity_weight = 0.1
        self.student_steps = 50
        self.distill = True
        
        # ======= Analysis Parameters =======
        # These parameters can be adjusted for testing vs production analysis
        
        # TESTING PARAMETERS (for quick iterations)
        # Set to True to use fast testing settings, False for complete thorough analysis
        self.analysis_testing_mode = True
        
        # Number of samples to use for visualizations
        # Original: 5, Testing: 1
        self.analysis_num_samples = 1 if self.analysis_testing_mode else 5
        
        # Denoising specific settings
        # Always use 1 sample for denoising plot regardless of testing mode
        self.denoising_num_samples = 1
        
        # For MSE Heatmap analysis
        # TESTING: Use only smallest model, PRODUCTION: Use all models
        self.mse_test_mode = self.analysis_testing_mode
        # Number of models to use (original: all models, testing: just smallest)
        self.mse_size_factors_limit = False  # Use all size factors by default
        self.mse_image_size_factors_limit = False  # Use all image size factors by default
        # Number of timesteps to test in MSE calculation
        self.mse_test_timesteps = [800] if self.mse_test_mode else [200, 400, 600, 800, 999]
        # Number of test samples for MSE calculation
        self.mse_num_test_samples = 5 if self.mse_test_mode else 20
        
        # For Time-dependent analysis
        self.time_test_mode = self.analysis_testing_mode
        self.time_analysis_sample_count = 1 if self.time_test_mode else 5
        
        # For Plot settings (DPI and figure size)
        self.plot_highres_dpi = 200  # Original setting
        self.plot_lowres_dpi = 120   # For quick viewing
        self.plot_highres_fig_size = (20, 12)  # Original size for detailed plots
        self.plot_lowres_fig_size = (10, 6)    # Reduced size for quick plots
        
        # Resolution factors for denoising comparison
        # Original: All resolution factors in student_image_size_factors
        # Testing: Just a few resolution factors
        self.denoising_resolution_factors = [1.0, 0.25] if self.analysis_testing_mode else None

        # ======= Directory Structure =======
        # Base Directories
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(self.base_dir, "output")
        self.data_dir = os.path.join(self.base_dir, "data")
        
        # Model Directories (Used in train_teacher.py)
        self.models_dir = os.path.join(self.output_dir, "models")
        self.teacher_models_dir = os.path.join(self.models_dir, "teacher")
        self.results_dir = os.path.join(self.output_dir, "results")
        
        # Additional Directories (Not used in train_teacher.py)
        self.student_models_dir = os.path.join(self.models_dir, "students")
        self.trajectory_dir = os.path.join(self.data_dir, "trajectories")
        self.analysis_dir = os.path.join(self.output_dir, "analysis")
        
        # Raw Data and Results
        self.metrics_dir = os.path.join(self.analysis_dir, "metrics")
        self.metrics_raw_dir = os.path.join(self.metrics_dir, "raw")
        self.metrics_processed_dir = os.path.join(self.metrics_dir, "processed")
        
        # Visualizations
        self.viz_dir = os.path.join(self.analysis_dir, "visualizations")
        self.sample_viz_dir = os.path.join(self.viz_dir, "samples")
        self.trajectory_viz_dir = os.path.join(self.viz_dir, "trajectories")
        self.comparison_viz_dir = os.path.join(self.viz_dir, "comparisons")
        
        # Model Analysis
        self.model_analysis_dir = os.path.join(self.analysis_dir, "model_analysis")
        self.attention_dir = os.path.join(self.model_analysis_dir, "attention")
        self.noise_dir = os.path.join(self.model_analysis_dir, "noise")
        self.dimensionality_dir = os.path.join(self.model_analysis_dir, "dimensionality")
        
        # Comparative Analysis
        self.comparative_dir = os.path.join(self.analysis_dir, "comparative")
        self.time_analysis_dir = os.path.join(self.comparative_dir, "time_dependent")
        self.size_analysis_dir = os.path.join(self.comparative_dir, "size_dependent")
        self.denoising_analysis_dir = os.path.join(self.comparative_dir, "denoising")

    def create_directories(self):
        """Create necessary directories for saving results"""
        # Main directories
        main_dirs = [
            self.output_dir,
            self.results_dir,
            self.models_dir,
            self.teacher_models_dir,
            self.student_models_dir,
            self.data_dir,
            self.trajectory_dir,
            self.analysis_dir,
        ]
        
        # Analysis directories
        analysis_dirs = [
            # Metrics
            self.metrics_dir,
            self.metrics_raw_dir,
            self.metrics_processed_dir,
            
            # Visualizations
            self.viz_dir,
            self.sample_viz_dir,
            self.trajectory_viz_dir,
            self.comparison_viz_dir,
            
            # Model Analysis
            self.model_analysis_dir,
            self.attention_dir,
            self.noise_dir,
            self.dimensionality_dir,
            
            # Comparative Analysis
            self.comparative_dir,
            self.time_analysis_dir,
            self.size_analysis_dir,
            self.denoising_analysis_dir,
        ]
        
        # Create all directories
        all_dirs = main_dirs + analysis_dirs
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
        elif self.dataset.lower() == 'stl10':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            return torchvision.datasets.STL10(
                root=self.data_dir,
                split='test',  # or 'train' depending on needs
                download=True,
                transform=transform
            )
        elif self.dataset.lower() == 'lsun':
            transform = transforms.Compose([
                transforms.Resize((self.teacher_image_size, self.teacher_image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            # For LSUN, we use a subset of training data for testing since there's no explicit test set
            return torchvision.datasets.LSUN(
                root=self.data_dir,
                classes=[self.high_res_category],
                transform=transform
            )
        elif self.dataset.lower() == 'afhq':
            transform = transforms.Compose([
                transforms.Resize((self.teacher_image_size, self.teacher_image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            return torchvision.datasets.ImageFolder(
                root=os.path.join(self.data_dir, 'test'),
                transform=transform
            )
        else:
            raise ValueError(f"Dataset {self.dataset} not supported")

    def get_student_architecture(self, image_size, size_factor):
        """
        Determine the appropriate architecture based on image size and student size factor.
        
        Args:
            image_size: Input image size (assumed square)
            size_factor: Student model size factor (0.0 to 1.0)
            
        Returns:
            List of channel dimensions for each layer
        """
        # For 8x8 images, always use tiny architecture with one downsampling to 4x4
        if image_size == 8:
            base_channels = self.student_base_channels['tiny']
            return [base_channels, base_channels * 2]  # 8x8 -> 4x4
            
        # For larger images, determine model size based on size_factor
        if size_factor <= 0.1:
            base_channels = self.student_base_channels['tiny']
        elif size_factor <= 0.3:
            base_channels = self.student_base_channels['small']
        elif size_factor <= 0.5:
            base_channels = self.student_base_channels['medium']
        elif size_factor <= 0.8:
            base_channels = self.student_base_channels['large']
        else:
            base_channels = self.student_base_channels['full']
        
        # Generate architecture with appropriate number of layers
        max_steps = self.max_downsample_steps[image_size]
        min_layers = self.min_layers[image_size]
        num_layers = max(max_steps + 1, min_layers)
        
        architecture = []
        current_channels = base_channels
        for i in range(num_layers):
            architecture.append(current_channels)
            current_channels = min(current_channels * 2, 512)  # Cap at 512 channels
            
        return architecture
