"""
Configuration file for the CTM project.
Contains all hyperparameters and settings for training and evaluation.
"""

import os
from dataclasses import dataclass
import torchvision
from torchvision import transforms

@dataclass
class Config:
    """Configuration class for CTM project"""
    
    def __init__(self):
        # Project structure
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(self.project_root, "output")
        self.models_dir = os.path.join(self.output_dir, "models")
        self.logs_dir = os.path.join(self.output_dir, "logs")
        self.results_dir = os.path.join(self.output_dir, "results")
        
        # Dataset configuration
        self.data_dir = os.path.join(self.project_root, "data")
        self.train_dir = os.path.join(self.data_dir, "train")
        self.test_dir = os.path.join(self.data_dir, "test")
        self.image_size = 32  # Size images will be resized to
        self.channels = 3  # RGB images
        self.batch_size = 128
        self.num_workers = 4
        
        # Model configuration
        self.base_channels = 64  # Base number of channels in UNet
        self.channel_multipliers = [1, 2, 4, 8]  # Channel multipliers for each UNet level
        self.num_res_blocks = 2  # Number of residual blocks per level
        self.attention_resolutions = [16]  # Resolutions at which to apply attention
        self.dropout = 0.1
        self.embedding_dim = 128  # Dimension of timestep embedding
        
        # Diffusion configuration
        self.timesteps = 1000
        self.beta_schedule = 'linear'  # Options: linear, cosine
        self.beta_start = 0.0001
        self.beta_end = 0.02
        
        # Training configuration
        self.epochs = 100
        self.lr = 2e-4
        self.weight_decay = 0.0
        self.ema_decay = 0.9999
        self.gradient_clip = 1.0
        self.seed = 42
        
        # CTM specific configuration
        self.ctm_size_factors = [0.2, 0.4, 0.6, 0.8, 1.0]
        self.ctm_trajectory_probs = [0.5, 0.7, 0.9]
        self.ctm_time_diffs = [0.3, 0.5, 0.7]
        self.early_stopping_patience = 10
        self.fid_evaluation_frequency = 5  # Calculate FID every N epochs
        self.fid_samples = 10000  # Number of samples for FID calculation
        
        # Evaluation configuration
        self.eval_batch_size = 256
        self.num_samples = 64  # Number of samples for visualization
        self.save_model_frequency = 10  # Save model every N epochs
        
        # Data transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.eval_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def create_directories(self):
        """Create necessary directories for the project"""
        directories = [
            self.output_dir,
            self.models_dir,
            self.logs_dir,
            self.results_dir,
            os.path.join(self.models_dir, "ctm"),
            os.path.join(self.models_dir, "teacher"),
            os.path.join(self.results_dir, "samples"),
            os.path.join(self.results_dir, "metrics"),
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
    
    def __repr__(self):
        """Pretty print configuration"""
        config_str = "CTM Configuration:\n"
        for key, value in self.__dict__.items():
            if isinstance(value, (list, tuple)):
                value = f"\n\t" + "\n\t".join(map(str, value))
            config_str += f"{key}: {value}\n"
        return config_str
