"""
Analysis package for trajectory distillation.
"""

from config.config import Config
from models import DiffusionUNet
from analysis.trajectory_engine import compare_trajectories
from analysis.metrics.trajectory_metrics import compute_trajectory_metrics, visualize_metrics, visualize_batch_metrics
from analysis.dimensionality.dimensionality_reduction import dimensionality_reduction_analysis
from analysis.noise_prediction.noise_analysis import analyze_noise_prediction
from analysis.dimensionality.latent_space import generate_latent_space_visualization
from analysis.visualization.model_size_viz import generate_3d_model_size_visualization
from analysis.metrics.fid_score import calculate_and_visualize_fid

__all__ = [
    'Config',
    'DiffusionUNet',
    'compare_trajectories',
    'compute_trajectory_metrics',
    'visualize_metrics',
    'visualize_batch_metrics',
    'dimensionality_reduction_analysis',
    'analyze_noise_prediction',
    'generate_latent_space_visualization',
    'generate_3d_model_size_visualization',
    'calculate_and_visualize_fid'
]

# Add missing function imports
# Time-dependent analysis functions
from analysis.metrics.time_dependent import analyze_time_dependent_distances
from analysis.visualization.time_dependent import plot_time_dependent_grid, plot_time_dependent_combined, plot_trajectory_divergence_vs_timestep

# Size-dependent analysis functions
from analysis.metrics.size_dependent import plot_mse_vs_size, plot_metrics_vs_size 