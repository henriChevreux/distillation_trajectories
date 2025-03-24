# Trajectory Distillation Analysis

This repository contains tools for analyzing and visualizing trajectory metrics in diffusion models, with a focus on model size comparisons and Classifier-Free Guidance (CFG) impact.

## Training

### Teacher Model
```bash
python scripts/train_teacher.py [OPTIONS]
```
Options:
- `--epochs N`: Number of training epochs (default: 10)
- `--image_size N`: Size of images (default: 16)
- `--batch_size N`: Batch size (default: 64)
- `--timesteps N`: Number of diffusion timesteps (default: 50)

### Student Models
```bash
python scripts/train_students.py [OPTIONS]
```
Options:
- `--epochs N`: Number of training epochs (default: 5, half of teacher epochs)
- `--custom_size_factors "0.1,0.5,0.9"`: Specific size factors to train (default: [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
- `--image_size N`: Size of images (default: 16)
- `--batch_size N`: Batch size (default: 64)

Student models are organized by size factor and architecture type:
- **Tiny** (< 0.1): 2 layers instead of 3
- **Small** (0.1-0.3): 3 layers with smaller dimensions
- **Medium** (0.3-0.7): 3 layers with 75% of teacher dimensions
- **Full** (0.7-1.0): Same architecture as teacher

## Analysis Scripts

### Trajectory Metrics Analysis
```bash
python scripts/analysis/analyze_trajectory_metrics.py --teacher_model model_epoch_200.pt
```
This script generates:
- Heatmaps showing the impact of CFG across different model sizes and guidance scales
- Radar plots comparing trajectory metrics across model sizes
- A composite radar plot showing all model sizes together

Key metrics analyzed:
- Path Length Similarity
- Trajectory MSE Similarity
- Directional Consistency
- Distribution Similarity

Output files will be saved in:
```
output/analysis/model_comparisons/
├── radar_plot_grid.png
├── composite_radar_plot.png
├── cfg_heatmap_path_length_similarity.png
├── cfg_heatmap_trajectory_mse.png
├── cfg_heatmap_mean_directional_consistency.png
├── cfg_heatmap_distribution_similarity.png
└── cfg_heatmap_combined.png
```

### Dimensionality Analysis
```bash
python scripts/analysis/analyze_dimensionality.py --teacher_model model_epoch_200.pt
```
Analyzes the dimensionality of the latent space and generates visualizations.

### Noise Prediction Analysis
```bash
python scripts/analysis/analyze_noise_prediction.py --teacher_model model_epoch_200.pt
```
Analyzes noise prediction patterns across different model sizes.

### Time-Dependent Analysis
```bash
python scripts/analysis/analyze_time_dependent.py --teacher_model model_epoch_200.pt
```
Analyzes how trajectory metrics evolve over time during the diffusion process.

### FID Score Analysis
```bash
python scripts/analysis/analyze_fid.py --teacher_model model_epoch_200.pt
```
Calculates and visualizes Fréchet Inception Distance (FID) scores across model sizes.

## Directory Structure
```
analysis/
├── metrics/           # Metric computation functions
├── dimensionality/    # Dimensionality analysis tools
├── noise_prediction/  # Noise prediction analysis
├── visualization/     # Visualization utilities
└── __init__.py       # Package initialization

scripts/
├── train_teacher.py   # Teacher model training
├── train_students.py  # Student models training
└── analysis/         # Analysis scripts

output/
└── analysis/         # Analysis results
    ├── metrics/      # Basic trajectory metrics
    ├── model_comparisons/  # Radar plots and heatmaps
    ├── time_dependent/    # Time-dependent analysis
    ├── size_dependent/    # Size-dependent analysis
    ├── dimensionality/    # Dimensionality analysis
    ├── latent_space/      # Latent space visualizations
    ├── attention/         # Attention analysis
    ├── noise_prediction/  # Noise prediction analysis
    ├── denoising/        # Denoising analysis
    └── fid/              # FID score analysis
```

## Dependencies
- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- scikit-learn

## Usage
1. Train the teacher model using `train_teacher.py`
2. Train student models using `train_students.py`
3. Run the desired analysis script with appropriate arguments
4. Results will be saved in the specified output directory

## Notes
- All metrics are normalized to [0,1] range for consistent visualization
- CFG impact is analyzed using guidance scales from 1.0 (no CFG) to 20.0
- Model sizes range from 0.05 to 1.0 (full size)
