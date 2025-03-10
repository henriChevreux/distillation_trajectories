# Diffusion Model Analysis Toolkit

## Project Overview

This project provides a comprehensive toolkit for analyzing Diffusion Models, with a focus on knowledge distillation and trajectory analysis. It supports in-depth exploration of teacher and student models trained on generative tasks, with particular emphasis on studying the impact of model size and architecture on performance.

### Key Features
- Model analysis and comparison across different model sizes
- Student models with varying architectures and complexity
- Flexible configuration for different model parameters
- Multiple analysis modules with comparative visualizations
- Support for various computational devices (CUDA, MPS, CPU)

## Prerequisites

### System Requirements
- Python 3.9+
- PyTorch
- CUDA or MPS (optional, but recommended for performance)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/diffusion-model-analysis.git
cd diffusion-model-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Model Training

The project supports training both teacher and student models with various configurations:

```bash
# Train teacher model (if not already trained)
python diffusion_training.py --train_teacher

# Train only student models (using existing teacher model)
python train_students.py
```

### Student Model Architecture

Student models can be created with different architectures and size factors:

- **Size Factors**: Range from extremely small (0.01) to full-sized (1.0)
- **Architecture Types**:
  - `tiny`: 2 layers instead of 3 (for very small models)
  - `small`: 3 layers with smaller dimensions
  - `medium`: 3 layers with 75% of teacher dimensions
  - `full`: Same architecture as teacher

The architecture type is automatically selected based on the size factor, but maintains the same input and output dimensions as the teacher model.

### Running Analysis

```bash
python run_analysis.py [OPTIONS]
```

### Command Line Options

#### Model Selection
- `--teacher_model`: Teacher model file (default: `model_epoch_1.pt`)
- `--student_model`: Student model file (default: `student_model_epoch_1.pt`)

#### Analysis Parameters
- `--analysis_dir`: Directory to save analysis results (default: `analysis`)
- `--num_samples`: Number of trajectory samples to generate (default: 50)

#### Diffusion Process
- `--teacher_steps`: Number of timesteps for teacher model (default: 50)
- `--student_steps`: Number of timesteps for student model (default: 50)

#### Analysis Module Control
Optionally skip specific analysis modules:
- `--skip_metrics`: Skip trajectory metrics analysis
- `--skip_dimensionality`: Skip dimensionality reduction analysis
- `--skip_noise`: Skip noise prediction analysis
- `--skip_attention`: Skip attention map analysis
- `--skip_3d`: Skip 3D visualization
- `--skip_fid`: Skip FID calculation

### Example Commands

1. Train only the teacher model:
```bash
python diffusion_training.py --train_teacher
```

2. Train student models using existing teacher model:
```bash
python train_students.py
```

3. Full analysis with default settings:
```bash
python run_analysis.py
```

4. Custom analysis with more samples and different timesteps:
```bash
python run_analysis.py --num_samples 10 --teacher_steps 100 --student_steps 50
```

5. Partial analysis, skipping some modules:
```bash
python run_analysis.py --skip_metrics --skip_3d
```

## Project Structure

- `diffusion_training.py`: Main script for training teacher and student models
- `train_students.py`: Script for training only student models with various size factors
- `run_analysis.py`: Script for running model analysis
- `diffusion_analysis.py`: Core analysis and model implementation
- `models/`: Directory for trained model weights
- `analysis/`: Output directory for analysis results
- `results/`: Directory for generated samples
- `trajectories/`: Directory for trajectory data

## Analysis Outputs

The analysis generates various visualizations and metrics:

- **Trajectory Metrics**: Path length, Wasserstein distance, endpoint distance, path efficiency
- **Comparative Visualizations**: Plots showing how metrics vary with model size
- **FID Scores**: Fréchet Inception Distance between real images, teacher samples, and student samples
- **3D Visualizations**: PCA projections of latent space trajectories
- **Sample Comparisons**: Visual comparison of generated samples

## Troubleshooting

### Common Issues
- Ensure PyTorch is correctly installed with GPU support if needed
- Check model file paths
- Verify computational device compatibility

### Device Support
The script automatically detects and uses the best available device:
1. CUDA (NVIDIA GPUs)
2. MPS (Apple Silicon)
3. CPU (Fallback)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@misc{diffusion_analysis_toolkit,
  title={Diffusion Model Analysis Toolkit},
  author={Your Name},
  year={2024}
}
```

## Contact

For questions or support, please open an issue on the GitHub repository.
