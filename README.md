# Diffusion Model Analysis Toolkit

A comprehensive toolkit for analyzing diffusion models with a focus on model size impact.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
  - [Training Models](#training-models)
  - [Running Analysis](#running-analysis)
  - [CPU Mode](#cpu-mode)
  - [Testing](#testing)
- [Configuration](#configuration)
- [Analysis Outputs](#analysis-outputs)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Citation](#citation)

## 🔍 Overview

This toolkit enables detailed comparison between teacher diffusion models and student models of varying sizes. It answers the question: "How does model size impact the quality, efficiency, and behavior of diffusion models?"

## ✨ Features

- **Performance analysis across model sizes**: Compare metrics as model size changes
- **Model efficiency evaluation**: Analyze performance-to-parameter ratio
- **Student models with varying architectures**: Size factors from 0.01 to 1.0
- **Detailed visualizations**: Charts showing performance trends
- **Comprehensive metrics**: Trajectory analysis, FID scores, efficiency measurements
- **Multi-device support**: CUDA, MPS, CPU

## 📦 Installation

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

## 📁 Project Structure

```
distillation_trajectories/
├── analysis/                  # Analysis modules
│   ├── dimensionality/        # Dimensionality reduction analysis
│   ├── metrics/               # Trajectory metrics
│   └── ...                    # Other analysis types
├── config/                    # Configuration
├── data/                      # Data handling
├── models/                    # Model definitions
├── output/                    # All output files
│   ├── models/                # Models directory
│   │   ├── teacher/           # Teacher models
│   │   └── students/          # Student models by size
│   └── results/               # Training results
├── scripts/                   # Executable scripts
├── testing/                   # Testing utilities
└── utils/                     # Utility functions
```

## 🚀 Quick Start

Train a teacher model, then student models, and run analysis:

```bash
# Train the teacher model
python scripts/train_teacher.py

# Train student models with various size factors
python scripts/train_students.py

# Run analysis
python scripts/run_analysis.py
```

## 📖 Usage Guide

### Training Models

#### Teacher Model

Train the full-sized teacher diffusion model:

```bash
python scripts/train_teacher.py [OPTIONS]
```

Options:
- `--epochs N`: Number of training epochs (default: 10)
- `--dataset [MNIST|CIFAR10]`: Dataset to use (default: CIFAR10)
- `--image_size N`: Size of images (default: 16)
- `--batch_size N`: Batch size (default: 64)
- `--timesteps N`: Number of diffusion timesteps (default: 50)

**Recommended for best results**:
```bash
# Train teacher with 1000 timesteps (best quality, slower training)
python scripts/train_teacher.py --timesteps 1000
```

#### Student Models

Train student models with various size factors:

```bash
python scripts/train_students.py [OPTIONS]
```

Options:
- `--epochs N`: Number of training epochs (default: 5, half of teacher epochs)
- `--custom_size_factors "0.1,0.5,0.9"`: Specific size factors to train (default: [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
- `--dataset [MNIST|CIFAR10]`: Dataset to use (default: CIFAR10)
- `--image_size N`: Size of images (default: 16)
- `--batch_size N`: Batch size (default: 64)

Student models are organized by size factor and architecture type:
- **Tiny** (< 0.1): 2 layers instead of 3
- **Small** (0.1-0.3): 3 layers with smaller dimensions
- **Medium** (0.3-0.7): 3 layers with 75% of teacher dimensions
- **Full** (0.7-1.0): Same architecture as teacher

#### Consistency Trajectory Model (CTM)

The Consistency Trajectory Model extends diffusion models to enable arbitrary timestep sampling and much faster generation. Unlike traditional diffusion models, CTM can generate high-quality samples in 5-20 steps instead of hundreds or thousands.

Train a CTM model:
```bash
python scripts/train_ctm.py
```

Or with custom parameters:
```bash
python scripts/train_ctm.py --size 0.2 --epochs 50
```

Test a CTM model:
```bash
python scripts/test_ctm.py
```

**Recommended settings**:
The CTM is configured to use DPM-solver with 20 steps by default, which offers an excellent balance of quality and speed.

Options:
- `--size N`: Model size factor (default: 0.2)
- `--batch-size N`: Batch size (default: 16)
- `--epochs N`: Number of training epochs (default: 50)
- `--trajectory-prob N`: Probability of using trajectory mode (default: 0.7)
- `--max-time-diff N`: Maximum time difference for trajectory (default: 0.5)
- `--teacher-path PATH`: Path to teacher model (default: output/models/teacher/model_epoch_1.pt)

Or test with custom parameters:
```bash
python scripts/test_ctm.py --size 0.2 --sample-steps 20
```

Options for testing:
- `--size N`: Model size factor (default: 0.2)
- `--sample-steps N`: Default number of sampling steps (default: 20)
- `--use-dpm`: Use DPM-solver for sampling (default: True)
- `--device [cuda|cpu]`: Device to use (default: cuda if available)

Generate MSE heatmaps for CTM models:
```bash
python scripts/test_ctm_mse.py --ctm-dir output/ctm_models
```

CTM models offer several advantages:
- **Faster sampling**: 5-20 steps vs 1000+ steps for traditional diffusion
- **Flexible quality-speed tradeoff**: Adjust steps at inference time
- **Better small models**: Small CTM models often match larger traditional models
- **Trajectory learning**: Jumping directly between any points in the diffusion process

### Running Analysis

Run comprehensive analysis on trained models:

```bash
python scripts/run_analysis.py [OPTIONS]
```

Options:
- `--teacher_model NAME`: Teacher model filename (default: "model_epoch_1.pt")
- `--num_samples N`: Number of trajectory samples (default: 50)
- `--teacher_steps N`: Teacher timesteps (default: 50)
- `--student_steps N`: Student timesteps (default: 50)
- `--analysis_dir NAME`: Directory to save analysis results (default: "analysis")
- `--focus_size_range "0.1-0.5"`: Focus on specific size range (default: all sizes)
- `--compare_specific_sizes "0.1,0.5,1.0"`: Compare specific sizes (default: all sizes)

Skip specific analysis modules:
- `--skip_metrics`: Skip trajectory metrics analysis (default: run this analysis)
- `--skip_dimensionality`: Skip dimensionality reduction analysis (default: run this analysis)
- `--skip_noise`: Skip noise prediction analysis (default: run this analysis)
- `--skip_attention`: Skip attention map analysis (default: run this analysis)
- `--skip_3d`: Skip 3D visualization (default: run this analysis)
- `--skip_fid`: Skip FID calculation (default: run this analysis)

### CPU Mode

Force any script to run on CPU (useful for systems without GPU):

```bash
python scripts/run_on_cpu.py SCRIPT [--args "SCRIPT_ARGS"]
```

Examples:
```bash
python scripts/run_on_cpu.py train_teacher
python scripts/run_on_cpu.py train_students --args "--custom_size_factors 0.1,0.5"
python scripts/run_on_cpu.py run_analysis --args "--num_samples 5 --skip_fid"
```

### Testing

Run tests to verify the diffusion model implementation:

```bash
python testing/test_diffusion.py
```

## ⚙️ Configuration

The project uses a centralized configuration system in `config/config.py`:

```python
from config.config import Config

config = Config()
config.create_directories()  # Creates all necessary directories

# Access paths
models_dir = config.models_dir
results_dir = config.results_dir
```

## 📊 Analysis Outputs

All analysis results are organized in the `output/analysis/` directory:

- **Trajectory Metrics** (`metrics/`): Path length, Wasserstein distance, endpoint distance
- **Comparative Visualizations** (`visualization/`): Plots showing metrics vs model size
- **FID Scores** (`fid/`): Fréchet Inception Distance measurements
- **Dimensionality Reduction** (`dimensionality/`): PCA, t-SNE, and UMAP projections
- **Attention Analysis** (`attention/`): Attention map visualizations
- **Noise Prediction** (`noise/`): Analysis of noise prediction patterns

## ❓ Troubleshooting

Common issues:
- **GPU not detected**: Use `scripts/run_on_cpu.py` to force CPU usage
- **Model not found**: Check paths in `config.py` and ensure models are trained
- **Memory errors**: Reduce batch size or number of samples

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📝 Citation

If you use this toolkit in your research, please cite:

```bibtex
@misc{diffusion_analysis_toolkit,
  title={Diffusion Model Analysis Toolkit},
  author={Your Name},
  year={2024}
}
