# Diffusion Model Analysis Toolkit

A comprehensive toolkit for analyzing diffusion models with a focus on model size impact.

## 📋 Table of Contents

- [Diffusion Model Analysis Toolkit](#diffusion-model-analysis-toolkit)
  - [📋 Table of Contents](#-table-of-contents)
  - [🔍 Overview](#-overview)
  - [✨ Features](#-features)
  - [📦 Installation](#-installation)
  - [📁 Project Structure](#-project-structure)
  - [🚀 Quick Start](#-quick-start)
  - [📖 Usage Guide](#-usage-guide)
    - [Training Models](#training-models)
      - [Teacher Model](#teacher-model)
      - [Student Models](#student-models)
    - [Running Analysis](#running-analysis)
    - [CPU Mode](#cpu-mode)
    - [Testing](#testing)
  - [⚙️ Configuration](#️-configuration)
  - [📊 Analysis Outputs](#-analysis-outputs)
  - [❓ Troubleshooting](#-troubleshooting)
  - [📄 License](#-license)
  - [📝 Citation](#-citation)

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

### Running Analysis

Run comprehensive analysis on trained models:

```bash
python scripts/run_analysis.py [OPTIONS]
```

Options:
- `--teacher_model NAME`: Teacher model filename (default: "model_epoch_{highest}.pt")
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

## 🏗️ UNet Architecture Specifications

This project uses the following UNet architecture parameters for CIFAR10 diffusion models:

### Model Configuration
| Parameter | Value |
|-----------|-------|
| Image size | 32 × 32 |
| Number of channels | 3 |
| Base channels | 128 |
| Channel multipliers | [1, 2, 2, 2] |
| Number of residual blocks | 3 |
| Dropout | 0.3 |

### Diffusion Process Parameters
| Parameter | Value |
|-----------|-------|
| Training diffusion steps | 4000 |
| Sampling timesteps | 50 |
| Noise schedule | cosine |

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Learning rate | 1 × 10⁻⁴ |
| Batch size | 128 |
| Optimizer | Adam (β₁=0.8, β₂=0.999) |
| EMA rate | 0.9999 |

### Progressive Distillation Parameters
| Parameter | Value |
|-----------|-------|
| Teacher timesteps | 50 |
| Student timesteps | 50 |

### Understanding Diffusion Steps vs. Sampling Timesteps

There's an important distinction between:
- **Training diffusion steps (4000)**: The total number of noise addition steps in the original diffusion process during training
- **Sampling timesteps (50)**: The actual number of steps used when generating images during inference

Progressive distillation allows us to train on the full 4000-step diffusion process, but generate high-quality samples using only 50 steps. This significantly accelerates the sampling process while maintaining image quality.

These parameters are based on research from:
1. Ho et al. (2020). Denoising diffusion probabilistic models. NeurIPS.
2. Nichol & Dhariwal (2021). Improved denoising diffusion probabilistic models. ICML.
3. Salimans & Ho (2022). Progressive distillation for fast sampling of diffusion models. ICLR.
4. Song et al. (2020). Denoising diffusion implicit models. ICLR.

A detailed LaTeX document with these specifications is available in `unet_architecture.tex`.

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
