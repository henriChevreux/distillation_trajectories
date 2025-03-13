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
│   ├── mse_heatmap.py        # MSE analysis and visualization
│   ├── denoising.py          # Denoising process analysis
│   ├── time_analysis.py      # Time-dependent metrics
│   └── trajectory_analysis.py # Trajectory comparison tools
├── config/                    # Configuration
│   └── config.py             # Centralized configuration
├── data/                      # Data handling
│   └── dataset.py            # Dataset loading and processing
├── models/                    # Model definitions
│   ├── simple_unet.py        # Teacher model architecture
│   └── student_unet.py       # Student model architectures
├── output/                    # All output files
│   ├── models/               # Trained models
│   │   ├── teacher/          # Teacher model checkpoints
│   │   └── students/         # Student models by size
│   └── analysis/             # Analysis results
├── scripts/                   # Executable scripts
│   ├── train_teacher.py      # Teacher model training
│   ├── train_students.py     # Student models training
│   └── run_analysis.py       # Analysis pipeline
└── utils/                     # Utility functions
    └── diffusion.py          # Diffusion process utilities
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
- `--epochs N`: Override the default number of training epochs (recommended: 15-20)
- `--dataset [MNIST|CIFAR10]`: Dataset to use for training
- `--image_size N`: Size of images to use for training (default: 512)
- `--batch_size N`: Batch size for training (default: 8)
- `--timesteps N`: Number of timesteps for diffusion process (default: 50)

Features:
- **Automatic Dataset Verification**: Checks dataset availability before training
- **Memory Check**: Performs GPU memory verification with a small batch
- **Device Selection**: Automatically selects CUDA > MPS > CPU
- **Early Stopping**: Stops training if no improvement for 10 epochs
- **Best Model Tracking**: Saves the best model based on loss
- **Sample Generation**: Generates and saves samples from the best model
- **Progress Monitoring**: Shows detailed progress bars and loss tracking

Training Process:
1. Dataset and memory verification
2. Model initialization and setup (512x512 resolution)
3. Training loop with loss tracking (15-20 epochs recommended)
4. Regular model checkpointing (every 5 epochs)
5. Sample generation from best model
6. Early stopping if no improvement for 10 epochs

#### Student Models

Train student models with various size factors:

```bash
python scripts/train_students.py [OPTIONS]
```

Features:
- **Flexible Architecture**: Automatically adapts model architecture based on size factor
- **Multi-Resolution Support**: Trains models at different image resolutions
- **Memory Optimization**: Adjusts batch size based on available GPU memory
- **Progress Tracking**: Detailed training progress and model size comparisons
- **Distillation Process**: Knowledge transfer from teacher to student models

The training process includes:
1. Teacher model loading and verification
2. Student model architecture selection
3. Progressive training across different sizes
4. Regular checkpointing and validation
5. Performance comparison with teacher

### Running Analysis

The analysis pipeline (`run_analysis.py`) provides comprehensive model evaluation:

```bash
python scripts/run_analysis.py [OPTIONS]
```

Analysis Types:
- `--mse`: Mean Squared Error analysis between teacher and students
- `--denoising`: Denoising process visualization and comparison
- `--time`: Time-dependent performance analysis
- `--trajectory`: Trajectory analysis of the diffusion process
- `--all`: Run all available analyses

Configuration Options:
- `--batch-size N`: Batch size for loading test data (default: 4)
- `--num-samples N`: Number of samples to use for analysis (default: 5)
- `--cpu`: Force CPU usage for analysis
- `--full-mse`: Run comprehensive MSE analysis with all model sizes
- `--verbose`: Print detailed error messages and debugging info

Features:
- **Automated Device Selection**: CUDA/CPU with manual override
- **Comprehensive Metrics**: MSE, trajectory similarity, time analysis
- **Visualization**: Automated generation of plots and heatmaps
- **Error Handling**: Robust error reporting and recovery
- **Flexible Analysis**: Support for partial or complete analysis runs

### CPU Mode

For systems without GPU or for debugging, force CPU usage:

```bash
python scripts/run_analysis.py --cpu [OTHER_OPTIONS]
```

All scripts support CPU execution, with automatic optimization for CPU-based processing.

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

# Key Configuration Parameters
config.teacher_image_size = 512      # Teacher model image size
config.student_image_size = 256      # Base student model image size
config.timesteps = 50               # Number of diffusion timesteps
config.student_steps = 50           # Student model timesteps
config.batch_size = 8               # Training batch size
config.epochs = 15                  # Number of training epochs (recommended: 15-20 for teacher)

# Model Architecture
config.channels = 3                 # Number of image channels (RGB)
config.base_channels = 64           # Base number of model channels
config.student_size_factors = [0.2, 0.4, 0.5, 0.6, 0.8, 1.0]  # Model size factors

# Training Parameters
config.lr = 1e-4                   # Learning rate
config.save_interval = 5           # Checkpoint save interval
config.early_stopping_patience = 10 # Stop if no improvement for 10 epochs

# Analysis Settings
config.num_samples = 5             # Number of samples for analysis
config.mse_size_factors_limit = False  # Use all size factors for MSE
```