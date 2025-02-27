# Diffusion Model Analysis Toolkit

## Project Overview

This project provides a comprehensive toolkit for analyzing Diffusion Models, with a focus on knowledge distillation and trajectory analysis. It supports in-depth exploration of teacher and student models trained on generative tasks.

### Key Features
- Model analysis and comparison
- Flexible configuration for different model parameters
- Multiple analysis modules
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
- `--num_samples`: Number of trajectory samples to generate (default: 5)

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

### Example Commands

1. Full analysis with default settings:
```bash
python run_analysis.py
```

2. Custom analysis with more samples and different timesteps:
```bash
python run_analysis.py --num_samples 10 --teacher_steps 100 --student_steps 50
```

3. Partial analysis, skipping some modules:
```bash
python run_analysis.py --skip_metrics --skip_3d
```

## Project Structure

- `run_analysis.py`: Main script for running model analysis
- `diffusion_analysis.py`: Core analysis and model implementation
- `models/`: Directory for trained model weights
- `analysis/`: Output directory for analysis results

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