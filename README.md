# Diffusion Model Distillation for MacOS

This project studies the impact of diffusion model distillation on latent trajectories, optimized for Intel MacBooks with limited computational resources.

## Overview

The code implements:
1. A lightweight UNet-based diffusion model architecture
2. Teacher-student knowledge distillation to reduce inference steps
3. Trajectory analysis tools to compare latent space paths
4. Visualization of how diffusion paths change through distillation

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## MPS Compatibility Issues

This code can be run with Metal Performance Shaders (MPS) acceleration on MacOS, but there are known issues with some PyTorch operations on MPS. If you encounter errors related to MPS, you have two options:

### Option 1: Modify the source code
Open `diffusion_distillation.py` and change:
```python
use_mps = False  # Change this from True to False
```

### Option 2: Use the CPU runner script
Run the model forcibly on CPU using:
```bash
python run_on_cpu.py
```

## Testing

Run the test script to verify your setup:
```bash
python test_diffusion.py
```

The tests will check various components:
- Model initialization
- Diffusion parameters
- Forward diffusion process
- Data loading
- Training step
- Sampling process

## Running the Full Pipeline

After testing, run the full pipeline:
```bash
python diffusion_distillation.py
```

This will:
1. Train a teacher diffusion model with 50 timesteps
2. Distill knowledge to a student model with 10 timesteps
3. Compare latent trajectories between models
4. Generate visualizations in the specified directories

## Output Directories

- `results/`: Generated samples during training
- `models/`: Saved model checkpoints
- `trajectories/`: Trajectory analysis visualizations

## Fine-tuning For Your Machine

If you're experiencing slow performance or memory issues:

1. Reduce model size:
```python
config.hidden_dims = [16, 32, 64]  # Smaller network
```

2. Reduce diffusion steps:
```python
config.timesteps = 20  # Fewer steps
config.teacher_steps = 20
config.student_steps = 5
```

3. Reduce batch size:
```python
config.batch_size = 32  # Smaller batches
```

## Analysis

The trajectory analysis compares how samples evolve in latent space between the teacher and student models through:

1. Direct comparisons of sample evolution
2. L2 distance measurements between trajectories  
3. PCA visualization of latent paths
4. Quantitative assessment of how distillation affects the denoising process