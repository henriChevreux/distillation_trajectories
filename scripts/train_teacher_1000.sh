#!/bin/bash
# Convenience script to train the teacher model with 1000 timesteps
echo "Starting teacher model training with 1000 timesteps..."
python scripts/train_teacher.py --timesteps 1000 