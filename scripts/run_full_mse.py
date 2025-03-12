#!/usr/bin/env python3
"""
Script to run full MSE heatmap analysis using all model and image sizes.
"""

import os
import sys
import argparse
import torch

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config
from models import SimpleUNet, StudentUNet
from analysis import mse_heatmap
from scripts.run_analysis import setup_analysis

def main():
    parser = argparse.ArgumentParser(description='Run full MSE heatmap analysis with all size factors')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for loading test data')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    
    print("Setting up environment for full MSE analysis...")
    
    # Create mock args for setup_analysis
    class MockArgs:
        def __init__(self):
            self.batch_size = args.batch_size
            self.num_samples = 1
    
    mock_args = MockArgs()
    
    # Load configuration, models, and test data
    config, teacher_model, student_models, test_samples = setup_analysis(mock_args)
    
    # Disable size factor limits
    config.mse_size_factors_limit = False
    config.mse_image_size_factors_limit = False
    
    # Force CPU usage if specified
    if args.cpu:
        print("Forcing CPU usage for MSE analysis")
        config.force_cpu = True
    
    print(f"Running FULL MSE analysis with:")
    print(f"- {len(config.student_size_factors)} model size factors: {config.student_size_factors}")
    print(f"- {len(config.student_image_size_factors)} image size factors: {config.student_image_size_factors}")
    print(f"- {len(test_samples)} test samples")
    
    # Run the MSE heatmap analysis
    try:
        mse_heatmap.create_mse_heatmap(config, test_samples)
        print("MSE heatmap generated successfully!")
    except Exception as e:
        print(f"Error in MSE analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 