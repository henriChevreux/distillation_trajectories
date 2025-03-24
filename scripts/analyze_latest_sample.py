#!/usr/bin/env python3
"""
Script to analyze the latest sample image and provide statistics.
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config

def main():
    # Create configuration
    config = Config()
    results_dir = config.results_dir
    
    # Find the latest sample image
    sample_files = glob.glob(os.path.join(results_dir, 'samples_epoch_*.png'))
    latest_sample = max(sample_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    epoch_num = int(latest_sample.split('_')[-1].split('.')[0])
    
    print(f"Analyzing latest sample image: {os.path.basename(latest_sample)} (Epoch {epoch_num})")
    
    # Load the sample image
    sample_img = np.array(Image.open(latest_sample))
    
    # Calculate basic statistics
    print("\nImage Statistics:")
    print(f"  Shape: {sample_img.shape}")
    print(f"  Min value: {sample_img.min()}")
    print(f"  Max value: {sample_img.max()}")
    print(f"  Mean value: {sample_img.mean():.2f}")
    print(f"  Std deviation: {sample_img.std():.2f}")
    
    # Check for structure formation
    print("\nStructure Analysis:")
    
    # Calculate variance in each color channel across the image
    if len(sample_img.shape) > 2 and sample_img.shape[2] >= 3:
        for i, channel in enumerate(['Red', 'Green', 'Blue']):
            channel_var = np.var(sample_img[:,:,i])
            print(f"  {channel} channel variance: {channel_var:.2f}")
    
    # Check for local structure by comparing adjacent patches
    structure_score = 0
    patch_size = 16
    rows, cols = sample_img.shape[0] // patch_size, sample_img.shape[1] // patch_size
    
    for i in range(rows-1):
        for j in range(cols-1):
            patch1 = sample_img[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patch2 = sample_img[(i+1)*patch_size:(i+2)*patch_size, j*patch_size:(j+1)*patch_size]
            
            # Calculate correlation between adjacent patches
            if len(patch1.shape) > 2:
                try:
                    corr = np.corrcoef(patch1.flatten(), patch2.flatten())[0, 1]
                    if not np.isnan(corr):
                        structure_score += abs(corr)
                except:
                    # Skip if correlation calculation fails
                    pass
    
    avg_structure_score = structure_score / ((rows-1) * (cols-1)) if ((rows-1) * (cols-1)) > 0 else 0
    print(f"  Local structure score: {avg_structure_score:.4f} (higher values indicate more structure)")
    
    # Progress assessment
    if avg_structure_score < 0.1:
        print("\nProgress Assessment: EARLY STAGE")
        print("Your samples still appear mostly random. This is normal for early training.")
        print("The model is still learning the basic color distributions.")
        
    elif avg_structure_score < 0.3:
        print("\nProgress Assessment: DEVELOPING STRUCTURE")
        print("Your samples are beginning to show basic patterns and color grouping.")
        print("Continue training to see more defined shapes emerging.")
        
    elif avg_structure_score < 0.5:
        print("\nProgress Assessment: INTERMEDIATE PROGRESS")
        print("Your samples now show recognizable patterns and shapes.")
        print("Further training will refine details and improve clarity.")
        
    else:
        print("\nProgress Assessment: ADVANCED STAGE")
        print("Your samples contain clear structure and possibly recognizable objects.")
        print("Further training will add finer details and improve realism.")
    
    print("\nRecommendation:")
    if epoch_num < 50:
        print(f"  Continue training for at least {50 - epoch_num} more epochs")
        print("  Or start a new training run with 1000 timesteps for potentially better results")
    else:
        print("  You've trained for a good number of epochs")
        print("  Consider trying a different noise schedule or increased timesteps if results aren't satisfactory")
    
    print("\nNote on Diffusion Training:")
    print("  1. Diffusion models often need 50-100+ epochs to show clear structure with default settings")
    print("  2. Visual quality can improve suddenly after many epochs of seemingly little progress")
    print("  3. A loss of ~0.815 at epoch 32 is actually quite normal and indicates learning is happening")
    print("  4. More timesteps (1000 vs your current 50/100) would likely show faster visual improvement")

if __name__ == "__main__":
    main() 