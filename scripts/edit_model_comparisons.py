#!/usr/bin/env python3
"""
Script to modify the model_comparisons.py file to use raw metrics without min-max scaling.
Only normalizes directional consistency from [-1,1] to [0,1] using (value + 1) / 2.
Uses MSE Similarity (1 - MSE) instead of MSE for radar plots.
"""

import os
import sys
import re

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def modify_model_comparisons():
    """
    Modify the model_comparisons.py file to:
    1. Use MSE Similarity (1 - MSE) instead of MSE
    2. Skip min-max scaling for all metrics
    3. Only normalize directional consistency from [-1,1] to [0,1]
    """
    # Path to the model_comparisons.py file
    file_path = os.path.join(project_root, 'analysis', 'metrics', 'model_comparisons.py')
    
    if not os.path.exists(file_path):
        print(f"ERROR: File not found at {file_path}")
        sys.exit(1)
    
    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 1. Replace endpoint distance with MSE in both functions
    content = content.replace("endpoint_distance = metrics.get('endpoint_distance', 0.0)", 
                             "mse = metrics.get('mse', 0.0)  # Use MSE instead of endpoint distance")
    
    # 2. Replace min-max scaling with direct use of raw metrics in create_radar_plot_grid
    grid_scaling_pattern = r"""        # Apply min-max scaling to all metrics.*?
        # Combine the metrics into the order we want for the radar plot"""
    
    grid_replacement = """        # Use raw metrics directly without scaling, except for directional consistency
        # For path_length_similarity (higher is better): already in [0,1]
        path_length_score = path_length_similarity
        
        # For MSE (lower is better): convert to MSE Similarity (1 - MSE)
        mse_similarity = 1.0 - mse
        
        # For directional_consistency (higher is better): normalize from [-1,1] to [0,1]
        directional_score = (directional_consistency + 1) / 2 if directional_consistency != float('nan') else 0.5
        
        # For distribution_similarity (higher is better): already in [0,1]
        distribution_score = distribution_similarity
        
        # Combine the metrics into the order we want for the radar plot"""
    
    content = re.sub(grid_scaling_pattern, grid_replacement, content, flags=re.DOTALL)
    
    # 3. Replace min-max scaling with direct use of raw metrics in create_composite_radar_plot
    composite_scaling_pattern = r"""        # Apply min-max scaling to all metrics.*?
        # Combine the metrics into the order we want for the radar plot"""
    
    composite_replacement = """        # Use raw metrics directly without scaling, except for directional consistency
        # For path_length_similarity (higher is better): already in [0,1]
        path_length_score = path_length_similarity
        
        # For MSE (lower is better): convert to MSE Similarity (1 - MSE)
        mse_similarity = 1.0 - mse
        
        # For directional_consistency (higher is better): normalize from [-1,1] to [0,1]
        directional_score = (directional_consistency + 1) / 2 if directional_consistency != float('nan') else 0.5
        
        # For distribution_similarity (higher is better): already in [0,1]
        distribution_score = distribution_similarity
        
        # Combine the metrics into the order we want for the radar plot"""
    
    content = re.sub(composite_scaling_pattern, composite_replacement, content, flags=re.DOTALL)
    
    # 4. Update the metric labels in both functions
    content = content.replace("'Endpoint Distance',", "'MSE Similarity',")
    content = content.replace("'Endpoint\\nDistance',", "'MSE\\nSimilarity',")
    content = content.replace("'Endpoint\\nAlignment',", "'MSE\\nSimilarity',")
    content = content.replace("'MSE',", "'MSE Similarity',")
    content = content.replace("'MSE\\nSimilarity',", "'MSE\\nSimilarity',")
    
    # 5. Update the metrics values array in both functions
    content = content.replace("endpoint_distance_score,", "mse_similarity,")
    content = content.replace("mse_score,", "mse_similarity,")
    
    # 6. Update the diagnostic print statements
    content = content.replace("    MSE: {mse:.4f} → Score: {mse_score:.4f}", 
                             "    MSE: {mse:.4f} → MSE Similarity: {mse_similarity:.4f}")
    
    # Write the modified content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Modified {file_path} to:")
    print("  - Use MSE Similarity (1 - MSE) instead of MSE")
    print("  - Skip min-max scaling for all metrics")
    print("  - Only normalize directional consistency from [-1,1] to [0,1]")

def main():
    """Main function"""
    modify_model_comparisons()
    print("Done.")

if __name__ == "__main__":
    main() 