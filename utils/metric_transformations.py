import numpy as np

def transform_metrics(path_length_similarity, trajectory_mse, directional_consistency, distribution_similarity):
    """
    Transform raw metric values to normalized scores in [0,1] range.
    Uses simpler transformations that preserve the intuitive meaning of each metric.
    
    Args:
        path_length_similarity: Raw path length similarity (already in [0,1])
        trajectory_mse: Raw MSE between trajectories (lower is better)
        directional_consistency: Raw directional consistency (in [-1,1])
        distribution_similarity: Raw distribution similarity
        
    Returns:
        Dictionary of transformed scores
    """
    # Path Length Similarity: Already in [0,1], no transformation needed
    path_length_score = path_length_similarity
    
    # Trajectory MSE: Apply log transformation for better visualization
    # Lower MSE is better, so we don't need to convert to similarity
    # Handle negative or invalid values
    trajectory_mse = np.clip(trajectory_mse, 0, None)  # Ensure non-negative
    mse_similarity = np.log1p(trajectory_mse)
    mse_similarity = np.clip(1 - (mse_similarity / np.log1p(1.0)), 0, 1)  # Invert so higher is better
    
    # Directional Consistency: Take absolute value to get [0,1] range
    directional_score = np.abs(directional_consistency)
    
    # Distribution Similarity: Apply log transformation for better visualization
    distribution_score = np.log1p(distribution_similarity)
    distribution_score = np.clip(distribution_score / np.log1p(1.0), 0, 1)  # Normalize to [0,1]
    
    return {
        'path_length_similarity': path_length_score,
        'trajectory_mse': mse_similarity,
        'mean_directional_consistency': directional_score,  # Changed key to match heatmap
        'distribution_similarity': distribution_score
    } 