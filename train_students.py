import os
import argparse
import torch

# Import from the diffusion_training module
from diffusion_training import main as train_models, Config

def print_size_factor_info():
    """Print information about the size factors that will be trained"""
    config = Config()
    size_factors = config.student_size_factors
    
    print("\n" + "="*80)
    print("MODEL SIZE SPECTRUM TRAINING")
    print("="*80)
    
    print("\nThis script trains student models with various size factors to enable")
    print("comprehensive analysis of how model size affects performance.")
    
    print(f"\nTraining {len(size_factors)} student models with size factors ranging from")
    print(f"{min(size_factors)} (tiny) to {max(size_factors)} (full size).")
    
    # Group size factors by category
    tiny = [sf for sf in size_factors if sf < 0.1]
    small = [sf for sf in size_factors if 0.1 <= sf < 0.3]
    medium = [sf for sf in size_factors if 0.3 <= sf < 0.7]
    large = [sf for sf in size_factors if sf >= 0.7]
    
    print("\nSize distribution:")
    print(f"  Tiny models (< 0.1x): {len(tiny)} models - {tiny}")
    print(f"  Small models (0.1-0.3x): {len(small)} models - {small}")
    print(f"  Medium models (0.3-0.7x): {len(medium)} models - {medium}")
    print(f"  Large models (0.7-1.0x): {len(large)} models - {large}")
    
    # Architecture types
    print("\nArchitecture types:")
    print("  - Tiny (< 0.1): 2 layers instead of 3 (for very small models)")
    print("  - Small (0.1-0.3): 3 layers with smaller dimensions")
    print("  - Medium (0.3-0.7): 3 layers with 75% of teacher dimensions")
    print("  - Full (0.7-1.0): Same architecture as teacher")
    
    # Parameter count approximation
    base_params = 1.0  # Normalized to teacher model size
    param_counts = {sf: sf**2 * base_params for sf in size_factors}
    
    print("\nApproximate parameter counts (relative to teacher model):")
    for category, factors in [("Tiny", tiny), ("Small", small), ("Medium", medium), ("Large", large)]:
        if factors:
            min_factor = min(factors)
            max_factor = max(factors)
            min_params = param_counts[min_factor]
            max_params = param_counts[max_factor]
            print(f"  {category}: {min_params:.4f}x to {max_params:.4f}x parameters")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train student diffusion models with various size factors for performance comparison',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override the number of epochs for training')
    parser.add_argument('--custom_size_factors', type=str, default=None,
                        help='Custom size factors to train (comma-separated, e.g., "0.1,0.5,0.9")')
    args = parser.parse_args()
    
    # Check if teacher model exists
    teacher_model_path = os.path.join('models', 'model_epoch_1.pt')
    if not os.path.exists(teacher_model_path):
        print("\nERROR: Teacher model not found at", teacher_model_path)
        print("Please train the teacher model first by running:")
        print("\n    python diffusion_training.py --train_teacher\n")
        exit(1)
    
    # Print information about the size factors
    print_size_factor_info()
    
    print("\nStarting student model training with various size factors...")
    print("Using existing teacher model from:", teacher_model_path)
    
    # Call the main function with skip_teacher_training=True
    train_models(skip_teacher_training=True)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print("\nStudent models with various size factors have been saved.")
    print("These models can now be analyzed to understand how model size affects performance.")
    print("\nTo run the comprehensive size impact analysis:")
    print("\n    python run_analysis.py\n")
    print("This will generate visualizations and metrics comparing performance across all model sizes.")
