import os
import argparse
import torch

# Import from the diffusion_training module
from diffusion_training import main as train_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train student diffusion models with various size factors')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override the number of epochs for training (default: use config value)')
    args = parser.parse_args()
    
    # Check if teacher model exists
    teacher_model_path = os.path.join('models', 'model_epoch_1.pt')
    if not os.path.exists(teacher_model_path):
        print("\nERROR: Teacher model not found at", teacher_model_path)
        print("Please train the teacher model first by running:")
        print("\n    python diffusion_training.py --train_teacher\n")
        exit(1)
    
    print("Starting student model training with various size factors...")
    print("Using existing teacher model from:", teacher_model_path)
    
    # Call the main function with skip_teacher_training=True
    train_models(skip_teacher_training=True)
    
    print("\nTraining complete! Student models with various size factors have been saved.")
    print("You can now run the analysis to compare the models:")
    print("\n    python run_analysis.py\n")
