#!/usr/bin/env python3
"""
Script to convert the newly trained teacher model to be compatible with the analysis script.
This script adds the missing 'final_decoder' layer to the state dictionary.
"""

import os
import sys
import torch
import argparse

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models import DiffusionUNet, SimpleUNet
from config.config import Config

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Convert model for analysis compatibility',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input_model', type=str, default='output/models/teacher/model_epoch_1.pt',
                        help='Path to the input model file')
    parser.add_argument('--output_model', type=str, default='output/models/teacher/model_epoch_1_converted.pt',
                        help='Path to save the converted model file')
    
    return parser.parse_args()

def convert_model(input_path, output_path):
    """
    Convert the model state dictionary to be compatible with the analysis script.
    
    Args:
        input_path: Path to the input model file
        output_path: Path to save the converted model file
    
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    print(f"Converting model from {input_path} to {output_path}")
    
    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        return False
    
    try:
        # Load the state dictionary
        state_dict = torch.load(input_path, map_location='cpu')
        print(f"Loaded state dictionary with {len(state_dict)} keys")
        
        # Create a new state dictionary with renamed keys
        new_state_dict = {}
        
        # Check if we need to convert (if 'enc4' exists but 'final_decoder' doesn't)
        has_enc4 = any('enc4' in key for key in state_dict.keys())
        has_final_decoder = any('final_decoder' in key for key in state_dict.keys())
        
        if has_enc4 and not has_final_decoder:
            print("Model already has the new format (enc4 but no final_decoder). No conversion needed.")
            # Just copy the state dict
            new_state_dict = state_dict.copy()
        else:
            # Convert from old format to new format
            print("Converting from old format to new format...")
            
            # Copy all existing keys
            for key, value in state_dict.items():
                # If the key is from final_decoder, map it to enc4
                if 'final_decoder' in key:
                    new_key = key.replace('final_decoder', 'enc4')
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            print(f"Converted state dictionary has {len(new_state_dict)} keys")
        
        # Save the new state dictionary
        torch.save(new_state_dict, output_path)
        print(f"Successfully saved converted model to {output_path}")
        return True
    
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return False

def main():
    """Main function"""
    args = parse_args()
    
    # Convert the model
    success = convert_model(args.input_model, args.output_model)
    
    if success:
        print("\nModel conversion completed successfully!")
        print(f"You can now use the converted model for analysis with:")
        print(f"python scripts/run_analysis.py --teacher_model={os.path.basename(args.output_model)}")
    else:
        print("\nModel conversion failed.")
        sys.exit(1)

if __name__ == "__main__":
    main() 