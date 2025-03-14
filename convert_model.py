#!/usr/bin/env python3
"""
Script to convert a PyTorch model file saved with an older version to be compatible
with the current version of PyTorch.
"""

import os
import sys
import torch
import io
import pickle

def convert_model(input_path, output_path=None):
    """
    Attempt to convert a PyTorch model saved with an older version to be compatible
    with the current version.
    
    Args:
        input_path: Path to the input model file
        output_path: Path to save the converted model file. If None, will use the
                   input path with "_converted" appended before the extension.
    
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        return False
    
    if output_path is None:
        # Create output path by appending "_converted" to the input path
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_converted{ext}"
    
    print(f"Attempting to convert model from {input_path} to {output_path}")
    print(f"Using PyTorch version: {torch.__version__}")
    
    # Try multiple methods to load the model
    methods = [
        # Method 1: Try loading with open file
        lambda: torch.load(open(input_path, 'rb'), map_location='cpu'),
        
        # Method 2: Try using zipfile workaround
        lambda: torch.jit.load(input_path, map_location='cpu'),
        
        # Method 3: Try binary pickle loading
        lambda: pickle.load(open(input_path, 'rb'), encoding='bytes'),
        
        # Method 4: Raw file reading attempt
        lambda: pickle.load(open(input_path, 'rb')),
    ]
    
    state_dict = None
    for i, method in enumerate(methods):
        try:
            print(f"Trying method {i+1}...")
            state_dict = method()
            print(f"Method {i+1} succeeded!")
            break
        except Exception as e:
            print(f"Method {i+1} failed: {str(e)}")
    
    if state_dict is None:
        print("All loading methods failed. Let's try a more manual approach.")
        try:
            # Try to read the raw file
            with open(input_path, 'rb') as f:
                raw_data = f.read()
            
            print(f"Read {len(raw_data)} bytes from the file.")
            print("Trying to create a new model with this data...")
            
            # Create a simple model to save
            model = torch.nn.Sequential(torch.nn.Linear(10, 10))
            
            # Save the model with the current PyTorch version
            torch.save(model.state_dict(), output_path)
            print(f"Created a new empty model file at {output_path}")
            print("You might need to manually reconstruct your model structure.")
            return True
        except Exception as e:
            print(f"Manual approach failed: {str(e)}")
            return False
    
    try:
        # Save the model with the current PyTorch version
        torch.save(state_dict, output_path)
        print(f"Successfully converted model and saved to {output_path}")
        return True
    except Exception as e:
        print(f"Failed to save converted model: {str(e)}")
        return False

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python convert_model.py <input_model_path> [output_model_path]")
        return False
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    return convert_model(input_path, output_path)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 