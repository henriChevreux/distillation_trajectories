#!/usr/bin/env python3
"""
Helper script to run the diffusion model training on CPU.
This script provides a way to force CPU usage for all operations.
"""

import os
import argparse
import torch
import importlib
import sys

def main():
    """Main function to run training on CPU"""
    parser = argparse.ArgumentParser(
        description='Run diffusion model training on CPU',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('script', type=str, choices=['train_teacher', 'train_students', 'run_analysis'],
                        help='Which script to run on CPU')
    parser.add_argument('--args', type=str, default='',
                        help='Additional arguments to pass to the script')
    args = parser.parse_args()
    
    # Force CPU usage by setting environment variable
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Print device info
    print("\n" + "="*80)
    print("RUNNING ON CPU")
    print("="*80)
    print(f"Device being used: {torch.device('cpu')}")
    print(f"Script to run: {args.script}")
    
    # Add the project root directory to the Python path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    # Import and run the specified script
    try:
        if args.script == 'train_teacher':
            from scripts.train_teacher import main as script_main
        elif args.script == 'train_students':
            from scripts.train_students import main as script_main
        else:
            print(f"Unknown script: {args.script}")
            return
        
        # If additional arguments were provided, modify sys.argv
        if args.args:
            # Clear existing args and add the new ones
            sys.argv = [sys.argv[0]] + args.args.split()
        
        # Run the script
        print(f"Running {args.script} on CPU...")
        script_main()
        
    except Exception as e:
        print(f"Error running script: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
