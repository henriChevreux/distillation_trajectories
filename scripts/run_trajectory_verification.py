#!/usr/bin/env python3
"""
Script to run trajectory verification between teacher and student models.
This script verifies that the teacher and 1.0 student models produce identical trajectories.
"""

import os
import sys
import argparse

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from analysis.trajectory_verification import main as run_verification

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Verify that teacher and 1.0 student models produce identical trajectories',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    return parser.parse_args()

def main():
    """Main function"""
    parse_args()  # Parse arguments
    run_verification()  # Run verification

if __name__ == "__main__":
    main() 