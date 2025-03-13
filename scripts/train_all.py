#!/usr/bin/env python3
"""
Script to run both teacher and student training sequentially.
This allows for unattended training of all models.
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_command(cmd, description):
    """Run a command and log its output"""
    print("\n" + "="*80)
    print(f"Starting {description} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            shell=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        
        if process.returncode != 0:
            print(f"\nError in {description}! Exit code: {process.returncode}")
            sys.exit(process.returncode)
            
        print(f"\n{description} completed successfully!")
        
    except Exception as e:
        print(f"\nError running {description}: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Run complete training pipeline (teacher + students)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    args = parser.parse_args()
    
    # Prepare CPU flag if needed
    cpu_flag = "--cpu" if args.cpu else ""
    
    print("\n" + "="*80)
    print("STARTING COMPLETE TRAINING PIPELINE")
    print("="*80)
    print("\nThis script will:")
    print("1. Train the teacher model")
    print("2. Train all student models")
    print(f"Device: {'CPU' if args.cpu else 'GPU if available'}")
    
    # 1. Train Teacher
    teacher_cmd = f"python scripts/train_teacher.py {cpu_flag}"
    run_command(teacher_cmd, "Teacher Training")
    
    # 2. Train Students
    student_cmd = f"python scripts/train_students.py {cpu_flag}"
    run_command(student_cmd, "Student Training")
    
    print("\n" + "="*80)
    print("COMPLETE TRAINING PIPELINE FINISHED SUCCESSFULLY!")
    print("="*80 + "\n")
    print("You can now run analysis with:")
    print("    python scripts/run_analysis.py --all")

if __name__ == "__main__":
    main() 