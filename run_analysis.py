#!/usr/bin/env python3
import os
import subprocess
import argparse
from pathlib import Path

def run_script(script_path, args=None):
    """Run a Python script and handle any errors."""
    try:
        cmd = ["python", script_path]
        if args:
            cmd.extend(args)
        print(f"\nRunning {script_path}...")
        subprocess.run(cmd, check=True)
        print(f"Successfully completed {script_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Run all analysis scripts")
    parser.add_argument("--skip", nargs="+", help="Scripts to skip (without .py extension)")
    parser.add_argument("--teacher_model", type=str, help="Path to teacher model (e.g., 'model_epoch_10.pt')")
    args = parser.parse_args()

    # Get the directory containing this script
    script_dir = Path(__file__).parent
    analysis_dir = script_dir / "scripts" / "analysis"

    # List of analysis scripts to run
    analysis_scripts = [
        "analyze_heatmaps.py",
        "analyze_trajectories.py",
        "analyze_radars.py",
        "analyze_effectiveness.py"
    ]

    # Filter out skipped scripts
    if args.skip:
        analysis_scripts = [s for s in analysis_scripts 
                          if Path(s).stem not in args.skip]

    # Prepare common arguments for all scripts
    script_args = []
    if args.teacher_model:
        script_args.extend(["--teacher_model", args.teacher_model])

    # Run each analysis script
    for script in analysis_scripts:
        script_path = analysis_dir / script
        if script_path.exists():
            run_script(script_path, script_args)
        else:
            print(f"Warning: {script} not found")

if __name__ == "__main__":
    main() 