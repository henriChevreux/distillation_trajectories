"""
Script to clean up and reorganize output files according to the new directory structure.
This script moves files from the old locations to the new output/analysis folder structure.
"""

import os
import shutil
import argparse
from tqdm import tqdm
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

def move_files(source_dir, target_dir, file_extensions=None, dry_run=False):
    """
    Move files from source directory to target directory.
    
    Args:
        source_dir: Source directory to move files from
        target_dir: Target directory to move files to
        file_extensions: List of file extensions to move (e.g., ['.png', '.jpg'])
        dry_run: If True, just print what would be done without actually moving files
    """
    if not os.path.exists(source_dir):
        print(f"Source directory does not exist: {source_dir}")
        return
    
    if not dry_run:
        os.makedirs(target_dir, exist_ok=True)
    
    # Get all files in the source directory
    files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    
    # Filter by extensions if specified
    if file_extensions:
        files = [f for f in files if any(f.endswith(ext) for ext in file_extensions)]
    
    # Move each file
    for file in tqdm(files, desc=f"Moving files from {source_dir} to {target_dir}"):
        source_path = os.path.join(source_dir, file)
        target_path = os.path.join(target_dir, file)
        
        if dry_run:
            print(f"Would move {source_path} to {target_path}")
        else:
            # If target file exists, remove it first
            if os.path.exists(target_path):
                os.remove(target_path)
            
            # Move the file
            shutil.move(source_path, target_path)

def remove_directory_if_empty(directory, dry_run=False):
    """
    Remove a directory if it is empty.
    
    Args:
        directory: Directory to check and remove if empty
        dry_run: If True, just print what would be done without actually removing
    """
    if not os.path.exists(directory):
        return
    
    # Check if directory is empty
    if not os.listdir(directory):
        if dry_run:
            print(f"Would remove empty directory: {directory}")
        else:
            print(f"Removing empty directory: {directory}")
            os.rmdir(directory)

def main():
    """Main function to reorganize output files"""
    parser = argparse.ArgumentParser(description='Reorganize output files to new directory structure')
    parser.add_argument('--dry_run', action='store_true', help='Print actions without actually moving files')
    args = parser.parse_args()
    
    # Get config
    config = Config()
    
    # Define old and new directories
    moves = [
        {
            'source': os.path.join(config.output_dir, "model_comparisons"),
            'target': config.model_comparisons_dir,
            'extensions': ['.png', '.jpg', '.csv', '.txt', '.json']
        },
        {
            'source': os.path.join(config.output_dir, "time_dependent"),
            'target': config.time_dependent_dir,
            'extensions': ['.png', '.jpg', '.csv', '.txt', '.json']
        },
        {
            'source': os.path.join(config.output_dir, "size_dependent"),
            'target': config.size_dependent_dir,
            'extensions': ['.png', '.jpg', '.csv', '.txt', '.json']
        },
        {
            'source': os.path.join(config.output_dir, "metrics"),
            'target': config.metrics_dir,
            'extensions': ['.png', '.jpg', '.csv', '.txt', '.json']
        },
        {
            'source': os.path.join(config.output_dir, "dimensionality"),
            'target': config.dimensionality_dir,
            'extensions': ['.png', '.jpg', '.csv', '.txt', '.json']
        },
        {
            'source': os.path.join(config.output_dir, "latent_space"),
            'target': config.latent_space_dir,
            'extensions': ['.png', '.jpg', '.csv', '.txt', '.json']
        }
    ]
    
    print(f"Starting cleanup at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Move files
    for move in moves:
        source_dir = move['source']
        target_dir = move['target']
        extensions = move.get('extensions', None)
        
        print(f"\nProcessing: {source_dir} -> {target_dir}")
        move_files(source_dir, target_dir, extensions, args.dry_run)
        
        # Check if the source directory has subdirectories
        if os.path.exists(source_dir):
            for subdir in os.listdir(source_dir):
                subdir_path = os.path.join(source_dir, subdir)
                if os.path.isdir(subdir_path):
                    target_subdir = os.path.join(target_dir, subdir)
                    print(f"  Subdir: {subdir_path} -> {target_subdir}")
                    move_files(subdir_path, target_subdir, extensions, args.dry_run)
                    remove_directory_if_empty(subdir_path, args.dry_run)
        
        # Try to remove the source directory if it's empty
        remove_directory_if_empty(source_dir, args.dry_run)
    
    print("\nCleanup complete!")
    if args.dry_run:
        print("This was a dry run. No files were actually moved.")
        print("Run without --dry_run to actually perform the operations.")

if __name__ == "__main__":
    main() 