#!/usr/bin/env python3
"""
Script to analyze the impact of editing on distilled and teacher diffusion models.
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.config import Config
from models import SimpleUNet, StudentUNet
from utils.diffusion import get_diffusion_params
from utils.trajectory_manager import TrajectoryManager

# Import editing modules
from editing.prompt_editing import apply_prompt_editing
from editing.masked_inpainting import apply_masked_inpainting
from editing.latent_manipulation import apply_latent_manipulation

# Import evaluation metrics
from evaluation.metrics import compute_lpips, compute_fid, compute_trajectory_divergence

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Analyze editing capabilities of diffusion models")
    parser.add_argument("--teacher_model", type=str, default=None, help="Path to teacher model")
    parser.add_argument("--student_model", type=str, default=None, help="Path to student model")
    parser.add_argument("--size_factor", type=float, default=1.0, help="Size factor of student model")
    parser.add_argument("--output_dir", type=str, default="results/editing", help="Output directory")
    parser.add_argument("--edit_mode", type=str, choices=["prompt", "inpainting", "latent", "all"], 
                        default="all", help="Editing mode to analyze")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()

def main():
    """Main function to run the editing analysis"""
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load configuration
    config = Config()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else 
        "mps" if torch.backends.mps.is_available() and config.mps_enabled else
        "cpu"
    )
    print(f"Using {device} device")
    
    # Load teacher model
    teacher_model_path = args.teacher_model or os.path.join('models/', 'model_epoch_10.pt')
    print(f"Loading teacher model from {teacher_model_path}...")
    teacher_model = SimpleUNet(config).to(device)
    teacher_model.load_state_dict(torch.load(teacher_model_path, map_location=device))
    teacher_model.eval()
    
    # Load student model
    student_model_path = args.student_model or os.path.join(
        'models/', f'student_model_size_{args.size_factor}_epoch_1.pt')
    print(f"Loading student model from {student_model_path}...")
    
    # Determine architecture type based on size factor
    architecture_type = 'full'
    if args.size_factor < 0.1:
        architecture_type = 'tiny'
    elif args.size_factor < 0.3:
        architecture_type = 'small'
    elif args.size_factor < 0.7:
        architecture_type = 'medium'
    
    student_model = StudentUNet(config, size_factor=args.size_factor, 
                               architecture_type=architecture_type).to(device)
    student_model.load_state_dict(torch.load(student_model_path, map_location=device))
    student_model.eval()
    
    # Create diffusion parameters
    teacher_params = get_diffusion_params(config.teacher_steps, config)
    student_params = get_diffusion_params(config.student_steps, config)
    
    # Run editing analysis based on selected mode
    results = {}
    
    if args.edit_mode in ["prompt", "all"]:
        print("\nRunning prompt-based editing analysis...")
        prompt_results = run_prompt_editing_analysis(
            teacher_model, student_model, 
            teacher_params, student_params,
            config, args, device
        )
        results["prompt"] = prompt_results
    
    if args.edit_mode in ["inpainting", "all"]:
        print("\nRunning masked inpainting analysis...")
        inpainting_results = run_inpainting_analysis(
            teacher_model, student_model, 
            teacher_params, student_params,
            config, args, device
        )
        results["inpainting"] = inpainting_results
    
    if args.edit_mode in ["latent", "all"]:
        print("\nRunning latent-space manipulation analysis...")
        latent_results = run_latent_manipulation_analysis(
            teacher_model, student_model, 
            teacher_params, student_params,
            config, args, device
        )
        results["latent"] = latent_results
    
    # Generate summary report
    generate_summary_report(results, args.output_dir, args.size_factor)
    
    print(f"\nEditing analysis complete. Results saved to {args.output_dir}")

def run_prompt_editing_analysis(teacher_model, student_model, 
                               teacher_params, student_params,
                               config, args, device):
    """Run prompt-based editing analysis"""
    print("\nRunning prompt-based editing analysis...")
    
    # Create output directory for prompt editing
    prompt_dir = os.path.join(args.output_dir, "prompt_editing")
    os.makedirs(prompt_dir, exist_ok=True)
    
    # Define original and edited prompts
    # In a real implementation, these would be used with a text-conditioned model
    prompt_pairs = [
        ("A dog", "A dog with a hat"),
        ("A cat", "A cat with sunglasses"),
        ("A house", "A house with a red roof"),
        ("A car", "A sports car"),
        ("A person", "A person smiling")
    ]
    
    # Run prompt editing for teacher model
    print("Running prompt editing for teacher model...")
    teacher_results = []
    
    for original_prompt, edited_prompt in prompt_pairs:
        result = apply_prompt_editing(
            teacher_model, 
            teacher_params, 
            original_prompt, 
            edited_prompt, 
            config, 
            device
        )
        teacher_results.append(result)
        
        # Visualize results
        teacher_prompt_dir = os.path.join(prompt_dir, "teacher", f"{original_prompt}_to_{edited_prompt}")
        os.makedirs(teacher_prompt_dir, exist_ok=True)
        visualize_prompt_editing(result, teacher_prompt_dir)
    
    # Run prompt editing for student model
    print("Running prompt editing for student model...")
    student_results = []
    
    for original_prompt, edited_prompt in prompt_pairs:
        result = apply_prompt_editing(
            student_model, 
            student_params, 
            original_prompt, 
            edited_prompt, 
            config, 
            device
        )
        student_results.append(result)
        
        # Visualize results
        student_prompt_dir = os.path.join(prompt_dir, "student", f"{original_prompt}_to_{edited_prompt}")
        os.makedirs(student_prompt_dir, exist_ok=True)
        visualize_prompt_editing(result, student_prompt_dir, args.size_factor)
    
    # Compute metrics
    metrics = {}
    
    # LPIPS
    lpips_distances = []
    for teacher_result, student_result in zip(teacher_results, student_results):
        # Compare edited images
        lpips_distance = compute_lpips(
            teacher_result["edited_image"], 
            student_result["edited_image"], 
            device
        )
        lpips_distances.append(lpips_distance)
    
    metrics["lpips"] = lpips_distances
    
    # FID
    teacher_edited_images = [result["edited_image"] for result in teacher_results]
    student_edited_images = [result["edited_image"] for result in student_results]
    
    fid_score = compute_fid(teacher_edited_images, student_edited_images, device)
    metrics["fid"] = fid_score
    
    # Trajectory divergence
    trajectory_divergences = []
    for teacher_result, student_result in zip(teacher_results, student_results):
        if "edited_trajectory" in teacher_result and "edited_trajectory" in student_result:
            divergence = compute_trajectory_divergence(
                teacher_result["edited_trajectory"],
                student_result["edited_trajectory"]
            )
            trajectory_divergences.append(divergence)
    
    if trajectory_divergences:
        # Compute average metrics across all trajectories
        avg_divergence = {
            "distances": np.mean([d["distances"] for d in trajectory_divergences], axis=0).tolist(),
            "similarities": np.mean([d["similarities"] for d in trajectory_divergences], axis=0).tolist(),
            "avg_distance": np.mean([d["avg_distance"] for d in trajectory_divergences]),
            "max_distance": np.mean([d["max_distance"] for d in trajectory_divergences]),
            "avg_similarity": np.mean([d["avg_similarity"] for d in trajectory_divergences]),
            "min_similarity": np.mean([d["min_similarity"] for d in trajectory_divergences]),
            "length_ratio": np.mean([d["length_ratio"] for d in trajectory_divergences])
        }
        metrics["trajectory_divergence"] = avg_divergence
    
    # Visualize metrics
    metrics_dir = os.path.join(prompt_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    visualize_metrics(metrics, metrics_dir, args.size_factor)
    
    return {
        "teacher_results": teacher_results,
        "student_results": student_results,
        "metrics": metrics,
        "status": "completed"
    }

def run_inpainting_analysis(teacher_model, student_model, 
                           teacher_params, student_params,
                           config, args, device):
    """Run masked inpainting analysis"""
    print("\nRunning masked inpainting analysis...")
    
    # Create output directory for inpainting
    inpainting_dir = os.path.join(args.output_dir, "inpainting")
    os.makedirs(inpainting_dir, exist_ok=True)
    
    # Generate original images
    print("Generating original images...")
    original_images = []
    
    for i in range(args.num_samples):
        # Set seed for reproducibility
        torch.manual_seed(args.seed + i)
        
        # Generate image
        image, _ = generate_image(teacher_model, teacher_params, config, device)
        original_images.append(image)
    
    # Create masks
    masks = []
    for i in range(args.num_samples):
        mask = create_random_mask(config.image_size, config.image_size)
        masks.append(mask)
    
    # Run inpainting for teacher model
    print("Running inpainting for teacher model...")
    teacher_results = []
    
    for i, (image, mask) in enumerate(zip(original_images, masks)):
        result = apply_masked_inpainting(
            teacher_model, 
            teacher_params, 
            image, 
            mask, 
            config, 
            device
        )
        teacher_results.append(result)
        
        # Visualize results
        teacher_inpainting_dir = os.path.join(inpainting_dir, "teacher", f"sample_{i+1}")
        os.makedirs(teacher_inpainting_dir, exist_ok=True)
        visualize_inpainting(result, teacher_inpainting_dir)
    
    # Run inpainting for student model
    print("Running inpainting for student model...")
    student_results = []
    
    for i, (image, mask) in enumerate(zip(original_images, masks)):
        result = apply_masked_inpainting(
            student_model, 
            student_params, 
            image, 
            mask, 
            config, 
            device
        )
        student_results.append(result)
        
        # Visualize results
        student_inpainting_dir = os.path.join(inpainting_dir, "student", f"sample_{i+1}")
        os.makedirs(student_inpainting_dir, exist_ok=True)
        visualize_inpainting(result, student_inpainting_dir, args.size_factor)
    
    # Compute metrics
    metrics = {}
    
    # LPIPS
    lpips_distances = []
    for teacher_result, student_result in zip(teacher_results, student_results):
        # Compare inpainted images
        lpips_distance = compute_lpips(
            teacher_result["inpainted_image"], 
            student_result["inpainted_image"], 
            device
        )
        lpips_distances.append(lpips_distance)
    
    metrics["lpips"] = lpips_distances
    
    # FID
    teacher_inpainted_images = [result["inpainted_image"] for result in teacher_results]
    student_inpainted_images = [result["inpainted_image"] for result in student_results]
    
    fid_score = compute_fid(teacher_inpainted_images, student_inpainted_images, device)
    metrics["fid"] = fid_score
    
    # Trajectory divergence
    trajectory_divergences = []
    for teacher_result, student_result in zip(teacher_results, student_results):
        if "trajectory" in teacher_result and "trajectory" in student_result:
            divergence = compute_trajectory_divergence(
                teacher_result["trajectory"],
                student_result["trajectory"]
            )
            trajectory_divergences.append(divergence)
    
    if trajectory_divergences:
        # Compute average metrics across all trajectories
        avg_divergence = {
            "distances": np.mean([d["distances"] for d in trajectory_divergences], axis=0).tolist(),
            "similarities": np.mean([d["similarities"] for d in trajectory_divergences], axis=0).tolist(),
            "avg_distance": np.mean([d["avg_distance"] for d in trajectory_divergences]),
            "max_distance": np.mean([d["max_distance"] for d in trajectory_divergences]),
            "avg_similarity": np.mean([d["avg_similarity"] for d in trajectory_divergences]),
            "min_similarity": np.mean([d["min_similarity"] for d in trajectory_divergences]),
            "length_ratio": np.mean([d["length_ratio"] for d in trajectory_divergences])
        }
        metrics["trajectory_divergence"] = avg_divergence
    
    # Visualize metrics
    metrics_dir = os.path.join(inpainting_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    visualize_metrics(metrics, metrics_dir, args.size_factor)
    
    return {
        "teacher_results": teacher_results,
        "student_results": student_results,
        "metrics": metrics,
        "status": "completed"
    }

def run_latent_manipulation_analysis(teacher_model, student_model, 
                                    teacher_params, student_params,
                                    config, args, device):
    """Run latent-space manipulation analysis"""
    print("\nRunning latent-space manipulation analysis...")
    
    # Create output directory for latent manipulation
    latent_dir = os.path.join(args.output_dir, "latent_manipulation")
    os.makedirs(latent_dir, exist_ok=True)
    
    # Define manipulation strengths
    strengths = [0.5, 1.0, 1.5]
    
    # Find semantic directions (optional)
    # In a real implementation, this would find meaningful directions
    # For simplicity, we'll use random directions
    print("Generating semantic directions...")
    directions = {}
    
    # Generate a few random directions
    for i in range(3):
        latent_dim = config.channels * config.image_size * config.image_size
        direction = torch.randn(latent_dim, device=device)
        # Normalize the direction
        direction = direction / torch.norm(direction)
        directions[f"direction_{i+1}"] = direction
    
    # Run latent manipulation for teacher model
    print("Running latent manipulation for teacher model...")
    teacher_results = {}
    
    for direction_name, direction in directions.items():
        direction_results = {}
        
        for strength in strengths:
            result = apply_latent_manipulation(
                teacher_model, 
                teacher_params, 
                direction, 
                strength, 
                config, 
                device,
                num_samples=args.num_samples
            )
            direction_results[strength] = result
            
            # Visualize results
            teacher_latent_dir = os.path.join(latent_dir, "teacher", direction_name, f"strength_{strength}")
            os.makedirs(teacher_latent_dir, exist_ok=True)
            visualize_latent_manipulation(result, teacher_latent_dir)
        
        teacher_results[direction_name] = direction_results
    
    # Run latent manipulation for student model
    print("Running latent manipulation for student model...")
    student_results = {}
    
    for direction_name, direction in directions.items():
        direction_results = {}
        
        for strength in strengths:
            result = apply_latent_manipulation(
                student_model, 
                student_params, 
                direction, 
                strength, 
                config, 
                device,
                num_samples=args.num_samples
            )
            direction_results[strength] = result
            
            # Visualize results
            student_latent_dir = os.path.join(latent_dir, "student", direction_name, f"strength_{strength}")
            os.makedirs(student_latent_dir, exist_ok=True)
            visualize_latent_manipulation(result, student_latent_dir, args.size_factor)
        
        student_results[direction_name] = direction_results
    
    # Compute metrics
    all_metrics = {}
    
    for direction_name in directions:
        direction_metrics = {}
        
        for strength in strengths:
            teacher_result = teacher_results[direction_name][strength]
            student_result = student_results[direction_name][strength]
            
            metrics = {}
            
            # LPIPS
            lpips_distances = []
            for t_img, s_img in zip(teacher_result["manipulated_images"], student_result["manipulated_images"]):
                lpips_distance = compute_lpips(t_img, s_img, device)
                lpips_distances.append(lpips_distance)
            
            metrics["lpips"] = lpips_distances
            
            # FID
            fid_score = compute_fid(
                teacher_result["manipulated_images"], 
                student_result["manipulated_images"], 
                device
            )
            metrics["fid"] = fid_score
            
            # Trajectory divergence
            if "trajectories" in teacher_result and "trajectories" in student_result:
                trajectory_divergences = []
                
                for t_traj, s_traj in zip(teacher_result["trajectories"], student_result["trajectories"]):
                    divergence = compute_trajectory_divergence(
                        t_traj["manipulated"],
                        s_traj["manipulated"]
                    )
                    trajectory_divergences.append(divergence)
                
                if trajectory_divergences:
                    # Compute average metrics across all trajectories
                    avg_divergence = {
                        "distances": np.mean([d["distances"] for d in trajectory_divergences], axis=0).tolist(),
                        "similarities": np.mean([d["similarities"] for d in trajectory_divergences], axis=0).tolist(),
                        "avg_distance": np.mean([d["avg_distance"] for d in trajectory_divergences]),
                        "max_distance": np.mean([d["max_distance"] for d in trajectory_divergences]),
                        "avg_similarity": np.mean([d["avg_similarity"] for d in trajectory_divergences]),
                        "min_similarity": np.mean([d["min_similarity"] for d in trajectory_divergences]),
                        "length_ratio": np.mean([d["length_ratio"] for d in trajectory_divergences])
                    }
                    metrics["trajectory_divergence"] = avg_divergence
            
            # Visualize metrics
            metrics_dir = os.path.join(latent_dir, "metrics", direction_name, f"strength_{strength}")
            os.makedirs(metrics_dir, exist_ok=True)
            visualize_metrics(metrics, metrics_dir, args.size_factor)
            
            direction_metrics[strength] = metrics
        
        all_metrics[direction_name] = direction_metrics
    
    return {
        "teacher_results": teacher_results,
        "student_results": student_results,
        "metrics": all_metrics,
        "status": "completed"
    }

def generate_summary_report(results, output_dir, size_factor):
    """Generate a summary report of editing analysis results"""
    print("Generating summary report...")
    
    # Create summary directory
    summary_dir = os.path.join(output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Write summary text file
    with open(os.path.join(summary_dir, "summary.txt"), "w") as f:
        f.write(f"Editing Analysis Summary (Size Factor: {size_factor})\n")
        f.write("=" * 50 + "\n\n")
        
        for edit_mode, mode_results in results.items():
            f.write(f"{edit_mode.upper()} EDITING MODE\n")
            f.write("-" * 30 + "\n")
            f.write(f"Status: {mode_results.get('status', 'Unknown')}\n\n")
    
    print(f"Summary report saved to {summary_dir}")

if __name__ == "__main__":
    main() 