"""
Attention map analysis for diffusion models
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

class AttentionHook:
    """Hook to extract attention maps from models"""
    def __init__(self):
        self.attention_maps = []
    
    def __call__(self, module, input, output):
        # Extract attention maps from the output
        # This assumes the attention mechanism returns attention weights
        # Adjust based on the actual model architecture
        self.attention_maps.append(output.detach())
    
    def clear(self):
        self.attention_maps = []

def register_attention_hooks(model):
    """
    Register hooks to extract attention maps from the model
    
    Args:
        model: Diffusion model
        
    Returns:
        List of hooks and list of handles
    """
    hooks = []
    handles = []
    
    # Find all attention modules in the model
    for name, module in model.named_modules():
        # This needs to be adjusted based on the actual model architecture
        # Here we're assuming the attention modules are named with 'attention'
        if 'attention' in name.lower():
            hook = AttentionHook()
            handle = module.register_forward_hook(hook)
            hooks.append(hook)
            handles.append(handle)
    
    return hooks, handles

def remove_hooks(handles):
    """
    Remove hooks from the model
    
    Args:
        handles: List of hook handles
    """
    for handle in handles:
        handle.remove()

def visualize_attention_maps(teacher_attention, student_attention, timestep, output_dir, size_factor):
    """
    Visualize attention maps
    
    Args:
        teacher_attention: Attention maps from teacher model
        student_attention: Attention maps from student model
        timestep: Timestep for the attention maps
        output_dir: Directory to save visualizations
        size_factor: Size factor of the student model for labeling
    """
    # Number of attention maps to visualize
    n_maps = min(4, len(teacher_attention))
    
    # Create figure
    fig, axes = plt.subplots(2, n_maps, figsize=(n_maps * 4, 8))
    fig.suptitle(f'Attention Maps Comparison (Size Factor: {size_factor}, Timestep: {timestep})', fontsize=16)
    
    # Create a custom colormap
    cmap = sns.color_palette("viridis", as_cmap=True)
    
    # Plot teacher attention maps
    for i in range(n_maps):
        # Get attention map
        attn_map = teacher_attention[i]
        
        # If attention map is multi-dimensional, take the mean across all dimensions except the last two
        if len(attn_map.shape) > 2:
            # Reshape to (batch_size, seq_len, seq_len)
            attn_map = attn_map.mean(dim=0)  # Average across batch
        
        # Plot attention map
        im = axes[0, i].imshow(attn_map.cpu().numpy(), cmap=cmap)
        axes[0, i].set_title(f'Teacher Layer {i+1}')
        axes[0, i].axis('off')
    
    # Plot student attention maps
    for i in range(n_maps):
        # Get attention map
        if i < len(student_attention):
            attn_map = student_attention[i]
            
            # If attention map is multi-dimensional, take the mean across all dimensions except the last two
            if len(attn_map.shape) > 2:
                # Reshape to (batch_size, seq_len, seq_len)
                attn_map = attn_map.mean(dim=0)  # Average across batch
            
            # Plot attention map
            im = axes[1, i].imshow(attn_map.cpu().numpy(), cmap=cmap)
            axes[1, i].set_title(f'Student Layer {i+1}')
        axes[1, i].axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7)
    cbar.set_label('Attention Weight')
    
    # Add row labels
    axes[0, 0].set_ylabel('Teacher', fontsize=14)
    axes[1, 0].set_ylabel('Student', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'attention_maps_t{timestep}_size_{size_factor}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def calculate_attention_similarity(teacher_attention, student_attention):
    """
    Calculate similarity between teacher and student attention maps
    
    Args:
        teacher_attention: Attention maps from teacher model
        student_attention: Attention maps from student model
        
    Returns:
        Dictionary of similarity metrics
    """
    # Number of attention maps to compare
    n_maps = min(len(teacher_attention), len(student_attention))
    
    # Calculate similarity metrics
    cosine_similarities = []
    mse_values = []
    
    for i in range(n_maps):
        # Get attention maps
        teacher_map = teacher_attention[i]
        student_map = student_attention[i]
        
        # If attention maps are multi-dimensional, take the mean across all dimensions except the last two
        if len(teacher_map.shape) > 2:
            teacher_map = teacher_map.mean(dim=0)  # Average across batch
        if len(student_map.shape) > 2:
            student_map = student_map.mean(dim=0)  # Average across batch
        
        # Flatten attention maps
        teacher_flat = teacher_map.view(-1)
        student_flat = student_map.view(-1)
        
        # Calculate cosine similarity
        teacher_norm = torch.nn.functional.normalize(teacher_flat, p=2, dim=0)
        student_norm = torch.nn.functional.normalize(student_flat, p=2, dim=0)
        cosine_sim = torch.sum(teacher_norm * student_norm).item()
        
        # Calculate MSE
        mse = torch.mean((teacher_map - student_map) ** 2).item()
        
        cosine_similarities.append(cosine_sim)
        mse_values.append(mse)
    
    # Calculate average metrics
    avg_cosine = np.mean(cosine_similarities)
    avg_mse = np.mean(mse_values)
    
    return {
        "cosine_similarities": cosine_similarities,
        "mse_values": mse_values,
        "avg_cosine": avg_cosine,
        "avg_mse": avg_mse
    }

def plot_attention_metrics_by_timestep(metrics_by_timestep, output_dir, size_factor):
    """
    Plot attention metrics by timestep
    
    Args:
        metrics_by_timestep: Dictionary of metrics by timestep
        output_dir: Directory to save visualizations
        size_factor: Size factor of the student model for labeling
    """
    # Extract timesteps and metrics
    timesteps = sorted(metrics_by_timestep.keys())
    cosine_values = [metrics_by_timestep[t]['avg_cosine'] for t in timesteps]
    mse_values = [metrics_by_timestep[t]['avg_mse'] for t in timesteps]
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    fig.suptitle(f'Attention Map Similarity by Timestep (Size Factor: {size_factor})', fontsize=16)
    
    # Plot cosine similarity
    axes[0].plot(timesteps, cosine_values, 'o-', color='blue', linewidth=2)
    axes[0].set_ylabel('Cosine Similarity')
    axes[0].set_title('Cosine Similarity (higher is better)')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot MSE
    axes[1].plot(timesteps, mse_values, 'o-', color='red', linewidth=2)
    axes[1].set_ylabel('MSE')
    axes[1].set_title('Mean Squared Error (lower is better)')
    axes[1].set_xlabel('Timestep')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'attention_metrics_by_timestep_size_{size_factor}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_attention_maps(teacher_model, student_model, config, output_dir=None, size_factor=None, fixed_samples=None):
    """
    Analyze attention maps of models
    
    Args:
        teacher_model: Teacher diffusion model
        student_model: Student diffusion model
        config: Configuration object
        output_dir: Directory to save visualizations
        size_factor: Size factor of the student model for labeling
        fixed_samples: Fixed samples to use for analysis (for consistent comparison)
        
    Returns:
        Dictionary of analysis results
    """
    if output_dir is None:
        output_dir = os.path.join(config.analysis_dir, "attention_maps", f"size_{size_factor}")
    
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Analyzing attention maps for size factor {size_factor}...")
    
    # Get device
    device = next(teacher_model.parameters()).device
    
    # Set models to evaluation mode
    teacher_model.eval()
    student_model.eval()
    
    # Get test dataset or use fixed samples
    if fixed_samples is not None:
        print(f"Using {len(fixed_samples)} fixed samples for consistent comparison")
        images = fixed_samples[:1].to(device)  # Use just one sample for attention analysis
    else:
        test_dataset = config.get_test_dataset()
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        images, _ = next(iter(test_loader))
        images = images.to(device)
    
    # Number of timesteps to analyze
    n_timesteps = 5
    timesteps_to_analyze = torch.linspace(0, config.timesteps - 1, n_timesteps, dtype=torch.long).to(device)
    
    # Register hooks to extract attention maps
    teacher_hooks, teacher_handles = register_attention_hooks(teacher_model)
    student_hooks, student_handles = register_attention_hooks(student_model)
    
    # Store metrics by timestep
    metrics_by_timestep = {}
    
    try:
        # Analyze each timestep
        for t in timesteps_to_analyze:
            # Create batch of same timestep
            timesteps = torch.full((images.size(0),), t, device=device, dtype=torch.long)
            
            # Clear hooks
            for hook in teacher_hooks:
                hook.clear()
            for hook in student_hooks:
                hook.clear()
            
            # Forward pass through teacher model
            with torch.no_grad():
                _ = teacher_model(images, timesteps)
            
            # Get teacher attention maps
            teacher_attention = []
            for hook in teacher_hooks:
                teacher_attention.extend(hook.attention_maps)
            
            # Forward pass through student model
            with torch.no_grad():
                _ = student_model(images, timesteps)
            
            # Get student attention maps
            student_attention = []
            for hook in student_hooks:
                student_attention.extend(hook.attention_maps)
            
            # Check if we have attention maps
            if not teacher_attention or not student_attention:
                print("  No attention maps found. The model may not have attention layers or they are not accessible.")
                return {"status": "no_attention_maps"}
            
            # Visualize attention maps
            visualize_attention_maps(teacher_attention, student_attention, t.item(), output_dir, size_factor)
            
            # Calculate similarity metrics
            metrics = calculate_attention_similarity(teacher_attention, student_attention)
            
            # Store metrics
            metrics_by_timestep[t.item()] = metrics
    
    finally:
        # Remove hooks
        remove_hooks(teacher_handles)
        remove_hooks(student_handles)
    
    # Plot metrics by timestep
    if metrics_by_timestep:
        plot_attention_metrics_by_timestep(metrics_by_timestep, output_dir, size_factor)
    
    # Calculate average metrics across all timesteps
    if metrics_by_timestep:
        avg_cosine = np.mean([metrics['avg_cosine'] for metrics in metrics_by_timestep.values()])
        avg_mse = np.mean([metrics['avg_mse'] for metrics in metrics_by_timestep.values()])
        
        # Save metrics
        results = {
            "avg_cosine": avg_cosine,
            "avg_mse": avg_mse,
            "metrics_by_timestep": metrics_by_timestep
        }
        
        # Save metrics to file
        with open(os.path.join(output_dir, f"attention_metrics_size_{size_factor}.txt"), "w") as f:
            f.write(f"Average Cosine Similarity: {avg_cosine:.6f}\n")
            f.write(f"Average MSE: {avg_mse:.6f}\n\n")
            f.write("Metrics by Timestep:\n")
            for t, metrics in sorted(metrics_by_timestep.items()):
                f.write(f"  Timestep {t}:\n")
                f.write(f"    Avg Cosine Similarity: {metrics['avg_cosine']:.6f}\n")
                f.write(f"    Avg MSE: {metrics['avg_mse']:.6f}\n")
        
        print(f"  Average Cosine Similarity: {avg_cosine:.6f}")
        print(f"  Average MSE: {avg_mse:.6f}")
        
        return results
    else:
        print("  No attention metrics calculated.")
        return {"status": "no_metrics"} 