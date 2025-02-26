import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm

# Import the main diffusion code file
# Assuming the main code is saved as diffusion_distillation.py
try:
    import diffusion_distillation as dd
except Exception as e:
    print(f"Error importing diffusion_distillation module: {e}")
    print("Make sure diffusion_distillation.py is in the same directory as this test script.")
    exit(1)

# Set up a smaller configuration for quick tests
class TestConfig(dd.Config):
    def __init__(self):
        super().__init__()
        # Override with smaller values for quick testing
        self.batch_size = 16
        self.timesteps = 20
        self.teacher_steps = 20
        self.student_steps = 5
        self.epochs = 2
        self.save_interval = 1
        
        # Create test directories
        self.results_dir = "test_results"
        self.models_dir = "test_models"
        self.trajectory_dir = "test_trajectories"
        self.create_directories()

# Test 1: Check if the device setup works correctly
def test_device_setup():
    print("\n--- Testing Device Setup ---")
    original_device = dd.device
    
    # Force CPU for testing to avoid MPS issues
    print("Forcing CPU usage for testing to avoid MPS-related errors")
    dd.device = torch.device("cpu")
    
    test_tensor = torch.randn(10, 10)
    test_tensor = test_tensor.to(dd.device)
    print(f"✓ Successfully created tensor on {dd.device}")
    
    print(f"Original device was: {original_device}")
    print(f"Testing device is: {dd.device}")
    
    return True

# Test 2: Check model initialization
def test_model_init():
    print("\n--- Testing Model Initialization ---")
    config = TestConfig()
    try:
        model = dd.SimpleUNet(config).to(dd.device)
        print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass with proper error handling
        try:
            dummy_input = torch.randn(2, config.channels, config.image_size, config.image_size).to(dd.device)
            dummy_t = torch.randint(0, config.timesteps, (2,)).to(dd.device)
            
            output = model(dummy_input, dummy_t)
            print(f"✓ Forward pass successful, output shape: {output.shape}")
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            print("  This might be due to dimension mismatches in the model architecture.")
            print("  Attempting to fix common issues...")
            
            # Try to diagnose and fix the issue
            if "match the size of tensor" in str(e) and "dimension" in str(e):
                print("  Issue detected: Tensor dimension mismatch in model.")
                print("  Please update the time embedding method in the SimpleUNet class.")
            
            return None, None
        
        return model, config
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return None, None

# Test 3: Check diffusion parameters
def test_diffusion_params():
    print("\n--- Testing Diffusion Parameters ---")
    config = TestConfig()
    try:
        params = dd.get_diffusion_params(config.timesteps)
        print("✓ Diffusion parameters generated")
        
        # Check if all expected keys are present
        expected_keys = ['betas', 'alphas_cumprod', 'sqrt_recip_alphas', 
                         'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 
                         'posterior_variance']
        
        all_keys_present = all(key in params for key in expected_keys)
        print(f"✓ All expected parameters are present: {all_keys_present}")
        
        return params
    except Exception as e:
        print(f"✗ Diffusion parameters failed: {e}")
        return None

# Test 4: Check forward diffusion process
def test_forward_diffusion(config, params):
    print("\n--- Testing Forward Diffusion Process ---")
    try:
        # Create a sample image (batch of 1)
        x_start = torch.zeros(1, config.channels, config.image_size, config.image_size).to(dd.device)
        x_start[:, :, 10:20, 10:20] = 1.0  # Create a simple white square on black background
        
        # Apply diffusion at different timesteps
        timesteps = [0, 5, 10, 15, 19]  # Using the max timestep from TestConfig
        
        fig, axs = plt.subplots(1, len(timesteps), figsize=(15, 3))
        for i, t_step in enumerate(timesteps):
            t = torch.tensor([t_step]).to(dd.device)
            x_noisy, _ = dd.q_sample(x_start, t, params)
            
            # Visualize
            img = x_noisy[0].detach().cpu()
            img = (img + 1) / 2  # Convert from [-1, 1] to [0, 1]
            axs[i].imshow(img[0], cmap='gray')
            axs[i].set_title(f"t={t_step}")
            axs[i].axis('off')
            
        plt.tight_layout()
        os.makedirs('test_results', exist_ok=True)
        plt.savefig('test_results/forward_diffusion_test.png')
        plt.close()
        
        print("✓ Forward diffusion process tested and visualization saved")
        return True
    except Exception as e:
        print(f"✗ Forward diffusion test failed: {e}")
        return False

# Test 5: Check data loader
def test_data_loader():
    print("\n--- Testing Data Loader ---")
    config = TestConfig()
    try:
        loader = dd.get_data_loader(config)
        batch = next(iter(loader))
        images, labels = batch
        
        print(f"✓ Data loader working, batch shape: {images.shape}")
        
        # Visualize a few examples
        fig, axs = plt.subplots(1, 4, figsize=(12, 3))
        for i in range(4):
            img = images[i].detach().cpu()
            img = (img + 1) / 2  # Convert from [-1, 1] to [0, 1]
            axs[i].imshow(img[0], cmap='gray')
            axs[i].set_title(f"Label: {labels[i].item()}")
            axs[i].axis('off')
            
        plt.tight_layout()
        os.makedirs('test_results', exist_ok=True)
        plt.savefig('test_results/dataloader_test.png')
        plt.close()
        
        print("✓ Data visualization saved")
        return loader
    except Exception as e:
        print(f"✗ Data loader test failed: {e}")
        return None

# Test 6: Check a single training step
def test_training_step(model, config, params):
    print("\n--- Testing Single Training Step ---")
    try:
        # Make sure we're on CPU for this test to avoid MPS issues
        model = model.to("cpu")
        
        # Get a single batch
        loader = dd.get_data_loader(config)
        images, _ = next(iter(loader))
        images = images.to("cpu")  # Move to CPU
        
        # Sample random timesteps
        t = torch.randint(0, config.timesteps, (images.shape[0],), device="cpu").long()
        
        # Calculate loss
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        optimizer.zero_grad()
        
        loss = dd.p_losses(model, images, t, params)
        loss.backward()
        optimizer.step()
        
        print(f"✓ Training step completed with loss: {loss.item()}")
        return True
    except Exception as e:
        print(f"✗ Training step failed: {e}")
        print("  This may be due to MPS compatibility issues. Try running the full code with CPU:")
        print("  1. Set 'use_mps = False' in diffusion_distillation.py")
        print("  2. Run the main script with CPU: python diffusion_distillation.py")
        return False

# Test 7: Quick sampling test
def test_sampling(model, config, params):
    print("\n--- Testing Sampling Process ---")
    try:
        # Set model to eval mode
        model.eval()
        
        # Generate a small sample
        with torch.no_grad():
            samples = dd.p_sample_loop(
                model, 
                shape=(4, config.channels, config.image_size, config.image_size),
                timesteps=5,  # Use fewer steps for quick testing
                diffusion_params=params
            )
        
        # Visualize samples
        fig, axs = plt.subplots(1, 4, figsize=(12, 3))
        for i in range(4):
            img = samples[i].detach().cpu()
            img = (img + 1) / 2  # Convert from [-1, 1] to [0, 1]
            axs[i].imshow(img[0], cmap='gray')
            axs[i].set_title(f"Sample {i+1}")
            axs[i].axis('off')
            
        plt.tight_layout()
        plt.savefig('test_results/sampling_test.png')
        plt.close()
        
        print("✓ Sampling process tested and visualization saved")
        return True
    except Exception as e:
        print(f"✗ Sampling test failed: {e}")
        return False

# Main test function
def run_tests():
    print("Starting tests for diffusion model distillation code...")
    
    # Run all tests
    test_device_setup()
    model, config = test_model_init()
    
    if model is None or config is None:
        print("❌ Critical test failed. Stopping further tests.")
        return
    
    params = test_diffusion_params()
    if params is None:
        print("❌ Critical test failed. Stopping further tests.")
        return
    
    test_forward_diffusion(config, params)
    test_data_loader()
    
    if test_training_step(model, config, params):
        test_sampling(model, config, params)
    
    print("\nAll tests complete!")
    print("To run the full training pipeline, execute the main diffusion_distillation.py script.")

if __name__ == "__main__":
    run_tests()