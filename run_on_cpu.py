"""
Helper script to run the diffusion model distillation on CPU.
This script imports the main module and forces CPU usage.
"""

import torch
import importlib

# Import the main module
try:
    import diffusion_distillation as dd
    print("Successfully imported diffusion_distillation module")
except Exception as e:
    print(f"Error importing module: {e}")
    exit(1)

# Force CPU usage
print("Original device setting:", dd.device)
dd.device = torch.device("cpu")
print("Forced device to CPU:", dd.device)

# Make sure all device-specific code uses the new setting
print("Running the model on CPU...")

# Call the main function
if __name__ == "__main__":
    try:
        dd.main()
    except Exception as e:
        print(f"Error running main function: {e}")
        import traceback
        traceback.print_exc()