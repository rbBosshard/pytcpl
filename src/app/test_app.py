import torch

# Check if a GPU is available
if torch.cuda.is_available():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"GPU(s) are available. Number of GPUs: {num_gpus}")
else:
    print("No GPU available. Using CPU.")
