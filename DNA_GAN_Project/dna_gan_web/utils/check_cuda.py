"""
Utility script to check CUDA availability.
"""

import torch
import sys


def check_cuda():
    """
    Check if CUDA is available and print GPU information.
    
    Returns:
        bool: True if CUDA is available, False otherwise.
    """
    cuda_available = torch.cuda.is_available()
    
    print("CUDA Available:", cuda_available)
    
    if cuda_available:
        print("CUDA Version:", torch.version.cuda)
        print("Number of GPUs:", torch.cuda.device_count())
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
        # Print GPU memory
        try:
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
        except:
            print("Could not get GPU memory information")
    
    return cuda_available


if __name__ == "__main__":
    cuda_available = check_cuda()
    
    # Exit with status code 0 if CUDA is available, 1 otherwise
    sys.exit(0 if cuda_available else 1)
