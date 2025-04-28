"""
Script to check CUDA availability and display GPU information.

This script:
1. Checks if CUDA is available
2. Displays information about available GPUs
3. Runs a simple test to verify CUDA functionality
"""

import torch
import sys


def check_cuda():
    """
    Check CUDA availability and display GPU information.
    """
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("Number of GPUs:", torch.cuda.device_count())
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Capability: {torch.cuda.get_device_capability(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024 / 1024 / 1024:.2f} GB")
        
        # Run a simple test
        print("\nRunning a simple CUDA test...")
        try:
            x = torch.rand(1000, 1000).cuda()
            y = torch.rand(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("CUDA test successful!")
        except Exception as e:
            print(f"CUDA test failed: {e}")
    else:
        print("\nCUDA is not available. Possible reasons:")
        print("1. No NVIDIA GPU is installed")
        print("2. NVIDIA drivers are not installed or outdated")
        print("3. PyTorch was installed without CUDA support")
        
        print("\nTo fix this:")
        print("1. If you have an NVIDIA GPU, make sure the drivers are installed")
        print("2. Reinstall PyTorch with CUDA support:")
        print("   pip uninstall torch")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("   (Replace cu118 with your desired CUDA version)")


def main():
    """
    Main function to check CUDA availability.
    """
    check_cuda()


if __name__ == "__main__":
    main()
