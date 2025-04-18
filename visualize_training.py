"""
Visualize training progress for DNA sequence generation using Gumbel-Softmax GAN.

This script loads the latest checkpoint and visualizes the training progress.
"""

import os
import re
import argparse
import torch
import matplotlib.pyplot as plt

def find_latest_checkpoint(checkpoint_dir='checkpoints'):
    """Find the latest checkpoint in the checkpoint directory."""
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist.")
        return None
    
    # Get all checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                        if (f.startswith('checkpoint_epoch_') or f == 'final_model.pt') 
                        and f.endswith('.pt')]
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}.")
        return None
    
    # Check for final model
    if 'final_model.pt' in checkpoint_files:
        return os.path.join(checkpoint_dir, 'final_model.pt')
    
    # Extract epoch numbers
    epoch_numbers = []
    for file in checkpoint_files:
        match = re.search(r'checkpoint_epoch_(\d+)\.pt', file)
        if match:
            epoch_numbers.append(int(match.group(1)))
    
    if not epoch_numbers:
        print(f"Could not extract epoch numbers from checkpoint files.")
        return None
    
    # Find the latest epoch
    latest_epoch = max(epoch_numbers)
    latest_checkpoint = os.path.join(checkpoint_dir, f"checkpoint_epoch_{latest_epoch}.pt")
    
    print(f"Found latest checkpoint: {latest_checkpoint} (Epoch {latest_epoch})")
    return latest_checkpoint

def plot_training_history(history, save_path=None):
    """
    Plot training history.
    
    Args:
        history (dict): Dictionary containing training history.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(history['generator_loss'], label='Generator Loss')
    plt.plot(history['discriminator_loss'], label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator and Discriminator Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot discriminator accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['discriminator_accuracy'], label='Discriminator Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Discriminator Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot GC content
    plt.subplot(2, 2, 3)
    plt.plot(history['gc_content'], label='GC Content')
    plt.xlabel('Epoch')
    plt.ylabel('GC Content')
    plt.title('GC Content')
    plt.legend()
    plt.grid(True)
    
    # Plot Moran's I and diversity
    plt.subplot(2, 2, 4)
    plt.plot(history['morans_i'], label="Moran's I")
    plt.plot(history['diversity'], label='Diversity')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title("Moran's I and Diversity")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

def main():
    """Main function to visualize training progress."""
    parser = argparse.ArgumentParser(description='Visualize DNA sequence generation training progress')
    
    # Parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', 
                        help='Directory containing checkpoints')
    parser.add_argument('--save_path', type=str, default='images/training_history.png', 
                        help='Path to save the plot')
    
    args = parser.parse_args()
    
    # Find the latest checkpoint
    latest_checkpoint = find_latest_checkpoint(args.checkpoint_dir)
    
    if latest_checkpoint is None:
        print("No checkpoint found. Cannot visualize training progress.")
        return
    
    # Load the checkpoint
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')
    
    # Extract the history
    history = checkpoint.get('history', None)
    
    if history is None:
        print(f"No training history found in {latest_checkpoint}.")
        return
    
    # Plot the training history
    plot_training_history(history, args.save_path)

if __name__ == '__main__':
    main()
