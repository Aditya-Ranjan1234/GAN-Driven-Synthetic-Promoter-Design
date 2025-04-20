"""
Script for visualizing the training progress of the improved DNA sequence generation model.

This script provides a command-line interface for visualizing the training progress.
"""

import argparse
import torch
import os
import glob
import re
import matplotlib.pyplot as plt
import numpy as np

from ImprovedDNAGAN.models import ImprovedGenerator, ImprovedDiscriminator
from ImprovedDNAGAN.train import plot_training_history


def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in the checkpoint directory.
    
    Args:
        checkpoint_dir (str): Directory containing checkpoints.
        
    Returns:
        str: Path to the latest checkpoint.
    """
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt'))
    
    if not checkpoint_files:
        return None
    
    # Extract epoch numbers from checkpoint files
    epoch_numbers = []
    for file in checkpoint_files:
        match = re.search(r'checkpoint_epoch_(\d+)\.pt', file)
        if match:
            epoch_numbers.append(int(match.group(1)))
    
    if not epoch_numbers:
        return None
    
    # Find the latest epoch
    latest_epoch = max(epoch_numbers)
    latest_checkpoint = os.path.join(checkpoint_dir, f'checkpoint_epoch_{latest_epoch}.pt')
    
    return latest_checkpoint


def visualize_generated_sequences(generator, noise_dim=100, num_samples=10, device='cpu'):
    """
    Visualize generated DNA sequences.
    
    Args:
        generator (nn.Module): Generator model.
        noise_dim (int): Dimension of the input noise vector.
        num_samples (int): Number of sequences to generate.
        device (str): Device to use for generation.
    """
    generator.eval()
    
    # Generate sequences
    with torch.no_grad():
        noise = torch.randn(num_samples, noise_dim).to(device)
        generated_sequences = generator(noise, temperature=1.0, hard=True)
    
    # Convert one-hot to indices
    indices = torch.argmax(generated_sequences, dim=2).cpu().numpy()
    
    # Map indices to nucleotides
    nucleotides = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    
    # Print generated sequences
    print("Generated DNA Sequences:")
    print("-" * 50)
    
    for i, seq_indices in enumerate(indices):
        # Convert indices to nucleotides
        seq = ''.join([nucleotides[idx] for idx in seq_indices])
        
        # Print sequence
        print(f"Sequence {i+1}: {seq}")
    
    print("-" * 50)
    
    # Visualize nucleotide distribution
    nucleotide_counts = np.zeros((4,))
    
    for seq_indices in indices:
        for idx in seq_indices:
            nucleotide_counts[idx] += 1
    
    nucleotide_counts /= nucleotide_counts.sum()
    
    plt.figure(figsize=(8, 6))
    plt.bar(['A', 'C', 'G', 'T'], nucleotide_counts)
    plt.title('Nucleotide Distribution in Generated Sequences')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(args.checkpoint_dir), 'images', 'nucleotide_distribution.png'))
    plt.show()


def main():
    """
    Main function for visualizing the training progress.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Visualize training progress of improved DNA sequence generation model')
    
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/improved_dna_gan',
                        help='Directory containing checkpoints')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to specific checkpoint to visualize (if None, use latest)')
    parser.add_argument('--save_path', type=str, default=None,
                        help='Path to save the plot (if None, use default)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--noise_dim', type=int, default=100,
                        help='Dimension of the input noise vector')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Dimension of the LSTM hidden state')
    parser.add_argument('--seq_len', type=int, default=150,
                        help='Length of DNA sequences')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers')
    
    global args
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Find checkpoint to visualize
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
        
    if checkpoint_path is None:
        print("No checkpoint found. Please train the model first.")
        return
    
    print(f"Visualizing training progress from checkpoint: {checkpoint_path}")
    
    # Create images directory
    images_dir = os.path.join(args.checkpoint_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Set save path
    save_path = args.save_path
    if save_path is None:
        save_path = os.path.join(images_dir, 'training_history.png')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    history = checkpoint.get('history', {})
    
    # Plot training history
    plot_training_history(history, save_path=save_path)
    
    # Initialize models
    generator = ImprovedGenerator(
        noise_dim=args.noise_dim,
        hidden_dim=args.hidden_dim,
        seq_len=args.seq_len,
        num_layers=args.num_layers
    ).to(device)
    
    discriminator = ImprovedDiscriminator(
        seq_len=args.seq_len
    ).to(device)
    
    # Load model weights
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    # Visualize generated sequences
    visualize_generated_sequences(generator, args.noise_dim, num_samples=10, device=device)
    
    print(f"Training history plot saved to {save_path}")


if __name__ == '__main__':
    main()
