"""
Script to generate DNA sequences using the Gumbel-Softmax GAN model.
"""

import os
import sys
import argparse
import torch
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the model
try:
    from models.gumbel_softmax_gan.model import Generator, save_sequences_to_fasta
except ImportError:
    print("Error importing model. Please make sure the model file exists.")
    sys.exit(1)


def generate_sequences(num_sequences, output_file, checkpoint_path=None):
    """
    Generate DNA sequences using the Gumbel-Softmax GAN model.
    
    Args:
        num_sequences (int): Number of sequences to generate.
        output_file (str): Path to save the generated sequences.
        checkpoint_path (str, optional): Path to the checkpoint file.
    """
    # Parameters
    noise_dim = 100
    hidden_dim = 256
    seq_len = 150
    vocab_size = 4  # A, C, G, T
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create generator
    generator = Generator(
        noise_dim=noise_dim,
        hidden_dim=hidden_dim,
        seq_len=seq_len,
        vocab_size=vocab_size
    ).to(device)
    
    # Load checkpoint if available
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Try different key patterns
            if 'generator_state_dict' in checkpoint:
                generator.load_state_dict(checkpoint['generator_state_dict'])
            elif 'generator' in checkpoint:
                generator.load_state_dict(checkpoint['generator'])
            else:
                generator.load_state_dict(checkpoint)
                
            print("Checkpoint loaded successfully")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Using randomly initialized generator")
    else:
        print("No checkpoint found, using randomly initialized generator")
    
    # Generate sequences
    generator.eval()
    print(f"Generating {num_sequences} sequences...")
    
    sequences = generator.generate(num_sequences, device)
    
    # Save sequences to FASTA file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_sequences_to_fasta(sequences, output_file)
    
    print(f"Generated sequences saved to {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate DNA sequences using Gumbel-Softmax GAN')
    parser.add_argument('--num_sequences', type=int, default=100, help='Number of sequences to generate')
    parser.add_argument('--output_file', type=str, default='data/gumbel_generated_sequences.fasta', help='Output file path')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint file path')
    args = parser.parse_args()
    
    # Generate sequences
    generate_sequences(args.num_sequences, args.output_file, args.checkpoint)


if __name__ == '__main__':
    main()
