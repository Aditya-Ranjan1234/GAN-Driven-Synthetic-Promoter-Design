"""
Script to generate DNA sequences from a trained Gumbel-Softmax GAN model.
"""

import argparse
import torch
import os
import sys
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the Generator model and save_sequences_to_fasta function
# The exact import path may need to be adjusted based on your project structure
try:
    from models.gumbel_softmax_gan.dna_gan.models import Generator
    from models.gumbel_softmax_gan.dna_gan.data import save_sequences_to_fasta
except ImportError:
    print("Error importing modules. Please make sure the dna_gan package is properly installed.")
    sys.exit(1)


def main():
    """
    Main function to generate DNA sequences from a trained Gumbel-Softmax GAN model.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate DNA sequences from a trained Gumbel-Softmax GAN model')
    
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_epoch_1000.pt',
                        help='Path to the checkpoint file')
    parser.add_argument('--num_sequences', type=int, default=1000,
                        help='Number of sequences to generate')
    parser.add_argument('--output_file', type=str, default='data/gumbel_generated_sequences.fasta',
                        help='Path to save the generated sequences')
    parser.add_argument('--seq_len', type=int, default=150,
                        help='Length of DNA sequences')
    parser.add_argument('--noise_dim', type=int, default=100,
                        help='Dimension of the input noise vector')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Dimension of the LSTM hidden state')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature parameter for Gumbel-Softmax')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Initialize generator
    generator = Generator(
        noise_dim=args.noise_dim,
        hidden_dim=args.hidden_dim,
        seq_len=args.seq_len,
        vocab_size=4  # A, C, G, T
    ).to(device)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'generator_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['generator_state_dict'])
        else:
            generator.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {args.checkpoint}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Set generator to evaluation mode
    generator.eval()
    
    # Generate sequences
    print(f"Generating {args.num_sequences} sequences...")
    with torch.no_grad():
        # Generate in batches to avoid memory issues
        batch_size = 100
        num_batches = (args.num_sequences + batch_size - 1) // batch_size
        
        all_sequences = []
        
        for i in range(num_batches):
            current_batch_size = min(batch_size, args.num_sequences - i * batch_size)
            noise = torch.randn(current_batch_size, args.noise_dim).to(device)
            sequences = generator(noise, args.temperature, hard=True)
            all_sequences.append(sequences)
        
        # Concatenate all batches
        all_sequences = torch.cat(all_sequences, dim=0)
    
    # Save sequences to FASTA file
    save_sequences_to_fasta(all_sequences, args.output_file)
    print(f"Generated sequences saved to {args.output_file}")


if __name__ == "__main__":
    main()
