"""
Run script for DNA sequence generation using Gumbel-Softmax GAN.

This script runs the DNA sequence generation model on the provided data file
for 1000 epochs, with checkpointing and visualization.
"""

import os
import sys
import argparse

def main():
    """Main function to run the DNA sequence generation model."""
    parser = argparse.ArgumentParser(description='Run DNA sequence generation model')
    
    # Data parameters
    parser.add_argument('--fasta_file', type=str, default='data/clean_all_dna_sequences.fasta', 
                        help='Path to the FASTA file')
    parser.add_argument('--seq_len', type=int, default=150, 
                        help='Length of DNA sequences')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=1000, 
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size')
    parser.add_argument('--save_interval', type=int, default=10, 
                        help='Interval for saving checkpoints')
    parser.add_argument('--resume', type=str, default=None, 
                        help='Path to checkpoint to resume training')
    parser.add_argument('--discriminator_type', type=str, default='cnn', 
                        choices=['cnn', 'lstm'], help='Type of discriminator')
    parser.add_argument('--no_cuda', action='store_true', 
                        help='Disable CUDA')
    
    args = parser.parse_args()
    
    # Construct the command to run the main module
    cmd = [
        'python', '-m', 'dna_gan.main',
        '--fasta_file', args.fasta_file,
        '--seq_len', str(args.seq_len),
        '--num_epochs', str(args.num_epochs),
        '--batch_size', str(args.batch_size),
        '--save_interval', str(args.save_interval),
        '--discriminator_type', args.discriminator_type,
        '--checkpoint_dir', 'checkpoints'
    ]
    
    # Add resume parameter if specified
    if args.resume:
        cmd.extend(['--resume', args.resume])
    
    # Add no_cuda parameter if specified
    if args.no_cuda:
        cmd.append('--no_cuda')
    
    # Print the command
    print("Running command:", ' '.join(cmd))
    
    # Run the command
    os.system(' '.join(cmd))

if __name__ == '__main__':
    main()
