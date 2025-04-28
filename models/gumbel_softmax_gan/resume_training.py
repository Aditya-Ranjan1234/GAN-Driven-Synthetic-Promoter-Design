"""
Resume training script for DNA sequence generation using Gumbel-Softmax GAN.

This script finds the latest checkpoint and resumes training from there.
"""

import os
import re
import argparse

def find_latest_checkpoint(checkpoint_dir='checkpoints'):
    """Find the latest checkpoint in the checkpoint directory."""
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist.")
        return None
    
    # Get all checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                        if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}.")
        return None
    
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

def main():
    """Main function to resume training."""
    parser = argparse.ArgumentParser(description='Resume DNA sequence generation training')
    
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
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', 
                        help='Directory containing checkpoints')
    parser.add_argument('--discriminator_type', type=str, default='cnn', 
                        choices=['cnn', 'lstm'], help='Type of discriminator')
    parser.add_argument('--no_cuda', action='store_true', 
                        help='Disable CUDA')
    
    args = parser.parse_args()
    
    # Find the latest checkpoint
    latest_checkpoint = find_latest_checkpoint(args.checkpoint_dir)
    
    if latest_checkpoint is None:
        print("No checkpoint found. Starting training from scratch.")
        resume_arg = []
    else:
        print(f"Resuming training from {latest_checkpoint}")
        resume_arg = ['--resume', latest_checkpoint]
    
    # Construct the command to run the main module
    cmd = [
        'python', '-m', 'dna_gan.main',
        '--fasta_file', args.fasta_file,
        '--seq_len', str(args.seq_len),
        '--num_epochs', str(args.num_epochs),
        '--batch_size', str(args.batch_size),
        '--save_interval', str(args.save_interval),
        '--discriminator_type', args.discriminator_type,
        '--checkpoint_dir', args.checkpoint_dir
    ]
    
    # Add resume parameter if a checkpoint was found
    if resume_arg:
        cmd.extend(resume_arg)
    
    # Add no_cuda parameter if specified
    if args.no_cuda:
        cmd.append('--no_cuda')
    
    # Print the command
    print("Running command:", ' '.join(cmd))
    
    # Run the command
    os.system(' '.join(cmd))

if __name__ == '__main__':
    main()
