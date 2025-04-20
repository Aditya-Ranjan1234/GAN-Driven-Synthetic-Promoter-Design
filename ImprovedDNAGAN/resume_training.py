"""
Script for resuming training of the improved DNA sequence generation model.

This script provides a command-line interface for resuming training from a checkpoint.
"""

import argparse
import torch
import os
import logging
import glob
import re

from ImprovedDNAGAN.models import ImprovedGenerator, ImprovedDiscriminator
from ImprovedDNAGAN.data import get_data_loader
from ImprovedDNAGAN.train import train_wgan_gp, plot_training_history, load_checkpoint


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


def main():
    """
    Main function for resuming training of the improved DNA sequence generation model.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Resume training of improved DNA sequence generation model')
    
    # Data parameters
    parser.add_argument('--fasta_file', type=str, default='data/clean_all_dna_sequences.fasta',
                        help='Path to the FASTA file')
    parser.add_argument('--seq_len', type=int, default=150,
                        help='Length of DNA sequences')
    
    # Model parameters
    parser.add_argument('--noise_dim', type=int, default=100,
                        help='Dimension of the input noise vector')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Dimension of the LSTM hidden state')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='Number of additional training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr_g', type=float, default=1e-4,
                        help='Learning rate for the generator')
    parser.add_argument('--lr_d', type=float, default=1e-4,
                        help='Learning rate for the discriminator')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature parameter for Gumbel-Softmax')
    parser.add_argument('--n_critic', type=int, default=5,
                        help='Number of discriminator updates per generator update')
    parser.add_argument('--lambda_gp', type=float, default=10.0,
                        help='Weight for gradient penalty')
    
    # Checkpoint parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/improved_dna_gan',
                        help='Directory containing checkpoints')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to specific checkpoint to resume training (if None, use latest)')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='Interval for logging training progress')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Interval for saving checkpoints')
    
    # Other parameters
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes for data loading')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Find checkpoint to resume from
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
        
    if checkpoint_path is None:
        print("No checkpoint found. Please train the model first.")
        return
    
    print(f"Resuming training from checkpoint: {checkpoint_path}")
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.checkpoint_dir, 'resume_training.log')),
            logging.StreamHandler()
        ]
    )
    
    # Log arguments
    logging.info(f"Arguments: {args}")
    
    # Load data
    data_loader = get_data_loader(
        args.fasta_file,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Initialize models
    generator = ImprovedGenerator(
        noise_dim=args.noise_dim,
        hidden_dim=args.hidden_dim,
        seq_len=args.seq_len,
        num_layers=args.num_layers
    )
    
    discriminator = ImprovedDiscriminator(
        seq_len=args.seq_len
    )
    
    # Load checkpoint
    generator, discriminator, history = load_checkpoint(
        checkpoint_path, generator, discriminator, device
    )
    
    # Extract epoch number from checkpoint path
    match = re.search(r'checkpoint_epoch_(\d+)\.pt', checkpoint_path)
    start_epoch = int(match.group(1)) if match else 0
    
    logging.info(f"Resumed training from epoch {start_epoch}")
    
    # Train model
    history = train_wgan_gp(
        generator=generator,
        discriminator=discriminator,
        data_loader=data_loader,
        num_epochs=args.num_epochs,
        noise_dim=args.noise_dim,
        batch_size=args.batch_size,
        device=device,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        temperature=args.temperature,
        start_epoch=start_epoch,
        history=history,
        n_critic=args.n_critic,
        lambda_gp=args.lambda_gp
    )
    
    # Plot training history
    images_dir = os.path.join(args.checkpoint_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    plot_path = os.path.join(images_dir, 'training_history.png')
    plot_training_history(history, save_path=plot_path)
    
    logging.info("Training resumed and completed successfully!")


if __name__ == '__main__':
    main()
