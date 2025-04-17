"""
Main script for training DNA sequence generation model.

This script provides the main entry point for training the GAN model.
"""

import argparse
import torch
import os
import logging
import matplotlib.pyplot as plt
from datetime import datetime

from .models import Generator, CNNDiscriminator, LSTMDiscriminator
from .data import get_data_loader, one_hot_to_sequence, sequences_to_fasta
from .train import train_gan, plot_training_history, load_checkpoint
from .metrics import evaluate_model


def main():
    """Main function for training the GAN model."""
    parser = argparse.ArgumentParser(description='Train DNA sequence generation model')
    
    # Data parameters
    parser.add_argument('--fasta_file', type=str, required=True, help='Path to the FASTA file')
    parser.add_argument('--seq_len', type=int, default=150, help='Length of DNA sequences')
    
    # Model parameters
    parser.add_argument('--noise_dim', type=int, default=100, help='Dimension of the noise vector')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Dimension of the hidden state')
    parser.add_argument('--discriminator_type', type=str, default='cnn', choices=['cnn', 'lstm'], help='Type of discriminator')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--lr_g', type=float, default=1e-4, help='Learning rate for generator')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='Learning rate for discriminator')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for Gumbel-Softmax')
    
    # Checkpoint parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=1, help='Interval for logging training progress')
    parser.add_argument('--save_interval', type=int, default=10, help='Interval for saving checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for data loading')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = os.path.join(args.checkpoint_dir, f"{timestamp}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(checkpoint_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    # Log arguments
    logging.info(f"Arguments: {args}")
    
    # Load data
    data_loader = get_data_loader(
        args.fasta_file,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_workers=args.num_workers
    )
    
    # Create models
    generator = Generator(
        noise_dim=args.noise_dim,
        hidden_dim=args.hidden_dim,
        seq_len=args.seq_len,
        vocab_size=4
    ).to(device)
    
    if args.discriminator_type == 'cnn':
        discriminator = CNNDiscriminator(
            seq_len=args.seq_len,
            vocab_size=4
        ).to(device)
    else:
        discriminator = LSTMDiscriminator(
            seq_len=args.seq_len,
            vocab_size=4,
            hidden_dim=args.hidden_dim
        ).to(device)
    
    # Resume from checkpoint if specified
    if args.resume:
        generator, discriminator, history = load_checkpoint(
            args.resume, generator, discriminator, device
        )
        logging.info(f"Resumed from checkpoint: {args.resume}")
    
    # Train the model
    history = train_gan(
        generator=generator,
        discriminator=discriminator,
        data_loader=data_loader,
        num_epochs=args.num_epochs,
        noise_dim=args.noise_dim,
        batch_size=args.batch_size,
        device=device,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        checkpoint_dir=checkpoint_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        temperature=args.temperature
    )
    
    # Plot training history
    plot_path = os.path.join(checkpoint_dir, 'training_history.png')
    plot_training_history(history, save_path=plot_path)
    
    # Evaluate the model
    metrics = evaluate_model(
        generator=generator,
        data_loader=data_loader,
        noise_dim=args.noise_dim,
        device=device
    )
    
    # Log evaluation metrics
    logging.info(f"Evaluation metrics: {metrics}")
    
    # Generate and save sequences
    with torch.no_grad():
        noise = torch.randn(100, args.noise_dim).to(device)
        generated_sequences = generator(noise, temperature=1.0, hard=True)
        
        # Convert to DNA sequences
        dna_sequences = []
        for i in range(generated_sequences.size(0)):
            seq = one_hot_to_sequence(generated_sequences[i])
            dna_sequences.append(seq)
        
        # Save to FASTA file
        fasta_path = os.path.join(checkpoint_dir, 'generated_sequences.fasta')
        sequences_to_fasta(dna_sequences, fasta_path)


if __name__ == '__main__':
    main()
