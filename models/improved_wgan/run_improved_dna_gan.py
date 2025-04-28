"""
Main script for training the improved DNA sequence generation model.

This script provides a command-line interface for training the improved GAN model.
"""

import argparse
import torch
import os
import logging
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime

# Import from local modules
from models.improved_wgan.models import ImprovedGenerator, ImprovedDiscriminator
from models.improved_wgan.data import get_data_loader, save_sequences_to_fasta
from models.improved_wgan.train import train_wgan_gp, plot_training_history, load_checkpoint
from models.improved_wgan.metrics import evaluate_model


def set_seed(seed):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """
    Main function for training the improved DNA sequence generation model.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train improved DNA sequence generation model')

    # Data parameters
    parser.add_argument('--fasta_file', type=str, default='data/preprocessed_dna_sequences.fasta',
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
                        help='Number of training epochs')
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
                        help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='Interval for logging training progress')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Interval for saving checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')

    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker processes for data loading')

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Create images directory
    images_dir = os.path.join(args.checkpoint_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.checkpoint_dir, 'training.log')),
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
    ).to(device)

    discriminator = ImprovedDiscriminator(
        seq_len=args.seq_len
    ).to(device)

    # Print model architecture
    logging.info(f"Generator architecture:\n{generator}")
    logging.info(f"Discriminator architecture:\n{discriminator}")

    # Resume training if checkpoint is provided
    start_epoch = 0
    history = None

    if args.resume:
        generator, discriminator, history = load_checkpoint(
            args.resume, generator, discriminator, device
        )
        start_epoch = int(args.resume.split('_')[-1].split('.')[0])
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
    plot_path = os.path.join(images_dir, 'training_history.png')
    plot_training_history(history, save_path=plot_path)

    # Generate sequences
    with torch.no_grad():
        noise = torch.randn(100, args.noise_dim).to(device)
        generated_sequences = generator(noise, temperature=1.0, hard=True)

    # Save generated sequences
    output_file = os.path.join(args.checkpoint_dir, 'generated_sequences.fasta')
    save_sequences_to_fasta(generated_sequences, output_file)

    # Evaluate model
    metrics = evaluate_model(
        generator=generator,
        data_loader=data_loader,
        noise_dim=args.noise_dim,
        device=device
    )

    # Log evaluation metrics
    logging.info("Evaluation metrics:")
    for metric, value in metrics.items():
        logging.info(f"{metric}: {value:.4f}")

    # Plot evaluation metrics
    plt.figure(figsize=(10, 6))

    metrics_to_plot = [
        ('GC Content', 'generated_gc_content', 'real_gc_content'),
        ("Moran's I", 'generated_morans_i', 'real_morans_i'),
        ('Diversity', 'generated_diversity', 'real_diversity'),
        ('K-mer Diversity', 'generated_kmer_diversity', 'real_kmer_diversity')
    ]

    for i, (title, gen_key, real_key) in enumerate(metrics_to_plot):
        plt.subplot(2, 2, i+1)
        plt.bar(['Generated', 'Real'], [metrics[gen_key], metrics[real_key]])
        plt.title(title)
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'evaluation_metrics.png'))

    logging.info("Training completed successfully!")


if __name__ == '__main__':
    main()
