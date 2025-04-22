"""
Example script for training a DNA sequence generation model.

This script demonstrates how to use the dna_gan package to train a model
for generating DNA sequences.
"""

import torch
import os
import argparse
from dna_gan.models import Generator, CNNDiscriminator, LSTMDiscriminator
from dna_gan.data import get_data_loader, one_hot_to_sequence, sequences_to_fasta
from dna_gan.train import train_gan, plot_training_history
from dna_gan.metrics import evaluate_model


def main():
    """Main function for the example script."""
    parser = argparse.ArgumentParser(description='Train DNA sequence generation model')
    parser.add_argument('--fasta_file', type=str, required=True, help='Path to the FASTA file')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save output')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--seq_len', type=int, default=150, help='Length of DNA sequences')
    parser.add_argument('--discriminator_type', type=str, default='cnn', choices=['cnn', 'lstm'], help='Type of discriminator')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    data_loader = get_data_loader(
        args.fasta_file,
        batch_size=args.batch_size,
        seq_len=args.seq_len
    )
    
    # Create models
    generator = Generator(
        noise_dim=100,
        hidden_dim=256,
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
            hidden_dim=256
        ).to(device)
    
    # Train the model
    history = train_gan(
        generator=generator,
        discriminator=discriminator,
        data_loader=data_loader,
        num_epochs=args.num_epochs,
        device=device,
        checkpoint_dir=args.output_dir
    )
    
    # Plot training history
    plot_path = os.path.join(args.output_dir, 'training_history.png')
    plot_training_history(history, save_path=plot_path)
    
    # Evaluate the model
    metrics = evaluate_model(
        generator=generator,
        data_loader=data_loader,
        device=device
    )
    
    print(f"Evaluation metrics: {metrics}")
    
    # Generate and save sequences
    with torch.no_grad():
        noise = torch.randn(100, 100).to(device)
        generated_sequences = generator(noise, temperature=1.0, hard=True)
        
        # Convert to DNA sequences
        dna_sequences = []
        for i in range(generated_sequences.size(0)):
            seq = one_hot_to_sequence(generated_sequences[i])
            dna_sequences.append(seq)
        
        # Save to FASTA file
        fasta_path = os.path.join(args.output_dir, 'generated_sequences.fasta')
        sequences_to_fasta(dna_sequences, fasta_path)


if __name__ == '__main__':
    main()
