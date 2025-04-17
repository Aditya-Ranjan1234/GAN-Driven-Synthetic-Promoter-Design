"""
Test script for the DNA Sequence GAN.

This script tests the PyTorch implementation of the DNA Sequence GAN.
"""

import os
import torch
from data_loader import DNADataLoader
from gan_model import DNAGAN
from dna_utils import one_hot_decode

def main():
    """
    Main function to test the DNA Sequence GAN.
    """
    print("Testing DNA Sequence GAN with PyTorch...")
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create data loader
    data_loader = DNADataLoader(sequence_length=50, batch_size=16)
    
    # Generate dummy data
    print("Generating dummy data...")
    sequences = data_loader.generate_dummy_data(num_sequences=100, min_length=40, max_length=50)
    print(f"Generated {len(sequences)} sequences")
    print(f"Sample sequence: {sequences[0]}")
    
    # Preprocess data
    print("Preprocessing data...")
    encoded_sequences = data_loader.preprocess()
    print(f"Encoded sequences shape: {encoded_sequences.shape}")
    
    # Create dataset
    print("Creating dataset...")
    dataset = data_loader.create_dataset()
    print(f"Dataset size: {len(dataset)}")
    
    # Initialize GAN
    print("Initializing GAN...")
    gan = DNAGAN(
        sequence_length=data_loader.sequence_length,
        batch_size=16,
        latent_dim=100,
        device=device
    )
    
    # Load data
    print("Loading data into GAN...")
    gan.load_data_from_dataset(dataset)
    
    # Train GAN for a few epochs
    print("Training GAN for 5 epochs...")
    gan.train(epochs=5, save_interval=5, verbose=True)
    
    # Generate sequences
    print("Generating synthetic sequences...")
    synthetic_sequences = gan.generate(num_sequences=5)
    print("Generated sequences:")
    for i, seq in enumerate(synthetic_sequences):
        print(f"Sequence {i+1}: {seq}")
    
    # Save checkpoint
    print("Saving checkpoint...")
    gan.save_checkpoint(epoch=5)
    
    print("Test completed successfully!")

if __name__ == "__main__":
    main()
