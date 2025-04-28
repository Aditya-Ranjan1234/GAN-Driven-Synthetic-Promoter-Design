"""
Utility functions for preparing data for the Streamlit app.

This module provides functions for:
1. Cleaning and preprocessing DNA sequences
2. Converting between different formats
3. Generating training history JSON files for visualization
"""

import os
import json
import re
import numpy as np
from Bio import SeqIO
import torch


def clean_fasta_file(input_path, output_path, min_length=50, max_length=None):
    """
    Clean a FASTA file by removing non-standard nucleotides and filtering by length.
    
    Args:
        input_path (str): Path to the input FASTA file.
        output_path (str): Path to the output FASTA file.
        min_length (int): Minimum sequence length to include.
        max_length (int): Maximum sequence length to include.
    """
    cleaned_records = []
    
    for record in SeqIO.parse(input_path, "fasta"):
        seq = str(record.seq).upper()
        
        # Remove non-standard nucleotides
        seq = re.sub(r'[^ACGT]', '', seq)
        
        # Filter by length
        if min_length and len(seq) < min_length:
            continue
        if max_length and len(seq) > max_length:
            continue
        
        # Update record
        record.seq = seq
        cleaned_records.append(record)
    
    # Write cleaned records to output file
    with open(output_path, 'w') as f:
        SeqIO.write(cleaned_records, f, "fasta")
    
    print(f"Cleaned {len(cleaned_records)} sequences and saved to {output_path}")


def convert_checkpoint_to_history(checkpoint_path, output_path):
    """
    Convert a checkpoint file to a training history JSON file.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file.
        output_path (str): Path to the output JSON file.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'history' in checkpoint:
        history = checkpoint['history']
        
        # Convert tensors to Python types
        for key in history:
            if isinstance(history[key], list):
                history[key] = [float(x) if isinstance(x, torch.Tensor) else x for x in history[key]]
        
        # Save history to JSON file
        with open(output_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Training history saved to {output_path}")
    else:
        print(f"No training history found in checkpoint: {checkpoint_path}")


def generate_dummy_history(output_path, num_epochs=1000, model_type='gumbel'):
    """
    Generate dummy training history for visualization.
    
    Args:
        output_path (str): Path to the output JSON file.
        num_epochs (int): Number of epochs.
        model_type (str): Type of model ('gumbel' or 'improved').
    """
    np.random.seed(42)
    
    if model_type == 'gumbel':
        # Gumbel-Softmax GAN history
        history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'discriminator_accuracy': [],
            'gc_content': [],
            'morans_i': [],
            'diversity': []
        }
        
        # Generate dummy data
        for i in range(num_epochs):
            # Generator loss increases over time
            g_loss = 1.0 + 9.0 * (i / num_epochs) + np.random.normal(0, 0.5)
            history['generator_loss'].append(max(0, g_loss))
            
            # Discriminator loss decreases quickly
            d_loss = 1.5 * np.exp(-i / 50) + np.random.normal(0, 0.05)
            history['discriminator_loss'].append(max(0, d_loss))
            
            # Discriminator accuracy increases quickly to 1.0
            d_acc = 1.0 - 0.5 * np.exp(-i / 30) + np.random.normal(0, 0.01)
            history['discriminator_accuracy'].append(min(1.0, max(0, d_acc)))
            
            # GC content decreases over time
            gc = 0.5 * np.exp(-i / 100) + np.random.normal(0, 0.01)
            history['gc_content'].append(max(0, gc))
            
            # Moran's I increases then decreases
            morans_i = 0.3 * np.exp(-((i - 100) / 100) ** 2) + np.random.normal(0, 0.01)
            history['morans_i'].append(max(0, morans_i))
            
            # Diversity decreases over time
            diversity = 0.7 * np.exp(-i / 200) + np.random.normal(0, 0.01)
            history['diversity'].append(max(0, diversity))
    
    else:  # improved WGAN-GP
        # Improved WGAN-GP history
        history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'wasserstein_distance': [],
            'gradient_penalty': [],
            'gc_content': [],
            'morans_i': [],
            'diversity': []
        }
        
        # Generate dummy data
        for i in range(num_epochs):
            # Generator loss decreases over time
            g_loss = 5.0 * np.exp(-i / 500) + np.random.normal(0, 0.2)
            history['generator_loss'].append(max(0, g_loss))
            
            # Discriminator loss decreases over time
            d_loss = 3.0 * np.exp(-i / 400) + np.random.normal(0, 0.1)
            history['discriminator_loss'].append(max(0, d_loss))
            
            # Wasserstein distance decreases over time
            w_dist = 2.0 * np.exp(-i / 300) + np.random.normal(0, 0.1)
            history['wasserstein_distance'].append(max(0, w_dist))
            
            # Gradient penalty remains stable
            gp = 10.0 + np.random.normal(0, 0.5)
            history['gradient_penalty'].append(max(0, gp))
            
            # GC content stabilizes around 0.4
            gc = 0.4 + 0.1 * np.exp(-i / 200) + np.random.normal(0, 0.01)
            history['gc_content'].append(max(0, gc))
            
            # Moran's I stabilizes around 0.2
            morans_i = 0.2 + 0.1 * np.exp(-i / 200) + np.random.normal(0, 0.01)
            history['morans_i'].append(max(0, morans_i))
            
            # Diversity remains high
            diversity = 0.8 - 0.1 * np.exp(-i / 300) + np.random.normal(0, 0.01)
            history['diversity'].append(min(1.0, max(0, diversity)))
    
    # Save history to JSON file
    with open(output_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Dummy training history saved to {output_path}")


def generate_dummy_sequences(output_path, num_sequences=100, seq_len=150, model_type='gumbel'):
    """
    Generate dummy DNA sequences for visualization.
    
    Args:
        output_path (str): Path to the output FASTA file.
        num_sequences (int): Number of sequences to generate.
        seq_len (int): Length of each sequence.
        model_type (str): Type of model ('gumbel', 'improved', or 'original').
    """
    np.random.seed(42)
    
    nucleotides = ['A', 'C', 'G', 'T']
    
    if model_type == 'gumbel':
        # Gumbel-Softmax GAN tends to generate sequences with low GC content
        probabilities = [0.45, 0.05, 0.05, 0.45]  # A, C, G, T
    elif model_type == 'improved':
        # Improved WGAN-GP generates sequences with more balanced nucleotide distribution
        probabilities = [0.25, 0.25, 0.25, 0.25]  # A, C, G, T
    else:  # original
        # Original data has slightly higher GC content
        probabilities = [0.2, 0.3, 0.3, 0.2]  # A, C, G, T
    
    with open(output_path, 'w') as f:
        for i in range(num_sequences):
            # Generate sequence
            sequence = ''.join(np.random.choice(nucleotides, size=seq_len, p=probabilities))
            
            # Write to FASTA file
            f.write(f">{model_type}_sequence_{i+1}\n")
            f.write(f"{sequence}\n")
    
    print(f"Generated {num_sequences} dummy sequences and saved to {output_path}")


if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs("../../data", exist_ok=True)
    
    # Generate dummy data for visualization
    generate_dummy_sequences("../../data/clean_all_dna_sequences.fasta", model_type='original')
    generate_dummy_sequences("../../data/gumbel_generated_sequences.fasta", model_type='gumbel')
    generate_dummy_sequences("../../data/improved_generated_sequences.fasta", model_type='improved')
    
    # Generate dummy training history
    os.makedirs("../../models/gumbel_softmax_gan/checkpoints", exist_ok=True)
    os.makedirs("../../models/improved_wgan/checkpoints/improved_dna_gan", exist_ok=True)
    
    generate_dummy_history("../../models/gumbel_softmax_gan/checkpoints/training_history.json", model_type='gumbel')
    generate_dummy_history("../../models/improved_wgan/checkpoints/improved_dna_gan/training_history.json", model_type='improved')
