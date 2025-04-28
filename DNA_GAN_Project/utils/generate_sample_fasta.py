"""
Script to generate a sample FASTA file with valid DNA sequences.

This script:
1. Generates random DNA sequences
2. Saves them in FASTA format
3. Provides a valid FASTA file for testing
"""

import os
import random
import numpy as np


def generate_random_dna_sequence(length=150):
    """
    Generate a random DNA sequence.
    
    Args:
        length (int): Length of the sequence.
        
    Returns:
        str: Random DNA sequence.
    """
    nucleotides = ['A', 'C', 'G', 'T']
    return ''.join(random.choice(nucleotides) for _ in range(length))


def generate_sample_fasta(output_path, num_sequences=1000, seq_length=150):
    """
    Generate a sample FASTA file with random DNA sequences.
    
    Args:
        output_path (str): Path to save the FASTA file.
        num_sequences (int): Number of sequences to generate.
        seq_length (int): Length of each sequence.
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate sequences
    with open(output_path, 'w') as f:
        for i in range(num_sequences):
            # Generate a random DNA sequence
            sequence = generate_random_dna_sequence(seq_length)
            
            # Write to FASTA file
            f.write(f">sequence_{i+1}\n")
            
            # Write sequence in lines of 60 characters
            for j in range(0, len(sequence), 60):
                f.write(f"{sequence[j:j+60]}\n")
    
    print(f"Generated {num_sequences} random DNA sequences and saved to {output_path}")


def main():
    """
    Main function to generate a sample FASTA file.
    """
    # Define output path
    output_path = "data/sample_dna_sequences.fasta"
    
    # Generate sample FASTA file
    generate_sample_fasta(output_path, num_sequences=1000, seq_length=150)
    
    print("Sample FASTA file generation completed successfully.")


if __name__ == "__main__":
    main()
