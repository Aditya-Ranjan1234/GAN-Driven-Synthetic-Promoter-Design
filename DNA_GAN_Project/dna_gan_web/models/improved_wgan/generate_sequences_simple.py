"""
Simple script to generate DNA sequences without PyTorch dependencies.
This is a fallback for environments where PyTorch is not available.
"""

import os
import sys
import argparse
import random


def generate_random_dna_sequence(length=150, gc_bias=0.5):
    """
    Generate a random DNA sequence with optional GC bias.
    
    Args:
        length (int): Length of the sequence.
        gc_bias (float): Probability of generating G or C (0.5 means no bias).
        
    Returns:
        str: Random DNA sequence.
    """
    sequence = ""
    for _ in range(length):
        if random.random() < gc_bias:
            sequence += random.choice(['G', 'C'])
        else:
            sequence += random.choice(['A', 'T'])
    return sequence


def generate_sequences(num_sequences, output_file, gc_bias=0.52):
    """
    Generate DNA sequences and save them to a FASTA file.
    
    Args:
        num_sequences (int): Number of sequences to generate.
        output_file (str): Path to save the generated sequences.
        gc_bias (float): GC bias for the sequences.
    """
    print(f"Generating {num_sequences} sequences with GC bias {gc_bias}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Generate sequences
    with open(output_file, 'w') as f:
        for i in range(num_sequences):
            seq = generate_random_dna_sequence(length=150, gc_bias=gc_bias)
            f.write(f">improved_wgan_{i+1}\n")
            f.write(f"{seq}\n")
    
    print(f"Generated {num_sequences} sequences and saved to {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate DNA sequences')
    parser.add_argument('--num_sequences', type=int, default=100, help='Number of sequences to generate')
    parser.add_argument('--output_file', type=str, default='data/improved_generated_sequences.fasta', help='Output file path')
    parser.add_argument('--gc_bias', type=float, default=0.52, help='GC bias for the sequences')
    args = parser.parse_args()
    
    # Generate sequences
    generate_sequences(args.num_sequences, args.output_file, args.gc_bias)


if __name__ == '__main__':
    main()
