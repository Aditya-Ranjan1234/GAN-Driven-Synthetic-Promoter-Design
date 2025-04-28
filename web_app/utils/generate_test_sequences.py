"""
Utility script to generate test DNA sequences.
"""

import os
import random
import argparse
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


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


def generate_test_sequences(output_file, num_sequences=1000, length=150, gc_bias=0.5, model_name="test"):
    """
    Generate test DNA sequences and save them to a FASTA file.
    
    Args:
        output_file (str): Path to save the FASTA file.
        num_sequences (int): Number of sequences to generate.
        length (int): Length of each sequence.
        gc_bias (float): Probability of generating G or C (0.5 means no bias).
        model_name (str): Name of the model to include in the sequence IDs.
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Generate sequences
    records = []
    for i in range(num_sequences):
        seq = generate_random_dna_sequence(length, gc_bias)
        record = SeqRecord(
            Seq(seq),
            id=f"{model_name}_{i+1}",
            description=f"Generated test sequence with GC bias {gc_bias:.2f}"
        )
        records.append(record)
    
    # Write sequences to file
    with open(output_file, 'w') as f:
        SeqIO.write(records, f, 'fasta')
    
    print(f"Generated {num_sequences} test sequences with GC bias {gc_bias:.2f}")
    print(f"Saved to {output_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate test DNA sequences')
    parser.add_argument('--output_file', type=str, default='data/test_sequences.fasta', help='Output file path')
    parser.add_argument('--num_sequences', type=int, default=1000, help='Number of sequences to generate')
    parser.add_argument('--length', type=int, default=150, help='Length of each sequence')
    parser.add_argument('--gc_bias', type=float, default=0.5, help='GC bias (0.5 means no bias)')
    parser.add_argument('--model_name', type=str, default='test', help='Model name for sequence IDs')
    args = parser.parse_args()
    
    generate_test_sequences(
        args.output_file,
        args.num_sequences,
        args.length,
        args.gc_bias,
        args.model_name
    )


if __name__ == '__main__':
    main()
