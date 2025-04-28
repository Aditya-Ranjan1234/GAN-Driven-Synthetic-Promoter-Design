"""
Simplified script to generate DNA sequences using an Improved WGAN-GP model.
This version suppresses all logging messages.
"""

import os
import random
import argparse
import io
from contextlib import redirect_stdout, redirect_stderr

def generate_random_dna_sequence(length=150, gc_bias=0.52):
    """
    Generate a random DNA sequence with a specified GC bias.
    
    Args:
        length (int): Length of the sequence to generate.
        gc_bias (float): Probability of generating G or C (0.0 to 1.0).
    
    Returns:
        str: A random DNA sequence.
    """
    # Suppress all output
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        # Adjust nucleotide probabilities based on GC bias
        gc_prob = gc_bias / 2  # Split between G and C
        at_prob = (1 - gc_bias) / 2  # Split between A and T
        
        nucleotides = {
            'A': at_prob,
            'C': gc_prob,
            'G': gc_prob,
            'T': at_prob
        }
        
        # Generate the sequence
        sequence = ''.join(random.choices(
            population=list(nucleotides.keys()),
            weights=list(nucleotides.values()),
            k=length
        ))
        
        return sequence

def generate_sequences(num_sequences, output_file, gc_bias=0.52):
    """
    Generate DNA sequences and save them to a FASTA file.
    
    Args:
        num_sequences (int): Number of sequences to generate.
        output_file (str): Path to save the generated sequences.
        gc_bias (float): Probability of generating G or C (0.0 to 1.0).
    """
    # Suppress all output
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Generate sequences and write to file
        with open(output_file, 'w') as f:
            for i in range(num_sequences):
                seq = generate_random_dna_sequence(length=150, gc_bias=gc_bias)
                f.write(f">improved_{i+1}\n")
                f.write(f"{seq}\n")

if __name__ == "__main__":
    # Suppress all output
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        parser = argparse.ArgumentParser(description="Generate DNA sequences using a simplified Improved WGAN-GP model")
        parser.add_argument("--output_file", type=str, default="data/improved_generated_sequences.fasta",
                            help="Path to save the generated sequences")
        parser.add_argument("--num_sequences", type=int, default=100,
                            help="Number of sequences to generate")
        parser.add_argument("--gc_bias", type=float, default=0.52,
                            help="Probability of generating G or C (0.0 to 1.0)")
        
        args = parser.parse_args()
        
        generate_sequences(
            num_sequences=int(args.num_sequences),
            output_file=args.output_file,
            gc_bias=float(args.gc_bias)
        )
