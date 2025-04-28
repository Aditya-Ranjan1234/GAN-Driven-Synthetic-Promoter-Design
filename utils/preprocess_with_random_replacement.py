"""
Script to preprocess DNA sequences by replacing 'N' characters with random nucleotides.

This script:
1. Reads the FASTA file
2. Replaces 'N' characters with random nucleotides (A, C, G, T)
3. Removes any other non-standard nucleotides
4. Saves the cleaned sequences to a new FASTA file
"""

import os
import re
import random
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def replace_n_with_random(seq):
    """
    Replace 'N' characters with random nucleotides.
    
    Args:
        seq (str): DNA sequence.
        
    Returns:
        str: DNA sequence with 'N' characters replaced.
    """
    nucleotides = ['A', 'C', 'G', 'T']
    result = ''
    
    for char in seq:
        if char == 'N':
            result += random.choice(nucleotides)
        else:
            result += char
    
    return result


def preprocess_fasta_file(input_path, output_path, min_length=50, max_length=150):
    """
    Preprocess a FASTA file by replacing 'N' characters with random nucleotides.
    
    Args:
        input_path (str): Path to the input FASTA file.
        output_path (str): Path to the output FASTA file.
        min_length (int): Minimum sequence length to include.
        max_length (int): Maximum sequence length to include.
    """
    cleaned_records = []
    
    for record in SeqIO.parse(input_path, "fasta"):
        # Get the sequence as a string
        seq = str(record.seq).upper()
        
        # Replace 'N' characters with random nucleotides
        seq = replace_n_with_random(seq)
        
        # Remove any other non-standard nucleotides
        seq = re.sub(r'[^ACGT]', '', seq)
        
        # Filter by length
        if min_length and len(seq) < min_length:
            continue
        
        # Truncate to max_length if needed
        if max_length and len(seq) > max_length:
            # Randomly select a subsequence
            start = np.random.randint(0, len(seq) - max_length + 1)
            seq = seq[start:start + max_length]
        
        # Create a new record with the cleaned sequence
        new_record = SeqRecord(
            Seq(seq),
            id=record.id,
            description=record.description
        )
        
        cleaned_records.append(new_record)
    
    # Write cleaned records to output file
    with open(output_path, 'w') as f:
        SeqIO.write(cleaned_records, f, "fasta")
    
    print(f"Preprocessed {len(cleaned_records)} sequences and saved to {output_path}")
    print(f"Average sequence length: {sum(len(r.seq) for r in cleaned_records) / len(cleaned_records) if cleaned_records else 0:.2f}")


def main():
    """
    Main function to preprocess the data.
    """
    # Define input and output paths
    input_path = "data/clean_seq_download.fasta"
    output_path = "data/preprocessed_dna_sequences.fasta"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Preprocess the FASTA file
    preprocess_fasta_file(input_path, output_path, min_length=50, max_length=150)
    
    print("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()
