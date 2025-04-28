"""
Utility script to preprocess the seq_download.pl.fasta file for GAN training.

This script:
1. Reads the FASTA file
2. Removes 'N' characters
3. Extracts clean DNA sequences
4. Saves the cleaned sequences to a new FASTA file
"""

import os
import re
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def clean_fasta_file(input_path, output_path, min_length=50, max_length=None):
    """
    Clean a FASTA file by removing 'N' characters and filtering by length.
    
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
        
        # Remove 'N' characters
        seq = seq.replace('N', '')
        
        # Remove any other non-standard nucleotides
        seq = re.sub(r'[^ACGT]', '', seq)
        
        # Filter by length
        if min_length and len(seq) < min_length:
            continue
        if max_length and len(seq) > max_length:
            # Truncate to max_length
            seq = seq[:max_length]
        
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
    
    print(f"Cleaned {len(cleaned_records)} sequences and saved to {output_path}")
    print(f"Average sequence length: {sum(len(r.seq) for r in cleaned_records) / len(cleaned_records):.2f}")


def main():
    """
    Main function to preprocess the data.
    """
    # Define input and output paths
    input_path = "../data/seq_download.pl.fasta"
    output_path = "../data/clean_dna_sequences.fasta"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Clean the FASTA file
    clean_fasta_file(input_path, output_path, min_length=50, max_length=150)


if __name__ == "__main__":
    main()
