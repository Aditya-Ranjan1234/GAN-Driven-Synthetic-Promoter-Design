"""
Clean all sequences from FASTA file

This script reads a FASTA file, cleans up the data, and creates a properly
formatted FASTA file that can be loaded by bioinformatics tools.
"""

import re
import os
import sys

def clean_fasta_file(input_file, output_file):
    """
    Clean and properly format a FASTA file.
    
    Args:
        input_file (str): Path to the input FASTA file.
        output_file (str): Path to the output FASTA file.
    """
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove any HTML or XML tags at the end of the file
    content = re.sub(r'</[^>]+>$', '', content)
    
    # Split the content by '>' to get individual sequences
    entries = content.split('>')
    
    # Remove the first empty entry
    if entries and not entries[0].strip():
        entries = entries[1:]
    
    # Count valid sequences
    valid_count = 0
    
    # Open the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in entries:
            if not entry.strip():
                continue
            
            # Skip entries that start with 'band' or 'gene'
            if entry.strip().startswith(('band', 'gene')):
                continue
            
            # Split the entry into lines
            lines = entry.strip().split('\n')
            if not lines:
                continue
            
            # First line is the header
            header = lines[0].strip()
            
            # Join the remaining lines to get the sequence
            sequence = ''.join(lines[1:]).strip()
            
            # Skip entries with empty sequences
            if not sequence:
                continue
            
            # Write the header and sequence
            f.write(f">{header}\n")
            
            # Write sequence with line wrapping at 60 characters
            for i in range(0, len(sequence), 60):
                f.write(f"{sequence[i:i+60]}\n")
            
            valid_count += 1
    
    print(f"Cleaned FASTA file saved to {output_file}")
    print(f"Total valid sequences: {valid_count}")
    return valid_count

def main():
    # Input and output files
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = 'all_dna_sequences.fasta'
    
    output_file = 'clean_all_dna_sequences.fasta'
    
    # Clean the FASTA file
    clean_fasta_file(input_file, output_file)

if __name__ == "__main__":
    main()
