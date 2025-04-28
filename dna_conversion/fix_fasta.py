"""
Fix FASTA file

This script reads a FASTA file, removes any problematic entries,
and creates a properly formatted FASTA file.
"""

import re
import os
import sys

def fix_fasta_file(input_file, output_file):
    """
    Fix a FASTA file by removing problematic entries.
    
    Args:
        input_file (str): Path to the input FASTA file.
        output_file (str): Path to the output FASTA file.
    """
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split the content by '>' to get individual sequences
    entries = content.split('>')
    
    # Remove the first empty entry
    if entries and not entries[0].strip():
        entries = entries[1:]
    
    # Open the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in entries:
            if not entry.strip():
                continue
            
            # Skip entries that start with 'band'
            if entry.strip().startswith('band'):
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
            
            # Add an extra newline between sequences
            f.write("\n")
    
    print(f"Fixed FASTA file saved to {output_file}")
    return True

def validate_fasta(fasta_file):
    """
    Validate a FASTA file.
    
    Args:
        fasta_file (str): Path to the FASTA file.
        
    Returns:
        bool: True if the file is valid, False otherwise.
    """
    try:
        # Read the FASTA file
        with open(fasta_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if the file starts with '>'
        if not content.strip().startswith('>'):
            print(f"Error: {fasta_file} does not start with '>'")
            return False
        
        # Split the content by '>' to get individual sequences
        entries = content.split('>')
        
        # Remove the first empty entry
        if entries and not entries[0].strip():
            entries = entries[1:]
        
        # Check each entry
        for i, entry in enumerate(entries):
            if not entry.strip():
                print(f"Warning: Empty entry found at position {i+1}")
                continue
            
            # Split the entry into lines
            lines = entry.strip().split('\n')
            if not lines:
                print(f"Error: Entry at position {i+1} has no lines")
                return False
            
            # First line is the header
            header = lines[0].strip()
            if not header:
                print(f"Error: Entry at position {i+1} has an empty header")
                return False
            
            # Check the sequence
            sequence = ''.join(lines[1:]).strip()
            if not sequence:
                print(f"Error: Entry at position {i+1} has an empty sequence")
                return False
            
            # Check for invalid characters in the sequence
            invalid_chars = re.findall(r'[^ACGTN\s]', sequence)
            if invalid_chars:
                print(f"Warning: Entry at position {i+1} has invalid characters: {set(invalid_chars)}")
        
        print(f"FASTA file {fasta_file} is valid")
        print(f"Total sequences: {len(entries)}")
        return True
    
    except Exception as e:
        print(f"Error validating {fasta_file}: {e}")
        return False

def main():
    # Input and output files
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = 'clean_dna_sequences.fasta'
    
    output_file = 'final_dna_sequences.fasta'
    
    # Fix the FASTA file
    if fix_fasta_file(input_file, output_file):
        # Validate the fixed FASTA file
        validate_fasta(output_file)

if __name__ == "__main__":
    main()
