"""
Validate FASTA file

This script validates a FASTA file to ensure it can be loaded properly.
"""

import re
import os

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

        # Filter out entries that start with 'band' or are empty
        valid_entries = []
        for entry in entries:
            if not entry.strip():
                continue
            if entry.strip().startswith('band'):
                continue
            valid_entries.append(entry)

        # Check each valid entry
        for i, entry in enumerate(valid_entries):
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
        print(f"Total valid sequences: {len(valid_entries)}")
        return True

    except Exception as e:
        print(f"Error validating {fasta_file}: {e}")
        return False

def main():
    # Validate the formatted FASTA file
    import sys

    if len(sys.argv) > 1:
        fasta_file = sys.argv[1]
    else:
        fasta_file = 'formatted_sequences.fasta'

    validate_fasta(fasta_file)

if __name__ == "__main__":
    main()
