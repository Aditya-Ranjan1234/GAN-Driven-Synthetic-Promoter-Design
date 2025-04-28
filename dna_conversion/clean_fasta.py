"""
Clean FASTA file

This script reads the original FASTA file, removes any problematic entries,
and creates a properly formatted FASTA file.
"""

import re
import os

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

    # Check if it's an HTML file
    if '<pre>' in content and '</pre>' in content:
        # Extract the pre tag content
        pre_match = re.search(r'<pre>(.*?)</pre>', content, re.DOTALL)
        if pre_match:
            content = pre_match.group(1)

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

            # Extract ID and description
            id_match = re.match(r'(EP\d+|P\d+)\s+(.*)', header)
            if id_match:
                sequence_id = id_match.group(1)
                description = id_match.group(2).strip()
                # Write the header
                f.write(f">{sequence_id} {description}\n")
            else:
                # If no EP ID found, check if it's an unnamed sequence
                if header.startswith('; range'):
                    f.write(f">Unnamed {header}\n")
                else:
                    # Skip problematic headers
                    continue

            # Join the remaining lines to get the sequence
            sequence = ''.join(lines[1:]).strip()

            # Clean up the sequence - remove non-DNA characters
            sequence = re.sub(r'[^ACGTN]', '', sequence.upper())

            # Skip entries with empty sequences
            if not sequence:
                continue

            # Write sequence with line wrapping at 60 characters
            for i in range(0, len(sequence), 60):
                f.write(f"{sequence[i:i+60]}\n")

            # Add an extra newline between sequences
            f.write("\n")

    print(f"Cleaned FASTA file saved to {output_file}")

def main():
    # Input and output files
    input_file = 'seq_download.pl.fasta'
    output_file = 'clean_sequences.fasta'

    # Clean and format the FASTA file
    clean_fasta_file(input_file, output_file)

if __name__ == "__main__":
    main()
