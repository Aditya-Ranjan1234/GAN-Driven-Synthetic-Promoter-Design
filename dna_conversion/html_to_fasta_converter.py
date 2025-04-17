"""
HTML to CSV Converter

This script extracts DNA sequences from an HTML file and converts them directly to CSV format.
"""

import re
import os
import pandas as pd

def extract_sequences_from_html(html_file):
    """
    Extract DNA sequences from an HTML file.

    Args:
        html_file (str): Path to the HTML file containing sequence data.

    Returns:
        list: List of dictionaries with sequence_id and sequence.
    """
    # Read the HTML file
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Extract the pre tag content which contains the sequence data
    pre_content = re.search(r'<pre>(.*?)</pre>', html_content, re.DOTALL)
    if not pre_content:
        raise ValueError("No sequence data found in the HTML file")

    content = pre_content.group(1)

    # Parse the content line by line
    lines = content.split('\n')
    sequences = []
    current_id = None
    current_sequence = ''

    for line in lines:
        line = line.strip()

        # Check if this is a header line (contains sequence ID)
        if '>EP' in line:
            # Save previous sequence if exists
            if current_id and current_sequence:
                sequences.append({
                    'sequence_id': current_id,
                    'sequence': current_sequence
                })

            # Extract new ID
            match = re.search(r'>(EP\d+)', line)
            if match:
                current_id = match.group(1)
                current_sequence = ''
        elif current_id and line and not line.startswith(';'):
            # Add to current sequence if not a comment line
            # Remove any non-sequence characters
            cleaned_line = re.sub(r'[^ACGT]', '', line.upper())
            current_sequence += cleaned_line

    # Add the last sequence
    if current_id and current_sequence:
        sequences.append({
            'sequence_id': current_id,
            'sequence': current_sequence
        })

    return sequences

def main():
    # Input and output files
    html_file = 'seq_download.pl.fasta.html'
    csv_file = 'data/dna_sequences.csv'

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    # Extract sequences from HTML
    sequences = extract_sequences_from_html(html_file)

    # Create DataFrame
    df = pd.DataFrame(sequences)

    # Save to CSV
    df.to_csv(csv_file, index=False)

    print(f"Converted HTML to CSV format. Saved to {csv_file}")
    print(f"Total sequences: {len(sequences)}")

    # Print sample sequences
    if sequences:
        print("\nSample sequences:")
        for i, seq in enumerate(sequences[:3]):
            print(f"Sequence {i+1}: {seq['sequence_id']} - {seq['sequence'][:50]}...")

if __name__ == "__main__":
    main()
