"""
HTML to CSV Converter for FASTA files

This script extracts FASTA sequences from an HTML file and converts them to CSV format.
"""

import re
import pandas as pd
import os

def extract_fasta_from_html(html_file):
    """
    Extract FASTA sequences from an HTML file.

    Args:
        html_file (str): Path to the HTML file containing FASTA data.

    Returns:
        list: List of dictionaries with sequence_id, description, and sequence.
    """
    # Read the HTML file
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Extract the pre tag content which contains the FASTA data
    pre_content = re.search(r'<pre>(.*?)</pre>', html_content, re.DOTALL)
    if not pre_content:
        raise ValueError("No FASTA data found in the HTML file")

    fasta_content = pre_content.group(1)

    # Use regex to extract FASTA entries
    # Pattern to match: >ID description\nsequence\n>next or end
    pattern = r'>(\S+)\s+([^\n]*)\n([^>]*?)(?=>|$)'
    matches = re.findall(pattern, fasta_content, re.DOTALL)

    sequences = []

    for match in matches:
        sequence_id = match[0]
        description = match[1]
        sequence_lines = match[2].strip().split('\n')
        sequence = ''.join(sequence_lines)

        # Remove 'N' characters (ambiguous nucleotides) from the sequence
        sequence = sequence.replace('N', '')

        # Skip entries with empty sequences or problematic entries
        if sequence and not sequence_id.startswith('</band'):
            sequences.append({
                'sequence_id': sequence_id,
                'description': description,
                'sequence': sequence
            })

    return sequences

def convert_to_csv(sequences, output_file):
    """
    Convert sequences to CSV format.

    Args:
        sequences (list): List of dictionaries with sequence_id, description, and sequence.
        output_file (str): Path to the output CSV file.
    """
    # Create a DataFrame
    df = pd.DataFrame(sequences)

    # Save to CSV
    df.to_csv(output_file, index=False)

    print(f"Converted {len(sequences)} sequences to {output_file}")

def main():
    # Input and output files
    html_file = 'seq_download.pl.fasta.html'
    output_file = 'data/dna_sequences.csv'

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Extract sequences from HTML
    sequences = extract_fasta_from_html(html_file)

    # Convert to CSV
    convert_to_csv(sequences, output_file)

    # Print some statistics
    print(f"Total sequences: {len(sequences)}")

    # Print sample sequences
    print("\nSample sequences:")
    for i, seq in enumerate(sequences[:5]):
        print(f"Sequence {i+1}: {seq['sequence_id']} - {seq['sequence'][:50]}...")

if __name__ == "__main__":
    main()
