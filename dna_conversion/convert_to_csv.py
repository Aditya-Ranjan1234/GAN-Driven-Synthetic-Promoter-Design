"""
Convert HTML FASTA file to CSV

This script extracts DNA sequences from an HTML file containing FASTA data
and converts them to CSV format.
"""

import re
import os
import pandas as pd

def extract_sequences(html_file):
    """
    Extract DNA sequences from an HTML file containing FASTA data.
    
    Args:
        html_file (str): Path to the HTML file.
        
    Returns:
        list: List of dictionaries with sequence_id and sequence.
    """
    # Read the HTML file
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract the pre tag content
    pre_match = re.search(r'<pre>(.*?)</pre>', content, re.DOTALL)
    if not pre_match:
        raise ValueError("No pre tag found in the HTML file")
    
    pre_content = pre_match.group(1)
    
    # Split the content by '>' to get individual sequences
    entries = pre_content.split('>')
    
    # Remove the first empty entry
    if entries and not entries[0].strip():
        entries = entries[1:]
    
    sequences = []
    
    for entry in entries:
        if not entry.strip():
            continue
        
        # Split the entry into lines
        lines = entry.strip().split('\n')
        
        # First line is the header
        header = lines[0]
        
        # Extract sequence ID
        id_match = re.search(r'(EP\d+)', header)
        if not id_match:
            continue
        
        sequence_id = id_match.group(1)
        
        # Join the remaining lines to get the sequence
        sequence = ''.join(lines[1:])
        
        # Remove 'N' characters (ambiguous nucleotides)
        sequence = sequence.replace('N', '')
        
        sequences.append({
            'sequence_id': sequence_id,
            'sequence': sequence
        })
    
    return sequences

def main():
    # Input and output files
    html_file = 'seq_download.pl.fasta.html'
    csv_file = 'data/dna_sequences.csv'
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    # Extract sequences
    sequences = extract_sequences(html_file)
    
    if not sequences:
        print("No sequences found in the HTML file")
        return
    
    # Create DataFrame
    df = pd.DataFrame(sequences)
    
    # Save to CSV
    df.to_csv(csv_file, index=False)
    
    print(f"Converted {len(sequences)} sequences to CSV format")
    print(f"Saved to {csv_file}")
    
    # Print sample sequences
    print("\nSample sequences:")
    for i, seq in enumerate(sequences[:3]):
        print(f"Sequence {i+1}: {seq['sequence_id']} - {seq['sequence'][:50]}...")

if __name__ == "__main__":
    main()
