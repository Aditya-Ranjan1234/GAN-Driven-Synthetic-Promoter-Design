"""
Extract DNA sequences from HTML file and convert to CSV

This script extracts DNA sequences from an HTML file containing FASTA data,
cleans up the data, and converts it to CSV format.
"""

import re
import os
import pandas as pd

def extract_fasta_from_html(html_file):
    """
    Extract FASTA content from an HTML file.
    
    Args:
        html_file (str): Path to the HTML file.
        
    Returns:
        str: FASTA formatted content
    """
    # Read the HTML file
    with open(html_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract the pre tag content
    pre_match = re.search(r'<pre>(.*?)</pre>', content, re.DOTALL)
    if not pre_match:
        raise ValueError("No pre tag found in the HTML file")
    
    return pre_match.group(1)

def parse_fasta(fasta_content):
    """
    Parse FASTA content into a list of records.
    
    Args:
        fasta_content (str): FASTA formatted content.
        
    Returns:
        list: List of dictionaries with sequence_id, description, and sequence.
    """
    # Split the content by '>' to get individual sequences
    entries = fasta_content.split('>')
    
    # Remove the first empty entry
    if entries and not entries[0].strip():
        entries = entries[1:]
    
    records = []
    
    for entry in entries:
        if not entry.strip():
            continue
        
        # Split the entry into lines
        lines = entry.strip().split('\n')
        if not lines:
            continue
        
        # First line is the header
        header = lines[0].strip()
        
        # Clean up the header - remove HTML entities
        header = header.replace('&gt;', '')
        
        # Extract ID and description from header
        id_match = re.match(r'(EP\d+|P\d+)\s+(.*)', header)
        if id_match:
            sequence_id = id_match.group(1)
            description = id_match.group(2).strip()
        else:
            # If no EP ID found, use the whole header as ID
            sequence_id = header
            description = ""
        
        # Join the remaining lines to get the sequence
        sequence = ''.join(lines[1:]).strip()
        
        # Clean up the sequence - remove HTML entities and non-DNA characters
        sequence = re.sub(r'&gt;.*$', '', sequence)  # Remove any trailing HTML entities
        sequence = re.sub(r'[^ACGTN]', '', sequence.upper())  # Keep only DNA characters
        
        # Skip entries with empty sequences
        if sequence:
            records.append({
                'sequence_id': sequence_id,
                'description': description,
                'sequence': sequence
            })
    
    return records

def main():
    # Input and output files
    html_file = 'seq_download.pl.fasta.html'
    csv_file = 'data/dna_sequences.csv'
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    # Extract FASTA content from HTML
    fasta_content = extract_fasta_from_html(html_file)
    
    # Parse FASTA records
    records = parse_fasta(fasta_content)
    
    if not records:
        print("No sequences found in the HTML file")
        return
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Save to CSV
    df.to_csv(csv_file, index=False)
    
    print(f"Converted {len(records)} sequences to CSV format")
    print(f"Saved to {csv_file}")
    
    # Print sample sequences
    print("\nSample sequences:")
    for i, record in enumerate(records[:3]):
        print(f"Sequence {i+1}: {record['sequence_id']} - {record['description']}")
        print(f"   Sequence length: {len(record['sequence'])} bp")
        print(f"   First 50 bp: {record['sequence'][:50]}...")

if __name__ == "__main__":
    main()
