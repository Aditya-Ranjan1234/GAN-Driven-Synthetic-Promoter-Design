"""
Extract all sequences from HTML file and convert to FASTA

This script extracts all DNA sequences from an HTML file containing FASTA data,
cleans up the data, and converts it to a properly formatted FASTA file.
"""

import re
import os
import sys

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

def clean_and_save_fasta(fasta_content, output_file):
    """
    Clean and save FASTA content to a file.
    
    Args:
        fasta_content (str): FASTA formatted content.
        output_file (str): Path to the output FASTA file.
    """
    # Replace HTML entities
    fasta_content = fasta_content.replace('&gt;', '>')
    
    # Write the cleaned content to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(fasta_content)
    
    print(f"Extracted FASTA content saved to {output_file}")

def count_sequences(fasta_file):
    """
    Count the number of sequences in a FASTA file.
    
    Args:
        fasta_file (str): Path to the FASTA file.
        
    Returns:
        int: Number of sequences in the file.
    """
    with open(fasta_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Count the number of sequences
    sequences = re.findall(r'^>.*$', content, re.MULTILINE)
    
    return len(sequences)

def main():
    # Input and output files
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = 'seq_download.pl.fasta'
    
    output_file = 'all_dna_sequences.fasta'
    
    # Extract FASTA content from HTML
    try:
        fasta_content = extract_fasta_from_html(input_file)
        
        # Clean and save FASTA content
        clean_and_save_fasta(fasta_content, output_file)
        
        # Count the number of sequences
        num_sequences = count_sequences(output_file)
        
        print(f"Extracted {num_sequences} sequences from {input_file}")
    except Exception as e:
        print(f"Error extracting sequences: {e}")

if __name__ == "__main__":
    main()
