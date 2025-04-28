"""
Script to check the content of the FASTA file.

This script:
1. Reads the FASTA file
2. Prints information about the content
3. Helps diagnose issues with the file format
"""

import os
import sys
from Bio import SeqIO


def check_fasta_file(file_path):
    """
    Check the content of a FASTA file.
    
    Args:
        file_path (str): Path to the FASTA file.
    """
    print(f"Checking FASTA file: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File does not exist: {file_path}")
        return
    
    # Check file size
    file_size = os.path.getsize(file_path)
    print(f"File size: {file_size} bytes")
    
    # Read the first 500 characters of the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read(500)
    
    print("\nFirst 500 characters of the file:")
    print("-" * 50)
    print(content)
    print("-" * 50)
    
    # Try to parse the file with BioPython
    try:
        records = list(SeqIO.parse(file_path, "fasta"))
        print(f"\nNumber of sequences found: {len(records)}")
        
        if len(records) > 0:
            print("\nFirst sequence:")
            print(f"ID: {records[0].id}")
            print(f"Description: {records[0].description}")
            print(f"Sequence: {records[0].seq[:50]}...")
            print(f"Length: {len(records[0].seq)}")
        else:
            print("\nNo sequences found in the file.")
            
            # Try to parse with different formats
            print("\nTrying different FASTA formats:")
            
            records = list(SeqIO.parse(file_path, "fasta-2line"))
            print(f"fasta-2line format: {len(records)} sequences found")
            
            records = list(SeqIO.parse(file_path, "fasta-pearson"))
            print(f"fasta-pearson format: {len(records)} sequences found")
            
            records = list(SeqIO.parse(file_path, "fasta-blast"))
            print(f"fasta-blast format: {len(records)} sequences found")
    
    except Exception as e:
        print(f"\nError parsing FASTA file: {e}")
        
        # Print more of the file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read(2000)
        
        print("\nFirst 2000 characters of the file:")
        print("-" * 50)
        print(content)
        print("-" * 50)


def main():
    """
    Main function to check the FASTA file.
    """
    # Define input path
    input_path = "data/clean_seq_download.fasta"
    
    # Check the FASTA file
    check_fasta_file(input_path)


if __name__ == "__main__":
    main()
