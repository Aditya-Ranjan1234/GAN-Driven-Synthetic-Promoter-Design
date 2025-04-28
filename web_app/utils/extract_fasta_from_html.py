"""
Utility to extract FASTA content from HTML files.
"""

import os
import re
import argparse
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def extract_fasta_from_html(html_file, output_file=None):
    """
    Extract FASTA content from an HTML file.
    
    Args:
        html_file (str): Path to the HTML file.
        output_file (str, optional): Path to save the extracted FASTA content.
            If None, the content is returned as a string.
            
    Returns:
        str or None: If output_file is None, returns the extracted FASTA content.
            Otherwise, returns None.
    """
    # Read the HTML file
    with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
        html_content = f.read()
    
    # Extract content between <pre> tags
    pre_pattern = re.compile(r'<pre>(.*?)</pre>', re.DOTALL)
    pre_match = pre_pattern.search(html_content)
    
    if pre_match:
        fasta_content = pre_match.group(1)
    else:
        # If no <pre> tags, try to find FASTA-like content
        fasta_pattern = re.compile(r'(>.*?\n.*?(?=>|\Z))', re.DOTALL)
        fasta_matches = fasta_pattern.findall(html_content)
        
        if fasta_matches:
            fasta_content = '\n'.join(fasta_matches)
        else:
            raise ValueError("No FASTA content found in the HTML file.")
    
    # Clean up the content
    fasta_content = fasta_content.strip()
    
    # Save to file if output_file is provided
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(fasta_content)
        print(f"Extracted FASTA content saved to {output_file}")
        return None
    
    return fasta_content


def extract_and_validate_fasta(html_file, output_file=None):
    """
    Extract FASTA content from an HTML file and validate it.
    
    Args:
        html_file (str): Path to the HTML file.
        output_file (str, optional): Path to save the extracted FASTA content.
            
    Returns:
        bool: True if extraction and validation were successful, False otherwise.
    """
    try:
        # Extract FASTA content
        fasta_content = extract_fasta_from_html(html_file, None)
        
        # Parse the FASTA content to validate it
        records = []
        for record in SeqIO.parse(fasta_content.splitlines(), 'fasta'):
            records.append(record)
        
        if not records:
            print(f"No valid FASTA records found in {html_file}")
            return False
        
        print(f"Found {len(records)} valid FASTA records")
        
        # Save to file if output_file is provided
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                f.write(fasta_content)
            print(f"Validated FASTA content saved to {output_file}")
        
        return True
    
    except Exception as e:
        print(f"Error extracting FASTA content: {str(e)}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Extract FASTA content from HTML files')
    parser.add_argument('--input', type=str, required=True, help='Input HTML file')
    parser.add_argument('--output', type=str, required=True, help='Output FASTA file')
    args = parser.parse_args()
    
    success = extract_and_validate_fasta(args.input, args.output)
    
    if success:
        print("Extraction successful")
    else:
        print("Extraction failed")


if __name__ == '__main__':
    main()
