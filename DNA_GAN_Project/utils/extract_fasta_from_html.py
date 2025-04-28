"""
Script to extract FASTA content from HTML file.

This script:
1. Reads the HTML file containing FASTA content
2. Extracts the content between <pre> and </pre> tags
3. Saves it as a proper FASTA file
"""

import os
import re
import sys


def extract_fasta_from_html(input_path, output_path):
    """
    Extract FASTA content from HTML file.
    
    Args:
        input_path (str): Path to the HTML file containing FASTA content.
        output_path (str): Path to save the extracted FASTA content.
    """
    try:
        # Read the HTML file
        with open(input_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Extract content between <pre> and </pre> tags
        match = re.search(r'<pre>(.*?)</pre>', html_content, re.DOTALL)
        
        if match:
            fasta_content = match.group(1)
            
            # Write the extracted content to a new file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(fasta_content)
            
            print(f"Successfully extracted FASTA content and saved to {output_path}")
            return True
        else:
            print("Error: Could not find FASTA content between <pre> tags.")
            return False
    
    except Exception as e:
        print(f"Error extracting FASTA content: {e}")
        return False


def main():
    """
    Main function to extract FASTA content from HTML file.
    """
    # Define input and output paths
    input_path = "data/seq_download.pl.fasta"
    output_path = "data/clean_seq_download.fasta"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract FASTA content
    success = extract_fasta_from_html(input_path, output_path)
    
    if success:
        print("FASTA content extraction completed successfully.")
    else:
        print("FASTA content extraction failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
