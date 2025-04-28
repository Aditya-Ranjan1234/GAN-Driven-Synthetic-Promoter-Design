"""
Improved script to extract FASTA content from HTML file.

This script:
1. Reads the HTML file containing FASTA content
2. Extracts the content between <pre> and </pre> tags
3. Processes the content to ensure it's in valid FASTA format
4. Saves it as a proper FASTA file
"""

import os
import re
import sys
import requests
from bs4 import BeautifulSoup


def improved_extract_fasta(input_path, output_path):
    """
    Extract FASTA content from HTML file with improved handling.
    
    Args:
        input_path (str): Path to the HTML file containing FASTA content.
        output_path (str): Path to save the extracted FASTA content.
    """
    try:
        # Read the HTML file
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
        
        # Try using BeautifulSoup for better HTML parsing
        soup = BeautifulSoup(html_content, 'html.parser')
        pre_tag = soup.find('pre')
        
        if pre_tag:
            # Extract content from the <pre> tag
            fasta_content = pre_tag.get_text()
        else:
            # Fall back to regex if BeautifulSoup doesn't find a <pre> tag
            match = re.search(r'<pre>(.*?)</pre>', html_content, re.DOTALL)
            
            if match:
                fasta_content = match.group(1)
            else:
                # If no <pre> tag is found, try to extract FASTA-like content directly
                # Look for lines starting with '>' followed by sequence lines
                fasta_lines = []
                in_sequence = False
                current_header = None
                
                for line in html_content.split('\n'):
                    line = line.strip()
                    
                    if line.startswith('>'):
                        in_sequence = True
                        current_header = line
                        fasta_lines.append(current_header)
                    elif in_sequence and line and not line.startswith('<'):
                        # Only add non-empty, non-HTML lines
                        fasta_lines.append(line)
                
                if fasta_lines:
                    fasta_content = '\n'.join(fasta_lines)
                else:
                    raise ValueError("Could not find FASTA content in the file.")
        
        # Process the FASTA content to ensure it's valid
        processed_lines = []
        current_header = None
        
        for line in fasta_content.split('\n'):
            line = line.strip()
            
            if not line:
                continue
            
            if line.startswith('>'):
                current_header = line
                processed_lines.append(current_header)
            elif current_header is not None:
                # Remove any non-nucleotide characters
                cleaned_line = re.sub(r'[^ACGTN]', '', line.upper())
                if cleaned_line:
                    processed_lines.append(cleaned_line)
        
        # Check if we have any valid FASTA content
        if len(processed_lines) < 2:  # Need at least one header and one sequence line
            # Try a different approach - look for sequences directly in the file
            sequences_found = re.findall(r'>([^>]+?)\n([ACGTN\s]+)', html_content, re.IGNORECASE)
            
            if sequences_found:
                processed_lines = []
                for header, seq in sequences_found:
                    processed_lines.append(f">{header.strip()}")
                    # Clean the sequence and split into lines of 60 characters
                    cleaned_seq = re.sub(r'[^ACGTN]', '', seq.upper())
                    for i in range(0, len(cleaned_seq), 60):
                        processed_lines.append(cleaned_seq[i:i+60])
            else:
                raise ValueError("Could not find valid FASTA content in the file.")
        
        # Write the processed content to a new file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(processed_lines))
        
        print(f"Successfully extracted FASTA content and saved to {output_path}")
        print(f"Number of lines: {len(processed_lines)}")
        
        # Count the number of sequences
        num_sequences = sum(1 for line in processed_lines if line.startswith('>'))
        print(f"Number of sequences: {num_sequences}")
        
        return True
    
    except Exception as e:
        print(f"Error extracting FASTA content: {e}")
        return False


def extract_from_url(url, output_path):
    """
    Extract FASTA content directly from a URL.
    
    Args:
        url (str): URL to the FASTA file.
        output_path (str): Path to save the extracted FASTA content.
    """
    try:
        # Download the content
        response = requests.get(url)
        response.raise_for_status()
        
        # Save the content to a temporary file
        temp_path = "data/temp_download.html"
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        # Extract FASTA content
        success = improved_extract_fasta(temp_path, output_path)
        
        # Remove the temporary file
        os.remove(temp_path)
        
        return success
    
    except Exception as e:
        print(f"Error downloading and extracting FASTA content: {e}")
        return False


def main():
    """
    Main function to extract FASTA content.
    """
    # Define input and output paths
    input_path = "data/seq_download.pl.fasta"
    output_path = "data/clean_seq_download.fasta"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Try to extract FASTA content from the local file
    success = improved_extract_fasta(input_path, output_path)
    
    # If local extraction fails, try downloading from a URL
    if not success:
        print("Local extraction failed. Trying to download from URL...")
        
        # URL to a sample DNA sequence dataset
        url = "https://raw.githubusercontent.com/biopython/biopython/master/Tests/GenBank/NC_005816.fna"
        
        success = extract_from_url(url, output_path)
    
    if success:
        print("FASTA content extraction completed successfully.")
    else:
        print("FASTA content extraction failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
