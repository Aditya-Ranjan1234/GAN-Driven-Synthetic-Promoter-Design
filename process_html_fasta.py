"""
Process HTML FASTA file to extract clean FASTA content.
This version suppresses all logging messages.
"""

import os
import re
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

def extract_fasta_from_html(input_file, output_file):
    """
    Extract FASTA content from an HTML file.
    
    Args:
        input_file (str): Path to the input HTML file.
        output_file (str): Path to save the extracted FASTA content.
    
    Returns:
        bool: True if extraction was successful, False otherwise.
    """
    try:
        # Read the input file
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract content between <pre> tags
        pre_pattern = re.compile(r'<pre>(.*?)</pre>', re.DOTALL)
        pre_match = pre_pattern.search(content)
        
        if pre_match:
            fasta_content = pre_match.group(1)
            
            # Clean up the content
            fasta_content = fasta_content.replace('&gt;', '>')
            
            # Write to output file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(fasta_content)
            
            return True
        
        # If no <pre> tags, try to extract FASTA-like content directly
        fasta_pattern = re.compile(r'(>.*?\n(?:[ACGTN]+\n)+)', re.DOTALL)
        fasta_matches = fasta_pattern.findall(content)
        
        if fasta_matches:
            with open(output_file, 'w', encoding='utf-8') as f:
                for match in fasta_matches:
                    f.write(match)
            
            return True
        
        return False
    
    except Exception:
        return False

def validate_fasta(file_path):
    """
    Validate that a file contains FASTA format sequences.
    
    Args:
        file_path (str): Path to the FASTA file.
    
    Returns:
        bool: True if the file contains valid FASTA sequences, False otherwise.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Check if the file contains at least one FASTA header
        if not re.search(r'^>\S+', content, re.MULTILINE):
            return False
        
        # Check if the file contains sequence data
        if not re.search(r'^[ACGTN]+$', content, re.MULTILINE):
            return False
        
        return True
    
    except Exception:
        return False

def extract_and_validate_fasta(input_file, output_file):
    """
    Extract FASTA content from an HTML file and validate it.
    
    Args:
        input_file (str): Path to the input HTML file.
        output_file (str): Path to save the extracted FASTA content.
    
    Returns:
        bool: True if extraction and validation were successful, False otherwise.
    """
    # Suppress all output during extraction
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        # Extract FASTA content
        extraction_success = extract_fasta_from_html(input_file, output_file)
        
        if not extraction_success:
            return False
        
        # Validate the extracted FASTA content
        validation_success = validate_fasta(output_file)
        
        return validation_success

if __name__ == "__main__":
    # Suppress all output
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        if len(sys.argv) != 3:
            sys.exit(1)
        
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        
        success = extract_and_validate_fasta(input_file, output_file)
        
        sys.exit(0 if success else 1)
