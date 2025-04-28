"""
Script to directly process the HTML FASTA file.
"""

import os
import re
import sys


def extract_fasta_from_html(html_file, output_file):
    """
    Extract FASTA content from an HTML file.
    
    Args:
        html_file (str): Path to the HTML file.
        output_file (str): Path to save the extracted FASTA content.
            
    Returns:
        bool: True if extraction was successful, False otherwise.
    """
    try:
        print(f"Processing HTML file: {html_file}")
        
        # Read the HTML file
        with open(html_file, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
        
        print(f"HTML file size: {len(html_content)} bytes")
        
        # Extract content between <pre> tags
        pre_pattern = re.compile(r'<pre>(.*?)</pre>', re.DOTALL)
        pre_match = pre_pattern.search(html_content)
        
        if pre_match:
            print("Found <pre> tags in HTML")
            fasta_content = pre_match.group(1)
        else:
            print("No <pre> tags found, trying to find FASTA-like content")
            # If no <pre> tags, try to find FASTA-like content
            fasta_pattern = re.compile(r'(>.*?\n.*?(?=>|\Z))', re.DOTALL)
            fasta_matches = fasta_pattern.findall(html_content)
            
            if fasta_matches:
                print(f"Found {len(fasta_matches)} FASTA-like entries")
                fasta_content = '\n'.join(fasta_matches)
            else:
                # Last resort: look for any line starting with '>'
                lines = html_content.splitlines()
                fasta_lines = []
                in_sequence = False
                
                for line in lines:
                    line = line.strip()
                    if line.startswith('>'):
                        fasta_lines.append(line)
                        in_sequence = True
                    elif in_sequence and line and not line.startswith('<'):
                        fasta_lines.append(line)
                
                if fasta_lines:
                    print(f"Found {len(fasta_lines)} lines of FASTA-like content")
                    fasta_content = '\n'.join(fasta_lines)
                else:
                    print("No FASTA content found in the HTML file")
                    return False
        
        # Clean up the content
        fasta_content = fasta_content.strip()
        
        # Validate the FASTA content
        lines = fasta_content.splitlines()
        valid_records = 0
        current_header = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('>'):
                current_header = line
                valid_records += 1
        
        if valid_records == 0:
            print(f"No valid FASTA records found in {html_file}")
            return False
        
        print(f"Found {valid_records} valid FASTA records")
        
        # Save to file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(fasta_content)
        
        print(f"Saved FASTA content to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error extracting FASTA content: {str(e)}")
        return False


def main():
    """Main function."""
    if len(sys.argv) != 3:
        print("Usage: python process_html_fasta.py <input_html_file> <output_fasta_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    success = extract_fasta_from_html(input_file, output_file)
    
    if success:
        print("Extraction successful")
        sys.exit(0)
    else:
        print("Extraction failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
