#!/bin/bash
# Script to preprocess DNA sequences and check the results

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Python is not installed. Please install Python 3.10."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "env310" ]; then
    echo "Creating virtual environment with Python 3.10..."
    python -m venv env310
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Please ensure Python 3.10 is installed."
        exit 1
    fi
fi

# Activate virtual environment
source env310/bin/activate

# Install required packages if not already installed
pip install -q torch numpy matplotlib tqdm scikit-learn biopython beautifulsoup4 requests

# Create directories if they don't exist
mkdir -p data

# Check if the data file exists
if [ ! -f "data/seq_download.pl.fasta" ]; then
    echo "Error: data/seq_download.pl.fasta not found."
    echo "Please make sure the file is in the correct location."
    exit 1
fi

# Try to extract FASTA content using the improved script
python -m utils.improved_extract_fasta

# Check if extraction was successful
if [ ! -f "data/clean_seq_download.fasta" ]; then
    echo "Warning: Failed to extract FASTA content from HTML file."
    echo "Generating sample FASTA file instead..."
    python -m utils.generate_sample_fasta
    
    # Use the sample FASTA file instead
    cp data/sample_dna_sequences.fasta data/clean_seq_download.fasta
fi

# Check the content of the extracted FASTA file
python -m utils.check_fasta_content

# Preprocess the sequences by replacing 'N' characters with random nucleotides
python -m utils.preprocess_with_random_replacement

# Check the content of the preprocessed FASTA file
echo
echo "Checking preprocessed FASTA file:"
python -c "from utils.check_fasta_content import check_fasta_file; check_fasta_file('data/preprocessed_dna_sequences.fasta')"

# Deactivate virtual environment when done
deactivate
