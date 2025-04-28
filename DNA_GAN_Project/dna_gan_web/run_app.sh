#!/bin/bash
# Script to run the DNA GAN web interface

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Python is not installed. Please install Python 3.10."
    exit 1
fi

# Install required packages if not already installed
pip install -q streamlit numpy matplotlib pandas seaborn biopython pillow

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "Creating data directory..."
    mkdir -p data
fi

# Check if the data files exist
if [ ! -f "data/preprocessed_dna_sequences.fasta" ]; then
    echo "Warning: Original data file not found."
    echo "Generating sample data..."
    python -c "from utils.generate_test_sequences import generate_test_sequences; generate_test_sequences('data/preprocessed_dna_sequences.fasta', num_sequences=1000, length=150, gc_bias=0.5, model_name='original')"
fi

if [ ! -f "data/gumbel_generated_sequences.fasta" ]; then
    echo "Warning: Gumbel-Softmax generated sequences not found."
    echo "Generating sample data..."
    python -c "from utils.generate_test_sequences import generate_test_sequences; generate_test_sequences('data/gumbel_generated_sequences.fasta', num_sequences=1000, length=150, gc_bias=0.45, model_name='gumbel')"
fi

if [ ! -f "data/improved_generated_sequences.fasta" ]; then
    echo "Warning: Improved WGAN-GP generated sequences not found."
    echo "Generating sample data..."
    python -c "from utils.generate_test_sequences import generate_test_sequences; generate_test_sequences('data/improved_generated_sequences.fasta', num_sequences=1000, length=150, gc_bias=0.52, model_name='improved')"
fi

# Run the Streamlit app
echo "Running the DNA GAN web interface..."
streamlit run templates/app.py
