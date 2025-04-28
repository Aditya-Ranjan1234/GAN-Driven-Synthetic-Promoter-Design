#!/bin/bash
# Script to run the DNA sequence evaluation

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
pip install -q torch numpy matplotlib tqdm scikit-learn biopython pandas seaborn umap-learn

# Create directories if they don't exist
mkdir -p data
mkdir -p evaluation/results
mkdir -p evaluation/visualization/plots

# Check if the data file exists
if [ ! -f "data/preprocessed_dna_sequences.fasta" ]; then
    echo "Warning: Preprocessed data file not found."
    echo "Running preprocessing script..."
    python -m utils.preprocess_with_random_replacement
fi

# Check if the generated sequences exist
if [ ! -f "data/gumbel_generated_sequences.fasta" ]; then
    echo "Warning: Gumbel-Softmax generated sequences not found."
    echo "Please generate sequences first using the Streamlit app."
fi

if [ ! -f "data/improved_generated_sequences.fasta" ]; then
    echo "Warning: Improved WGAN-GP generated sequences not found."
    echo "Please generate sequences first using the Streamlit app."
fi

# Run the evaluation
python -m evaluation.evaluate_sequences

# Run the visualization
python -m evaluation.visualization.plot_results

# Deactivate virtual environment when done
deactivate

echo "Evaluation completed. You can view the results in the Streamlit app."
