#!/bin/bash
# Script to run the DNA sequence evaluation with fixed paths

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

# Check CUDA availability
echo "Checking CUDA availability..."
python -m utils.check_cuda

# Install required packages
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q numpy matplotlib tqdm scikit-learn biopython pandas seaborn umap-learn

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
    echo "Generating test sequences for evaluation..."
    python -m utils.generate_test_sequences
fi

# Run the evaluation
echo "Running evaluation..."
python -c "import sys; sys.path.insert(0, '.'); from evaluation.evaluate_sequences import main; main()"

# Run the visualization
echo "Running visualization..."
python -c "import sys; sys.path.insert(0, '.'); from evaluation.visualization.plot_results import main; main()"

# Deactivate virtual environment when done
deactivate

echo "Evaluation completed. You can view the results in the Streamlit app."
