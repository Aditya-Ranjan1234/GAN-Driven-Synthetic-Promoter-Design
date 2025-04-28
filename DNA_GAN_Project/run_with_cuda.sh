#!/bin/bash
# Script to run the DNA sequence generation with CUDA

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

# Install PyTorch with CUDA support if needed
echo ""
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other required packages
pip install -q numpy matplotlib tqdm scikit-learn biopython pandas seaborn umap-learn

# Create directories if they don't exist
mkdir -p data
mkdir -p checkpoints/improved_dna_gan
mkdir -p checkpoints/improved_dna_gan/images

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

# Preprocess the sequences by replacing 'N' characters with random nucleotides
python -m utils.preprocess_with_random_replacement

# Run the improved DNA GAN with CUDA
echo ""
echo "Running the improved DNA GAN with CUDA..."
python -m models.improved_wgan.run_improved_dna_gan --fasta_file data/preprocessed_dna_sequences.fasta --num_epochs 500 --seq_len 150

# Deactivate virtual environment when done
deactivate
