#!/bin/bash
# Script to run the enhanced DNA GAN web interface with all logging suppressed

# Activate the Python environment if it exists
if [ -f "env310/bin/activate" ]; then
    source env310/bin/activate > /dev/null 2>&1
elif [ -f "env/bin/activate" ]; then
    source env/bin/activate > /dev/null 2>&1
fi

# Install required packages silently
pip install -q streamlit numpy matplotlib pandas biopython plotly scipy > /dev/null 2>&1
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu > /dev/null 2>&1

# Create directories if they don't exist
mkdir -p data utils models/gumbel_softmax_gan models/improved_wgan > /dev/null 2>&1

# Set environment variables
export PYTHONPATH="$PWD/.."
export DATA_PATH="data/clean_all_dna_sequences.fasta"

# Run the Streamlit app with all warnings suppressed
streamlit run web_app/templates/simplified_app.py --logger.level=error
