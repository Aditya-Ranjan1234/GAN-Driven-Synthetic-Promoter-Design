#!/bin/bash
# Script to run the enhanced DNA GAN web interface

echo "===== DNA Sequence Analysis Dashboard ====="
echo

# Activate the Python environment if it exists
if [ -f "../env310/bin/activate" ]; then
    source ../env310/bin/activate
    echo "[SUCCESS] Activated Python environment"
elif [ -f "../env/bin/activate" ]; then
    source ../env/bin/activate
    echo "[SUCCESS] Activated Python environment"
else
    echo "[WARNING] Python environment not found. Using system Python."
fi

# Install required packages if not already installed
echo "Installing required packages..."
pip install -q streamlit numpy matplotlib pandas seaborn biopython pillow plotly scipy altair
echo "[SUCCESS] Basic packages installed"

echo "Installing PyTorch..."
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
echo "[SUCCESS] PyTorch installed"

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "Creating data directory..."
    mkdir -p data
    echo "[SUCCESS] Created data directory"
fi

# Check if utils directory exists
if [ ! -d "utils" ]; then
    echo "Creating utils directory..."
    mkdir -p utils
    echo "[SUCCESS] Created utils directory"
fi

# Check if the utility scripts exist
if [ ! -f "utils/generate_test_sequences.py" ]; then
    echo "Copying utility scripts..."
    cp ../utils/generate_test_sequences.py utils/ 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "[WARNING] Could not copy generate_test_sequences.py"
    else
        echo "[SUCCESS] Copied utility scripts"
    fi
fi

# Create models directory structure
if [ ! -d "models" ]; then
    echo "Creating models directory structure..."
    mkdir -p models/gumbel_softmax_gan
    mkdir -p models/improved_wgan
    echo "[SUCCESS] Created models directory structure"
fi

# Copy simplified model scripts
echo "Copying simplified model scripts..."
cp ../dna_gan_web/models/gumbel_softmax_gan/generate_sequences_simple.py models/gumbel_softmax_gan/ 2>/dev/null
cp ../dna_gan_web/models/improved_wgan/generate_sequences_simple.py models/improved_wgan/ 2>/dev/null
echo "[SUCCESS] Copied simplified model scripts"

# Set environment variables
export PYTHONPATH="$(pwd)/.."
export ORIGINAL_DATA_PATH="D:/Experiential Learning/Biotech/data/seq_download.pl.fasta"
export PROCESSED_DATA_PATH="$(pwd)/data/preprocessed_dna_sequences.fasta"

echo
echo "Environment variables set:"
echo "PYTHONPATH=$PYTHONPATH"
echo "ORIGINAL_DATA_PATH=$ORIGINAL_DATA_PATH"
echo "PROCESSED_DATA_PATH=$PROCESSED_DATA_PATH"
echo

# Process the original data file if it exists
if [ -f "$ORIGINAL_DATA_PATH" ]; then
    echo "Processing original data file..."

    # Check if the file is an HTML file
    if grep -q "<html" "$ORIGINAL_DATA_PATH"; then
        echo "Original data appears to be an HTML file. Extracting FASTA content..."
        python -c "from utils.extract_fasta_from_html import extract_and_validate_fasta; extract_and_validate_fasta('$ORIGINAL_DATA_PATH', '$PROCESSED_DATA_PATH')"
        if [ $? -ne 0 ]; then
            echo "[WARNING] Failed to extract FASTA content from HTML file."
        else
            echo "[SUCCESS] Extracted FASTA content from HTML file."
            export DATA_PATH="$PROCESSED_DATA_PATH"
        fi
    else
        echo "Original data appears to be a regular FASTA file."
        cp "$ORIGINAL_DATA_PATH" "$PROCESSED_DATA_PATH"
        echo "[SUCCESS] Copied original data to processed data path."
        export DATA_PATH="$PROCESSED_DATA_PATH"
    fi
else
    echo "[WARNING] Original data file not found at $ORIGINAL_DATA_PATH"
    export DATA_PATH="$PROCESSED_DATA_PATH"
fi

echo "Using data from: $DATA_PATH"
echo

# Run the Streamlit app
echo "Running the DNA GAN web interface..."
echo
echo "Press Ctrl+C to stop the application"
echo
streamlit run templates/simplified_app.py
