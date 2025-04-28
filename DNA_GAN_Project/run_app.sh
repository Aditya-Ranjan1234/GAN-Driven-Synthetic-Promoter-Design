#!/bin/bash
# Script to run the DNA GAN Streamlit app

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
pip install -q torch numpy matplotlib tqdm scikit-learn biopython streamlit

# Run the Streamlit app with dummy data preparation
python run_streamlit_app.py --prepare-data

# Deactivate virtual environment when done
deactivate
