@echo off
REM Script to run the DNA GAN web interface

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in PATH. Please install Python 3.10.
    exit /b 1
)

REM Install required packages if not already installed
pip install -q streamlit numpy matplotlib pandas seaborn biopython pillow

REM Check if data directory exists
if not exist data (
    echo Creating data directory...
    mkdir data
)

REM Check if the data files exist
if not exist data\preprocessed_dna_sequences.fasta (
    echo Warning: Original data file not found.
    echo Generating sample data...
    python -c "from utils.generate_test_sequences import generate_test_sequences; generate_test_sequences('data/preprocessed_dna_sequences.fasta', num_sequences=1000, length=150, gc_bias=0.5, model_name='original')"
)

if not exist data\gumbel_generated_sequences.fasta (
    echo Warning: Gumbel-Softmax generated sequences not found.
    echo Generating sample data...
    python -c "from utils.generate_test_sequences import generate_test_sequences; generate_test_sequences('data/gumbel_generated_sequences.fasta', num_sequences=1000, length=150, gc_bias=0.45, model_name='gumbel')"
)

if not exist data\improved_generated_sequences.fasta (
    echo Warning: Improved WGAN-GP generated sequences not found.
    echo Generating sample data...
    python -c "from utils.generate_test_sequences import generate_test_sequences; generate_test_sequences('data/improved_generated_sequences.fasta', num_sequences=1000, length=150, gc_bias=0.52, model_name='improved')"
)

REM Run the Streamlit app
echo Running the DNA GAN web interface...
streamlit run templates\app.py
