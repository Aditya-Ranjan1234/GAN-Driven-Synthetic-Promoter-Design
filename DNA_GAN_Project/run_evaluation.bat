@echo off
REM Script to run the DNA sequence evaluation

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in PATH. Please install Python 3.10.
    exit /b 1
)

REM Check if virtual environment exists
if not exist env310\ (
    echo Creating virtual environment with Python 3.10...
    python -m venv env310
    if %ERRORLEVEL% neq 0 (
        echo Failed to create virtual environment. Please ensure Python 3.10 is installed.
        exit /b 1
    )
)

REM Activate virtual environment
call env310\Scripts\activate

REM Install required packages if not already installed
pip install -q torch numpy matplotlib tqdm scikit-learn biopython pandas seaborn umap-learn

REM Create directories if they don't exist
mkdir data 2>nul
mkdir evaluation\results 2>nul
mkdir evaluation\visualization\plots 2>nul

REM Check if the data file exists
if not exist data\preprocessed_dna_sequences.fasta (
    echo Warning: Preprocessed data file not found.
    echo Running preprocessing script...
    python -m utils.preprocess_with_random_replacement
)

REM Check if the generated sequences exist
if not exist data\gumbel_generated_sequences.fasta (
    echo Warning: Gumbel-Softmax generated sequences not found.
    echo Please generate sequences first using the Streamlit app.
)

if not exist data\improved_generated_sequences.fasta (
    echo Warning: Improved WGAN-GP generated sequences not found.
    echo Please generate sequences first using the Streamlit app.
)

REM Run the evaluation
python -m evaluation.evaluate_sequences

REM Run the visualization
python -m evaluation.visualization.plot_results

REM Deactivate virtual environment when done
call deactivate

echo Evaluation completed. You can view the results in the Streamlit app.
