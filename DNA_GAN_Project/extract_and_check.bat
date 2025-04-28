@echo off
REM Script to extract FASTA content and check the results

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
pip install -q torch numpy matplotlib tqdm scikit-learn biopython beautifulsoup4 requests

REM Create directories if they don't exist
mkdir data 2>nul

REM Check if the data file exists
if not exist data\seq_download.pl.fasta (
    echo Error: data\seq_download.pl.fasta not found.
    echo Please make sure the file is in the correct location.
    exit /b 1
)

REM Try to extract FASTA content using the improved script
python -m utils.improved_extract_fasta

REM Check if extraction was successful
if not exist data\clean_seq_download.fasta (
    echo Warning: Failed to extract FASTA content from HTML file.
    echo Generating sample FASTA file instead...
    python -m utils.generate_sample_fasta
    
    REM Use the sample FASTA file instead
    copy data\sample_dna_sequences.fasta data\clean_seq_download.fasta
)

REM Check the content of the FASTA file
python -m utils.check_fasta_content

REM Deactivate virtual environment when done
call deactivate
