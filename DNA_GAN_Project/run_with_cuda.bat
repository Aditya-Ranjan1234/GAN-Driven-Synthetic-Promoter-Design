@echo off
REM Script to run the DNA sequence generation with CUDA

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

REM Check CUDA availability
echo Checking CUDA availability...
python -m utils.check_cuda

REM Install PyTorch with CUDA support if needed
echo.
echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

REM Install other required packages
pip install -q numpy matplotlib tqdm scikit-learn biopython pandas seaborn umap-learn

REM Create directories if they don't exist
mkdir data 2>nul
mkdir checkpoints\improved_dna_gan 2>nul
mkdir checkpoints\improved_dna_gan\images 2>nul

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

REM Preprocess the sequences by replacing 'N' characters with random nucleotides
python -m utils.preprocess_with_random_replacement

REM Run the improved DNA GAN with CUDA
echo.
echo Running the improved DNA GAN with CUDA...
python -m models.improved_wgan.run_improved_dna_gan --fasta_file data/preprocessed_dna_sequences.fasta --num_epochs 500 --seq_len 150

REM Deactivate virtual environment when done
call deactivate
