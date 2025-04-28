@echo off
REM Script to run the enhanced DNA GAN web interface

echo ===== DNA Sequence Analysis Dashboard =====
echo.

REM Activate the Python environment if it exists
if exist ..\env310\Scripts\activate.bat (
    call ..\env310\Scripts\activate.bat
    echo [SUCCESS] Activated Python environment
) else if exist ..\env\Scripts\activate.bat (
    call ..\env\Scripts\activate.bat
    echo [SUCCESS] Activated Python environment
) else (
    echo [WARNING] Python environment not found. Using system Python.
)

REM Install required packages if not already installed
echo Installing required packages...
pip install -q streamlit numpy matplotlib pandas seaborn biopython pillow plotly scipy altair
echo [SUCCESS] Basic packages installed

echo Installing PyTorch...
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
echo [SUCCESS] PyTorch installed

REM Check if data directory exists
if not exist data (
    echo Creating data directory...
    mkdir data
    echo [SUCCESS] Created data directory
)

REM Check if utils directory exists
if not exist utils (
    echo Creating utils directory...
    mkdir utils
    echo [SUCCESS] Created utils directory
)

REM Check if the utility scripts exist
if not exist utils\generate_test_sequences.py (
    echo Copying utility scripts...
    copy ..\utils\generate_test_sequences.py utils\ >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] Could not copy generate_test_sequences.py
    ) else (
        echo [SUCCESS] Copied utility scripts
    )
)

REM Create models directory structure
if not exist models (
    echo Creating models directory structure...
    mkdir models
    mkdir models\gumbel_softmax_gan
    mkdir models\improved_wgan
    echo [SUCCESS] Created models directory structure
)

REM Copy simplified model scripts
echo Copying simplified model scripts...
copy ..\dna_gan_web\models\gumbel_softmax_gan\generate_sequences_simple.py models\gumbel_softmax_gan\ >nul 2>&1
copy ..\dna_gan_web\models\improved_wgan\generate_sequences_simple.py models\improved_wgan\ >nul 2>&1
echo [SUCCESS] Copied simplified model scripts

REM Set environment variables
set PYTHONPATH=%CD%\..
set ORIGINAL_DATA_PATH=D:\Experiential Learning\Biotech\data\seq_download.pl.fasta
set PROCESSED_DATA_PATH=%CD%\data\preprocessed_dna_sequences.fasta

echo.
echo Environment variables set:
echo PYTHONPATH=%PYTHONPATH%
echo ORIGINAL_DATA_PATH=%ORIGINAL_DATA_PATH%
echo PROCESSED_DATA_PATH=%PROCESSED_DATA_PATH%
echo.

REM Process the original data file if it exists
if exist "%ORIGINAL_DATA_PATH%" (
    echo Processing original data file...

    REM Check if the file is an HTML file
    findstr "<html" "%ORIGINAL_DATA_PATH%" >nul
    if not errorlevel 1 (
        echo Original data appears to be an HTML file. Extracting FASTA content...
        python process_html_fasta.py "%ORIGINAL_DATA_PATH%" "%PROCESSED_DATA_PATH%"
        if errorlevel 1 (
            echo [WARNING] Failed to extract FASTA content from HTML file.
            echo Generating sample data...
            python -c "from utils.generate_test_sequences import generate_test_sequences; generate_test_sequences('%PROCESSED_DATA_PATH%', num_sequences=100, length=150, gc_bias=0.5, model_name='original')"
            echo [SUCCESS] Generated sample data.
        ) else (
            echo [SUCCESS] Extracted FASTA content from HTML file.
        )
        set DATA_PATH=%PROCESSED_DATA_PATH%
    ) else (
        echo Original data appears to be a regular FASTA file.
        copy "%ORIGINAL_DATA_PATH%" "%PROCESSED_DATA_PATH%" >nul
        echo [SUCCESS] Copied original data to processed data path.
        set DATA_PATH=%PROCESSED_DATA_PATH%
    )
) else (
    echo [WARNING] Original data file not found at %ORIGINAL_DATA_PATH%
    echo Generating sample data...
    python -c "from utils.generate_test_sequences import generate_test_sequences; generate_test_sequences('%PROCESSED_DATA_PATH%', num_sequences=100, length=150, gc_bias=0.5, model_name='original')"
    echo [SUCCESS] Generated sample data.
    set DATA_PATH=%PROCESSED_DATA_PATH%
)

echo Using data from: %DATA_PATH%
echo.

REM Run the Streamlit app
echo Running the DNA GAN web interface...
echo.
echo Press Ctrl+C to stop the application
echo.
streamlit run templates\simplified_app.py
