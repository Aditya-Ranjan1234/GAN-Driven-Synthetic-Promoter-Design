@echo off
REM Script to run the DNA GAN Comparison Streamlit app

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
pip install -q torch numpy matplotlib tqdm scikit-learn biopython streamlit

REM Run the Streamlit comparison app with dummy data preparation
python run_comparison_app.py --prepare-data

REM Deactivate virtual environment when done
call deactivate
