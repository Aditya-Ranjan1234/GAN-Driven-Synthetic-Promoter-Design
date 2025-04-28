@echo off
REM Script to run the enhanced DNA GAN web interface with all logging suppressed

REM Activate the Python environment if it exists
if exist env310\Scripts\activate.bat (
    call env310\Scripts\activate.bat >nul 2>&1
) else if exist env\Scripts\activate.bat (
    call env\Scripts\activate.bat >nul 2>&1
)

REM Install required packages silently
pip install -q streamlit numpy matplotlib pandas biopython plotly scipy >nul 2>&1
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu >nul 2>&1

REM Create directories if they don't exist
if not exist data mkdir data >nul 2>&1
if not exist utils mkdir utils >nul 2>&1
if not exist models mkdir models >nul 2>&1
if not exist models\gumbel_softmax_gan mkdir models\gumbel_softmax_gan >nul 2>&1
if not exist models\improved_wgan mkdir models\improved_wgan >nul 2>&1

REM Set environment variables
set PYTHONPATH=%CD%\..
set DATA_PATH=data/clean_all_dna_sequences.fasta

REM Run the Streamlit app with all warnings suppressed
streamlit run web_app/templates/simplified_app.py --logger.level=error
