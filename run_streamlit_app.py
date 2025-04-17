"""
Launcher script for the DNA Sequence GAN Streamlit application.
"""

import os
import subprocess
import sys

def main():
    """
    Launch the Streamlit application.
    """
    print("Launching DNA Sequence GAN Streamlit application...")
    
    # Ensure the required directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Launch the Streamlit application
    cmd = [sys.executable, "-m", "streamlit", "run", "web_app/app.py"]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
