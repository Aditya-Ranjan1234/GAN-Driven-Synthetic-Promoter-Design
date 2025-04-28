# DNA Sequence GAN Streamlit Application

This directory contains the Streamlit web application for the DNA Sequence GAN system.

## Running the Application

To run the application, use the launcher script:

```bash
python run_streamlit_app.py
```

This will launch the Streamlit application in your default web browser. If it doesn't open automatically, navigate to the URL shown in the terminal (typically `http://localhost:8501`).

## Application Structure

- `app.py`: Main Streamlit application file
- `static/`: Directory for static assets (images, etc.)

## Features

The Streamlit application provides a user-friendly interface for:

1. **Data Management**
   - Upload FASTA or CSV files
   - Generate dummy data for testing
   - View data statistics and visualizations

2. **Model Training**
   - Initialize the GAN model with customizable parameters
   - Train the model with real-time progress tracking
   - Load saved model checkpoints

3. **Sequence Generation**
   - Generate synthetic DNA sequences
   - Download generated sequences as CSV
   - View sequence statistics

4. **Evaluation**
   - Evaluate the quality of synthetic sequences
   - View comprehensive evaluation reports
   - Download evaluation reports

5. **Visualization**
   - View sequence logos and interactive sequence viewers
   - Compare k-mer distributions and GC content
   - Visualize training history

## Navigation

The application uses a sidebar for navigation between different sections. Each section provides a user-friendly interface for the corresponding functionality.

## Requirements

The application requires the following Python packages:

- streamlit
- torch
- numpy
- pandas
- matplotlib
- seaborn
- plotly
- biopython
- scikit-learn
- tqdm

These dependencies are listed in the main `requirements.txt` file.
