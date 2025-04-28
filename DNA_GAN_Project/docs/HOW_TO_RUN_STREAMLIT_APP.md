# How to Run the DNA GAN Streamlit App

This guide provides step-by-step instructions on how to run the Streamlit app for visualizing DNA sequence generation results.

## Prerequisites

- Python 3.10
- pip (Python package installer)

## Option 1: Using the Provided Scripts

### On Windows

1. Open a command prompt
2. Navigate to the project directory:
   ```
   cd path\to\DNA_GAN_Project
   ```
3. Run the batch script:
   ```
   run_app.bat
   ```

### On Unix/MacOS

1. Open a terminal
2. Navigate to the project directory:
   ```
   cd path/to/DNA_GAN_Project
   ```
3. Make the script executable:
   ```
   chmod +x run_app.sh
   ```
4. Run the shell script:
   ```
   ./run_app.sh
   ```

## Option 2: Manual Setup

### Step 1: Create and Activate a Virtual Environment

#### On Windows

```bash
python -m venv env310
env310\Scripts\activate
```

#### On Unix/MacOS

```bash
python -m venv env310
source env310/bin/activate
```

### Step 2: Install Required Packages

```bash
pip install torch numpy matplotlib tqdm scikit-learn biopython streamlit
```

### Step 3: Run the Streamlit App

```bash
python run_streamlit_app.py --prepare-data
```

The `--prepare-data` flag generates dummy data for visualization if you don't have the actual data files.

## Accessing the App

Once the app is running, it will be available at http://localhost:8501 in your web browser.

## App Features

The Streamlit app includes the following pages:

1. **Home**: Overview of the project and available models
2. **Original Data**: Visualization of the original DNA sequences
3. **Generated Data**: Visualization of the generated DNA sequences
4. **Model Comparison**: Comparison between the original and generated sequences
5. **Training Progress**: Visualization of the training progress

## Troubleshooting

### Port Already in Use

If port 8501 is already in use, you can specify a different port:

```bash
python run_streamlit_app.py --prepare-data --port 8502
```

### Missing Data Files

If you see warnings about missing data files, make sure you've used the `--prepare-data` flag or that your data files are in the correct location:

```
DNA_GAN_Project/data/clean_all_dna_sequences.fasta
DNA_GAN_Project/data/gumbel_generated_sequences.fasta
DNA_GAN_Project/data/improved_generated_sequences.fasta
```

### Python Version

This project requires Python 3.10. If you have multiple Python versions installed, make sure you're using the correct one:

```bash
python --version
```

If needed, specify the Python version explicitly:

```bash
python3.10 -m venv env310
```
