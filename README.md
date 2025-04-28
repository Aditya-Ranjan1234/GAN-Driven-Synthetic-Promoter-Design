# DNA Sequence GAN

This project provides a comprehensive system for visualizing DNA sequences and generating synthetic DNA sequences using Generative Adversarial Networks (GANs) with PyTorch.

## Features

- **DNA Sequence Visualization**: Interactive visualization of DNA sequences with various representations
- **PyTorch GAN Implementation**: Generate synthetic DNA sequences that mimic real biological sequences using PyTorch
- **Evaluation and Analysis**: Compare real and synthetic sequences using various metrics
- **Web Interface**: User-friendly web interface for visualization and analysis
- **Modular Design**: Well-documented, modular codebase for easy extension and modification
- **Checkpointing**: Save and load model state for continued training

## Project Structure

```
.
├── README.md                 # Project documentation
├── WORKFLOW.md               # Detailed workflow guide
├── data_loader.py            # Functions for loading and preprocessing DNA sequences
├── dna_utils.py              # Utility functions for DNA sequence manipulation
├── gan_model.py              # GAN architecture and training functions using PyTorch
├── visualization.py          # Functions for sequence visualization
├── evaluation.py             # Functions for comparing real and synthetic sequences
├── complete_workflow.py      # Example script demonstrating the complete workflow
├── run_streamlit_app.py      # Launcher script for the Streamlit application
├── web_app/                  # Streamlit web application
│   ├── app.py                # Main Streamlit application file
│   └── static/               # Static assets for the web application
├── models/                   # Directory for saved model checkpoints
├── data/                     # Directory for datasets
└── results/                  # Directory for generated results
```

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Training the GAN

```python
from gan_model import DNAGAN
from data_loader import DNADataLoader

# Initialize the data loader
data_loader = DNADataLoader(sequence_length=100, batch_size=32)

# Load and preprocess data
data_loader.load_fasta('data/sequences.fasta')
data_loader.preprocess()
dataset = data_loader.create_dataset()

# Initialize the GAN
gan = DNAGAN(sequence_length=100, batch_size=32)

# Load data into the GAN
gan.load_data_from_dataset(dataset)

# Train the model
gan.train(epochs=100)

# Generate synthetic sequences
synthetic_sequences = gan.generate(num_sequences=10)

# Save the model for later use
gan.save_checkpoint(epoch=100)
```

### Running the Web Interface

```bash
python run_streamlit_app.py
```

This will launch the Streamlit web application in your default browser. If it doesn't open automatically, navigate to the URL shown in the terminal (typically `http://localhost:8501`).

## Evaluation Metrics

The system provides several metrics for evaluating the quality of synthetic sequences:

1. **K-mer Distribution Analysis**: Compare the distribution of k-mers between real and synthetic sequences
2. **Motif Enrichment Analysis**: Analyze the presence of known DNA motifs
3. **Sequence Diversity**: Measure the diversity of generated sequences
4. **Biological Plausibility**: Assess whether synthetic sequences maintain biological constraints

## Streamlit Web Application

The project includes a comprehensive Streamlit web application that provides a user-friendly interface for working with the DNA Sequence GAN system. The application includes the following features:

- **Data Management**: Upload FASTA or CSV files, or generate dummy data for testing
- **Model Training**: Initialize and train the GAN model with customizable parameters
- **Sequence Generation**: Generate synthetic DNA sequences using the trained model
- **Evaluation**: Evaluate the quality of synthetic sequences using various metrics
- **Visualization**: Create interactive visualizations to analyze and compare sequences

To run the Streamlit application:

```bash
python run_streamlit_app.py
```

This will launch the application in your default web browser. The application provides an intuitive interface for all the functionality of the DNA Sequence GAN system.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
