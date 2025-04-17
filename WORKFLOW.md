# DNA Sequence GAN Workflow Guide

This document provides a comprehensive guide to using the DNA Sequence GAN system for generating and analyzing synthetic DNA sequences.

## Table of Contents

1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Training the GAN](#training-the-gan)
4. [Generating Synthetic Sequences](#generating-synthetic-sequences)
5. [Evaluating Results](#evaluating-results)
6. [Visualization](#visualization)
7. [Web Interface](#web-interface)
8. [Advanced Usage](#advanced-usage)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd dna-sequence-gan
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify the installation by running the test script:
   ```bash
   python test_gan.py
   ```

## Data Preparation

### Using Your Own Data

1. Prepare your DNA sequence data in FASTA or CSV format.
   - FASTA files should contain DNA sequences with headers.
   - CSV files should have a column named 'sequence' containing the DNA sequences.

2. Place your data file in the `data/` directory.

### Using Dummy Data

If you don't have your own data, you can generate dummy data for testing:

```python
from data_loader import DNADataLoader

# Create data loader
data_loader = DNADataLoader(sequence_length=50)

# Generate dummy data
sequences = data_loader.generate_dummy_data(
    num_sequences=100,
    min_length=40,
    max_length=50
)

# Preprocess data
encoded_sequences = data_loader.preprocess()
```

## Training the GAN

### Basic Training

1. Create a Python script (e.g., `train_gan.py`) with the following code:

```python
import torch
from data_loader import DNADataLoader
from gan_model import DNAGAN

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Create data loader
data_loader = DNADataLoader(sequence_length=50, batch_size=32)

# Load data
data_loader.load_fasta('data/your_sequences.fasta')  # or load_csv for CSV files
data_loader.preprocess()
dataset = data_loader.create_dataset()

# Initialize GAN
gan = DNAGAN(
    sequence_length=data_loader.sequence_length,
    batch_size=32,
    latent_dim=100,
    device=device
)

# Load data into GAN
gan.load_data_from_dataset(dataset)

# Train the model
gan.train(epochs=100, save_interval=10, verbose=True)

# Save the final model
gan.save_checkpoint(epoch=100)
```

2. Run the training script:
   ```bash
   python train_gan.py
   ```

### Resuming Training

To resume training from a checkpoint:

```python
# Load checkpoint
gan.load_checkpoint()  # Loads the latest checkpoint
# or
gan.load_checkpoint(epoch=50)  # Loads a specific checkpoint

# Continue training
gan.train(epochs=50, save_interval=10, verbose=True)
```

## Generating Synthetic Sequences

After training, you can generate synthetic DNA sequences:

```python
# Generate 10 synthetic sequences
synthetic_sequences = gan.generate(num_sequences=10)

# Print the sequences
for i, seq in enumerate(synthetic_sequences):
    print(f"Sequence {i+1}: {seq}")

# Save sequences to a file
import pandas as pd
df = pd.DataFrame({
    'sequence_id': range(len(synthetic_sequences)),
    'sequence': synthetic_sequences
})
df.to_csv('data/synthetic_sequences.csv', index=False)
```

## Evaluating Results

The system provides several methods for evaluating the quality of synthetic sequences:

```python
from evaluation import DNAEvaluator

# Ensure sequences have the same length
if len(set(len(seq) for seq in real_sequences)) > 1 or len(set(len(seq) for seq in synthetic_sequences)) > 1:
    from dna_utils import pad_sequences
    max_length = max(max(len(seq) for seq in real_sequences), max(len(seq) for seq in synthetic_sequences))
    real_sequences_padded = pad_sequences(real_sequences, max_length)
    synthetic_sequences_padded = pad_sequences(synthetic_sequences, max_length)
else:
    real_sequences_padded = real_sequences
    synthetic_sequences_padded = synthetic_sequences

# Comprehensive evaluation
evaluation_results = DNAEvaluator.comprehensive_evaluation(
    real_sequences_padded, synthetic_sequences_padded
)

# Generate a report
report = DNAEvaluator.generate_report(evaluation_results)
print(report)

# Individual evaluations
kmer_similarity = DNAEvaluator.kmer_distribution_similarity(
    real_sequences_padded, synthetic_sequences_padded, k=3
)
gc_similarity = DNAEvaluator.gc_content_similarity(
    real_sequences_padded, synthetic_sequences_padded
)
novelty = DNAEvaluator.novelty_check(
    real_sequences_padded, synthetic_sequences_padded
)
```

## Visualization

The system provides various visualization methods:

```python
from visualization import DNAVisualizer
import matplotlib.pyplot as plt

# Sequence logo
fig = DNAVisualizer.sequence_logo(
    synthetic_sequences[:10], 
    title="Synthetic DNA Sequence Logo"
)
plt.show()

# K-mer distribution
fig = DNAVisualizer.kmer_distribution(
    synthetic_sequences, 
    k=3, 
    title="3-mer Distribution in Synthetic Sequences"
)
plt.show()

# Compare k-mer distributions
fig = DNAVisualizer.compare_kmer_distributions(
    real_sequences, 
    synthetic_sequences, 
    k=3, 
    title="3-mer Distribution Comparison"
)
plt.show()

# GC content comparison
fig = DNAVisualizer.compare_gc_distributions(
    real_sequences, 
    synthetic_sequences, 
    title="GC Content Comparison"
)
plt.show()

# Training history
fig = gan.plot_training_history()
plt.show()
```

For interactive visualizations, you can use the Plotly-based methods:

```python
# Interactive k-mer distribution comparison
fig = DNAVisualizer.plotly_compare_kmer_distributions(
    real_sequences, 
    synthetic_sequences, 
    k=3, 
    title="3-mer Distribution Comparison"
)
fig.show()

# Interactive sequence viewer
fig = DNAVisualizer.plotly_sequence_viewer(
    synthetic_sequences[:5], 
    title="Synthetic DNA Sequences"
)
fig.show()
```

## Web Interface

The system includes a web interface for interactive visualization and analysis:

1. Start the web server:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to `http://localhost:5000`.

3. Use the web interface to:
   - Upload DNA sequence data
   - Train the GAN model
   - Generate synthetic sequences
   - Visualize and compare real and synthetic sequences
   - Evaluate the quality of synthetic sequences
   - Download synthetic sequences

## Advanced Usage

### Customizing the GAN Architecture

You can customize the GAN architecture by modifying the `Generator` and `Discriminator` classes in `gan_model.py`:

```python
# Example: Adding more layers to the generator
class Generator(nn.Module):
    def __init__(self, latent_dim, sequence_length, num_classes):
        super(Generator, self).__init__()
        
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        
        # Main network with more layers
        self.main = nn.Sequential(
            # Input layer
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            
            # Additional hidden layer
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            
            # Another hidden layer
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            
            # Output layer
            nn.Linear(1024, sequence_length * num_classes),
        )
        
        # Softmax to get probability distribution over nucleotides
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, z):
        # Pass through main network
        x = self.main(z)
        
        # Reshape to (batch_size, sequence_length, num_classes)
        x = x.view(-1, self.sequence_length, self.num_classes)
        
        # Apply softmax to get probability distribution
        x = self.softmax(x)
        
        return x
```

### Hyperparameter Tuning

You can experiment with different hyperparameters to improve the GAN performance:

```python
# Example: Different hyperparameters
gan = DNAGAN(
    sequence_length=data_loader.sequence_length,
    batch_size=64,  # Larger batch size
    latent_dim=200,  # Larger latent dimension
    device=device
)

# Custom learning rates
gan.generator_optimizer = torch.optim.Adam(
    gan.generator.parameters(), 
    lr=2e-4,  # Different learning rate
    betas=(0.5, 0.999)
)
gan.discriminator_optimizer = torch.optim.Adam(
    gan.discriminator.parameters(), 
    lr=1e-4,
    betas=(0.5, 0.999)
)

# Train with different parameters
gan.train(
    epochs=200,  # More epochs
    save_interval=20,
    verbose=True
)
```

### Creating Custom Evaluation Metrics

You can add custom evaluation metrics by extending the `DNAEvaluator` class in `evaluation.py`:

```python
# Example: Custom evaluation metric
@staticmethod
def custom_metric(real_sequences, synthetic_sequences):
    """
    Custom evaluation metric for DNA sequences.
    
    Args:
        real_sequences (List[str]): List of real DNA sequences.
        synthetic_sequences (List[str]): List of synthetic DNA sequences.
        
    Returns:
        Dict[str, float]: Dictionary with custom metric results.
    """
    # Implement your custom metric here
    # ...
    
    return {
        'custom_score': score,
        'custom_metric_detail': detail
    }
```

## Workflow Summary

1. **Prepare Data**: Load or generate DNA sequence data
2. **Train GAN**: Train the GAN model on the prepared data
3. **Generate Sequences**: Use the trained model to generate synthetic DNA sequences
4. **Evaluate Results**: Compare real and synthetic sequences using various metrics
5. **Visualize**: Create visualizations to analyze the results
6. **Web Interface**: Use the web interface for interactive analysis

By following this workflow, you can effectively use the DNA Sequence GAN system to generate and analyze synthetic DNA sequences for your research or applications.
