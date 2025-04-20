# DNA Sequence Generation with Gumbel-Softmax GAN - Documentation

This documentation provides a comprehensive overview of all files and methods in the DNA Sequence Generation project, explaining what each component does and how to use it.

## Project Overview

This project implements a Gumbel-Softmax GAN for generating DNA sequences. It uses LSTM generators with 256 units and provides options for CNN or LSTM discriminators. The implementation includes CUDA support, comprehensive metrics tracking, and visualization tools.

## File Structure

```
dna_gan/
├── __init__.py
├── models.py
├── data.py
├── metrics.py
├── train.py
├── main.py
├── README.md
├── example.py
└── setup.py
```

## Detailed Documentation

### 1. `dna_gan/models.py`

This file contains the neural network model definitions for the Generator and Discriminator.

#### Classes:

##### `Generator`
- **Purpose**: Generates DNA sequences from random noise using LSTM and Gumbel-Softmax.
- **Attributes**:
  - `noise_dim` (int): Dimension of the input noise vector (default: 100).
  - `hidden_dim` (int): Dimension of the LSTM hidden state (default: 256).
  - `seq_len` (int): Length of the generated DNA sequence.
  - `vocab_size` (int): Size of the vocabulary (4 for DNA: A, C, G, T).
  - `fc_hidden`, `fc_cell`: Linear layers to convert noise to initial hidden/cell states.
  - `lstm`: LSTM layer for sequence generation.
  - `fc`: Output projection layer.
  - `dropout`: Dropout layer for regularization.
- **Methods**:
  - `__init__(noise_dim=100, hidden_dim=256, seq_len=150, vocab_size=4)`: Initializes the Generator.
  - `forward(noise, temperature=1.0, hard=False)`: Forward pass generating sequences.
  - `sample(batch_size, device)`: Convenience method to sample sequences.

##### `CNNDiscriminator`
- **Purpose**: Discriminates between real and generated DNA sequences using CNN.
- **Attributes**:
  - `seq_len` (int): Length of the input DNA sequence.
  - `vocab_size` (int): Size of the vocabulary.
  - `conv1`, `conv2`, `conv3`: Convolutional layers.
  - `pool`: Max pooling layer.
  - `fc1`, `fc2`: Fully connected layers.
  - `dropout`: Dropout layer for regularization.
- **Methods**:
  - `__init__(seq_len=150, vocab_size=4)`: Initializes the CNN Discriminator.
  - `forward(x)`: Forward pass returning probability that input is real.

##### `LSTMDiscriminator`
- **Purpose**: Discriminates between real and generated DNA sequences using LSTM.
- **Attributes**:
  - `seq_len` (int): Length of the input DNA sequence.
  - `vocab_size` (int): Size of the vocabulary.
  - `hidden_dim` (int): Dimension of the LSTM hidden state.
  - `lstm`: LSTM layer for sequence processing.
  - `fc1`, `fc2`: Fully connected layers.
  - `dropout`: Dropout layer for regularization.
- **Methods**:
  - `__init__(seq_len=150, vocab_size=4, hidden_dim=256)`: Initializes the LSTM Discriminator.
  - `forward(x)`: Forward pass returning probability that input is real.

### 2. `dna_gan/data.py`

This file provides utilities for loading, preprocessing, and batching DNA sequences.

#### Classes:

##### `DNASequenceDataset`
- **Purpose**: PyTorch Dataset for DNA sequences from FASTA files.
- **Attributes**:
  - `sequences` (list): List of DNA sequences.
  - `seq_len` (int): Length of each sequence.
  - `vocab` (dict): Mapping from nucleotides to indices.
  - `vocab_size` (int): Size of the vocabulary.
- **Methods**:
  - `__init__(fasta_file, seq_len=150)`: Initializes the dataset from a FASTA file.
  - `_load_fasta(fasta_file)`: Loads sequences from a FASTA file.
  - `__len__()`: Returns the number of sequences.
  - `__getitem__(idx)`: Returns a one-hot encoded sequence.

#### Functions:

##### `get_data_loader(fasta_file, batch_size=64, seq_len=150, shuffle=True, num_workers=4)`
- **Purpose**: Creates a DataLoader for DNA sequences.
- **Parameters**:
  - `fasta_file` (str): Path to the FASTA file.
  - `batch_size` (int): Batch size.
  - `seq_len` (int): Length of each sequence.
  - `shuffle` (bool): Whether to shuffle the data.
  - `num_workers` (int): Number of worker processes.
- **Returns**: PyTorch DataLoader.

##### `nucleotide_to_idx(nucleotide)`
- **Purpose**: Converts a nucleotide to its index.
- **Parameters**: `nucleotide` (str): Nucleotide (A, C, G, T, or N).
- **Returns**: Index of the nucleotide.

##### `idx_to_nucleotide(idx)`
- **Purpose**: Converts an index to its nucleotide.
- **Parameters**: `idx` (int): Index of the nucleotide.
- **Returns**: Nucleotide (A, C, G, or T).

##### `one_hot_to_sequence(one_hot)`
- **Purpose**: Converts a one-hot encoded tensor to a DNA sequence.
- **Parameters**: `one_hot` (torch.Tensor): One-hot encoded tensor.
- **Returns**: DNA sequence as a string.

##### `sequences_to_fasta(sequences, file_path, prefix="generated")`
- **Purpose**: Saves sequences to a FASTA file.
- **Parameters**:
  - `sequences` (list): List of DNA sequences.
  - `file_path` (str): Path to save the FASTA file.
  - `prefix` (str): Prefix for sequence names.

### 3. `dna_gan/metrics.py`

This file provides functions for calculating various metrics to evaluate the quality of generated DNA sequences.

#### Functions:

##### `calculate_gc_content(sequences)`
- **Purpose**: Calculates the GC content of sequences.
- **Parameters**: `sequences` (torch.Tensor): One-hot encoded sequences.
- **Returns**: Average GC content as a float.

##### `calculate_morans_i(sequences, k=5)`
- **Purpose**: Calculates Moran's I spatial autocorrelation for sequences.
- **Parameters**:
  - `sequences` (torch.Tensor): One-hot encoded sequences.
  - `k` (int): Number of nearest neighbors to consider.
- **Returns**: Average Moran's I value as a float.

##### `calculate_diversity(sequences)`
- **Purpose**: Calculates the diversity of sequences.
- **Parameters**: `sequences` (torch.Tensor): One-hot encoded sequences.
- **Returns**: Diversity score as a float.

##### `evaluate_model(generator, data_loader, noise_dim=100, device='cuda', num_samples=1000)`
- **Purpose**: Evaluates the generator model.
- **Parameters**:
  - `generator` (nn.Module): Generator model.
  - `data_loader` (DataLoader): DataLoader for real DNA sequences.
  - `noise_dim` (int): Dimension of the input noise vector.
  - `device` (str): Device to use for evaluation.
  - `num_samples` (int): Number of samples to generate.
- **Returns**: Dictionary containing evaluation metrics.

### 4. `dna_gan/train.py`

This file provides functions for training the GAN model and evaluating its performance.

#### Functions:

##### `train_gan(generator, discriminator, data_loader, num_epochs=500, noise_dim=100, batch_size=64, device='cuda', lr_g=1e-4, lr_d=1e-4, checkpoint_dir='checkpoints', log_interval=10, save_interval=10, temperature=1.0)`
- **Purpose**: Trains the GAN model.
- **Parameters**:
  - `generator` (nn.Module): Generator model.
  - `discriminator` (nn.Module): Discriminator model.
  - `data_loader` (DataLoader): DataLoader for real DNA sequences.
  - `num_epochs` (int): Number of training epochs.
  - `noise_dim` (int): Dimension of the input noise vector.
  - `batch_size` (int): Batch size.
  - `device` (str): Device to use for training.
  - `lr_g` (float): Learning rate for the generator.
  - `lr_d` (float): Learning rate for the discriminator.
  - `checkpoint_dir` (str): Directory to save checkpoints.
  - `log_interval` (int): Interval for logging training progress.
  - `save_interval` (int): Interval for saving checkpoints.
  - `temperature` (float): Temperature parameter for Gumbel-Softmax.
- **Returns**: Dictionary containing training history.

##### `plot_training_history(history, save_path=None)`
- **Purpose**: Plots training history.
- **Parameters**:
  - `history` (dict): Dictionary containing training history.
  - `save_path` (str): Path to save the plot.

##### `load_checkpoint(checkpoint_path, generator, discriminator, device)`
- **Purpose**: Loads a checkpoint.
- **Parameters**:
  - `checkpoint_path` (str): Path to the checkpoint.
  - `generator` (nn.Module): Generator model.
  - `discriminator` (nn.Module): Discriminator model.
  - `device` (str): Device to load the models on.
- **Returns**: Tuple containing the loaded generator, discriminator, and history.

### 5. `dna_gan/main.py`

This file provides the main entry point for training the GAN model via command line.

#### Functions:

##### `main()`
- **Purpose**: Main function for training the GAN model.
- **Command-line Arguments**:
  - `--fasta_file` (str): Path to the FASTA file (required).
  - `--seq_len` (int): Length of DNA sequences (default: 150).
  - `--noise_dim` (int): Dimension of the noise vector (default: 100).
  - `--hidden_dim` (int): Dimension of the hidden state (default: 256).
  - `--discriminator_type` (str): Type of discriminator ('cnn' or 'lstm', default: 'cnn').
  - `--batch_size` (int): Batch size (default: 64).
  - `--num_epochs` (int): Number of epochs (default: 500).
  - `--lr_g` (float): Learning rate for generator (default: 1e-4).
  - `--lr_d` (float): Learning rate for discriminator (default: 1e-4).
  - `--temperature` (float): Temperature for Gumbel-Softmax (default: 1.0).
  - `--checkpoint_dir` (str): Directory to save checkpoints (default: 'checkpoints').
  - `--log_interval` (int): Interval for logging training progress (default: 1).
  - `--save_interval` (int): Interval for saving checkpoints (default: 10).
  - `--resume` (str): Path to checkpoint to resume training (default: None).
  - `--seed` (int): Random seed (default: 42).
  - `--no_cuda`: Disable CUDA (default: False).
  - `--num_workers` (int): Number of worker processes for data loading (default: 4).

### 6. `dna_gan/__init__.py`

This file initializes the package and imports key components for easy access.

### 7. `example.py`

This file provides a simplified example of how to use the package.

#### Functions:

##### `main()`
- **Purpose**: Main function for the example script.
- **Command-line Arguments**:
  - `--fasta_file` (str): Path to the FASTA file (required).
  - `--output_dir` (str): Directory to save output (default: 'output').
  - `--num_epochs` (int): Number of epochs (default: 500).
  - `--batch_size` (int): Batch size (default: 64).
  - `--seq_len` (int): Length of DNA sequences (default: 150).
  - `--discriminator_type` (str): Type of discriminator ('cnn' or 'lstm', default: 'cnn').
  - `--no_cuda`: Disable CUDA (default: False).

### 8. `setup.py`

This file provides the setup script for installing the package.

## How to Use

### Training a Model

To train a model, you can use either the example script or the main module:

#### Using the Example Script:

```bash
python example.py --fasta_file path/to/your/sequences.fasta --num_epochs 500 --batch_size 64
```

#### Using the Main Module:

```bash
python -m dna_gan.main --fasta_file path/to/your/sequences.fasta --seq_len 150 --num_epochs 500 --batch_size 64
```

### Resuming Training from a Checkpoint:

```bash
python -m dna_gan.main --fasta_file path/to/your/sequences.fasta --resume path/to/checkpoint.pt
```

### Visualizing Training Progress:

Training progress is automatically visualized and saved as a PNG file in the checkpoint directory. The visualization includes:
- Generator and discriminator loss
- Discriminator accuracy
- GC content
- Moran's I and diversity

### Generating Sequences:

After training, sequences are automatically generated and saved as a FASTA file in the checkpoint directory.

### Using the API:

You can also use the package as a Python API:

```python
import torch
from dna_gan.models import Generator, CNNDiscriminator
from dna_gan.data import get_data_loader
from dna_gan.train import train_gan, plot_training_history

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
data_loader = get_data_loader('path/to/your/sequences.fasta', batch_size=64)

# Create models
generator = Generator(noise_dim=100, hidden_dim=256, seq_len=150, vocab_size=4).to(device)
discriminator = CNNDiscriminator(seq_len=150, vocab_size=4).to(device)

# Train the model
history = train_gan(
    generator=generator,
    discriminator=discriminator,
    data_loader=data_loader,
    num_epochs=500,
    device=device
)

# Plot training history
plot_training_history(history)

# Generate sequences
with torch.no_grad():
    noise = torch.randn(100, 100).to(device)
    generated_sequences = generator(noise, temperature=1.0, hard=True)
```

## Recommended Training Parameters

- **Number of epochs**: 500-800 (with early stopping)
- **Batch size**: 64
- **Learning rates**: 1e-4 for both generator and discriminator
- **Temperature**: 1.0 for Gumbel-Softmax
- **Checkpointing**: Every 10 epochs

## Output Files

After training, the following files will be created in the checkpoint directory:
- Checkpoints saved every `save_interval` epochs (`checkpoint_epoch_X.pt`)
- Final model saved at the end of training (`final_model.pt`)
- Training history plot (`training_history.png`)
- Log file with training progress (`training.log`)
- Generated sequences in FASTA format (`generated_sequences.fasta`)

## Metrics Tracked During Training

- **Generator loss**: Loss function of the generator
- **Discriminator loss**: Loss function of the discriminator
- **Discriminator accuracy**: Accuracy of the discriminator in distinguishing real from fake sequences
- **GC content**: Proportion of G and C nucleotides in the generated sequences
- **Moran's I**: Spatial autocorrelation measure for sequence patterns
- **Diversity**: Measure of sequence diversity in the generated samples

This comprehensive documentation should help you understand and use all components of the DNA Sequence Generation project effectively.
