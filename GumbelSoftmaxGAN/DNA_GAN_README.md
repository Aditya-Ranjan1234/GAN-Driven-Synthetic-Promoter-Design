# DNA Sequence Generation with Gumbel-Softmax GAN

This project provides a framework for generating DNA sequences using a Gumbel-Softmax GAN with LSTM generators. It's designed for training on DNA sequence datasets and generating new, biologically plausible sequences.

## Features

- **Gumbel-Softmax GAN**: Uses the Gumbel-Softmax trick to handle discrete outputs (DNA nucleotides)
- **LSTM Generator**: 256-unit LSTM architecture for sequence generation
- **Flexible Discriminator**: Choice between CNN or LSTM-based discriminator
- **CUDA Support**: GPU acceleration for faster training
- **Comprehensive Metrics**: Tracks GC content, Moran's I spatial autocorrelation, and sequence diversity
- **Checkpointing**: Save and resume training with model checkpoints
- **Visualization**: Plot training progress and metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dna-gan.git
cd dna-gan

# Install the package
pip install -e .
```

## Quick Start

```bash
# Train a model using the example script
python example.py --fasta_file path/to/your/sequences.fasta --num_epochs 500 --batch_size 64
```

## Usage

### Command-line Interface

The package provides a command-line interface for training models:

```bash
python -m dna_gan.main --fasta_file path/to/your/sequences.fasta --seq_len 150 --num_epochs 500 --batch_size 64
```

### Python API

You can also use the package as a Python API:

```python
import torch
from dna_gan.models import Generator, CNNDiscriminator
from dna_gan.data import get_data_loader
from dna_gan.train import train_gan

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
```

## Training Parameters

The training process can be customized with the following parameters:

- **Number of epochs**: Typically 500-800 epochs are needed for convergence
- **Batch size**: Default is 64
- **Learning rates**: Default is 1e-4 for both generator and discriminator
- **Temperature**: Controls the sharpness of the Gumbel-Softmax distribution
- **Checkpointing**: Save checkpoints every 10 epochs by default

## Monitoring Training

During training, the following metrics are tracked:

- **Generator loss**: Loss function of the generator
- **Discriminator loss**: Loss function of the discriminator
- **Discriminator accuracy**: Accuracy of the discriminator in distinguishing real from fake sequences
- **GC content**: Proportion of G and C nucleotides in the generated sequences
- **Moran's I**: Spatial autocorrelation measure for sequence patterns
- **Diversity**: Measure of sequence diversity in the generated samples

## Project Structure

- `dna_gan/`: Main package directory
  - `models.py`: Generator and Discriminator model definitions
  - `data.py`: Data loading and processing utilities
  - `train.py`: Training functions and utilities
  - `metrics.py`: Metrics for evaluating generated sequences
  - `main.py`: Command-line interface for training
- `example.py`: Example script for training a model
- `setup.py`: Package installation script

## Requirements

- Python 3.6+
- PyTorch 1.7+
- NumPy
- Matplotlib
- tqdm
- scikit-learn

## Citation

If you use this code in your research, please cite:

```
@software{dna_gan,
  author = {Your Name},
  title = {DNA Sequence Generation with Gumbel-Softmax GAN},
  year = {2023},
  url = {https://github.com/yourusername/dna-gan}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
