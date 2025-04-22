# DNA Sequence Generation with Gumbel-Softmax GAN

This package provides tools for generating DNA sequences using a Gumbel-Softmax GAN with LSTM generators.

## Features

- Gumbel-Softmax trick for backpropagation through discrete outputs
- LSTM-based generator with 256 hidden units
- Choice of CNN or LSTM-based discriminator
- CUDA support for faster training
- Comprehensive metrics including GC content and Moran's I
- Checkpointing for continued training
- Visualization of training progress

## Requirements

- Python 3.6+
- PyTorch 1.7+
- NumPy
- Matplotlib
- tqdm
- scikit-learn

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dna-gan.git
cd dna-gan

# Install the package
pip install -e .
```

## Usage

### Training a model

```bash
python -m dna_gan.main --fasta_file path/to/your/sequences.fasta --seq_len 150 --num_epochs 500 --batch_size 64
```

### Command-line arguments

- `--fasta_file`: Path to the FASTA file containing DNA sequences (required)
- `--seq_len`: Length of DNA sequences (default: 150)
- `--noise_dim`: Dimension of the noise vector (default: 100)
- `--hidden_dim`: Dimension of the hidden state (default: 256)
- `--discriminator_type`: Type of discriminator ('cnn' or 'lstm', default: 'cnn')
- `--batch_size`: Batch size (default: 64)
- `--num_epochs`: Number of epochs (default: 500)
- `--lr_g`: Learning rate for generator (default: 1e-4)
- `--lr_d`: Learning rate for discriminator (default: 1e-4)
- `--temperature`: Temperature for Gumbel-Softmax (default: 1.0)
- `--checkpoint_dir`: Directory to save checkpoints (default: 'checkpoints')
- `--log_interval`: Interval for logging training progress (default: 1)
- `--save_interval`: Interval for saving checkpoints (default: 10)
- `--resume`: Path to checkpoint to resume training (default: None)
- `--seed`: Random seed (default: 42)
- `--no_cuda`: Disable CUDA (default: False)
- `--num_workers`: Number of worker processes for data loading (default: 4)

### Using the API

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
```

## Output

The training process will create a directory in the specified checkpoint directory with:

- Checkpoints saved every `save_interval` epochs
- A final model saved at the end of training
- A plot of the training history
- A log file with training progress
- A FASTA file with generated sequences

## Metrics

The following metrics are tracked during training:

- Generator loss
- Discriminator loss
- Discriminator accuracy
- GC content
- Moran's I spatial autocorrelation
- Sequence diversity

## References

- [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient](https://arxiv.org/abs/1609.05473)
- [Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144)
- [The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables](https://arxiv.org/abs/1611.00712)
