# Improved DNA Sequence Generation using Wasserstein GAN

This project implements an improved DNA sequence generation model using Wasserstein GAN with gradient penalty (WGAN-GP). The model uses LSTM layers with attention mechanisms to generate DNA sequences and CNN layers with spectral normalization for the discriminator.

## Features

- **Wasserstein GAN with Gradient Penalty**: Provides more stable training and avoids mode collapse
- **LSTM Generator with Attention**: Captures long-range dependencies in DNA sequences
- **CNN Discriminator with Spectral Normalization**: Provides stable gradients during training
- **Gumbel-Softmax Trick**: Enables backpropagation through discrete sampling
- **Comprehensive Biological Metrics**: Evaluates generated sequences using GC content, Moran's I, and diversity metrics

## Requirements

- Python 3.10
- PyTorch
- NumPy
- Matplotlib
- tqdm
- BioPython
- scikit-learn

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/improved-dna-gan.git
   cd improved-dna-gan
   ```

2. Create a virtual environment:
   ```bash
   python -m venv env310
   ```

3. Activate the virtual environment:
   ```bash
   # On Windows
   env310\Scripts\activate
   
   # On Unix or MacOS
   source env310/bin/activate
   ```

4. Install the required packages:
   ```bash
   pip install torch numpy matplotlib tqdm scikit-learn biopython
   ```

## Usage

### Training the Model

To train the model from scratch:

```bash
python -m ImprovedDNAGAN.run_improved_dna_gan --fasta_file data/clean_all_dna_sequences.fasta --num_epochs 500
```

This will:
- Train the model for 500 epochs
- Save checkpoints every 10 epochs in the `checkpoints/improved_dna_gan` directory
- Save visualization plots in the `checkpoints/improved_dna_gan/images` directory

### Command-line Arguments

- `--fasta_file`: Path to the FASTA file (default: 'data/clean_all_dna_sequences.fasta')
- `--seq_len`: Length of DNA sequences (default: 150)
- `--noise_dim`: Dimension of the input noise vector (default: 100)
- `--hidden_dim`: Dimension of the LSTM hidden state (default: 512)
- `--num_layers`: Number of LSTM layers (default: 2)
- `--num_epochs`: Number of training epochs (default: 500)
- `--batch_size`: Batch size (default: 64)
- `--lr_g`: Learning rate for the generator (default: 1e-4)
- `--lr_d`: Learning rate for the discriminator (default: 1e-4)
- `--temperature`: Temperature parameter for Gumbel-Softmax (default: 1.0)
- `--n_critic`: Number of discriminator updates per generator update (default: 5)
- `--lambda_gp`: Weight for gradient penalty (default: 10.0)
- `--checkpoint_dir`: Directory to save checkpoints (default: 'checkpoints/improved_dna_gan')
- `--log_interval`: Interval for logging training progress (default: 1)
- `--save_interval`: Interval for saving checkpoints (default: 10)
- `--resume`: Path to checkpoint to resume training (default: None)
- `--seed`: Random seed (default: 42)
- `--no_cuda`: Disable CUDA (default: False)
- `--num_workers`: Number of worker processes for data loading (default: 4)

### Resuming Training

If your training was interrupted, you can resume from the latest checkpoint:

```bash
python -m ImprovedDNAGAN.resume_training
```

This script will:
1. Find the latest checkpoint in the `checkpoints/improved_dna_gan` directory
2. Resume training from that checkpoint
3. Continue for the specified number of epochs (default: 500)

### Visualizing Training Progress

To visualize the training progress:

```bash
python -m ImprovedDNAGAN.visualize_training
```

This will:
1. Load the latest checkpoint
2. Plot the training metrics (losses, Wasserstein distance, gradient penalty, GC content, etc.)
3. Generate and display sample DNA sequences
4. Save the plots to the `checkpoints/improved_dna_gan/images` directory

## Model Architecture

### Generator

The generator consists of:
- An input layer that takes a 100-dimensional noise vector
- Linear layers to transform the noise into initial hidden and cell states
- LSTM layers with 512 hidden units
- Self-attention mechanism to capture long-range dependencies
- Layer normalization for stable training
- Dropout for regularization
- Gumbel-Softmax sampling to generate discrete outputs while maintaining differentiability

### Discriminator

The discriminator consists of:
- Three convolutional layers with spectral normalization
- Layer normalization after each convolution
- Max pooling layers
- Leaky ReLU activation functions
- Dropout for regularization

## Improvements over the Previous Model

1. **Wasserstein Loss**: Uses Wasserstein distance with gradient penalty instead of binary cross-entropy loss, providing more stable training and avoiding mode collapse.

2. **Attention Mechanism**: Incorporates self-attention to capture long-range dependencies in DNA sequences.

3. **Spectral Normalization**: Applies spectral normalization to discriminator layers for more stable gradients.

4. **Layer Normalization**: Uses layer normalization for more stable training.

5. **Multiple LSTM Layers**: Uses multiple LSTM layers for more expressive sequence generation.

6. **Improved Biological Metrics**: Includes additional metrics such as k-mer diversity to better evaluate the quality of generated sequences.

## Results

The improved model shows significant improvements over the previous model:

1. **Stable Training**: The Wasserstein loss with gradient penalty provides more stable training, avoiding the mode collapse observed in the previous model.

2. **Better Diversity**: The generated sequences show higher diversity, as measured by both sequence diversity and k-mer diversity metrics.

3. **More Realistic GC Content**: The GC content of generated sequences is closer to that of real DNA sequences.

4. **Improved Spatial Patterns**: The Moran's I spatial autocorrelation of generated sequences is closer to that of real DNA sequences.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
