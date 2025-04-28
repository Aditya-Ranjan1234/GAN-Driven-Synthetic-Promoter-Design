# DNA GAN Project

A comprehensive implementation of Generative Adversarial Networks (GANs) for DNA sequence generation, featuring two different architectures: Gumbel-Softmax GAN and Improved WGAN-GP.

![DNA Sequence Generation](https://cdn-icons-png.flaticon.com/512/2942/2942256.png)

## Introduction

The DNA GAN Project explores the application of deep learning techniques, specifically Generative Adversarial Networks (GANs), to generate synthetic DNA sequences. This has significant applications in biotechnology, including protein design, gene therapy, and drug discovery.

Traditional methods for DNA sequence generation often rely on hand-crafted rules or statistical models that may not capture the complex patterns and dependencies present in biological sequences. GANs offer a promising approach to learn these patterns directly from data.

This project implements and compares two different GAN architectures:

1. **Gumbel-Softmax GAN**: Uses LSTM generators with the Gumbel-Softmax trick to handle discrete outputs
2. **Improved WGAN-GP**: Uses Wasserstein GAN with gradient penalty, LSTM generators with attention, and CNN discriminators with spectral normalization

## GAN Architecture

### Gumbel-Softmax GAN

The Gumbel-Softmax GAN architecture consists of:

#### Generator
- An input layer that takes a 100-dimensional noise vector
- Linear layers to transform the noise into initial hidden and cell states
- An LSTM layer with 256 hidden units
- A dropout layer with a rate of 0.2 for regularization
- A fully connected output layer with softmax activation
- Gumbel-Softmax sampling to generate discrete outputs while maintaining differentiability

#### Discriminator
We implemented two types of discriminators:

**CNN Discriminator:**
- Three convolutional layers with 64, 128, and 256 filters
- Max pooling layers after each convolution
- A fully connected layer with 256 units
- A dropout layer with a rate of 0.3
- A sigmoid output layer

**LSTM Discriminator:**
- A bidirectional LSTM with 256 hidden units
- A fully connected layer with 256 units
- A dropout layer with a rate of 0.3
- A sigmoid output layer

### Improved WGAN-GP

The Improved WGAN-GP architecture consists of:

#### Generator
- An input layer that takes a 128-dimensional noise vector
- Linear layers to transform the noise into initial hidden and cell states
- An LSTM layer with 512 hidden units
- An attention mechanism to capture long-range dependencies
- A dropout layer with a rate of 0.2
- A fully connected output layer with softmax activation

#### Discriminator
- A 1D CNN with spectral normalization
- Four convolutional layers with 64, 128, 256, and 512 filters
- Layer normalization after each convolution
- A fully connected layer with 512 units
- A linear output layer (no sigmoid, as required by WGAN)

## Training Process

### Gumbel-Softmax GAN Training

The Gumbel-Softmax GAN was trained using the following procedure:
- Adam optimizer with learning rates of 1e-4 for both generator and discriminator
- Binary cross-entropy loss function
- Batch size of 64
- Training for 1,000 epochs
- Checkpointing every 10 epochs
- Temperature parameter of 1.0 for Gumbel-Softmax

### Improved WGAN-GP Training

The Improved WGAN-GP was trained using the following procedure:
- Adam optimizer with learning rates of 1e-4 for both generator and discriminator
- Wasserstein loss with gradient penalty
- Batch size of 64
- Training for 500 epochs
- 5 discriminator updates per generator update
- Gradient penalty weight of 10.0
- Checkpointing every 10 epochs

## Results

The models were evaluated using several metrics:

1. **GC Content**: Measures the proportion of G and C nucleotides in the sequences
2. **Moran's I Spatial Autocorrelation**: Measures the spatial autocorrelation of nucleotides
3. **Sequence Diversity**: Measures the diversity of generated sequences
4. **k-mer Distribution**: Compares the distribution of k-mers between original and generated sequences

The Improved WGAN-GP outperformed the Gumbel-Softmax GAN in all metrics, producing sequences that more closely resemble the original DNA sequences.

## Web Interface

The project includes a Streamlit web interface for visualizing and comparing DNA sequences. The interface provides:

1. **Home**: Overview of the project and available models
2. **Original Data**: Visualization of the original DNA sequences
3. **Generated Data**: Visualization of the generated DNA sequences
4. **Model Comparison**: Comparison between the original and generated sequences

## Directory Structure

```
DNA_GAN_Project/
├── models/                    # Model implementations
│   ├── gumbel_softmax_gan/    # Original GAN implementation
│   └── improved_wgan/         # Improved WGAN-GP implementation
├── data/                      # Data files
├── utils/                     # Utility functions
├── evaluation/                # Evaluation scripts and results
├── visualization/             # Visualization tools
├── docs/                      # Documentation
├── web_app/                   # Streamlit web interface
│   └── templates/             # Streamlit app templates
└── README.md                  # Project overview
```

## Installation and Setup

### Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DNA_GAN_Project.git
cd DNA_GAN_Project
```

2. Create and activate a virtual environment:
```bash
# On Windows
python -m venv env310
env310\Scripts\activate

# On Unix/MacOS
python -m venv env310
source env310/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Web Interface

To run the Streamlit web interface:

```bash
cd web_app
streamlit run templates/simplified_app.py
```

The app will be available at http://localhost:8501 in your web browser.

## Training the Models

### Training the Gumbel-Softmax GAN

```bash
cd models/gumbel_softmax_gan
python run_dna_gan.py --data_path ../../data/clean_all_dna_sequences.fasta --epochs 1000 --batch_size 64 --checkpoint_dir ../../checkpoints/gumbel_softmax_gan
```

### Training the Improved WGAN-GP

```bash
cd models/improved_wgan
python run_improved_dna_gan.py --data_path ../../data/clean_all_dna_sequences.fasta --epochs 500 --batch_size 64 --checkpoint_dir ../../checkpoints/improved_wgan
```

## Generating Sequences

### Generating Sequences with Gumbel-Softmax GAN

```bash
cd models/gumbel_softmax_gan
python generate_sequences.py --checkpoint_path ../../checkpoints/gumbel_softmax_gan/checkpoint_latest.pt --num_sequences 100 --output_file ../../data/gumbel_generated_sequences.fasta
```

### Generating Sequences with Improved WGAN-GP

```bash
cd models/improved_wgan
python generate_sequences.py --checkpoint_path ../../checkpoints/improved_wgan/checkpoint_latest.pt --num_sequences 100 --output_file ../../data/improved_generated_sequences.fasta
```

## Conclusion

This project demonstrates the application of GANs to DNA sequence generation. The Improved WGAN-GP architecture shows promising results in generating biologically plausible DNA sequences. However, there are still challenges to overcome, such as ensuring biological constraints and improving sequence diversity.

Future work could focus on:
1. Incorporating biological constraints specific to DNA sequences
2. Extending the model to conditional generation based on specific biological properties
3. Exploring alternative GAN formulations for improved stability and diversity
4. Adding attention mechanisms to better capture long-range dependencies
5. Implementing curriculum learning approaches for more effective training

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

## Acknowledgments

- The Gumbel-Softmax trick was introduced by Jang et al. (2016) and Maddison et al. (2016)
- The Wasserstein GAN with gradient penalty was introduced by Gulrajani et al. (2017)
- The DNA sequence data is from the Eukaryotic Promoter Database (EPD)
