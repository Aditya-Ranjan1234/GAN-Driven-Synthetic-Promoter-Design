# How to Run the Updated WGAN-GP for DNA Sequence Generation

This guide provides step-by-step instructions on how to run the improved Wasserstein GAN with gradient penalty (WGAN-GP) for DNA sequence generation.

## Prerequisites

- Python 3.10
- PyTorch
- NumPy
- Matplotlib
- tqdm
- scikit-learn
- BioPython

## Step 1: Set Up the Environment

Create and activate a Python 3.10 virtual environment:

```bash
# Create virtual environment
python -m venv env310

# Activate on Windows
env310\Scripts\activate

# Activate on Unix/MacOS
source env310/bin/activate
```

Install the required packages:

```bash
pip install torch numpy matplotlib tqdm scikit-learn biopython
```

## Step 2: Prepare the Data

The `seq_download.pl.fasta` file is actually an HTML file with FASTA-formatted content inside a `<pre>` tag. We need to extract and preprocess the data:

```bash
cd DNA_GAN_Project
./preprocess_and_check.bat  # On Windows
./preprocess_and_check.sh   # On Unix/MacOS
```

This will:

1. Extract the FASTA content from the HTML file and save it to `data/clean_seq_download.fasta`
2. Check the content of the extracted file
3. Preprocess the sequences by replacing 'N' characters with random nucleotides
4. Save the preprocessed sequences to `data/preprocessed_dna_sequences.fasta`
5. Check the content of the preprocessed file

If the extraction fails (which can happen if the HTML format is non-standard), the script will automatically generate a sample FASTA file with random DNA sequences.

## Step 3: Run the Improved WGAN-GP

Navigate to the project directory and run the improved WGAN-GP using the provided scripts:

### On Windows:
```bash
cd DNA_GAN_Project
run_updated_gan.bat
```

### On Unix/MacOS:
```bash
cd DNA_GAN_Project
chmod +x run_updated_gan.sh
./run_updated_gan.sh
```

### Or manually:
```bash
cd DNA_GAN_Project
python -m models.improved_wgan.run_improved_dna_gan --fasta_file data/preprocessed_dna_sequences.fasta --num_epochs 500 --seq_len 150
```

Note: The provided scripts will automatically extract and preprocess the FASTA content from the HTML file before running the model.

### Command-line Arguments

You can customize the training with the following arguments:

- `--fasta_file`: Path to the FASTA file (default: 'data/preprocessed_dna_sequences.fasta')
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
- `--no_cuda`: Disable CUDA (default: False)

For example, to train with a larger hidden dimension and for more epochs:

```bash
python -m models.improved_wgan.run_improved_dna_gan --fasta_file data/clean_all_dna_sequences.fasta --hidden_dim 1024 --num_epochs 1000
```

## Step 3: Generate Sequences from the Trained Model

After training, you can generate sequences using the trained model:

```bash
python -m models.improved_wgan.generate_sequences --checkpoint checkpoints/improved_dna_gan/final_model.pt --num_sequences 1000 --output_file data/improved_generated_sequences.fasta
```

### Command-line Arguments for Generation

- `--checkpoint`: Path to the checkpoint file (default: 'checkpoints/improved_dna_gan/final_model.pt')
- `--num_sequences`: Number of sequences to generate (default: 1000)
- `--output_file`: Path to save the generated sequences (default: 'data/improved_generated_sequences.fasta')
- `--seq_len`: Length of DNA sequences (default: 150)
- `--noise_dim`: Dimension of the input noise vector (default: 100)
- `--hidden_dim`: Dimension of the LSTM hidden state (default: 512)
- `--num_layers`: Number of LSTM layers (default: 2)
- `--temperature`: Temperature parameter for Gumbel-Softmax (default: 1.0)
- `--no_cuda`: Disable CUDA (default: False)

## Step 4: Verify the Generated Sequences

You can verify that the sequences were generated correctly by checking the output file:

```bash
head -n 10 data/improved_generated_sequences.fasta
```

## Step 5: Run the Evaluation

After generating sequences from both models, you can run a comprehensive evaluation to compare them with the original data:

### On Windows:
```bash
cd DNA_GAN_Project
run_evaluation.bat
```

### On Unix/MacOS:
```bash
cd DNA_GAN_Project
chmod +x run_evaluation.sh
./run_evaluation.sh
```

This will:

1. Evaluate the sequences using various metrics:
   - Feature-based comparisons (GC content, k-mer frequencies, motif analysis, DNA structural properties)
   - Statistical and machine learning approaches (dimensionality reduction, classifier-based discrimination, distributional tests)
   - Functional and predictive analyses (promoter strength prediction, downstream model training)

2. Generate visualization plots for all metrics

3. Save the results to `evaluation/results/evaluation_results.json`

## Step 6: View the Evaluation Results

You can view the evaluation results in the Streamlit app:

```bash
cd DNA_GAN_Project
python -m streamlit run visualization/streamlit_app_comparison.py
```

Then navigate to the "Evaluation" page in the sidebar.

## Advantages of the Improved WGAN-GP

The improved WGAN-GP offers several advantages over the previous Gumbel-Softmax GAN:

1. **More Stable Training**: The Wasserstein loss with gradient penalty provides more stable training, avoiding the mode collapse observed in the previous model.

2. **Better Diversity**: The generated sequences show higher diversity, as measured by both sequence diversity and k-mer diversity metrics.

3. **More Realistic GC Content**: The GC content of generated sequences is closer to that of real DNA sequences.

4. **Improved Spatial Patterns**: The Moran's I spatial autocorrelation of generated sequences is closer to that of real DNA sequences.

5. **Attention Mechanism**: The self-attention mechanism helps capture long-range dependencies in DNA sequences.

6. **Spectral Normalization**: Spectral normalization in the discriminator provides more stable gradients during training.

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA out of memory errors, try reducing the batch size:

```bash
python -m models.improved_wgan.run_improved_dna_gan --batch_size 32
```

### CPU-Only Training

If you don't have a GPU or want to use CPU only:

```bash
python -m models.improved_wgan.run_improved_dna_gan --no_cuda
```

### Slow Training

If training is too slow, you can try:
- Reducing the sequence length: `--seq_len 100`
- Reducing the hidden dimension: `--hidden_dim 256`
- Reducing the number of LSTM layers: `--num_layers 1`
- Increasing the batch size (if memory allows): `--batch_size 128`
