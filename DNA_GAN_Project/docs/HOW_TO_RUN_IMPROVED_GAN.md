# How to Run the Improved DNA Sequence Generation Model

This guide provides step-by-step instructions on how to run the improved DNA sequence generation model using Wasserstein GAN with gradient penalty.

## Prerequisites

1. Make sure you have Python 3.10 installed. You can check your Python version with:
   ```bash
   python --version
   ```

2. Create a virtual environment with Python 3.10:
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

4. Install the required dependencies:
   ```bash
   pip install torch numpy matplotlib tqdm scikit-learn biopython
   ```

5. Ensure your DNA sequence data is in FASTA format and located at:
   ```
   data/clean_all_dna_sequences.fasta
   ```

## Running the Improved Model

### Step 1: Prepare Your Data

Make sure your DNA sequence data is in FASTA format. The model expects sequences of similar length. If your sequences have varying lengths, the model will handle this by truncating or padding them to the specified length (default: 150).

### Step 2: Train the Model

To start training the model from scratch:

```bash
python -m ImprovedDNAGAN.run_improved_dna_gan --fasta_file data/clean_all_dna_sequences.fasta --num_epochs 500
```

This will:
- Train the model for 500 epochs
- Save checkpoints every 10 epochs in the `checkpoints/improved_dna_gan` directory
- Save visualization plots in the `checkpoints/improved_dna_gan/images` directory
- Generate sample sequences and save them to `checkpoints/improved_dna_gan/generated_sequences.fasta`

### Step 3: Monitor Training Progress

During training, you can monitor the progress in the terminal. The script will display:
- Generator loss
- Discriminator loss
- Wasserstein distance
- Gradient penalty
- GC content
- Moran's I spatial autocorrelation
- Sequence diversity

These metrics are also logged to `checkpoints/improved_dna_gan/training.log`.

### Step 4: Visualize Training Progress

To visualize the training progress at any time:

```bash
python -m ImprovedDNAGAN.visualize_training
```

This will:
1. Load the latest checkpoint
2. Plot the training metrics
3. Generate and display sample DNA sequences
4. Save the plots to the `checkpoints/improved_dna_gan/images` directory

### Step 5: Resume Training (If Needed)

If your training was interrupted, you can resume from the latest checkpoint:

```bash
python -m ImprovedDNAGAN.resume_training --num_epochs 200
```

This will:
1. Find the latest checkpoint in the `checkpoints/improved_dna_gan` directory
2. Resume training from that checkpoint
3. Continue for the specified number of epochs (in this example, 200 more epochs)

## Advanced Usage

### Customizing Training Parameters

You can customize various training parameters using command-line arguments:

```bash
python -m ImprovedDNAGAN.run_improved_dna_gan \
    --fasta_file data/clean_all_dna_sequences.fasta \
    --seq_len 200 \
    --hidden_dim 1024 \
    --num_layers 3 \
    --num_epochs 1000 \
    --batch_size 128 \
    --lr_g 2e-4 \
    --lr_d 2e-4 \
    --temperature 0.8 \
    --n_critic 3 \
    --lambda_gp 15.0 \
    --checkpoint_dir checkpoints/custom_model \
    --save_interval 20
```

### Using a Specific Checkpoint

To resume training from a specific checkpoint:

```bash
python -m ImprovedDNAGAN.resume_training \
    --checkpoint checkpoints/improved_dna_gan/checkpoint_epoch_100.pt \
    --num_epochs 300
```

### Visualizing a Specific Checkpoint

To visualize the training progress from a specific checkpoint:

```bash
python -m ImprovedDNAGAN.visualize_training \
    --checkpoint checkpoints/improved_dna_gan/checkpoint_epoch_100.pt
```

## Comparing with the Previous Model

The improved model offers several advantages over the previous Gumbel-Softmax GAN:

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
python -m ImprovedDNAGAN.run_improved_dna_gan --batch_size 32
```

### CPU-Only Training

If you don't have a GPU or want to use CPU only:

```bash
python -m ImprovedDNAGAN.run_improved_dna_gan --no_cuda
```

### Slow Training

If training is too slow, you can try:
- Reducing the sequence length: `--seq_len 100`
- Reducing the hidden dimension: `--hidden_dim 256`
- Reducing the number of LSTM layers: `--num_layers 1`
- Increasing the batch size (if memory allows): `--batch_size 128`

### Unstable Training

If training is unstable, you can try:
- Increasing the gradient penalty weight: `--lambda_gp 15.0`
- Increasing the number of discriminator updates: `--n_critic 7`
- Reducing the learning rates: `--lr_g 5e-5 --lr_d 5e-5`
- Adjusting the temperature: `--temperature 0.5`

## Next Steps

After training the model, you can:

1. **Generate More Sequences**: Use the trained model to generate more DNA sequences.
2. **Analyze Generated Sequences**: Analyze the properties of the generated sequences using bioinformatics tools.
3. **Fine-Tune the Model**: Fine-tune the model for specific types of DNA sequences.
4. **Extend the Model**: Extend the model to generate other types of biological sequences, such as RNA or protein sequences.

## Conclusion

The improved DNA sequence generation model using Wasserstein GAN with gradient penalty offers significant improvements over the previous model. By following this guide, you should be able to train the model and generate high-quality synthetic DNA sequences.
