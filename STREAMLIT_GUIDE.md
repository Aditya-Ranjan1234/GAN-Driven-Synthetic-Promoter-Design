# DNA Sequence GAN Streamlit Application Guide

This guide provides detailed instructions for using the DNA Sequence GAN Streamlit application.

## Getting Started

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit application:
   ```bash
   python run_streamlit_app.py
   ```

3. The application will open in your default web browser. If it doesn't open automatically, navigate to the URL shown in the terminal (typically `http://localhost:8501`).

## Application Workflow

The application follows a sequential workflow for generating and analyzing synthetic DNA sequences:

1. **Data Management**: Load or generate DNA sequence data
2. **Model Training**: Train the GAN model on the prepared data
3. **Sequence Generation**: Generate synthetic DNA sequences
4. **Evaluation**: Evaluate the quality of synthetic sequences
5. **Visualization**: Visualize and compare real and synthetic sequences

## Navigation

The application uses a sidebar for navigation between different sections. Each section provides a user-friendly interface for the corresponding functionality.

## Detailed Usage Instructions

### 1. Data Management

In this section, you can:

- **Upload Data**: Upload FASTA or CSV files containing DNA sequences
  - FASTA files should contain DNA sequences with headers
  - CSV files should have a column named 'sequence' containing the DNA sequences
  - You can specify a fixed sequence length or let the application determine it automatically

- **Generate Dummy Data**: Generate random DNA sequences for testing
  - Specify the number of sequences, minimum length, and maximum length
  - Useful for testing the application without real data

After loading or generating data, the application displays statistics and visualizations about the data, including:
- Number of sequences
- Sequence length distribution
- Nucleotide composition

### 2. Model Training

In this section, you can:

- **Initialize Model**: Set up the GAN model with customizable parameters
  - Latent dimension: Size of the random noise vector input to the generator
  - Device: Choose between CPU and GPU (if available)

- **Train Model**: Train the GAN model on the loaded data
  - Number of epochs: How many times to iterate through the entire dataset
  - Save interval: How often to save model checkpoints
  - The application displays real-time training progress and loss curves

- **Load Checkpoint**: Load a previously saved model checkpoint
  - Select from available checkpoints by epoch number
  - Useful for resuming training or using a previously trained model

### 3. Sequence Generation

In this section, you can:

- **Generate Sequences**: Create synthetic DNA sequences using the trained model
  - Specify the number of sequences to generate
  - The application displays the generated sequences and provides a download option

- **View Statistics**: Analyze the generated sequences
  - Number of sequences
  - Sequence length distribution
  - Nucleotide composition

### 4. Evaluation

In this section, you can:

- **Run Evaluation**: Compare real and synthetic sequences using various metrics
  - K-mer distribution similarity
  - GC content similarity
  - Sequence diversity
  - Novelty analysis

- **View Report**: Examine a comprehensive evaluation report
  - Detailed metrics and statistics
  - Overall assessment of synthetic sequence quality
  - Download the report as a Markdown file

### 5. Visualization

In this section, you can:

- **Sequence Visualization**: View DNA sequences in various formats
  - Sequence logos showing nucleotide frequencies at each position
  - Interactive sequence viewers for detailed examination

- **Distribution Comparison**: Compare distributions between real and synthetic sequences
  - K-mer distribution comparison with adjustable k-mer length
  - GC content distribution comparison

- **Training History**: Visualize the training process
  - Generator and discriminator loss curves
  - Track model convergence and stability

## Tips for Best Results

1. **Data Quality**: Use high-quality DNA sequences for training. The model can only learn patterns present in the training data.

2. **Sequence Length**: For best results, use sequences of similar length or set a fixed sequence length.

3. **Training Time**: GAN training can be time-consuming. Start with a small number of epochs and increase as needed.

4. **Hyperparameter Tuning**: Experiment with different latent dimensions and batch sizes to find the optimal configuration.

5. **Evaluation**: Always evaluate synthetic sequences against real sequences to ensure quality.

6. **Visualization**: Use visualizations to gain insights into the model's performance and the characteristics of synthetic sequences.

## Troubleshooting

- **Memory Issues**: If you encounter memory issues, try reducing the batch size or sequence length.

- **Training Instability**: GAN training can be unstable. If you observe erratic loss curves, try adjusting the learning rate or using a different model architecture.

- **Slow Performance**: If the application is slow, consider using a smaller dataset for initial testing.

- **File Upload Issues**: Ensure your files are in the correct format (FASTA or CSV) and contain valid DNA sequences.

## Next Steps

After generating and evaluating synthetic DNA sequences, you can:

1. **Use in Research**: Apply synthetic sequences in research projects
2. **Further Analysis**: Perform additional analyses using external tools
3. **Model Improvement**: Experiment with different GAN architectures or training strategies
4. **Integration**: Integrate the synthetic sequences into larger bioinformatics pipelines

The DNA Sequence GAN system provides a powerful tool for generating and analyzing synthetic DNA sequences, with the Streamlit application making it accessible through a user-friendly interface.
