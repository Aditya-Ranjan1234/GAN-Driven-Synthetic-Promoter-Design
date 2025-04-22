# DNA GAN Evaluation

This directory contains scripts and results for evaluating the DNA sequence generation models.

## Contents

- Evaluation scripts for comparing the quality of generated sequences
- Results of evaluation metrics
- Comparison between different models

## Evaluation Metrics

The following metrics are used to evaluate the quality of generated DNA sequences:

1. **GC Content**: The proportion of G and C nucleotides in the sequences
2. **Sequence Diversity**: The diversity of generated sequences
3. **Moran's I Spatial Autocorrelation**: A measure of spatial patterns in the sequences
4. **k-mer Diversity**: The diversity of k-mers in the sequences

## Running Evaluations

Evaluation metrics are calculated automatically during training and can be visualized using the Streamlit app:

```bash
python ../run_streamlit_app.py
```

For detailed evaluation results, please refer to the Model Comparison page in the Streamlit app.
