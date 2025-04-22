"""
Metrics for evaluating DNA sequence generation.

This module provides functions for calculating various metrics to evaluate
the quality of generated DNA sequences.
"""

import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform


def calculate_gc_content(sequences):
    """
    Calculate the GC content of sequences.
    
    Args:
        sequences (torch.Tensor): One-hot encoded sequences of shape (batch_size, seq_len, vocab_size).
        
    Returns:
        float: Average GC content.
    """
    # Get the indices of G and C (1 and 2)
    g_indices = sequences[:, :, 2].sum(dim=1)
    c_indices = sequences[:, :, 1].sum(dim=1)
    
    # Calculate GC content
    gc_content = (g_indices + c_indices) / sequences.size(1)
    
    return gc_content.mean().item()


def calculate_morans_i(sequences, k=5):
    """
    Calculate Moran's I spatial autocorrelation for sequences.
    
    Args:
        sequences (torch.Tensor): One-hot encoded sequences of shape (batch_size, seq_len, vocab_size).
        k (int): Number of nearest neighbors to consider.
        
    Returns:
        float: Average Moran's I value.
    """
    batch_size, seq_len, vocab_size = sequences.size()
    
    # Convert one-hot to indices
    indices = torch.argmax(sequences, dim=2).cpu().numpy()
    
    # Calculate Moran's I for each sequence
    morans_i_values = []
    
    for i in range(batch_size):
        seq_indices = indices[i]
        
        # Create a distance matrix
        positions = np.arange(seq_len).reshape(-1, 1)
        distances = squareform(pdist(positions, 'euclidean'))
        
        # Create a weight matrix (1 for k nearest neighbors, 0 otherwise)
        weights = np.zeros_like(distances)
        for j in range(seq_len):
            # Get indices of k nearest neighbors
            nearest_indices = np.argsort(distances[j])[1:k+1]
            weights[j, nearest_indices] = 1
        
        # Normalize weights
        row_sums = weights.sum(axis=1)
        weights = weights / row_sums[:, np.newaxis]
        
        # Calculate Moran's I
        y = seq_indices
        y_mean = np.mean(y)
        y_centered = y - y_mean
        
        numerator = np.sum(weights * np.outer(y_centered, y_centered))
        denominator = np.sum(y_centered ** 2)
        
        if denominator == 0:
            morans_i = 0
        else:
            morans_i = (seq_len / np.sum(weights)) * (numerator / denominator)
        
        morans_i_values.append(morans_i)
    
    return np.mean(morans_i_values)


def calculate_diversity(sequences):
    """
    Calculate the diversity of sequences.
    
    Args:
        sequences (torch.Tensor): One-hot encoded sequences of shape (batch_size, seq_len, vocab_size).
        
    Returns:
        float: Diversity score.
    """
    batch_size, seq_len, vocab_size = sequences.size()
    
    # Convert one-hot to indices
    indices = torch.argmax(sequences, dim=2).cpu().numpy()
    
    # Calculate pairwise Hamming distances
    distances = []
    for i in range(batch_size):
        for j in range(i+1, batch_size):
            distance = np.sum(indices[i] != indices[j]) / seq_len
            distances.append(distance)
    
    # Return average distance
    return np.mean(distances) if distances else 0.0


def evaluate_model(generator, data_loader, noise_dim=100, device='cuda', num_samples=1000):
    """
    Evaluate the generator model.
    
    Args:
        generator (nn.Module): Generator model.
        data_loader (DataLoader): DataLoader for real DNA sequences.
        noise_dim (int): Dimension of the input noise vector.
        device (str): Device to use for evaluation.
        num_samples (int): Number of samples to generate.
        
    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    generator.eval()
    
    # Generate sequences
    with torch.no_grad():
        noise = torch.randn(num_samples, noise_dim).to(device)
        generated_sequences = generator(noise, temperature=1.0, hard=True)
    
    # Get real sequences
    real_sequences = []
    for batch in data_loader:
        real_sequences.append(batch)
        if len(real_sequences) * batch.size(0) >= num_samples:
            break
    
    real_sequences = torch.cat(real_sequences, dim=0)[:num_samples].to(device)
    
    # Calculate metrics
    metrics = {
        'generated_gc_content': calculate_gc_content(generated_sequences),
        'real_gc_content': calculate_gc_content(real_sequences),
        'generated_morans_i': calculate_morans_i(generated_sequences),
        'real_morans_i': calculate_morans_i(real_sequences),
        'generated_diversity': calculate_diversity(generated_sequences),
        'real_diversity': calculate_diversity(real_sequences)
    }
    
    return metrics
