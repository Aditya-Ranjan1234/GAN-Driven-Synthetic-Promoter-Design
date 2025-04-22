"""
Data processing utilities for DNA sequence generation.

This module provides functions for loading, preprocessing, and batching DNA sequences.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os


class DNASequenceDataset(Dataset):
    """
    Dataset for DNA sequences.
    
    Attributes:
        sequences (list): List of DNA sequences.
        seq_len (int): Length of each sequence.
        vocab (dict): Mapping from nucleotides to indices.
        vocab_size (int): Size of the vocabulary.
    """
    
    def __init__(self, fasta_file, seq_len=150):
        """
        Initialize the dataset from a FASTA file.
        
        Args:
            fasta_file (str): Path to the FASTA file.
            seq_len (int): Length of each sequence.
        """
        self.seq_len = seq_len
        self.vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}  # Map N to A for simplicity
        self.vocab_size = len(set(self.vocab.values()))
        
        # Load sequences from FASTA file
        self.sequences = self._load_fasta(fasta_file)
        
        print(f"Loaded {len(self.sequences)} sequences from {fasta_file}")
    
    def _load_fasta(self, fasta_file):
        """
        Load sequences from a FASTA file.
        
        Args:
            fasta_file (str): Path to the FASTA file.
            
        Returns:
            list: List of DNA sequences.
        """
        sequences = []
        current_seq = ""
        
        with open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        # Process and add the previous sequence
                        if len(current_seq) >= self.seq_len:
                            sequences.append(current_seq[:self.seq_len])
                    current_seq = ""
                else:
                    current_seq += line
            
            # Add the last sequence
            if current_seq and len(current_seq) >= self.seq_len:
                sequences.append(current_seq[:self.seq_len])
        
        return sequences
    
    def __len__(self):
        """Return the number of sequences in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Get a sequence by index.
        
        Args:
            idx (int): Index of the sequence.
            
        Returns:
            torch.Tensor: One-hot encoded sequence.
        """
        seq = self.sequences[idx]
        
        # Convert to one-hot encoding
        one_hot = np.zeros((self.seq_len, self.vocab_size), dtype=np.float32)
        for i, nucleotide in enumerate(seq):
            if i >= self.seq_len:
                break
            if nucleotide in self.vocab:
                one_hot[i, self.vocab[nucleotide]] = 1.0
            else:
                one_hot[i, 0] = 1.0  # Default to 'A' for unknown nucleotides
        
        return torch.tensor(one_hot, dtype=torch.float32)


def get_data_loader(fasta_file, batch_size=64, seq_len=150, shuffle=True, num_workers=4):
    """
    Create a DataLoader for DNA sequences.
    
    Args:
        fasta_file (str): Path to the FASTA file.
        batch_size (int): Batch size.
        seq_len (int): Length of each sequence.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker processes.
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for the dataset.
    """
    dataset = DNASequenceDataset(fasta_file, seq_len)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def nucleotide_to_idx(nucleotide):
    """
    Convert a nucleotide to its index.
    
    Args:
        nucleotide (str): Nucleotide (A, C, G, T, or N).
        
    Returns:
        int: Index of the nucleotide.
    """
    vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 0}
    return vocab.get(nucleotide, 0)


def idx_to_nucleotide(idx):
    """
    Convert an index to its nucleotide.
    
    Args:
        idx (int): Index of the nucleotide.
        
    Returns:
        str: Nucleotide (A, C, G, or T).
    """
    idx_to_vocab = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    return idx_to_vocab.get(idx, 'N')


def one_hot_to_sequence(one_hot):
    """
    Convert a one-hot encoded tensor to a DNA sequence.
    
    Args:
        one_hot (torch.Tensor): One-hot encoded tensor of shape (seq_len, vocab_size).
        
    Returns:
        str: DNA sequence.
    """
    indices = torch.argmax(one_hot, dim=1).cpu().numpy()
    return ''.join([idx_to_nucleotide(idx) for idx in indices])


def sequences_to_fasta(sequences, file_path, prefix="generated"):
    """
    Save sequences to a FASTA file.
    
    Args:
        sequences (list): List of DNA sequences.
        file_path (str): Path to save the FASTA file.
        prefix (str): Prefix for sequence names.
    """
    with open(file_path, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f">{prefix}_{i}\n")
            f.write(f"{seq}\n")
    
    print(f"Saved {len(sequences)} sequences to {file_path}")
