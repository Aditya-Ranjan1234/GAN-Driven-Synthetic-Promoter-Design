"""
Data loading and preprocessing utilities for DNA sequence generation.

This module provides functions for loading and preprocessing DNA sequences from FASTA files.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import re
from Bio import SeqIO


class DNASequenceDataset(Dataset):
    """
    Dataset for DNA sequences.
    
    Attributes:
        sequences (torch.Tensor): One-hot encoded DNA sequences.
        seq_len (int): Length of each sequence.
        vocab_size (int): Size of the vocabulary (4 for DNA: A, C, G, T).
    """
    
    def __init__(self, fasta_file, seq_len=150, min_seq_len=None, max_seq_len=None):
        """
        Initialize the dataset.
        
        Args:
            fasta_file (str): Path to the FASTA file.
            seq_len (int): Length of each sequence.
            min_seq_len (int): Minimum sequence length to include.
            max_seq_len (int): Maximum sequence length to include.
        """
        self.seq_len = seq_len
        self.vocab_size = 4  # A, C, G, T
        
        # Load sequences from FASTA file
        raw_sequences = []
        
        for record in SeqIO.parse(fasta_file, "fasta"):
            seq = str(record.seq).upper()
            
            # Filter by sequence length if specified
            if min_seq_len is not None and len(seq) < min_seq_len:
                continue
            if max_seq_len is not None and len(seq) > max_seq_len:
                continue
            
            # Remove non-standard nucleotides
            seq = re.sub(r'[^ACGT]', '', seq)
            
            if len(seq) >= seq_len:
                raw_sequences.append(seq)
        
        print(f"Loaded {len(raw_sequences)} sequences from {fasta_file}")
        
        # Convert sequences to one-hot encoding
        self.sequences = self._one_hot_encode(raw_sequences)
    
    def _one_hot_encode(self, raw_sequences):
        """
        Convert raw sequences to one-hot encoding.
        
        Args:
            raw_sequences (list): List of raw DNA sequences.
            
        Returns:
            torch.Tensor: One-hot encoded sequences.
        """
        # Create mapping from nucleotides to indices
        nucleotide_to_idx = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
        
        # Initialize one-hot encoded sequences
        one_hot = torch.zeros(len(raw_sequences), self.seq_len, self.vocab_size)
        
        for i, seq in enumerate(raw_sequences):
            # Truncate or pad sequence to seq_len
            if len(seq) > self.seq_len:
                # Randomly select a subsequence
                start = np.random.randint(0, len(seq) - self.seq_len + 1)
                seq = seq[start:start + self.seq_len]
            else:
                # Pad with random nucleotides
                pad_length = self.seq_len - len(seq)
                if pad_length > 0:
                    pad_nucleotides = np.random.choice(['A', 'C', 'G', 'T'], size=pad_length)
                    seq = seq + ''.join(pad_nucleotides)
            
            # Convert to one-hot encoding
            for j, nucleotide in enumerate(seq):
                if nucleotide in nucleotide_to_idx:
                    one_hot[i, j, nucleotide_to_idx[nucleotide]] = 1.0
        
        return one_hot
    
    def __len__(self):
        """
        Get the number of sequences in the dataset.
        
        Returns:
            int: Number of sequences.
        """
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Get a sequence from the dataset.
        
        Args:
            idx (int): Index of the sequence.
            
        Returns:
            torch.Tensor: One-hot encoded sequence.
        """
        return self.sequences[idx]


def get_data_loader(fasta_file, seq_len=150, batch_size=64, shuffle=True, num_workers=4):
    """
    Get a DataLoader for DNA sequences.
    
    Args:
        fasta_file (str): Path to the FASTA file.
        seq_len (int): Length of each sequence.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker processes.
        
    Returns:
        DataLoader: DataLoader for DNA sequences.
    """
    dataset = DNASequenceDataset(fasta_file, seq_len)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


def save_sequences_to_fasta(sequences, output_file):
    """
    Save generated sequences to a FASTA file.
    
    Args:
        sequences (torch.Tensor): One-hot encoded sequences.
        output_file (str): Path to the output FASTA file.
    """
    # Create mapping from indices to nucleotides
    idx_to_nucleotide = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    
    # Convert one-hot to indices
    indices = torch.argmax(sequences, dim=2).cpu().numpy()
    
    # Write sequences to FASTA file
    with open(output_file, 'w') as f:
        for i, seq_indices in enumerate(indices):
            # Convert indices to nucleotides
            seq = ''.join([idx_to_nucleotide[idx] for idx in seq_indices])
            
            # Write to FASTA file
            f.write(f">generated_sequence_{i+1}\n")
            f.write(f"{seq}\n")
    
    print(f"Saved {len(sequences)} sequences to {output_file}")
