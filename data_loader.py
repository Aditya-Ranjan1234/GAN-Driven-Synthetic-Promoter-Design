"""
DNA Sequence Data Loader

This module provides functions for loading and preprocessing DNA sequence data.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Optional
from Bio import SeqIO
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from dna_utils import one_hot_encode, pad_sequences

class DNADataLoader:
    """
    Class for loading and preprocessing DNA sequence data.
    """

    def __init__(self, sequence_length: Optional[int] = None, batch_size: int = 32):
        """
        Initialize the data loader.

        Args:
            sequence_length (Optional[int]): Fixed length for sequences. If None, uses the length of the longest sequence.
            batch_size (int): Batch size for training.
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.sequences = []
        self.encoded_sequences = None
        self.dataset = None

    def load_fasta(self, file_path: str) -> List[str]:
        """
        Load sequences from a FASTA file.

        Args:
            file_path (str): Path to the FASTA file.

        Returns:
            List[str]: List of DNA sequences.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        sequences = []
        for record in SeqIO.parse(file_path, "fasta"):
            sequences.append(str(record.seq))

        self.sequences = sequences
        return sequences

    def load_csv(self, file_path: str, sequence_column: str = 'sequence') -> List[str]:
        """
        Load sequences from a CSV file.

        Args:
            file_path (str): Path to the CSV file.
            sequence_column (str): Name of the column containing sequences.

        Returns:
            List[str]: List of DNA sequences.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        df = pd.read_csv(file_path)

        if sequence_column not in df.columns:
            raise ValueError(f"Column '{sequence_column}' not found in the CSV file")

        sequences = df[sequence_column].tolist()
        self.sequences = sequences
        return sequences

    def preprocess(self, sequences: Optional[List[str]] = None) -> np.ndarray:
        """
        Preprocess sequences: pad to uniform length and one-hot encode.

        Args:
            sequences (Optional[List[str]]): List of DNA sequences. If None, uses self.sequences.

        Returns:
            np.ndarray: One-hot encoded sequences with shape (num_sequences, sequence_length, 4).
        """
        if sequences is None:
            sequences = self.sequences

        if not sequences:
            raise ValueError("No sequences to preprocess")

        # Pad sequences to uniform length
        if self.sequence_length is None:
            self.sequence_length = max(len(seq) for seq in sequences)

        padded_sequences = pad_sequences(sequences, self.sequence_length)

        # One-hot encode sequences
        encoded_sequences = np.array([one_hot_encode(seq) for seq in padded_sequences])

        self.encoded_sequences = encoded_sequences
        return encoded_sequences

    def create_dataset(self, encoded_sequences: Optional[np.ndarray] = None) -> TensorDataset:
        """
        Create a PyTorch dataset from encoded sequences.

        Args:
            encoded_sequences (Optional[np.ndarray]): One-hot encoded sequences.
                                                     If None, uses self.encoded_sequences.

        Returns:
            TensorDataset: PyTorch dataset for training.
        """
        if encoded_sequences is None:
            encoded_sequences = self.encoded_sequences

        if encoded_sequences is None:
            raise ValueError("No encoded sequences available. Call preprocess() first.")

        # Convert numpy array to PyTorch tensor
        tensor_data = torch.tensor(encoded_sequences, dtype=torch.float32)
        dataset = TensorDataset(tensor_data)

        self.dataset = dataset
        return dataset

    def load_and_prepare(self, file_path: str, file_type: str = 'auto') -> TensorDataset:
        """
        Load, preprocess, and create a dataset in one step.

        Args:
            file_path (str): Path to the sequence file.
            file_type (str): Type of file ('fasta', 'csv', or 'auto' to detect from extension).

        Returns:
            TensorDataset: PyTorch dataset for training.
        """
        # Determine file type if auto
        if file_type == 'auto':
            _, ext = os.path.splitext(file_path)
            if ext.lower() in ['.fa', '.fasta']:
                file_type = 'fasta'
            elif ext.lower() == '.csv':
                file_type = 'csv'
            else:
                raise ValueError(f"Could not determine file type from extension: {ext}")

        # Load sequences
        if file_type == 'fasta':
            self.load_fasta(file_path)
        elif file_type == 'csv':
            self.load_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # Preprocess sequences
        self.preprocess()

        # Create dataset
        return self.create_dataset()

    def generate_dummy_data(self, num_sequences: int = 100, min_length: int = 50, max_length: int = 100) -> List[str]:
        """
        Generate dummy DNA sequences for testing.

        Args:
            num_sequences (int): Number of sequences to generate.
            min_length (int): Minimum sequence length.
            max_length (int): Maximum sequence length.

        Returns:
            List[str]: List of generated DNA sequences.
        """
        nucleotides = ['A', 'C', 'G', 'T']
        sequences = []

        for _ in range(num_sequences):
            length = np.random.randint(min_length, max_length + 1)
            sequence = ''.join(np.random.choice(nucleotides) for _ in range(length))
            sequences.append(sequence)

        self.sequences = sequences
        return sequences
