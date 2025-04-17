"""
DNA Sequence Utility Functions

This module provides utility functions for manipulating and analyzing DNA sequences.
"""

import numpy as np
from collections import Counter
import re
from typing import List, Dict, Tuple, Union, Optional

# Define the DNA alphabet
DNA_ALPHABET = ['A', 'C', 'G', 'T']

def one_hot_encode(sequence: str) -> np.ndarray:
    """
    Convert a DNA sequence to one-hot encoding.
    
    Args:
        sequence (str): A DNA sequence string containing A, C, G, T.
        
    Returns:
        np.ndarray: One-hot encoded sequence with shape (len(sequence), 4).
                   Each position is encoded as a one-hot vector [A, C, G, T].
    """
    # Create mapping from nucleotide to position in one-hot vector
    mapping = {nuc: i for i, nuc in enumerate(DNA_ALPHABET)}
    
    # Initialize the encoding matrix
    encoded = np.zeros((len(sequence), len(DNA_ALPHABET)), dtype=np.float32)
    
    # Fill in the encoding
    for i, nucleotide in enumerate(sequence):
        if nucleotide in mapping:
            encoded[i, mapping[nucleotide]] = 1.0
    
    return encoded

def one_hot_decode(encoded_sequence: np.ndarray) -> str:
    """
    Convert a one-hot encoded sequence back to a DNA string.
    
    Args:
        encoded_sequence (np.ndarray): One-hot encoded sequence with shape (len(sequence), 4).
        
    Returns:
        str: The DNA sequence string.
    """
    # Get the index of the maximum value in each position
    indices = np.argmax(encoded_sequence, axis=1)
    
    # Map indices back to nucleotides
    sequence = ''.join([DNA_ALPHABET[i] for i in indices])
    
    return sequence

def pad_sequences(sequences: List[str], max_length: Optional[int] = None) -> List[str]:
    """
    Pad sequences to a uniform length.
    
    Args:
        sequences (List[str]): List of DNA sequence strings.
        max_length (Optional[int]): Maximum length to pad to. If None, uses the length of the longest sequence.
        
    Returns:
        List[str]: List of padded sequences.
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    padded_sequences = [seq + 'N' * (max_length - len(seq)) for seq in sequences]
    
    return padded_sequences

def get_kmers(sequence: str, k: int = 3) -> List[str]:
    """
    Extract all k-mers from a sequence.
    
    Args:
        sequence (str): A DNA sequence string.
        k (int): The length of k-mers to extract.
        
    Returns:
        List[str]: List of k-mers.
    """
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def kmer_frequency(sequences: List[str], k: int = 3) -> Dict[str, int]:
    """
    Calculate the frequency of k-mers across a set of sequences.
    
    Args:
        sequences (List[str]): List of DNA sequence strings.
        k (int): The length of k-mers to count.
        
    Returns:
        Dict[str, int]: Dictionary mapping k-mers to their frequencies.
    """
    kmer_counts = Counter()
    
    for seq in sequences:
        kmer_counts.update(get_kmers(seq, k))
    
    return dict(kmer_counts)

def gc_content(sequence: str) -> float:
    """
    Calculate the GC content of a DNA sequence.
    
    Args:
        sequence (str): A DNA sequence string.
        
    Returns:
        float: The GC content as a fraction between 0 and 1.
    """
    gc_count = sequence.count('G') + sequence.count('C')
    total = len(sequence)
    
    return gc_count / total if total > 0 else 0

def find_motifs(sequence: str, motif_pattern: str) -> List[int]:
    """
    Find all occurrences of a motif in a sequence.
    
    Args:
        sequence (str): A DNA sequence string.
        motif_pattern (str): A regular expression pattern to search for.
        
    Returns:
        List[int]: List of starting positions of motif matches.
    """
    matches = []
    for match in re.finditer(motif_pattern, sequence):
        matches.append(match.start())
    
    return matches

def reverse_complement(sequence: str) -> str:
    """
    Generate the reverse complement of a DNA sequence.
    
    Args:
        sequence (str): A DNA sequence string.
        
    Returns:
        str: The reverse complement sequence.
    """
    complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return ''.join(complement_map.get(base, base) for base in reversed(sequence))

def sequence_similarity(seq1: str, seq2: str) -> float:
    """
    Calculate a simple similarity score between two sequences.
    
    Args:
        seq1 (str): First DNA sequence.
        seq2 (str): Second DNA sequence.
        
    Returns:
        float: Similarity score between 0 and 1.
    """
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of the same length")
    
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matches / len(seq1)
