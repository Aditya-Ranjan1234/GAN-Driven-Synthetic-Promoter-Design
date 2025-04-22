"""
Feature-based sequence comparison metrics.

This module provides functions for:
1. GC content analysis
2. k-mer frequency analysis
3. Motif analysis
4. DNA structural property analysis
"""

import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import entropy, pearsonr, spearmanr
from scipy.spatial.distance import jensenshannon
import itertools
import re
from Bio import SeqIO
from Bio.SeqUtils import GC


def calculate_gc_content(sequences):
    """
    Calculate GC content for a list of sequences.
    
    Args:
        sequences (list): List of DNA sequences.
        
    Returns:
        list: GC content for each sequence.
    """
    gc_content = []
    
    for seq in sequences:
        gc = GC(seq) / 100.0  # Convert from percentage to fraction
        gc_content.append(gc)
    
    return gc_content


def calculate_kmer_frequencies(sequences, k=4):
    """
    Calculate k-mer frequencies for a list of sequences.
    
    Args:
        sequences (list): List of DNA sequences.
        k (int): Length of k-mers.
        
    Returns:
        dict: Dictionary of k-mer frequencies.
    """
    # Generate all possible k-mers
    nucleotides = ['A', 'C', 'G', 'T']
    all_kmers = [''.join(p) for p in itertools.product(nucleotides, repeat=k)]
    
    # Initialize frequency dictionary
    kmer_freq = {kmer: 0 for kmer in all_kmers}
    
    # Count k-mers in sequences
    total_kmers = 0
    
    for seq in sequences:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if re.match(f'^[ACGT]{{{k}}}$', kmer):  # Only count valid k-mers
                kmer_freq[kmer] += 1
                total_kmers += 1
    
    # Normalize frequencies
    if total_kmers > 0:
        for kmer in kmer_freq:
            kmer_freq[kmer] /= total_kmers
    
    return kmer_freq


def calculate_kmer_divergence(real_kmer_freq, synthetic_kmer_freq, metric='js'):
    """
    Calculate divergence between k-mer frequency distributions.
    
    Args:
        real_kmer_freq (dict): Dictionary of k-mer frequencies for real sequences.
        synthetic_kmer_freq (dict): Dictionary of k-mer frequencies for synthetic sequences.
        metric (str): Divergence metric ('js' for Jensen-Shannon, 'kl' for Kullback-Leibler).
        
    Returns:
        float: Divergence value.
    """
    # Ensure both dictionaries have the same keys
    all_kmers = sorted(set(real_kmer_freq.keys()) | set(synthetic_kmer_freq.keys()))
    
    # Convert to arrays
    real_freq = np.array([real_kmer_freq.get(kmer, 0) for kmer in all_kmers])
    synthetic_freq = np.array([synthetic_kmer_freq.get(kmer, 0) for kmer in all_kmers])
    
    # Add small epsilon to avoid division by zero in KL divergence
    epsilon = 1e-10
    real_freq = real_freq + epsilon
    synthetic_freq = synthetic_freq + epsilon
    
    # Normalize
    real_freq = real_freq / real_freq.sum()
    synthetic_freq = synthetic_freq / synthetic_freq.sum()
    
    if metric == 'js':
        return jensenshannon(real_freq, synthetic_freq)
    elif metric == 'kl':
        return entropy(real_freq, synthetic_freq)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def find_motif_occurrences(sequences, motif_pattern):
    """
    Find occurrences of a motif in sequences.
    
    Args:
        sequences (list): List of DNA sequences.
        motif_pattern (str): Regular expression pattern for the motif.
        
    Returns:
        list: Number of occurrences for each sequence.
    """
    occurrences = []
    
    for seq in sequences:
        matches = re.findall(motif_pattern, seq)
        occurrences.append(len(matches))
    
    return occurrences


def calculate_motif_enrichment(real_sequences, synthetic_sequences, motif_patterns):
    """
    Calculate motif enrichment in synthetic vs. real sequences.
    
    Args:
        real_sequences (list): List of real DNA sequences.
        synthetic_sequences (list): List of synthetic DNA sequences.
        motif_patterns (dict): Dictionary of motif names and their regex patterns.
        
    Returns:
        dict: Dictionary of motif enrichment scores.
    """
    enrichment = {}
    
    for motif_name, pattern in motif_patterns.items():
        real_occurrences = find_motif_occurrences(real_sequences, pattern)
        synthetic_occurrences = find_motif_occurrences(synthetic_sequences, pattern)
        
        real_mean = np.mean(real_occurrences)
        synthetic_mean = np.mean(synthetic_occurrences)
        
        if real_mean > 0:
            enrichment[motif_name] = synthetic_mean / real_mean
        else:
            enrichment[motif_name] = float('inf') if synthetic_mean > 0 else 1.0
    
    return enrichment


def calculate_dna_structural_properties(sequences):
    """
    Calculate DNA structural properties for sequences.
    
    Args:
        sequences (list): List of DNA sequences.
        
    Returns:
        dict: Dictionary of structural property profiles.
    """
    # Define structural property parameters
    # Values from https://doi.org/10.1093/nar/gkh779
    bendability = {
        'AA': 0.019, 'AC': 0.050, 'AG': 0.027, 'AT': 0.038,
        'CA': 0.017, 'CC': 0.026, 'CG': 0.037, 'CT': 0.027,
        'GA': 0.019, 'GC': 0.025, 'GG': 0.026, 'GT': 0.050,
        'TA': 0.025, 'TC': 0.019, 'TG': 0.017, 'TT': 0.019
    }
    
    propeller_twist = {
        'AA': -18.66, 'AC': -13.10, 'AG': -14.00, 'AT': -15.01,
        'CA': -9.45, 'CC': -8.11, 'CG': -10.03, 'CT': -14.00,
        'GA': -13.48, 'GC': -11.08, 'GG': -8.11, 'GT': -13.10,
        'TA': -11.85, 'TC': -13.48, 'TG': -9.45, 'TT': -18.66
    }
    
    helix_twist = {
        'AA': 35.62, 'AC': 34.40, 'AG': 27.70, 'AT': 31.50,
        'CA': 34.50, 'CC': 33.67, 'CG': 29.80, 'CT': 27.70,
        'GA': 36.90, 'GC': 40.00, 'GG': 33.67, 'GT': 34.40,
        'TA': 36.00, 'TC': 36.90, 'TG': 34.50, 'TT': 35.62
    }
    
    # Initialize property profiles
    profiles = {
        'bendability': [],
        'propeller_twist': [],
        'helix_twist': []
    }
    
    for seq in sequences:
        # Calculate properties for each dinucleotide
        bend_profile = []
        propeller_profile = []
        helix_profile = []
        
        for i in range(len(seq) - 1):
            dinucleotide = seq[i:i+2]
            
            if dinucleotide in bendability:
                bend_profile.append(bendability[dinucleotide])
            else:
                bend_profile.append(np.nan)
                
            if dinucleotide in propeller_twist:
                propeller_profile.append(propeller_twist[dinucleotide])
            else:
                propeller_profile.append(np.nan)
                
            if dinucleotide in helix_twist:
                helix_profile.append(helix_twist[dinucleotide])
            else:
                helix_profile.append(np.nan)
        
        # Append profiles
        profiles['bendability'].append(bend_profile)
        profiles['propeller_twist'].append(propeller_profile)
        profiles['helix_twist'].append(helix_profile)
    
    return profiles


def calculate_structural_correlation(real_profiles, synthetic_profiles, method='pearson'):
    """
    Calculate correlation between structural profiles.
    
    Args:
        real_profiles (dict): Dictionary of structural profiles for real sequences.
        synthetic_profiles (dict): Dictionary of structural profiles for synthetic sequences.
        method (str): Correlation method ('pearson' or 'spearman').
        
    Returns:
        dict: Dictionary of correlation values.
    """
    correlations = {}
    
    for property_name in real_profiles:
        # Calculate average profiles
        real_avg = np.nanmean(real_profiles[property_name], axis=0)
        synthetic_avg = np.nanmean(synthetic_profiles[property_name], axis=0)
        
        # Remove NaN values
        valid_indices = ~np.isnan(real_avg) & ~np.isnan(synthetic_avg)
        real_valid = real_avg[valid_indices]
        synthetic_valid = synthetic_avg[valid_indices]
        
        # Calculate correlation
        if method == 'pearson':
            corr, _ = pearsonr(real_valid, synthetic_valid)
        elif method == 'spearman':
            corr, _ = spearmanr(real_valid, synthetic_valid)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        correlations[property_name] = corr
    
    return correlations
