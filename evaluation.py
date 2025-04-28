"""
DNA Sequence Evaluation

This module provides functions for evaluating and comparing real and synthetic DNA sequences.
"""

import numpy as np
from typing import List, Dict, Tuple, Union, Optional
from collections import Counter
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

from dna_utils import one_hot_encode, get_kmers, kmer_frequency, gc_content, sequence_similarity

class DNAEvaluator:
    """
    Class for evaluating and comparing real and synthetic DNA sequences.
    """
    
    @staticmethod
    def kmer_distribution_similarity(real_sequences: List[str], synthetic_sequences: List[str], 
                                    k: int = 3) -> Dict[str, float]:
        """
        Calculate similarity between k-mer distributions of real and synthetic sequences.
        
        Args:
            real_sequences (List[str]): List of real DNA sequences.
            synthetic_sequences (List[str]): List of synthetic DNA sequences.
            k (int): Length of k-mers.
            
        Returns:
            Dict[str, float]: Dictionary with similarity metrics.
        """
        if not real_sequences or not synthetic_sequences:
            raise ValueError("Both real and synthetic sequence lists must be non-empty")
        
        # Calculate k-mer frequencies
        real_kmer_counts = kmer_frequency(real_sequences, k)
        synth_kmer_counts = kmer_frequency(synthetic_sequences, k)
        
        # Get the union of all k-mers
        all_kmers = set(real_kmer_counts.keys()).union(set(synth_kmer_counts.keys()))
        
        # Create vectors for comparison
        real_vector = np.array([real_kmer_counts.get(kmer, 0) for kmer in all_kmers])
        synth_vector = np.array([synth_kmer_counts.get(kmer, 0) for kmer in all_kmers])
        
        # Normalize to get probability distributions
        real_vector = real_vector / real_vector.sum() if real_vector.sum() > 0 else real_vector
        synth_vector = synth_vector / synth_vector.sum() if synth_vector.sum() > 0 else synth_vector
        
        # Calculate similarity metrics
        # Cosine similarity
        cosine_sim = np.dot(real_vector, synth_vector) / (np.linalg.norm(real_vector) * np.linalg.norm(synth_vector))
        
        # Jensen-Shannon divergence
        m = (real_vector + synth_vector) / 2
        js_div = 0.5 * (stats.entropy(real_vector, m) + stats.entropy(synth_vector, m))
        
        # Pearson correlation
        pearson_corr = np.corrcoef(real_vector, synth_vector)[0, 1]
        
        # Chi-square test
        chi2_stat, chi2_p = stats.chisquare(synth_vector, real_vector)
        
        return {
            'cosine_similarity': float(cosine_sim),
            'jensen_shannon_divergence': float(js_div),
            'pearson_correlation': float(pearson_corr),
            'chi2_statistic': float(chi2_stat),
            'chi2_p_value': float(chi2_p)
        }
    
    @staticmethod
    def gc_content_similarity(real_sequences: List[str], synthetic_sequences: List[str]) -> Dict[str, float]:
        """
        Calculate similarity between GC content distributions of real and synthetic sequences.
        
        Args:
            real_sequences (List[str]): List of real DNA sequences.
            synthetic_sequences (List[str]): List of synthetic DNA sequences.
            
        Returns:
            Dict[str, float]: Dictionary with similarity metrics.
        """
        if not real_sequences or not synthetic_sequences:
            raise ValueError("Both real and synthetic sequence lists must be non-empty")
        
        # Calculate GC content for each sequence
        real_gc = np.array([gc_content(seq) for seq in real_sequences])
        synth_gc = np.array([gc_content(seq) for seq in synthetic_sequences])
        
        # Calculate similarity metrics
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.ks_2samp(real_gc, synth_gc)
        
        # Mann-Whitney U test
        mw_stat, mw_p = stats.mannwhitneyu(real_gc, synth_gc)
        
        # Mean and standard deviation differences
        real_mean = np.mean(real_gc)
        synth_mean = np.mean(synth_gc)
        real_std = np.std(real_gc)
        synth_std = np.std(synth_gc)
        
        return {
            'ks_statistic': float(ks_stat),
            'ks_p_value': float(ks_p),
            'mw_statistic': float(mw_stat),
            'mw_p_value': float(mw_p),
            'real_mean_gc': float(real_mean),
            'synthetic_mean_gc': float(synth_mean),
            'real_std_gc': float(real_std),
            'synthetic_std_gc': float(synth_std),
            'mean_difference': float(abs(real_mean - synth_mean)),
            'std_difference': float(abs(real_std - synth_std))
        }
    
    @staticmethod
    def sequence_diversity(sequences: List[str]) -> Dict[str, float]:
        """
        Calculate diversity metrics for a set of sequences.
        
        Args:
            sequences (List[str]): List of DNA sequences.
            
        Returns:
            Dict[str, float]: Dictionary with diversity metrics.
        """
        if not sequences:
            raise ValueError("Sequence list must be non-empty")
        
        # Ensure all sequences have the same length
        if len(set(len(seq) for seq in sequences)) > 1:
            raise ValueError("All sequences must have the same length for diversity calculation")
        
        # Calculate pairwise similarities
        n = len(sequences)
        similarities = []
        
        for i in range(n):
            for j in range(i+1, n):
                similarities.append(sequence_similarity(sequences[i], sequences[j]))
        
        # Calculate diversity metrics
        mean_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        min_similarity = np.min(similarities)
        max_similarity = np.max(similarities)
        
        return {
            'mean_similarity': float(mean_similarity),
            'std_similarity': float(std_similarity),
            'min_similarity': float(min_similarity),
            'max_similarity': float(max_similarity),
            'diversity_index': float(1 - mean_similarity)
        }
    
    @staticmethod
    def novelty_check(real_sequences: List[str], synthetic_sequences: List[str]) -> Dict[str, float]:
        """
        Check how novel the synthetic sequences are compared to real sequences.
        
        Args:
            real_sequences (List[str]): List of real DNA sequences.
            synthetic_sequences (List[str]): List of synthetic DNA sequences.
            
        Returns:
            Dict[str, float]: Dictionary with novelty metrics.
        """
        if not real_sequences or not synthetic_sequences:
            raise ValueError("Both real and synthetic sequence lists must be non-empty")
        
        # Ensure all sequences have the same length
        real_lengths = set(len(seq) for seq in real_sequences)
        synth_lengths = set(len(seq) for seq in synthetic_sequences)
        
        if len(real_lengths) > 1 or len(synth_lengths) > 1 or real_lengths != synth_lengths:
            raise ValueError("All sequences must have the same length for novelty calculation")
        
        # Calculate maximum similarity of each synthetic sequence to any real sequence
        max_similarities = []
        
        for synth_seq in synthetic_sequences:
            similarities = [sequence_similarity(synth_seq, real_seq) for real_seq in real_sequences]
            max_similarities.append(max(similarities))
        
        # Calculate novelty metrics
        mean_max_similarity = np.mean(max_similarities)
        std_max_similarity = np.std(max_similarities)
        min_max_similarity = np.min(max_similarities)
        max_max_similarity = np.max(max_similarities)
        
        # Count exact matches
        exact_matches = sum(1 for sim in max_similarities if sim == 1.0)
        
        return {
            'mean_max_similarity': float(mean_max_similarity),
            'std_max_similarity': float(std_max_similarity),
            'min_max_similarity': float(min_max_similarity),
            'max_max_similarity': float(max_max_similarity),
            'novelty_index': float(1 - mean_max_similarity),
            'exact_match_count': int(exact_matches),
            'exact_match_percentage': float(exact_matches / len(synthetic_sequences) * 100)
        }
    
    @staticmethod
    def dimensionality_reduction(real_sequences: List[str], synthetic_sequences: List[str], 
                                method: str = 'pca') -> Tuple[np.ndarray, np.ndarray, plt.Figure]:
        """
        Perform dimensionality reduction on real and synthetic sequences for visualization.
        
        Args:
            real_sequences (List[str]): List of real DNA sequences.
            synthetic_sequences (List[str]): List of synthetic DNA sequences.
            method (str): Dimensionality reduction method ('pca' or 'tsne').
            
        Returns:
            Tuple[np.ndarray, np.ndarray, plt.Figure]: Real and synthetic embeddings, and visualization figure.
        """
        if not real_sequences or not synthetic_sequences:
            raise ValueError("Both real and synthetic sequence lists must be non-empty")
        
        # Ensure all sequences have the same length
        real_lengths = set(len(seq) for seq in real_sequences)
        synth_lengths = set(len(seq) for seq in synthetic_sequences)
        
        if len(real_lengths) > 1 or len(synth_lengths) > 1 or real_lengths != synth_lengths:
            raise ValueError("All sequences must have the same length for dimensionality reduction")
        
        # One-hot encode sequences
        real_encoded = np.array([one_hot_encode(seq) for seq in real_sequences])
        synth_encoded = np.array([one_hot_encode(seq) for seq in synthetic_sequences])
        
        # Reshape for dimensionality reduction
        real_reshaped = real_encoded.reshape(real_encoded.shape[0], -1)
        synth_reshaped = synth_encoded.reshape(synth_encoded.shape[0], -1)
        
        # Combine for dimensionality reduction
        combined = np.vstack([real_reshaped, synth_reshaped])
        
        # Perform dimensionality reduction
        if method.lower() == 'pca':
            reducer = PCA(n_components=2)
        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unsupported method: {method}. Use 'pca' or 'tsne'.")
        
        embeddings = reducer.fit_transform(combined)
        
        # Split back into real and synthetic
        real_embeddings = embeddings[:len(real_sequences)]
        synth_embeddings = embeddings[len(real_sequences):]
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(real_embeddings[:, 0], real_embeddings[:, 1], c='blue', label='Real', alpha=0.7)
        ax.scatter(synth_embeddings[:, 0], synth_embeddings[:, 1], c='red', label='Synthetic', alpha=0.7)
        
        ax.set_title(f'Sequence Visualization using {method.upper()}')
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.legend()
        
        return real_embeddings, synth_embeddings, fig
    
    @staticmethod
    def comprehensive_evaluation(real_sequences: List[str], synthetic_sequences: List[str]) -> Dict[str, Dict]:
        """
        Perform a comprehensive evaluation of synthetic sequences.
        
        Args:
            real_sequences (List[str]): List of real DNA sequences.
            synthetic_sequences (List[str]): List of synthetic DNA sequences.
            
        Returns:
            Dict[str, Dict]: Dictionary with all evaluation metrics.
        """
        if not real_sequences or not synthetic_sequences:
            raise ValueError("Both real and synthetic sequence lists must be non-empty")
        
        # Perform all evaluations
        kmer_sim_3 = DNAEvaluator.kmer_distribution_similarity(real_sequences, synthetic_sequences, k=3)
        kmer_sim_4 = DNAEvaluator.kmer_distribution_similarity(real_sequences, synthetic_sequences, k=4)
        gc_sim = DNAEvaluator.gc_content_similarity(real_sequences, synthetic_sequences)
        
        # Only calculate diversity and novelty if sequences have the same length
        real_lengths = set(len(seq) for seq in real_sequences)
        synth_lengths = set(len(seq) for seq in synthetic_sequences)
        
        diversity_real = {}
        diversity_synth = {}
        novelty = {}
        
        if len(real_lengths) == 1 and len(synth_lengths) == 1 and real_lengths == synth_lengths:
            diversity_real = DNAEvaluator.sequence_diversity(real_sequences)
            diversity_synth = DNAEvaluator.sequence_diversity(synthetic_sequences)
            novelty = DNAEvaluator.novelty_check(real_sequences, synthetic_sequences)
        
        # Combine all metrics
        return {
            'kmer_similarity_3mer': kmer_sim_3,
            'kmer_similarity_4mer': kmer_sim_4,
            'gc_content_similarity': gc_sim,
            'diversity_real': diversity_real,
            'diversity_synthetic': diversity_synth,
            'novelty': novelty
        }
    
    @staticmethod
    def generate_report(evaluation_results: Dict[str, Dict]) -> str:
        """
        Generate a human-readable report from evaluation results.
        
        Args:
            evaluation_results (Dict[str, Dict]): Results from comprehensive_evaluation.
            
        Returns:
            str: Formatted report.
        """
        report = []
        
        report.append("# DNA Sequence Evaluation Report\n")
        
        # K-mer similarity
        report.append("## K-mer Distribution Similarity\n")
        
        # 3-mer
        kmer3 = evaluation_results.get('kmer_similarity_3mer', {})
        if kmer3:
            report.append("### 3-mer Similarity\n")
            report.append(f"- Cosine Similarity: {kmer3.get('cosine_similarity', 'N/A'):.4f}")
            report.append(f"- Jensen-Shannon Divergence: {kmer3.get('jensen_shannon_divergence', 'N/A'):.4f}")
            report.append(f"- Pearson Correlation: {kmer3.get('pearson_correlation', 'N/A'):.4f}")
            report.append(f"- Chi-square p-value: {kmer3.get('chi2_p_value', 'N/A'):.4f}\n")
        
        # 4-mer
        kmer4 = evaluation_results.get('kmer_similarity_4mer', {})
        if kmer4:
            report.append("### 4-mer Similarity\n")
            report.append(f"- Cosine Similarity: {kmer4.get('cosine_similarity', 'N/A'):.4f}")
            report.append(f"- Jensen-Shannon Divergence: {kmer4.get('jensen_shannon_divergence', 'N/A'):.4f}")
            report.append(f"- Pearson Correlation: {kmer4.get('pearson_correlation', 'N/A'):.4f}")
            report.append(f"- Chi-square p-value: {kmer4.get('chi2_p_value', 'N/A'):.4f}\n")
        
        # GC content similarity
        gc = evaluation_results.get('gc_content_similarity', {})
        if gc:
            report.append("## GC Content Similarity\n")
            report.append(f"- Real Mean GC: {gc.get('real_mean_gc', 'N/A'):.4f}")
            report.append(f"- Synthetic Mean GC: {gc.get('synthetic_mean_gc', 'N/A'):.4f}")
            report.append(f"- Mean Difference: {gc.get('mean_difference', 'N/A'):.4f}")
            report.append(f"- Kolmogorov-Smirnov p-value: {gc.get('ks_p_value', 'N/A'):.4f}")
            report.append(f"- Mann-Whitney U p-value: {gc.get('mw_p_value', 'N/A'):.4f}\n")
        
        # Diversity
        div_real = evaluation_results.get('diversity_real', {})
        div_synth = evaluation_results.get('diversity_synthetic', {})
        if div_real and div_synth:
            report.append("## Sequence Diversity\n")
            report.append(f"- Real Diversity Index: {div_real.get('diversity_index', 'N/A'):.4f}")
            report.append(f"- Synthetic Diversity Index: {div_synth.get('diversity_index', 'N/A'):.4f}")
            report.append(f"- Real Mean Similarity: {div_real.get('mean_similarity', 'N/A'):.4f}")
            report.append(f"- Synthetic Mean Similarity: {div_synth.get('mean_similarity', 'N/A'):.4f}\n")
        
        # Novelty
        nov = evaluation_results.get('novelty', {})
        if nov:
            report.append("## Novelty Analysis\n")
            report.append(f"- Novelty Index: {nov.get('novelty_index', 'N/A'):.4f}")
            report.append(f"- Mean Max Similarity: {nov.get('mean_max_similarity', 'N/A'):.4f}")
            report.append(f"- Exact Match Count: {nov.get('exact_match_count', 'N/A')}")
            report.append(f"- Exact Match Percentage: {nov.get('exact_match_percentage', 'N/A'):.2f}%\n")
        
        # Overall assessment
        report.append("## Overall Assessment\n")
        
        # Assess k-mer similarity
        kmer_quality = "Unknown"
        if kmer3 and 'cosine_similarity' in kmer3:
            cosine_sim = kmer3['cosine_similarity']
            if cosine_sim > 0.9:
                kmer_quality = "Excellent"
            elif cosine_sim > 0.7:
                kmer_quality = "Good"
            elif cosine_sim > 0.5:
                kmer_quality = "Fair"
            else:
                kmer_quality = "Poor"
        
        # Assess GC content similarity
        gc_quality = "Unknown"
        if gc and 'mean_difference' in gc:
            mean_diff = gc['mean_difference']
            if mean_diff < 0.02:
                gc_quality = "Excellent"
            elif mean_diff < 0.05:
                gc_quality = "Good"
            elif mean_diff < 0.1:
                gc_quality = "Fair"
            else:
                gc_quality = "Poor"
        
        # Assess novelty
        novelty_quality = "Unknown"
        if nov and 'novelty_index' in nov:
            novelty_index = nov['novelty_index']
            if novelty_index > 0.3:
                novelty_quality = "Excellent"
            elif novelty_index > 0.2:
                novelty_quality = "Good"
            elif novelty_index > 0.1:
                novelty_quality = "Fair"
            else:
                novelty_quality = "Poor"
        
        report.append(f"- K-mer Distribution Quality: {kmer_quality}")
        report.append(f"- GC Content Quality: {gc_quality}")
        report.append(f"- Novelty Quality: {novelty_quality}")
        
        return "\n".join(report)
