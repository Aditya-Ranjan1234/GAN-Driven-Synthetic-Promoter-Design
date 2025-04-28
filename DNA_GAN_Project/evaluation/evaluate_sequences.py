"""
Main script for evaluating real and synthetic DNA sequences.

This script combines all the evaluation metrics and saves the results.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Bio import SeqIO
import pickle
import time

from evaluation.metrics.sequence_features import (
    calculate_gc_content,
    calculate_kmer_frequencies,
    calculate_kmer_divergence,
    calculate_motif_enrichment,
    calculate_dna_structural_properties,
    calculate_structural_correlation
)
from evaluation.metrics.statistical_ml import (
    encode_sequences,
    perform_dimensionality_reduction,
    train_classifier,
    perform_ks_test,
    calculate_mmd
)
from evaluation.metrics.functional_analysis import (
    SimplePromoterStrengthPredictor,
    simulate_promoter_strengths,
    evaluate_augmentation
)


def load_sequences(file_path, max_sequences=None):
    """
    Load sequences from a FASTA file.
    
    Args:
        file_path (str): Path to the FASTA file.
        max_sequences (int): Maximum number of sequences to load.
        
    Returns:
        list: List of sequences.
    """
    sequences = []
    
    for record in SeqIO.parse(file_path, "fasta"):
        seq = str(record.seq).upper()
        sequences.append(seq)
        
        if max_sequences is not None and len(sequences) >= max_sequences:
            break
    
    return sequences


def evaluate_sequences(real_file, synthetic_files, output_dir, max_sequences=1000):
    """
    Evaluate real and synthetic sequences.
    
    Args:
        real_file (str): Path to the real sequences FASTA file.
        synthetic_files (dict): Dictionary of synthetic file paths (model_name -> file_path).
        output_dir (str): Directory to save the results.
        max_sequences (int): Maximum number of sequences to evaluate.
        
    Returns:
        dict: Dictionary of evaluation results.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load sequences
    real_sequences = load_sequences(real_file, max_sequences)
    synthetic_sequences_dict = {}
    
    for model_name, file_path in synthetic_files.items():
        synthetic_sequences_dict[model_name] = load_sequences(file_path, max_sequences)
    
    # Initialize results dictionary
    results = {
        'metadata': {
            'real_file': real_file,
            'synthetic_files': synthetic_files,
            'num_real_sequences': len(real_sequences),
            'num_synthetic_sequences': {model: len(seqs) for model, seqs in synthetic_sequences_dict.items()},
            'evaluation_time': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'feature_based': {},
        'statistical_ml': {},
        'functional': {}
    }
    
    # 1. Feature-based sequence comparisons
    print("Performing feature-based sequence comparisons...")
    
    # 1.1 GC content
    real_gc = calculate_gc_content(real_sequences)
    synthetic_gc = {model: calculate_gc_content(seqs) for model, seqs in synthetic_sequences_dict.items()}
    
    results['feature_based']['gc_content'] = {
        'real': {
            'mean': np.mean(real_gc),
            'std': np.std(real_gc),
            'min': np.min(real_gc),
            'max': np.max(real_gc)
        },
        'synthetic': {
            model: {
                'mean': np.mean(gc),
                'std': np.std(gc),
                'min': np.min(gc),
                'max': np.max(gc)
            } for model, gc in synthetic_gc.items()
        }
    }
    
    # 1.2 k-mer frequencies
    for k in [2, 3, 4, 5]:
        real_kmer_freq = calculate_kmer_frequencies(real_sequences, k=k)
        synthetic_kmer_freq = {model: calculate_kmer_frequencies(seqs, k=k) for model, seqs in synthetic_sequences_dict.items()}
        
        # Calculate divergence
        divergence = {
            model: {
                'js': calculate_kmer_divergence(real_kmer_freq, freq, metric='js'),
                'kl': calculate_kmer_divergence(real_kmer_freq, freq, metric='kl')
            } for model, freq in synthetic_kmer_freq.items()
        }
        
        results['feature_based'][f'{k}mer_divergence'] = divergence
    
    # 1.3 Motif analysis
    motif_patterns = {
        'TATA_box': r'TATA[AT]A',
        'GC_box': r'GGGCGG',
        'CAAT_box': r'CCAAT',
        'Inr': r'[CT][CT]A[ACGT][AT][CT][CT]',
        'DPE': r'[AG]G[AT][CT][GAC]',
        'BRE': r'G[CT][GA]CGCC'
    }
    
    motif_enrichment = {
        model: calculate_motif_enrichment(real_sequences, seqs, motif_patterns)
        for model, seqs in synthetic_sequences_dict.items()
    }
    
    results['feature_based']['motif_enrichment'] = motif_enrichment
    
    # 1.4 DNA structural properties
    real_structural = calculate_dna_structural_properties(real_sequences)
    synthetic_structural = {model: calculate_dna_structural_properties(seqs) for model, seqs in synthetic_sequences_dict.items()}
    
    structural_correlation = {
        model: calculate_structural_correlation(real_structural, struct, method='pearson')
        for model, struct in synthetic_structural.items()
    }
    
    results['feature_based']['structural_correlation'] = structural_correlation
    
    # 2. Statistical and machine learning approaches
    print("Performing statistical and machine learning analyses...")
    
    # 2.1 Dimensionality reduction
    real_encodings = encode_sequences(real_sequences, k=4)
    synthetic_encodings = {model: encode_sequences(seqs, k=4) for model, seqs in synthetic_sequences_dict.items()}
    
    for method in ['pca', 'tsne', 'umap']:
        real_reduced = perform_dimensionality_reduction(real_encodings, method=method)
        synthetic_reduced = {
            model: perform_dimensionality_reduction(encodings, method=method)
            for model, encodings in synthetic_encodings.items()
        }
        
        # Save reduced representations
        np.save(os.path.join(output_dir, f'real_{method}_reduced.npy'), real_reduced)
        for model, reduced in synthetic_reduced.items():
            np.save(os.path.join(output_dir, f'{model}_{method}_reduced.npy'), reduced)
    
    # 2.2 Classifier-based discrimination
    classifier_results = {
        model: train_classifier(real_encodings, encodings)
        for model, encodings in synthetic_encodings.items()
    }
    
    results['statistical_ml']['classifier'] = {
        model: {
            'accuracy': float(res['accuracy']),
            'auc': float(res['auc']),
            'cv_scores': [float(score) for score in res['cv_scores']]
        } for model, res in classifier_results.items()
    }
    
    # 2.3 Distributional tests
    # KS test on GC content
    ks_results = {
        model: perform_ks_test({'gc_content': real_gc}, {'gc_content': gc})
        for model, gc in synthetic_gc.items()
    }
    
    results['statistical_ml']['ks_test'] = {
        model: {
            'gc_content': {
                'statistic': float(res['gc_content']['statistic']),
                'p_value': float(res['gc_content']['p_value'])
            }
        } for model, res in ks_results.items()
    }
    
    # MMD on k-mer encodings
    mmd_results = {
        model: calculate_mmd(real_encodings, encodings)
        for model, encodings in synthetic_encodings.items()
    }
    
    results['statistical_ml']['mmd'] = {
        model: float(mmd) for model, mmd in mmd_results.items()
    }
    
    # 3. Functional and predictive analyses
    print("Performing functional and predictive analyses...")
    
    # 3.1 Promoter strength prediction
    real_strengths = simulate_promoter_strengths(real_sequences)
    synthetic_strengths = {
        model: simulate_promoter_strengths(seqs)
        for model, seqs in synthetic_sequences_dict.items()
    }
    
    results['functional']['promoter_strength'] = {
        'real': {
            'mean': float(np.mean(real_strengths)),
            'std': float(np.std(real_strengths)),
            'min': float(np.min(real_strengths)),
            'max': float(np.max(real_strengths))
        },
        'synthetic': {
            model: {
                'mean': float(np.mean(strengths)),
                'std': float(np.std(strengths)),
                'min': float(np.min(strengths)),
                'max': float(np.max(strengths))
            } for model, strengths in synthetic_strengths.items()
        }
    }
    
    # 3.2 Downstream model training (augmentation experiment)
    augmentation_results = {
        model: evaluate_augmentation(real_sequences, seqs, real_strengths)
        for model, seqs in synthetic_sequences_dict.items()
    }
    
    results['functional']['augmentation'] = {
        model: {
            'real_only_mse': float(res['real_only_mse']),
            'real_only_r2': float(res['real_only_r2']),
            'augmented_mse': float(res['augmented_mse']),
            'augmented_r2': float(res['augmented_r2']),
            'mse_improvement': float(res['mse_improvement']),
            'r2_improvement': float(res['r2_improvement'])
        } for model, res in augmentation_results.items()
    }
    
    # Save results
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation completed. Results saved to {os.path.join(output_dir, 'evaluation_results.json')}")
    
    return results


def main():
    """
    Main function to run the evaluation.
    """
    # Define file paths
    real_file = "data/preprocessed_dna_sequences.fasta"
    synthetic_files = {
        "gumbel_softmax": "data/gumbel_generated_sequences.fasta",
        "improved_wgan": "data/improved_generated_sequences.fasta"
    }
    output_dir = "evaluation/results"
    
    # Run evaluation
    results = evaluate_sequences(real_file, synthetic_files, output_dir)
    
    print("Evaluation completed successfully.")


if __name__ == "__main__":
    main()
