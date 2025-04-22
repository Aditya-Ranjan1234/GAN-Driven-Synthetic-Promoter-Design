"""
Visualization functions for evaluation results.

This module provides functions for plotting:
1. GC content distributions
2. k-mer divergence
3. Motif enrichment
4. Structural correlations
5. Dimensionality reduction
6. Classifier performance
7. Promoter strength distributions
8. Augmentation benefits
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, auc


def plot_gc_content(results, output_dir):
    """
    Plot GC content distributions.
    
    Args:
        results (dict): Evaluation results.
        output_dir (str): Directory to save the plots.
    """
    gc_results = results['feature_based']['gc_content']
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot real GC content
    real_mean = gc_results['real']['mean']
    real_std = gc_results['real']['std']
    x = np.linspace(real_mean - 3*real_std, real_mean + 3*real_std, 100)
    plt.plot(x, np.exp(-0.5*((x - real_mean)/real_std)**2)/(real_std*np.sqrt(2*np.pi)), 
             label='Real', linewidth=2)
    
    # Plot synthetic GC content
    for model, stats in gc_results['synthetic'].items():
        mean = stats['mean']
        std = stats['std']
        x = np.linspace(mean - 3*std, mean + 3*std, 100)
        plt.plot(x, np.exp(-0.5*((x - mean)/std)**2)/(std*np.sqrt(2*np.pi)), 
                 label=model, linewidth=2)
    
    plt.xlabel('GC Content')
    plt.ylabel('Density')
    plt.title('GC Content Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'gc_content.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_kmer_divergence(results, output_dir):
    """
    Plot k-mer divergence.
    
    Args:
        results (dict): Evaluation results.
        output_dir (str): Directory to save the plots.
    """
    # Extract k-mer divergence results
    kmer_keys = [key for key in results['feature_based'].keys() if 'mer_divergence' in key]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Prepare data
    k_values = [int(key[0]) for key in kmer_keys]
    models = list(results['feature_based'][kmer_keys[0]].keys())
    
    # Plot Jensen-Shannon divergence
    js_data = {model: [results['feature_based'][f'{k}mer_divergence'][model]['js'] for k in k_values] 
               for model in models}
    
    for model, values in js_data.items():
        axes[0].plot(k_values, values, marker='o', label=model, linewidth=2)
    
    axes[0].set_xlabel('k-mer Length')
    axes[0].set_ylabel('Jensen-Shannon Divergence')
    axes[0].set_title('Jensen-Shannon Divergence by k-mer Length')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot Kullback-Leibler divergence
    kl_data = {model: [results['feature_based'][f'{k}mer_divergence'][model]['kl'] for k in k_values] 
               for model in models}
    
    for model, values in kl_data.items():
        axes[1].plot(k_values, values, marker='o', label=model, linewidth=2)
    
    axes[1].set_xlabel('k-mer Length')
    axes[1].set_ylabel('Kullback-Leibler Divergence')
    axes[1].set_title('Kullback-Leibler Divergence by k-mer Length')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'kmer_divergence.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_motif_enrichment(results, output_dir):
    """
    Plot motif enrichment.
    
    Args:
        results (dict): Evaluation results.
        output_dir (str): Directory to save the plots.
    """
    motif_results = results['feature_based']['motif_enrichment']
    
    # Get motifs and models
    motifs = list(next(iter(motif_results.values())).keys())
    models = list(motif_results.keys())
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Prepare data
    data = {model: [motif_results[model][motif] for motif in motifs] for model in models}
    
    # Set up bar positions
    bar_width = 0.8 / len(models)
    positions = np.arange(len(motifs))
    
    # Plot bars
    for i, (model, values) in enumerate(data.items()):
        plt.bar(positions + i*bar_width - 0.4 + bar_width/2, values, 
                width=bar_width, label=model)
    
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Equal Enrichment')
    
    plt.xlabel('Motif')
    plt.ylabel('Enrichment (Synthetic/Real)')
    plt.title('Motif Enrichment in Synthetic vs. Real Sequences')
    plt.xticks(positions, motifs, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'motif_enrichment.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_structural_correlation(results, output_dir):
    """
    Plot structural correlations.
    
    Args:
        results (dict): Evaluation results.
        output_dir (str): Directory to save the plots.
    """
    structural_results = results['feature_based']['structural_correlation']
    
    # Get properties and models
    properties = list(next(iter(structural_results.values())).keys())
    models = list(structural_results.keys())
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Prepare data
    data = {model: [structural_results[model][prop] for prop in properties] for model in models}
    
    # Set up bar positions
    bar_width = 0.8 / len(models)
    positions = np.arange(len(properties))
    
    # Plot bars
    for i, (model, values) in enumerate(data.items()):
        plt.bar(positions + i*bar_width - 0.4 + bar_width/2, values, 
                width=bar_width, label=model)
    
    plt.xlabel('Structural Property')
    plt.ylabel('Pearson Correlation')
    plt.title('Structural Property Correlation between Synthetic and Real Sequences')
    plt.xticks(positions, properties, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'structural_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_dimensionality_reduction(output_dir, method='pca'):
    """
    Plot dimensionality reduction results.
    
    Args:
        output_dir (str): Directory containing the results.
        method (str): Dimensionality reduction method ('pca', 'tsne', or 'umap').
    """
    # Load reduced representations
    real_reduced = np.load(os.path.join(output_dir, f'real_{method}_reduced.npy'))
    
    # Find synthetic reduced representations
    synthetic_files = [f for f in os.listdir(output_dir) if f.endswith(f'_{method}_reduced.npy') and not f.startswith('real_')]
    synthetic_reduced = {}
    
    for file in synthetic_files:
        model = file.split('_')[0]
        synthetic_reduced[model] = np.load(os.path.join(output_dir, file))
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot real data
    plt.scatter(real_reduced[:, 0], real_reduced[:, 1], label='Real', alpha=0.7, s=30)
    
    # Plot synthetic data
    for model, reduced in synthetic_reduced.items():
        plt.scatter(reduced[:, 0], reduced[:, 1], label=model, alpha=0.7, s=30)
    
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.title(f'{method.upper()} Projection of DNA Sequences')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.savefig(os.path.join(output_dir, f'{method}_projection.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_classifier_performance(results, output_dir):
    """
    Plot classifier performance.
    
    Args:
        results (dict): Evaluation results.
        output_dir (str): Directory to save the plots.
    """
    classifier_results = results['statistical_ml']['classifier']
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Prepare data
    models = list(classifier_results.keys())
    accuracy = [classifier_results[model]['accuracy'] for model in models]
    auc = [classifier_results[model]['auc'] for model in models]
    
    # Plot accuracy
    axes[0].bar(models, accuracy)
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Classifier Accuracy (Real vs. Synthetic)')
    axes[0].set_ylim(0, 1)
    axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random Guess')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot AUC
    axes[1].bar(models, auc)
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('AUC')
    axes[1].set_title('Classifier AUC (Real vs. Synthetic)')
    axes[1].set_ylim(0, 1)
    axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random Guess')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'classifier_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_promoter_strength(results, output_dir):
    """
    Plot promoter strength distributions.
    
    Args:
        results (dict): Evaluation results.
        output_dir (str): Directory to save the plots.
    """
    strength_results = results['functional']['promoter_strength']
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot real promoter strength
    real_mean = strength_results['real']['mean']
    real_std = strength_results['real']['std']
    x = np.linspace(0, 1, 100)
    plt.plot(x, np.exp(-0.5*((x - real_mean)/real_std)**2)/(real_std*np.sqrt(2*np.pi)), 
             label='Real', linewidth=2)
    
    # Plot synthetic promoter strength
    for model, stats in strength_results['synthetic'].items():
        mean = stats['mean']
        std = stats['std']
        x = np.linspace(0, 1, 100)
        plt.plot(x, np.exp(-0.5*((x - mean)/std)**2)/(std*np.sqrt(2*np.pi)), 
                 label=model, linewidth=2)
    
    plt.xlabel('Promoter Strength')
    plt.ylabel('Density')
    plt.title('Promoter Strength Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'promoter_strength.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_augmentation_benefits(results, output_dir):
    """
    Plot augmentation benefits.
    
    Args:
        results (dict): Evaluation results.
        output_dir (str): Directory to save the plots.
    """
    augmentation_results = results['functional']['augmentation']
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Prepare data
    models = list(augmentation_results.keys())
    mse_improvement = [augmentation_results[model]['mse_improvement'] * 100 for model in models]
    r2_improvement = [augmentation_results[model]['r2_improvement'] * 100 for model in models]
    
    # Plot MSE improvement
    axes[0].bar(models, mse_improvement)
    axes[0].set_xlabel('Model')
    axes[0].set_ylabel('MSE Improvement (%)')
    axes[0].set_title('MSE Improvement from Data Augmentation')
    axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.7)
    axes[0].grid(True, alpha=0.3)
    
    # Plot R² improvement
    axes[1].bar(models, r2_improvement)
    axes[1].set_xlabel('Model')
    axes[1].set_ylabel('R² Improvement (%)')
    axes[1].set_title('R² Improvement from Data Augmentation')
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'augmentation_benefits.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_plot(results, output_dir):
    """
    Create a summary plot of all evaluation metrics.
    
    Args:
        results (dict): Evaluation results.
        output_dir (str): Directory to save the plots.
    """
    # Get models
    models = list(results['feature_based']['motif_enrichment'].keys())
    
    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # 1. GC Content Difference
    gc_results = results['feature_based']['gc_content']
    real_gc = gc_results['real']['mean']
    gc_diff = [abs(gc_results['synthetic'][model]['mean'] - real_gc) for model in models]
    
    axes[0, 0].bar(models, gc_diff)
    axes[0, 0].set_xlabel('Model')
    axes[0, 0].set_ylabel('Absolute Difference')
    axes[0, 0].set_title('GC Content Difference from Real')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. k-mer Divergence (k=4)
    kmer_div = [results['feature_based']['4mer_divergence'][model]['js'] for model in models]
    
    axes[0, 1].bar(models, kmer_div)
    axes[0, 1].set_xlabel('Model')
    axes[0, 1].set_ylabel('Jensen-Shannon Divergence')
    axes[0, 1].set_title('4-mer Divergence')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Motif Enrichment (average)
    motif_results = results['feature_based']['motif_enrichment']
    motifs = list(next(iter(motif_results.values())).keys())
    motif_avg = [np.mean([abs(1 - motif_results[model][motif]) for motif in motifs]) for model in models]
    
    axes[1, 0].bar(models, motif_avg)
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel('Average Absolute Deviation')
    axes[1, 0].set_title('Motif Enrichment Deviation from Ideal')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Structural Correlation (average)
    structural_results = results['feature_based']['structural_correlation']
    properties = list(next(iter(structural_results.values())).keys())
    structural_avg = [np.mean([structural_results[model][prop] for prop in properties]) for model in models]
    
    axes[1, 1].bar(models, structural_avg)
    axes[1, 1].set_xlabel('Model')
    axes[1, 1].set_ylabel('Average Correlation')
    axes[1, 1].set_title('Structural Property Correlation')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 5. Classifier Performance
    classifier_results = results['statistical_ml']['classifier']
    accuracy = [classifier_results[model]['accuracy'] for model in models]
    
    axes[2, 0].bar(models, accuracy)
    axes[2, 0].set_xlabel('Model')
    axes[2, 0].set_ylabel('Accuracy')
    axes[2, 0].set_title('Classifier Accuracy (Real vs. Synthetic)')
    axes[2, 0].set_ylim(0, 1)
    axes[2, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random Guess')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 6. Augmentation Benefits
    augmentation_results = results['functional']['augmentation']
    r2_improvement = [augmentation_results[model]['r2_improvement'] * 100 for model in models]
    
    axes[2, 1].bar(models, r2_improvement)
    axes[2, 1].set_xlabel('Model')
    axes[2, 1].set_ylabel('R² Improvement (%)')
    axes[2, 1].set_title('R² Improvement from Data Augmentation')
    axes[2, 1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'summary.png'), dpi=300, bbox_inches='tight')
    plt.close()


def visualize_results(results_file, output_dir):
    """
    Visualize evaluation results.
    
    Args:
        results_file (str): Path to the evaluation results JSON file.
        output_dir (str): Directory to save the plots.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create plots
    plot_gc_content(results, output_dir)
    plot_kmer_divergence(results, output_dir)
    plot_motif_enrichment(results, output_dir)
    plot_structural_correlation(results, output_dir)
    
    for method in ['pca', 'tsne', 'umap']:
        try:
            plot_dimensionality_reduction(os.path.dirname(results_file), method)
        except FileNotFoundError:
            print(f"Dimensionality reduction results for {method} not found.")
    
    plot_classifier_performance(results, output_dir)
    plot_promoter_strength(results, output_dir)
    plot_augmentation_benefits(results, output_dir)
    create_summary_plot(results, output_dir)
    
    print(f"Visualization completed. Plots saved to {output_dir}")


def main():
    """
    Main function to run the visualization.
    """
    # Define file paths
    results_file = "evaluation/results/evaluation_results.json"
    output_dir = "evaluation/visualization/plots"
    
    # Run visualization
    visualize_results(results_file, output_dir)
    
    print("Visualization completed successfully.")


if __name__ == "__main__":
    main()
