"""
Streamlit page for displaying evaluation results.

This module provides a Streamlit interface for:
1. Running the evaluation
2. Displaying the results
3. Interpreting the findings
"""

import streamlit as st
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import sys
import subprocess
from io import StringIO

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import evaluation functions
from evaluation.evaluate_sequences import evaluate_sequences


def run_evaluation_page():
    """
    Run the evaluation page.
    """
    st.title("DNA Sequence Evaluation")
    
    st.markdown("""
    This page provides a comprehensive evaluation of synthetic DNA sequences compared to real ones.
    The evaluation includes:
    
    1. **Feature-Based Sequence Comparisons**
       - GC content analysis
       - k-mer frequency analysis
       - Motif analysis
       - DNA structural property analysis
    
    2. **Statistical & Machine Learning Approaches**
       - Dimensionality reduction and clustering
       - Classifier-based discrimination
       - Distributional tests
    
    3. **Functional & Predictive Analyses**
       - Promoter strength prediction
       - Downstream model training
    """)
    
    # Check if evaluation results exist
    results_file = "evaluation/results/evaluation_results.json"
    plots_dir = "evaluation/visualization/plots"
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Display evaluation metadata
        st.subheader("Evaluation Metadata")
        st.write(f"**Real Data File:** {results['metadata']['real_file']}")
        st.write(f"**Number of Real Sequences:** {results['metadata']['num_real_sequences']}")
        st.write("**Synthetic Data Files:**")
        for model, count in results['metadata']['num_synthetic_sequences'].items():
            st.write(f"- {model}: {count} sequences")
        st.write(f"**Evaluation Time:** {results['metadata']['evaluation_time']}")
        
        # Display evaluation results
        st.subheader("Evaluation Results")
        
        tabs = st.tabs(["Summary", "Feature-Based", "Statistical & ML", "Functional"])
        
        with tabs[0]:
            st.markdown("### Summary of Evaluation Results")
            
            # Display summary plot if it exists
            summary_plot_path = os.path.join(plots_dir, "summary.png")
            if os.path.exists(summary_plot_path):
                summary_plot = Image.open(summary_plot_path)
                st.image(summary_plot, caption="Summary of Evaluation Metrics", use_column_width=True)
            else:
                st.warning("Summary plot not found. Please run the evaluation first.")
            
            st.markdown("""
            ### Key Findings
            
            1. **Fidelity Assessment**:
               - How closely do the synthetic sequences match the real ones in terms of sequence features?
               - Are there systematic differences that could affect downstream applications?
            
            2. **Bias Detection**:
               - Are there specific motifs or structural properties that are under-represented in the synthetic data?
               - Do the synthetic sequences cover the same range of GC content as the real ones?
            
            3. **Diversity Measurement**:
               - Do the synthetic sequences explore novel yet plausible regions of sequence space?
               - Is there sufficient diversity in the synthetic data to be useful for data augmentation?
            """)
            
            # Create a table of model rankings
            st.markdown("### Model Rankings")
            
            # Calculate rankings based on different metrics
            models = list(results['feature_based']['motif_enrichment'].keys())
            
            # GC content difference (lower is better)
            gc_results = results['feature_based']['gc_content']
            real_gc = gc_results['real']['mean']
            gc_diff = [abs(gc_results['synthetic'][model]['mean'] - real_gc) for model in models]
            gc_ranks = pd.Series(gc_diff, index=models).rank()
            
            # k-mer divergence (lower is better)
            kmer_div = [results['feature_based']['4mer_divergence'][model]['js'] for model in models]
            kmer_ranks = pd.Series(kmer_div, index=models).rank()
            
            # Motif enrichment (lower deviation from 1 is better)
            motif_results = results['feature_based']['motif_enrichment']
            motifs = list(next(iter(motif_results.values())).keys())
            motif_avg = [np.mean([abs(1 - motif_results[model][motif]) for motif in motifs]) for model in models]
            motif_ranks = pd.Series(motif_avg, index=models).rank()
            
            # Structural correlation (higher is better)
            structural_results = results['feature_based']['structural_correlation']
            properties = list(next(iter(structural_results.values())).keys())
            structural_avg = [np.mean([structural_results[model][prop] for prop in properties]) for model in models]
            structural_ranks = pd.Series(structural_avg, index=models).rank(ascending=False)
            
            # Classifier accuracy (lower is better - means more realistic)
            classifier_results = results['statistical_ml']['classifier']
            accuracy = [classifier_results[model]['accuracy'] for model in models]
            classifier_ranks = pd.Series(accuracy, index=models).rank()
            
            # Augmentation benefits (higher is better)
            augmentation_results = results['functional']['augmentation']
            r2_improvement = [augmentation_results[model]['r2_improvement'] for model in models]
            augmentation_ranks = pd.Series(r2_improvement, index=models).rank(ascending=False)
            
            # Overall ranking
            overall_ranks = gc_ranks + kmer_ranks + motif_ranks + structural_ranks + classifier_ranks + augmentation_ranks
            overall_ranks = overall_ranks.rank()
            
            # Create ranking table
            ranking_table = pd.DataFrame({
                'GC Content': gc_ranks,
                'k-mer Divergence': kmer_ranks,
                'Motif Enrichment': motif_ranks,
                'Structural Correlation': structural_ranks,
                'Classifier Accuracy': classifier_ranks,
                'Augmentation Benefits': augmentation_ranks,
                'Overall Rank': overall_ranks
            })
            
            st.dataframe(ranking_table)
            
            # Highlight the best model
            best_model = overall_ranks.idxmin()
            st.success(f"**Best Overall Model: {best_model}**")
        
        with tabs[1]:
            st.markdown("### Feature-Based Sequence Comparisons")
            
            # GC content
            st.markdown("#### GC Content Distribution")
            gc_plot_path = os.path.join(plots_dir, "gc_content.png")
            if os.path.exists(gc_plot_path):
                gc_plot = Image.open(gc_plot_path)
                st.image(gc_plot, caption="GC Content Distribution", use_column_width=True)
            else:
                st.warning("GC content plot not found. Please run the evaluation first.")
            
            # Display GC content statistics
            gc_stats = pd.DataFrame({
                'Mean': [results['feature_based']['gc_content']['real']['mean']] + 
                        [results['feature_based']['gc_content']['synthetic'][model]['mean'] for model in models],
                'Std Dev': [results['feature_based']['gc_content']['real']['std']] + 
                           [results['feature_based']['gc_content']['synthetic'][model]['std'] for model in models],
                'Min': [results['feature_based']['gc_content']['real']['min']] + 
                       [results['feature_based']['gc_content']['synthetic'][model]['min'] for model in models],
                'Max': [results['feature_based']['gc_content']['real']['max']] + 
                       [results['feature_based']['gc_content']['synthetic'][model]['max'] for model in models]
            }, index=['Real'] + models)
            
            st.dataframe(gc_stats)
            
            # k-mer divergence
            st.markdown("#### k-mer Frequency Divergence")
            kmer_plot_path = os.path.join(plots_dir, "kmer_divergence.png")
            if os.path.exists(kmer_plot_path):
                kmer_plot = Image.open(kmer_plot_path)
                st.image(kmer_plot, caption="k-mer Frequency Divergence", use_column_width=True)
            else:
                st.warning("k-mer divergence plot not found. Please run the evaluation first.")
            
            # Motif enrichment
            st.markdown("#### Motif Enrichment")
            motif_plot_path = os.path.join(plots_dir, "motif_enrichment.png")
            if os.path.exists(motif_plot_path):
                motif_plot = Image.open(motif_plot_path)
                st.image(motif_plot, caption="Motif Enrichment", use_column_width=True)
            else:
                st.warning("Motif enrichment plot not found. Please run the evaluation first.")
            
            # Structural correlation
            st.markdown("#### Structural Property Correlation")
            structural_plot_path = os.path.join(plots_dir, "structural_correlation.png")
            if os.path.exists(structural_plot_path):
                structural_plot = Image.open(structural_plot_path)
                st.image(structural_plot, caption="Structural Property Correlation", use_column_width=True)
            else:
                st.warning("Structural correlation plot not found. Please run the evaluation first.")
        
        with tabs[2]:
            st.markdown("### Statistical & Machine Learning Approaches")
            
            # Dimensionality reduction
            st.markdown("#### Dimensionality Reduction")
            
            dim_red_method = st.selectbox("Select Dimensionality Reduction Method", ["PCA", "t-SNE", "UMAP"])
            method_map = {"PCA": "pca", "t-SNE": "tsne", "UMAP": "umap"}
            
            dim_red_plot_path = os.path.join(plots_dir, f"{method_map[dim_red_method]}_projection.png")
            if os.path.exists(dim_red_plot_path):
                dim_red_plot = Image.open(dim_red_plot_path)
                st.image(dim_red_plot, caption=f"{dim_red_method} Projection", use_column_width=True)
            else:
                st.warning(f"{dim_red_method} projection plot not found. Please run the evaluation first.")
            
            # Classifier performance
            st.markdown("#### Classifier Performance")
            classifier_plot_path = os.path.join(plots_dir, "classifier_performance.png")
            if os.path.exists(classifier_plot_path):
                classifier_plot = Image.open(classifier_plot_path)
                st.image(classifier_plot, caption="Classifier Performance", use_column_width=True)
            else:
                st.warning("Classifier performance plot not found. Please run the evaluation first.")
        
        with tabs[3]:
            st.markdown("### Functional & Predictive Analyses")
            
            # Promoter strength
            st.markdown("#### Promoter Strength Distribution")
            strength_plot_path = os.path.join(plots_dir, "promoter_strength.png")
            if os.path.exists(strength_plot_path):
                strength_plot = Image.open(strength_plot_path)
                st.image(strength_plot, caption="Promoter Strength Distribution", use_column_width=True)
            else:
                st.warning("Promoter strength plot not found. Please run the evaluation first.")
            
            # Augmentation benefits
            st.markdown("#### Data Augmentation Benefits")
            augmentation_plot_path = os.path.join(plots_dir, "augmentation_benefits.png")
            if os.path.exists(augmentation_plot_path):
                augmentation_plot = Image.open(augmentation_plot_path)
                st.image(augmentation_plot, caption="Data Augmentation Benefits", use_column_width=True)
            else:
                st.warning("Augmentation benefits plot not found. Please run the evaluation first.")
        
        # Practical applications
        st.subheader("Practical Applications")
        
        st.markdown("""
        ### Data Augmentation
        
        Synthetic DNA sequences can be used to augment limited real datasets, improving the training of machine learning models for:
        - Promoter activity prediction
        - Transcription factor binding prediction
        - Gene expression modeling
        
        ### Rational Promoter Engineering
        
        Insights from the evaluation can guide the design of new promoters with:
        - Tailored strengths for specific applications
        - Cell-type specificities for targeted expression
        - Reduced immunogenicity for gene therapy applications
        
        ### Benchmarking Generative Methods
        
        The evaluation metrics provide a quantitative framework for:
        - Comparing different generative models (GANs, diffusion models, etc.)
        - Identifying areas for improvement in synthetic data generation
        - Establishing standards for synthetic DNA sequence quality
        """)
        
    else:
        st.warning("Evaluation results not found. Please run the evaluation first.")
        
        if st.button("Run Evaluation"):
            st.info("Running evaluation... This may take a few minutes.")
            
            # Define file paths
            real_file = "data/preprocessed_dna_sequences.fasta"
            synthetic_files = {
                "gumbel_softmax": "data/gumbel_generated_sequences.fasta",
                "improved_wgan": "data/improved_generated_sequences.fasta"
            }
            output_dir = "evaluation/results"
            
            # Check if files exist
            missing_files = []
            if not os.path.exists(real_file):
                missing_files.append(real_file)
            for model, file_path in synthetic_files.items():
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
            
            if missing_files:
                st.error(f"The following files are missing: {', '.join(missing_files)}")
                
                # Generate test sequences if needed
                if st.button("Generate Test Sequences"):
                    from utils.generate_test_sequences import main as generate_sequences
                    generate_sequences()
                    st.success("Test sequences generated successfully.")
                
                st.stop()
            
            # Run evaluation
            try:
                # Capture stdout to display progress
                stdout = StringIO()
                sys.stdout = stdout
                
                results = evaluate_sequences(real_file, synthetic_files, output_dir)
                
                # Reset stdout
                sys.stdout = sys.__stdout__
                
                # Display progress
                st.code(stdout.getvalue())
                
                st.success("Evaluation completed successfully. Please refresh the page to view the results.")
            except Exception as e:
                st.error(f"Error running evaluation: {e}")
                
                # Reset stdout
                sys.stdout = sys.__stdout__


if __name__ == "__main__":
    run_evaluation_page()
