"""
Complete Workflow for DNA Sequence GAN

This script demonstrates the complete workflow for training a GAN model,
generating synthetic DNA sequences, evaluating results, and creating visualizations.
"""

import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from data_loader import DNADataLoader
from gan_model import DNAGAN
from evaluation import DNAEvaluator
from visualization import DNAVisualizer
from dna_utils import pad_sequences

def main():
    """
    Main function demonstrating the complete workflow.
    """
    print("DNA Sequence GAN - Complete Workflow")
    
    # Create output directory for results
    os.makedirs("results", exist_ok=True)
    
    # Step 1: Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Step 2: Create data loader and generate dummy data
    print("\n--- Step 2: Data Preparation ---")
    data_loader = DNADataLoader(sequence_length=50, batch_size=32)
    
    # Generate dummy data
    print("Generating dummy data...")
    real_sequences = data_loader.generate_dummy_data(
        num_sequences=200,
        min_length=40,
        max_length=50
    )
    print(f"Generated {len(real_sequences)} sequences")
    print(f"Sample sequence: {real_sequences[0]}")
    
    # Preprocess data
    print("Preprocessing data...")
    encoded_sequences = data_loader.preprocess()
    print(f"Encoded sequences shape: {encoded_sequences.shape}")
    
    # Create dataset
    print("Creating dataset...")
    dataset = data_loader.create_dataset()
    print(f"Dataset size: {len(dataset)}")
    
    # Step 3: Initialize and train GAN
    print("\n--- Step 3: GAN Training ---")
    gan = DNAGAN(
        sequence_length=data_loader.sequence_length,
        batch_size=32,
        latent_dim=100,
        device=device
    )
    
    # Load data into GAN
    print("Loading data into GAN...")
    gan.load_data_from_dataset(dataset)
    
    # Train GAN
    print("Training GAN for 20 epochs...")
    gan.train(epochs=20, save_interval=10, verbose=True)
    
    # Save the final model
    print("Saving final model checkpoint...")
    gan.save_checkpoint(epoch=20)
    
    # Step 4: Generate synthetic sequences
    print("\n--- Step 4: Generating Synthetic Sequences ---")
    synthetic_sequences = gan.generate(num_sequences=100)
    print(f"Generated {len(synthetic_sequences)} synthetic sequences")
    print("Sample synthetic sequences:")
    for i, seq in enumerate(synthetic_sequences[:5]):
        print(f"Sequence {i+1}: {seq}")
    
    # Save synthetic sequences to CSV
    df = pd.DataFrame({
        'sequence_id': range(len(synthetic_sequences)),
        'sequence': synthetic_sequences
    })
    csv_path = os.path.join("results", "synthetic_sequences.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved synthetic sequences to {csv_path}")
    
    # Step 5: Evaluate results
    print("\n--- Step 5: Evaluating Results ---")
    
    # Ensure sequences have the same length
    if len(set(len(seq) for seq in real_sequences)) > 1 or len(set(len(seq) for seq in synthetic_sequences)) > 1:
        max_length = max(max(len(seq) for seq in real_sequences), max(len(seq) for seq in synthetic_sequences))
        real_sequences_padded = pad_sequences(real_sequences, max_length)
        synthetic_sequences_padded = pad_sequences(synthetic_sequences, max_length)
        print(f"Padded sequences to uniform length of {max_length}")
    else:
        real_sequences_padded = real_sequences
        synthetic_sequences_padded = synthetic_sequences
    
    # Comprehensive evaluation
    print("Performing comprehensive evaluation...")
    evaluation_results = DNAEvaluator.comprehensive_evaluation(
        real_sequences_padded, synthetic_sequences_padded
    )
    
    # Generate report
    report = DNAEvaluator.generate_report(evaluation_results)
    report_path = os.path.join("results", "evaluation_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved evaluation report to {report_path}")
    print("\nEvaluation Summary:")
    print(report.split("## Overall Assessment")[1])
    
    # Step 6: Create visualizations
    print("\n--- Step 6: Creating Visualizations ---")
    
    # K-mer distribution comparison
    print("Creating k-mer distribution comparison...")
    fig = DNAVisualizer.compare_kmer_distributions(
        real_sequences_padded, 
        synthetic_sequences_padded, 
        k=3, 
        title="3-mer Distribution Comparison"
    )
    fig_path = os.path.join("results", "kmer_comparison.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved k-mer comparison to {fig_path}")
    
    # GC content comparison
    print("Creating GC content comparison...")
    fig = DNAVisualizer.compare_gc_distributions(
        real_sequences_padded, 
        synthetic_sequences_padded, 
        title="GC Content Comparison"
    )
    fig_path = os.path.join("results", "gc_comparison.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved GC content comparison to {fig_path}")
    
    # Sequence logo
    print("Creating sequence logos...")
    fig = DNAVisualizer.sequence_logo(
        real_sequences_padded[:10], 
        title="Real DNA Sequence Logo"
    )
    fig_path = os.path.join("results", "real_sequence_logo.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved real sequence logo to {fig_path}")
    
    fig = DNAVisualizer.sequence_logo(
        synthetic_sequences_padded[:10], 
        title="Synthetic DNA Sequence Logo"
    )
    fig_path = os.path.join("results", "synthetic_sequence_logo.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved synthetic sequence logo to {fig_path}")
    
    # Training history
    print("Creating training history plot...")
    fig = gan.plot_training_history()
    fig_path = os.path.join("results", "training_history.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved training history to {fig_path}")
    
    print("\n--- Workflow Complete ---")
    print(f"All results saved to the 'results' directory")
    print("To run the web interface, execute: python app.py")

if __name__ == "__main__":
    main()
