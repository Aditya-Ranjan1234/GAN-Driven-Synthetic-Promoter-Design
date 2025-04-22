"""
Functional and predictive analyses for sequence comparison.

This module provides functions for:
1. Promoter strength prediction
2. Downstream model training
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import itertools


class SimplePromoterStrengthPredictor:
    """
    A simple model to predict promoter strength based on sequence features.
    
    This is a placeholder for more sophisticated models like Enformer.
    In a real-world scenario, you would use a pre-trained deep learning model.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the predictor.
        
        Args:
            random_state (int): Random state for reproducibility.
        """
        self.model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        self.trained = False
        
    def _extract_features(self, sequences):
        """
        Extract features from sequences.
        
        Args:
            sequences (list): List of DNA sequences.
            
        Returns:
            numpy.ndarray: Feature matrix.
        """
        # Calculate GC content
        gc_content = np.array([seq.count('G') + seq.count('C') for seq in sequences]) / np.array([len(seq) for seq in sequences])
        
        # Calculate 3-mer frequencies
        nucleotides = ['A', 'C', 'G', 'T']
        all_3mers = [''.join(p) for p in itertools.product(nucleotides, repeat=3)]
        
        X_3mers = np.zeros((len(sequences), len(all_3mers)))
        
        for i, seq in enumerate(sequences):
            for j in range(len(seq) - 2):
                kmer = seq[j:j+3]
                if kmer in all_3mers:
                    X_3mers[i, all_3mers.index(kmer)] += 1
            
            # Normalize by sequence length
            if len(seq) - 2 > 0:
                X_3mers[i, :] /= (len(seq) - 2)
        
        # Combine features
        X = np.column_stack([gc_content.reshape(-1, 1), X_3mers])
        
        return X
    
    def fit(self, sequences, strengths):
        """
        Train the model on sequences and their strengths.
        
        Args:
            sequences (list): List of DNA sequences.
            strengths (list): List of promoter strengths.
            
        Returns:
            self: The trained model.
        """
        X = self._extract_features(sequences)
        self.model.fit(X, strengths)
        self.trained = True
        return self
    
    def predict(self, sequences):
        """
        Predict promoter strengths for sequences.
        
        Args:
            sequences (list): List of DNA sequences.
            
        Returns:
            numpy.ndarray: Predicted strengths.
        """
        if not self.trained:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        X = self._extract_features(sequences)
        return self.model.predict(X)


def simulate_promoter_strengths(sequences, noise_level=0.1, random_state=42):
    """
    Simulate promoter strengths based on sequence features.
    
    This is a placeholder for real experimental data.
    In a real-world scenario, you would use measured promoter activities.
    
    Args:
        sequences (list): List of DNA sequences.
        noise_level (float): Level of noise to add.
        random_state (int): Random state for reproducibility.
        
    Returns:
        numpy.ndarray: Simulated strengths.
    """
    np.random.seed(random_state)
    
    # Calculate GC content
    gc_content = np.array([seq.count('G') + seq.count('C') for seq in sequences]) / np.array([len(seq) for seq in sequences])
    
    # Look for TATA box
    tata_scores = np.array([seq.count('TATAAA') + seq.count('TATAAT') for seq in sequences])
    
    # Combine features with some weights
    strengths = 0.5 * gc_content + 0.3 * (tata_scores > 0) + 0.2 * np.random.normal(0, noise_level, len(sequences))
    
    # Normalize to [0, 1]
    strengths = (strengths - np.min(strengths)) / (np.max(strengths) - np.min(strengths))
    
    return strengths


def evaluate_augmentation(real_sequences, synthetic_sequences, real_strengths=None, test_size=0.2, random_state=42):
    """
    Evaluate the benefit of augmenting real data with synthetic data.
    
    Args:
        real_sequences (list): List of real DNA sequences.
        synthetic_sequences (list): List of synthetic DNA sequences.
        real_strengths (list): List of real promoter strengths (if None, will be simulated).
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random state for reproducibility.
        
    Returns:
        dict: Dictionary of evaluation metrics.
    """
    # Simulate strengths if not provided
    if real_strengths is None:
        real_strengths = simulate_promoter_strengths(real_sequences, random_state=random_state)
    
    # Split real data into train and test sets
    real_train, real_test, strengths_train, strengths_test = train_test_split(
        real_sequences, real_strengths, test_size=test_size, random_state=random_state
    )
    
    # Train model on real data only
    real_only_model = SimplePromoterStrengthPredictor(random_state=random_state)
    real_only_model.fit(real_train, strengths_train)
    real_only_pred = real_only_model.predict(real_test)
    
    # Calculate metrics for real-only model
    real_only_mse = mean_squared_error(strengths_test, real_only_pred)
    real_only_r2 = r2_score(strengths_test, real_only_pred)
    
    # Generate synthetic strengths
    synthetic_strengths = simulate_promoter_strengths(synthetic_sequences, random_state=random_state)
    
    # Combine real and synthetic data
    augmented_sequences = real_train + synthetic_sequences[:len(real_train)]
    augmented_strengths = np.concatenate([strengths_train, synthetic_strengths[:len(real_train)]])
    
    # Train model on augmented data
    augmented_model = SimplePromoterStrengthPredictor(random_state=random_state)
    augmented_model.fit(augmented_sequences, augmented_strengths)
    augmented_pred = augmented_model.predict(real_test)
    
    # Calculate metrics for augmented model
    augmented_mse = mean_squared_error(strengths_test, augmented_pred)
    augmented_r2 = r2_score(strengths_test, augmented_pred)
    
    # Calculate improvement
    mse_improvement = (real_only_mse - augmented_mse) / real_only_mse
    r2_improvement = (augmented_r2 - real_only_r2) / (1 - real_only_r2) if real_only_r2 < 1 else 0
    
    return {
        'real_only_mse': real_only_mse,
        'real_only_r2': real_only_r2,
        'augmented_mse': augmented_mse,
        'augmented_r2': augmented_r2,
        'mse_improvement': mse_improvement,
        'r2_improvement': r2_improvement
    }
