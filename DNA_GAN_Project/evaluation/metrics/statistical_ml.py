"""
Statistical and machine learning approaches for sequence comparison.

This module provides functions for:
1. Dimensionality reduction and clustering
2. Classifier-based discrimination
3. Distributional tests
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp
import itertools


def encode_sequences(sequences, k=4):
    """
    Encode DNA sequences as k-mer frequency vectors.
    
    Args:
        sequences (list): List of DNA sequences.
        k (int): Length of k-mers.
        
    Returns:
        numpy.ndarray: Matrix of k-mer frequencies.
    """
    # Generate all possible k-mers
    nucleotides = ['A', 'C', 'G', 'T']
    all_kmers = [''.join(p) for p in itertools.product(nucleotides, repeat=k)]
    
    # Initialize frequency matrix
    n_sequences = len(sequences)
    n_kmers = len(all_kmers)
    X = np.zeros((n_sequences, n_kmers))
    
    # Map k-mers to indices
    kmer_to_idx = {kmer: i for i, kmer in enumerate(all_kmers)}
    
    # Count k-mers in sequences
    for i, seq in enumerate(sequences):
        for j in range(len(seq) - k + 1):
            kmer = seq[j:j+k]
            if kmer in kmer_to_idx:  # Only count valid k-mers
                X[i, kmer_to_idx[kmer]] += 1
    
    # Normalize by sequence length
    for i, seq in enumerate(sequences):
        if len(seq) - k + 1 > 0:
            X[i, :] /= (len(seq) - k + 1)
    
    return X


def perform_dimensionality_reduction(X, method='pca', n_components=2, random_state=42):
    """
    Perform dimensionality reduction on sequence encodings.
    
    Args:
        X (numpy.ndarray): Matrix of sequence encodings.
        method (str): Dimensionality reduction method ('pca', 'tsne', or 'umap').
        n_components (int): Number of components to reduce to.
        random_state (int): Random state for reproducibility.
        
    Returns:
        numpy.ndarray: Reduced representation.
    """
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if method == 'pca':
        model = PCA(n_components=n_components, random_state=random_state)
    elif method == 'tsne':
        model = TSNE(n_components=n_components, random_state=random_state)
    elif method == 'umap':
        model = umap.UMAP(n_components=n_components, random_state=random_state)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    X_reduced = model.fit_transform(X_scaled)
    
    return X_reduced


def train_classifier(real_encodings, synthetic_encodings, classifier='rf', test_size=0.2, random_state=42):
    """
    Train a classifier to distinguish real from synthetic sequences.
    
    Args:
        real_encodings (numpy.ndarray): Encodings of real sequences.
        synthetic_encodings (numpy.ndarray): Encodings of synthetic sequences.
        classifier (str): Classifier type ('rf' for Random Forest, 'svm' for SVM).
        test_size (float): Proportion of data to use for testing.
        random_state (int): Random state for reproducibility.
        
    Returns:
        dict: Dictionary containing classifier, accuracy, AUC, and confusion matrix.
    """
    # Combine real and synthetic encodings
    X = np.vstack([real_encodings, synthetic_encodings])
    y = np.hstack([np.zeros(len(real_encodings)), np.ones(len(synthetic_encodings))])
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Train classifier
    if classifier == 'rf':
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    elif classifier == 'svm':
        clf = SVC(probability=True, random_state=random_state)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    
    clf.fit(X_train, y_train)
    
    # Evaluate classifier
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(clf, X, y, cv=5)
    
    return {
        'classifier': clf,
        'accuracy': accuracy,
        'auc': auc,
        'confusion_matrix': cm,
        'cv_scores': cv_scores
    }


def perform_ks_test(real_features, synthetic_features):
    """
    Perform Kolmogorov-Smirnov test on feature distributions.
    
    Args:
        real_features (dict): Dictionary of features for real sequences.
        synthetic_features (dict): Dictionary of features for synthetic sequences.
        
    Returns:
        dict: Dictionary of KS test statistics and p-values.
    """
    results = {}
    
    for feature_name in real_features:
        statistic, p_value = ks_2samp(real_features[feature_name], synthetic_features[feature_name])
        results[feature_name] = {'statistic': statistic, 'p_value': p_value}
    
    return results


def calculate_mmd(X, Y, gamma=1.0):
    """
    Calculate Maximum Mean Discrepancy between two distributions.
    
    Args:
        X (numpy.ndarray): Samples from first distribution.
        Y (numpy.ndarray): Samples from second distribution.
        gamma (float): RBF kernel parameter.
        
    Returns:
        float: MMD value.
    """
    XX = np.dot(X, X.T)
    YY = np.dot(Y, Y.T)
    XY = np.dot(X, Y.T)
    
    n_X = X.shape[0]
    n_Y = Y.shape[0]
    
    # RBF kernel
    def rbf_kernel(A, B, gamma):
        return np.exp(-gamma * (A + B - 2 * np.dot(X, Y.T)))
    
    # Calculate MMD
    K_XX = np.exp(-gamma * (np.diag(XX)[:, np.newaxis] + np.diag(XX)[np.newaxis, :] - 2 * XX))
    K_YY = np.exp(-gamma * (np.diag(YY)[:, np.newaxis] + np.diag(YY)[np.newaxis, :] - 2 * YY))
    K_XY = np.exp(-gamma * (np.diag(XX)[:, np.newaxis] + np.diag(YY)[np.newaxis, :] - 2 * XY))
    
    mmd = np.mean(K_XX) - 2 * np.mean(K_XY) + np.mean(K_YY)
    
    return mmd
