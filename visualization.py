"""
DNA Sequence Visualization

This module provides functions for visualizing DNA sequences and GAN results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Union, Optional
from collections import Counter
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64

from dna_utils import get_kmers, kmer_frequency, gc_content, DNA_ALPHABET

class DNAVisualizer:
    """
    Class for visualizing DNA sequences and GAN results.
    """
    
    @staticmethod
    def sequence_logo(sequences: List[str], title: str = "Sequence Logo") -> plt.Figure:
        """
        Create a sequence logo visualization.
        
        Args:
            sequences (List[str]): List of DNA sequences.
            title (str): Title for the plot.
            
        Returns:
            plt.Figure: Matplotlib figure with sequence logo.
        """
        if not sequences:
            raise ValueError("No sequences provided")
        
        # Ensure all sequences have the same length
        seq_length = len(sequences[0])
        if not all(len(seq) == seq_length for seq in sequences):
            raise ValueError("All sequences must have the same length")
        
        # Calculate nucleotide frequencies at each position
        position_counts = []
        for i in range(seq_length):
            counts = Counter(seq[i] for seq in sequences)
            position_counts.append(counts)
        
        # Create a position weight matrix
        pwm = np.zeros((len(DNA_ALPHABET), seq_length))
        for i, counts in enumerate(position_counts):
            total = sum(counts.values())
            for j, nuc in enumerate(DNA_ALPHABET):
                pwm[j, i] = counts.get(nuc, 0) / total if total > 0 else 0
        
        # Create the sequence logo
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Plot stacked bars for each position
        bottom = np.zeros(seq_length)
        for i, nuc in enumerate(DNA_ALPHABET):
            ax.bar(range(seq_length), pwm[i], bottom=bottom, label=nuc, 
                   color={'A': 'green', 'C': 'blue', 'G': 'orange', 'T': 'red'}[nuc])
            bottom += pwm[i]
        
        ax.set_xlabel('Position')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.set_xticks(range(seq_length))
        ax.set_xticklabels(range(1, seq_length + 1))
        ax.legend(title='Nucleotide')
        
        return fig
    
    @staticmethod
    def kmer_distribution(sequences: List[str], k: int = 3, top_n: int = 20, 
                          title: str = "K-mer Distribution") -> plt.Figure:
        """
        Visualize the distribution of k-mers in a set of sequences.
        
        Args:
            sequences (List[str]): List of DNA sequences.
            k (int): Length of k-mers.
            top_n (int): Number of top k-mers to display.
            title (str): Title for the plot.
            
        Returns:
            plt.Figure: Matplotlib figure with k-mer distribution.
        """
        if not sequences:
            raise ValueError("No sequences provided")
        
        # Calculate k-mer frequencies
        kmer_counts = kmer_frequency(sequences, k)
        
        # Sort k-mers by frequency
        sorted_kmers = sorted(kmer_counts.items(), key=lambda x: x[1], reverse=True)
        top_kmers = sorted_kmers[:top_n]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        kmers, counts = zip(*top_kmers)
        ax.bar(kmers, counts)
        
        ax.set_xlabel(f'{k}-mer')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def compare_kmer_distributions(real_sequences: List[str], synthetic_sequences: List[str], 
                                  k: int = 3, top_n: int = 20, 
                                  title: str = "K-mer Distribution Comparison") -> plt.Figure:
        """
        Compare k-mer distributions between real and synthetic sequences.
        
        Args:
            real_sequences (List[str]): List of real DNA sequences.
            synthetic_sequences (List[str]): List of synthetic DNA sequences.
            k (int): Length of k-mers.
            top_n (int): Number of top k-mers to display.
            title (str): Title for the plot.
            
        Returns:
            plt.Figure: Matplotlib figure with k-mer distribution comparison.
        """
        if not real_sequences or not synthetic_sequences:
            raise ValueError("Both real and synthetic sequence lists must be non-empty")
        
        # Calculate k-mer frequencies
        real_kmer_counts = kmer_frequency(real_sequences, k)
        synth_kmer_counts = kmer_frequency(synthetic_sequences, k)
        
        # Get the union of top k-mers from both sets
        real_top = sorted(real_kmer_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        synth_top = sorted(synth_kmer_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        top_kmers = set(kmer for kmer, _ in real_top).union(set(kmer for kmer, _ in synth_top))
        
        # Create a DataFrame for plotting
        df = pd.DataFrame({
            'k-mer': list(top_kmers),
            'Real': [real_kmer_counts.get(kmer, 0) for kmer in top_kmers],
            'Synthetic': [synth_kmer_counts.get(kmer, 0) for kmer in top_kmers]
        })
        
        # Sort by real frequency
        df = df.sort_values('Real', ascending=False).head(top_n)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 7))
        
        x = np.arange(len(df))
        width = 0.35
        
        ax.bar(x - width/2, df['Real'], width, label='Real')
        ax.bar(x + width/2, df['Synthetic'], width, label='Synthetic')
        
        ax.set_xlabel(f'{k}-mer')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(df['k-mer'], rotation=90)
        ax.legend()
        
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def gc_content_distribution(sequences: List[str], title: str = "GC Content Distribution") -> plt.Figure:
        """
        Visualize the distribution of GC content across sequences.
        
        Args:
            sequences (List[str]): List of DNA sequences.
            title (str): Title for the plot.
            
        Returns:
            plt.Figure: Matplotlib figure with GC content distribution.
        """
        if not sequences:
            raise ValueError("No sequences provided")
        
        # Calculate GC content for each sequence
        gc_contents = [gc_content(seq) for seq in sequences]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.histplot(gc_contents, bins=20, kde=True, ax=ax)
        
        ax.set_xlabel('GC Content')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        
        return fig
    
    @staticmethod
    def compare_gc_distributions(real_sequences: List[str], synthetic_sequences: List[str],
                                title: str = "GC Content Comparison") -> plt.Figure:
        """
        Compare GC content distributions between real and synthetic sequences.
        
        Args:
            real_sequences (List[str]): List of real DNA sequences.
            synthetic_sequences (List[str]): List of synthetic DNA sequences.
            title (str): Title for the plot.
            
        Returns:
            plt.Figure: Matplotlib figure with GC content comparison.
        """
        if not real_sequences or not synthetic_sequences:
            raise ValueError("Both real and synthetic sequence lists must be non-empty")
        
        # Calculate GC content for each sequence
        real_gc = [gc_content(seq) for seq in real_sequences]
        synth_gc = [gc_content(seq) for seq in synthetic_sequences]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.histplot(real_gc, bins=20, kde=True, ax=ax, color='blue', label='Real', alpha=0.5)
        sns.histplot(synth_gc, bins=20, kde=True, ax=ax, color='red', label='Synthetic', alpha=0.5)
        
        ax.set_xlabel('GC Content')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()
        
        return fig
    
    @staticmethod
    def sequence_length_distribution(sequences: List[str], title: str = "Sequence Length Distribution") -> plt.Figure:
        """
        Visualize the distribution of sequence lengths.
        
        Args:
            sequences (List[str]): List of DNA sequences.
            title (str): Title for the plot.
            
        Returns:
            plt.Figure: Matplotlib figure with sequence length distribution.
        """
        if not sequences:
            raise ValueError("No sequences provided")
        
        # Calculate sequence lengths
        lengths = [len(seq) for seq in sequences]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.histplot(lengths, bins=20, kde=True, ax=ax)
        
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        
        return fig
    
    @staticmethod
    def nucleotide_composition(sequences: List[str], title: str = "Nucleotide Composition") -> plt.Figure:
        """
        Visualize the overall nucleotide composition.
        
        Args:
            sequences (List[str]): List of DNA sequences.
            title (str): Title for the plot.
            
        Returns:
            plt.Figure: Matplotlib figure with nucleotide composition.
        """
        if not sequences:
            raise ValueError("No sequences provided")
        
        # Count nucleotides
        all_nucleotides = ''.join(sequences)
        counts = Counter(all_nucleotides)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 8))
        
        labels = [f"{nuc} ({counts[nuc]})" for nuc in DNA_ALPHABET]
        sizes = [counts[nuc] for nuc in DNA_ALPHABET]
        colors = ['green', 'blue', 'orange', 'red']
        
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        ax.set_title(title)
        
        return fig
    
    @staticmethod
    def training_history(history: Dict, title: str = "GAN Training History") -> plt.Figure:
        """
        Visualize the training history of the GAN.
        
        Args:
            history (Dict): Training history dictionary with 'generator_loss' and 'discriminator_loss' keys.
            title (str): Title for the plot.
            
        Returns:
            plt.Figure: Matplotlib figure with training history.
        """
        if 'generator_loss' not in history or 'discriminator_loss' not in history:
            raise ValueError("History must contain 'generator_loss' and 'discriminator_loss' keys")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        epochs = range(1, len(history['generator_loss']) + 1)
        ax.plot(epochs, history['generator_loss'], 'b-', label='Generator Loss')
        ax.plot(epochs, history['discriminator_loss'], 'r-', label='Discriminator Loss')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        return fig
    
    # Plotly versions for web display
    
    @staticmethod
    def plotly_kmer_distribution(sequences: List[str], k: int = 3, top_n: int = 20, 
                                title: str = "K-mer Distribution") -> go.Figure:
        """
        Create an interactive k-mer distribution plot using Plotly.
        
        Args:
            sequences (List[str]): List of DNA sequences.
            k (int): Length of k-mers.
            top_n (int): Number of top k-mers to display.
            title (str): Title for the plot.
            
        Returns:
            go.Figure: Plotly figure with k-mer distribution.
        """
        if not sequences:
            raise ValueError("No sequences provided")
        
        # Calculate k-mer frequencies
        kmer_counts = kmer_frequency(sequences, k)
        
        # Sort k-mers by frequency
        sorted_kmers = sorted(kmer_counts.items(), key=lambda x: x[1], reverse=True)
        top_kmers = sorted_kmers[:top_n]
        
        # Create DataFrame for plotting
        df = pd.DataFrame(top_kmers, columns=['k-mer', 'Frequency'])
        
        # Create the plot
        fig = px.bar(df, x='k-mer', y='Frequency', title=title)
        fig.update_layout(xaxis_title=f'{k}-mer', yaxis_title='Frequency')
        
        return fig
    
    @staticmethod
    def plotly_compare_kmer_distributions(real_sequences: List[str], synthetic_sequences: List[str], 
                                         k: int = 3, top_n: int = 20, 
                                         title: str = "K-mer Distribution Comparison") -> go.Figure:
        """
        Create an interactive comparison of k-mer distributions using Plotly.
        
        Args:
            real_sequences (List[str]): List of real DNA sequences.
            synthetic_sequences (List[str]): List of synthetic DNA sequences.
            k (int): Length of k-mers.
            top_n (int): Number of top k-mers to display.
            title (str): Title for the plot.
            
        Returns:
            go.Figure: Plotly figure with k-mer distribution comparison.
        """
        if not real_sequences or not synthetic_sequences:
            raise ValueError("Both real and synthetic sequence lists must be non-empty")
        
        # Calculate k-mer frequencies
        real_kmer_counts = kmer_frequency(real_sequences, k)
        synth_kmer_counts = kmer_frequency(synthetic_sequences, k)
        
        # Get the union of top k-mers from both sets
        real_top = sorted(real_kmer_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        synth_top = sorted(synth_kmer_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        top_kmers = set(kmer for kmer, _ in real_top).union(set(kmer for kmer, _ in synth_top))
        
        # Create a DataFrame for plotting
        df = pd.DataFrame({
            'k-mer': list(top_kmers),
            'Real': [real_kmer_counts.get(kmer, 0) for kmer in top_kmers],
            'Synthetic': [synth_kmer_counts.get(kmer, 0) for kmer in top_kmers]
        })
        
        # Sort by real frequency
        df = df.sort_values('Real', ascending=False).head(top_n)
        
        # Melt the DataFrame for Plotly
        df_melted = pd.melt(df, id_vars=['k-mer'], value_vars=['Real', 'Synthetic'], 
                           var_name='Type', value_name='Frequency')
        
        # Create the plot
        fig = px.bar(df_melted, x='k-mer', y='Frequency', color='Type', barmode='group',
                    title=title)
        fig.update_layout(xaxis_title=f'{k}-mer', yaxis_title='Frequency')
        
        return fig
    
    @staticmethod
    def plotly_gc_content_comparison(real_sequences: List[str], synthetic_sequences: List[str],
                                    title: str = "GC Content Comparison") -> go.Figure:
        """
        Create an interactive comparison of GC content distributions using Plotly.
        
        Args:
            real_sequences (List[str]): List of real DNA sequences.
            synthetic_sequences (List[str]): List of synthetic DNA sequences.
            title (str): Title for the plot.
            
        Returns:
            go.Figure: Plotly figure with GC content comparison.
        """
        if not real_sequences or not synthetic_sequences:
            raise ValueError("Both real and synthetic sequence lists must be non-empty")
        
        # Calculate GC content for each sequence
        real_gc = [gc_content(seq) for seq in real_sequences]
        synth_gc = [gc_content(seq) for seq in synthetic_sequences]
        
        # Create the plot
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=real_gc,
            name='Real',
            opacity=0.7,
            nbinsx=20
        ))
        
        fig.add_trace(go.Histogram(
            x=synth_gc,
            name='Synthetic',
            opacity=0.7,
            nbinsx=20
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='GC Content',
            yaxis_title='Count',
            barmode='overlay'
        )
        
        return fig
    
    @staticmethod
    def plotly_training_history(history: Dict, title: str = "GAN Training History") -> go.Figure:
        """
        Create an interactive training history plot using Plotly.
        
        Args:
            history (Dict): Training history dictionary with 'generator_loss' and 'discriminator_loss' keys.
            title (str): Title for the plot.
            
        Returns:
            go.Figure: Plotly figure with training history.
        """
        if 'generator_loss' not in history or 'discriminator_loss' not in history:
            raise ValueError("History must contain 'generator_loss' and 'discriminator_loss' keys")
        
        # Create the plot
        fig = go.Figure()
        
        epochs = list(range(1, len(history['generator_loss']) + 1))
        
        fig.add_trace(go.Scatter(
            x=epochs,
            y=history['generator_loss'],
            mode='lines',
            name='Generator Loss'
        ))
        
        fig.add_trace(go.Scatter(
            x=epochs,
            y=history['discriminator_loss'],
            mode='lines',
            name='Discriminator Loss'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Epoch',
            yaxis_title='Loss',
            legend_title='Loss Type'
        )
        
        return fig
    
    @staticmethod
    def plotly_sequence_viewer(sequences: List[str], max_display: int = 5, 
                              title: str = "DNA Sequence Viewer") -> go.Figure:
        """
        Create an interactive DNA sequence viewer using Plotly.
        
        Args:
            sequences (List[str]): List of DNA sequences.
            max_display (int): Maximum number of sequences to display.
            title (str): Title for the plot.
            
        Returns:
            go.Figure: Plotly figure with sequence visualization.
        """
        if not sequences:
            raise ValueError("No sequences provided")
        
        # Limit the number of sequences to display
        display_sequences = sequences[:max_display]
        
        # Create a figure with subplots
        fig = make_subplots(rows=len(display_sequences), cols=1, 
                           subplot_titles=[f"Sequence {i+1}" for i in range(len(display_sequences))])
        
        # Color mapping for nucleotides
        color_map = {'A': 'green', 'C': 'blue', 'G': 'orange', 'T': 'red', 'N': 'gray'}
        
        # Add each sequence as a heatmap
        for i, seq in enumerate(display_sequences):
            # Convert sequence to numerical values for heatmap
            seq_values = np.zeros(len(seq))
            colors = []
            
            for j, nuc in enumerate(seq):
                colors.append(color_map.get(nuc, 'gray'))
            
            # Create a heatmap-like visualization
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(seq))),
                    y=[i] * len(seq),
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=colors,
                        symbol='square'
                    ),
                    text=[nuc for nuc in seq],
                    hoverinfo='text',
                    showlegend=False
                ),
                row=i+1, col=1
            )
            
            # Add nucleotide labels
            for j, nuc in enumerate(seq):
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=nuc,
                    showarrow=False,
                    font=dict(color='white', size=8),
                    row=i+1, col=1
                )
        
        # Update layout
        fig.update_layout(
            title=title,
            height=100 * len(display_sequences),
            showlegend=False,
            plot_bgcolor='white'
        )
        
        # Update axes
        for i in range(len(display_sequences)):
            fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=i+1, col=1)
            fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=i+1, col=1)
        
        return fig
    
    @staticmethod
    def fig_to_base64(fig: Union[plt.Figure, go.Figure]) -> str:
        """
        Convert a Matplotlib or Plotly figure to a base64 encoded string.
        
        Args:
            fig (Union[plt.Figure, go.Figure]): Figure to convert.
            
        Returns:
            str: Base64 encoded string of the figure.
        """
        if isinstance(fig, plt.Figure):
            # For Matplotlib figures
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            return f"data:image/png;base64,{img_str}"
        elif isinstance(fig, go.Figure):
            # For Plotly figures
            img_bytes = fig.to_image(format="png")
            img_str = base64.b64encode(img_bytes).decode('utf-8')
            return f"data:image/png;base64,{img_str}"
        else:
            raise ValueError("Unsupported figure type")
