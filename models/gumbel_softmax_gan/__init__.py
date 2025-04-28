"""
DNA sequence generation using Gumbel-Softmax GAN.

This package provides tools for generating DNA sequences using a Gumbel-Softmax GAN.
"""

from .models import Generator, CNNDiscriminator, LSTMDiscriminator
from .data import get_data_loader, one_hot_to_sequence, sequences_to_fasta
from .train import train_gan, plot_training_history, load_checkpoint
from .metrics import calculate_gc_content, calculate_morans_i, calculate_diversity, evaluate_model

__version__ = '0.1.0'
