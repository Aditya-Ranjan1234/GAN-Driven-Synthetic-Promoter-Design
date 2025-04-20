"""
Models for DNA sequence generation using Gumbel-Softmax GAN.

This module defines the Generator and Discriminator models for DNA sequence generation.
The Generator uses LSTM layers to generate DNA sequences from random noise.
The Discriminator uses CNN or LSTM layers to distinguish between real and generated sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Generator(nn.Module):
    """
    Generator model for DNA sequence generation.
    
    Uses LSTM layers to generate DNA sequences from random noise.
    Applies Gumbel-Softmax trick for backpropagation through discrete outputs.
    
    Attributes:
        noise_dim (int): Dimension of the input noise vector.
        hidden_dim (int): Dimension of the LSTM hidden state.
        seq_len (int): Length of the generated DNA sequence.
        vocab_size (int): Size of the vocabulary (4 for DNA: A, C, G, T).
        lstm (nn.LSTM): LSTM layer for sequence generation.
        fc (nn.Linear): Fully connected layer for output projection.
    """
    
    def __init__(self, noise_dim=100, hidden_dim=256, seq_len=150, vocab_size=4):
        """
        Initialize the Generator model.
        
        Args:
            noise_dim (int): Dimension of the input noise vector.
            hidden_dim (int): Dimension of the LSTM hidden state.
            seq_len (int): Length of the generated DNA sequence.
            vocab_size (int): Size of the vocabulary (4 for DNA: A, C, G, T).
        """
        super(Generator, self).__init__()
        
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Embedding layer to convert noise to initial hidden state
        self.fc_hidden = nn.Linear(noise_dim, hidden_dim)
        self.fc_cell = nn.Linear(noise_dim, hidden_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, noise, temperature=1.0, hard=False):
        """
        Forward pass of the Generator.
        
        Args:
            noise (torch.Tensor): Input noise tensor of shape (batch_size, noise_dim).
            temperature (float): Temperature parameter for Gumbel-Softmax.
            hard (bool): Whether to use hard or soft Gumbel-Softmax.
            
        Returns:
            torch.Tensor: Generated DNA sequences of shape (batch_size, seq_len, vocab_size).
        """
        batch_size = noise.size(0)
        
        # Initialize hidden state and cell state from noise
        h0 = self.fc_hidden(noise).unsqueeze(0)  # (1, batch_size, hidden_dim)
        c0 = self.fc_cell(noise).unsqueeze(0)    # (1, batch_size, hidden_dim)
        
        # Initialize input as zeros
        x = torch.zeros(batch_size, 1, self.hidden_dim).to(noise.device)
        
        # Generate sequence step by step
        outputs = []
        for i in range(self.seq_len):
            # LSTM step
            out, (h0, c0) = self.lstm(x, (h0, c0))
            
            # Apply dropout
            out = self.dropout(out)
            
            # Project to vocabulary space
            logits = self.fc(out.squeeze(1))
            
            # Apply Gumbel-Softmax
            gumbel_out = F.gumbel_softmax(logits, tau=temperature, hard=hard)
            outputs.append(gumbel_out)
            
            # Prepare input for next step
            x = torch.matmul(gumbel_out, self.fc.weight).unsqueeze(1)
        
        # Stack outputs to get the full sequence
        return torch.stack(outputs, dim=1)  # (batch_size, seq_len, vocab_size)
    
    def sample(self, batch_size, device):
        """
        Sample DNA sequences from the Generator.
        
        Args:
            batch_size (int): Number of sequences to generate.
            device (torch.device): Device to use for generation.
            
        Returns:
            torch.Tensor: Generated DNA sequences of shape (batch_size, seq_len, vocab_size).
        """
        # Sample random noise
        noise = torch.randn(batch_size, self.noise_dim).to(device)
        
        # Generate sequences
        with torch.no_grad():
            sequences = self.forward(noise, temperature=1.0, hard=True)
        
        return sequences


class CNNDiscriminator(nn.Module):
    """
    CNN-based Discriminator model for DNA sequence classification.
    
    Uses convolutional layers to distinguish between real and generated DNA sequences.
    
    Attributes:
        seq_len (int): Length of the input DNA sequence.
        vocab_size (int): Size of the vocabulary (4 for DNA: A, C, G, T).
        conv1, conv2, conv3 (nn.Conv1d): Convolutional layers.
        fc1, fc2 (nn.Linear): Fully connected layers.
    """
    
    def __init__(self, seq_len=150, vocab_size=4):
        """
        Initialize the CNN Discriminator model.
        
        Args:
            seq_len (int): Length of the input DNA sequence.
            vocab_size (int): Size of the vocabulary (4 for DNA: A, C, G, T).
        """
        super(CNNDiscriminator, self).__init__()
        
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(vocab_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        conv_out_size = 256 * (seq_len // 8)
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 256)
        self.fc2 = nn.Linear(256, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        Forward pass of the Discriminator.
        
        Args:
            x (torch.Tensor): Input DNA sequences of shape (batch_size, seq_len, vocab_size).
            
        Returns:
            torch.Tensor: Probability that each sequence is real (batch_size, 1).
        """
        # Transpose for conv1d: (batch_size, vocab_size, seq_len)
        x = x.transpose(1, 2)
        
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x


class LSTMDiscriminator(nn.Module):
    """
    LSTM-based Discriminator model for DNA sequence classification.
    
    Uses LSTM layers to distinguish between real and generated DNA sequences.
    
    Attributes:
        seq_len (int): Length of the input DNA sequence.
        vocab_size (int): Size of the vocabulary (4 for DNA: A, C, G, T).
        hidden_dim (int): Dimension of the LSTM hidden state.
        lstm (nn.LSTM): LSTM layer for sequence processing.
        fc1, fc2 (nn.Linear): Fully connected layers.
    """
    
    def __init__(self, seq_len=150, vocab_size=4, hidden_dim=256):
        """
        Initialize the LSTM Discriminator model.
        
        Args:
            seq_len (int): Length of the input DNA sequence.
            vocab_size (int): Size of the vocabulary (4 for DNA: A, C, G, T).
            hidden_dim (int): Dimension of the LSTM hidden state.
        """
        super(LSTMDiscriminator, self).__init__()
        
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # LSTM layer
        self.lstm = nn.LSTM(vocab_size, hidden_dim, batch_first=True, bidirectional=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, 256)  # *2 for bidirectional
        self.fc2 = nn.Linear(256, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        Forward pass of the Discriminator.
        
        Args:
            x (torch.Tensor): Input DNA sequences of shape (batch_size, seq_len, vocab_size).
            
        Returns:
            torch.Tensor: Probability that each sequence is real (batch_size, 1).
        """
        # LSTM layer
        _, (hidden, _) = self.lstm(x)
        
        # Concatenate forward and backward hidden states
        hidden = torch.cat((hidden[0], hidden[1]), dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(hidden))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x
