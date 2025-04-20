"""
Improved models for DNA sequence generation using Wasserstein GAN with gradient penalty.

This module defines the Generator and Discriminator models for DNA sequence generation.
The Generator uses LSTM layers with attention to generate DNA sequences from random noise.
The Discriminator uses CNN layers with spectral normalization to provide stable gradients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import spectral_norm


class SelfAttention(nn.Module):
    """
    Self-attention module for sequence generation.
    
    Attributes:
        query (nn.Linear): Query projection.
        key (nn.Linear): Key projection.
        value (nn.Linear): Value projection.
        gamma (nn.Parameter): Learnable scaling parameter.
    """
    
    def __init__(self, hidden_dim):
        """
        Initialize the self-attention module.
        
        Args:
            hidden_dim (int): Dimension of the hidden state.
        """
        super(SelfAttention, self).__init__()
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        """
        Forward pass of the self-attention module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_dim).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_dim).
        """
        batch_size, seq_len, hidden_dim = x.size()
        
        # Project inputs to queries, keys, and values
        q = self.query(x).view(batch_size, seq_len, hidden_dim)
        k = self.key(x).view(batch_size, seq_len, hidden_dim)
        v = self.value(x).view(batch_size, seq_len, hidden_dim)
        
        # Compute attention scores
        attention = torch.bmm(q, k.transpose(1, 2))
        attention = F.softmax(attention, dim=2)
        
        # Apply attention to values
        out = torch.bmm(attention, v)
        
        # Apply learnable scaling
        out = self.gamma * out + x
        
        return out


class ImprovedGenerator(nn.Module):
    """
    Improved Generator model for DNA sequence generation.
    
    Uses LSTM layers with attention to generate DNA sequences from random noise.
    Applies Gumbel-Softmax trick for backpropagation through discrete outputs.
    
    Attributes:
        noise_dim (int): Dimension of the input noise vector.
        hidden_dim (int): Dimension of the LSTM hidden state.
        seq_len (int): Length of the generated DNA sequence.
        vocab_size (int): Size of the vocabulary (4 for DNA: A, C, G, T).
        lstm (nn.LSTM): LSTM layer for sequence generation.
        attention (SelfAttention): Self-attention module.
        fc (nn.Linear): Fully connected layer for output projection.
    """
    
    def __init__(self, noise_dim=100, hidden_dim=512, seq_len=150, vocab_size=4, num_layers=2):
        """
        Initialize the Generator model.
        
        Args:
            noise_dim (int): Dimension of the input noise vector.
            hidden_dim (int): Dimension of the LSTM hidden state.
            seq_len (int): Length of the generated DNA sequence.
            vocab_size (int): Size of the vocabulary (4 for DNA: A, C, G, T).
            num_layers (int): Number of LSTM layers.
        """
        super(ImprovedGenerator, self).__init__()
        
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # Embedding layer to convert noise to initial hidden state
        self.fc_hidden = nn.Linear(noise_dim, hidden_dim * num_layers)
        self.fc_cell = nn.Linear(noise_dim, hidden_dim * num_layers)
        
        # LSTM layer
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=0.2)
        
        # Self-attention layer
        self.attention = SelfAttention(hidden_dim)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
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
        h0 = self.fc_hidden(noise).view(self.num_layers, batch_size, self.hidden_dim)
        c0 = self.fc_cell(noise).view(self.num_layers, batch_size, self.hidden_dim)
        
        # Initialize input as zeros
        x = torch.zeros(batch_size, 1, self.hidden_dim).to(noise.device)
        
        # Generate sequence step by step
        outputs = []
        hidden_states = []
        
        for i in range(self.seq_len):
            # LSTM step
            out, (h0, c0) = self.lstm(x, (h0, c0))
            hidden_states.append(out)
            
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
        sequence = torch.stack(outputs, dim=1)  # (batch_size, seq_len, vocab_size)
        
        # Apply self-attention to the hidden states
        hidden_states = torch.cat(hidden_states, dim=1)  # (batch_size, seq_len, hidden_dim)
        attended = self.attention(hidden_states)
        
        # Apply layer normalization
        normalized = self.layer_norm(attended)
        
        # Project to vocabulary space and apply Gumbel-Softmax
        logits = self.fc(normalized)
        sequence = F.gumbel_softmax(logits, tau=temperature, hard=hard)
        
        return sequence
    
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


class ImprovedDiscriminator(nn.Module):
    """
    Improved Discriminator model for DNA sequence classification.
    
    Uses CNN layers with spectral normalization to provide stable gradients.
    
    Attributes:
        seq_len (int): Length of the input DNA sequence.
        vocab_size (int): Size of the vocabulary (4 for DNA: A, C, G, T).
        conv1, conv2, conv3 (nn.Conv1d): Convolutional layers with spectral normalization.
        fc1, fc2 (nn.Linear): Fully connected layers with spectral normalization.
    """
    
    def __init__(self, seq_len=150, vocab_size=4):
        """
        Initialize the Discriminator model.
        
        Args:
            seq_len (int): Length of the input DNA sequence.
            vocab_size (int): Size of the vocabulary (4 for DNA: A, C, G, T).
        """
        super(ImprovedDiscriminator, self).__init__()
        
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Convolutional layers with spectral normalization
        self.conv1 = spectral_norm(nn.Conv1d(vocab_size, 64, kernel_size=3, padding=1))
        self.conv2 = spectral_norm(nn.Conv1d(64, 128, kernel_size=3, padding=1))
        self.conv3 = spectral_norm(nn.Conv1d(128, 256, kernel_size=3, padding=1))
        
        # Pooling layers
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Calculate the size after convolutions and pooling
        conv_out_size = 256 * (seq_len // 8)
        
        # Fully connected layers with spectral normalization
        self.fc1 = spectral_norm(nn.Linear(conv_out_size, 256))
        self.fc2 = spectral_norm(nn.Linear(256, 1))
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm([64, seq_len])
        self.layer_norm2 = nn.LayerNorm([128, seq_len // 2])
        self.layer_norm3 = nn.LayerNorm([256, seq_len // 4])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        """
        Forward pass of the Discriminator.
        
        Args:
            x (torch.Tensor): Input DNA sequences of shape (batch_size, seq_len, vocab_size).
            
        Returns:
            torch.Tensor: Critic score for each sequence (batch_size, 1).
        """
        # Transpose for conv1d: (batch_size, vocab_size, seq_len)
        x = x.transpose(1, 2)
        
        # Convolutional layers with layer normalization
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = self.layer_norm1(x)
        x = self.pool(x)
        
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = self.layer_norm2(x)
        x = self.pool(x)
        
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = self.layer_norm3(x)
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
