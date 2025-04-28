"""
Simplified Improved WGAN-GP model for DNA sequence generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Generator(nn.Module):
    """
    Generator model for Improved WGAN-GP.
    """
    def __init__(self, noise_dim=100, hidden_dim=256, seq_len=150, vocab_size=4):
        """
        Initialize the generator.
        
        Args:
            noise_dim (int): Dimension of the input noise vector.
            hidden_dim (int): Dimension of the hidden state.
            seq_len (int): Length of the generated sequences.
            vocab_size (int): Size of the vocabulary (4 for DNA: A, C, G, T).
        """
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Fully connected layer to project noise to hidden dimension
        self.fc = nn.Linear(noise_dim, hidden_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Output layer
        self.output = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, z):
        """
        Forward pass.
        
        Args:
            z (torch.Tensor): Input noise tensor of shape (batch_size, noise_dim).
            
        Returns:
            torch.Tensor: Generated sequences of shape (batch_size, seq_len, vocab_size).
        """
        batch_size = z.size(0)
        
        # Project noise to hidden dimension
        h = F.relu(self.fc(z))
        
        # Repeat hidden state for each position in the sequence
        h = h.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # Pass through LSTM
        h, _ = self.lstm(h)
        
        # Project to vocabulary size
        logits = self.output(h)
        
        # Apply softmax to get probabilities
        return F.softmax(logits, dim=2)
    
    def generate(self, num_sequences, device):
        """
        Generate DNA sequences.
        
        Args:
            num_sequences (int): Number of sequences to generate.
            device (torch.device): Device to use for generation.
            
        Returns:
            torch.Tensor: Generated sequences of shape (num_sequences, seq_len).
        """
        with torch.no_grad():
            # Generate random noise
            z = torch.randn(num_sequences, self.noise_dim).to(device)
            
            # Generate sequences
            probs = self.forward(z)
            
            # Convert to one-hot
            _, indices = torch.max(probs, dim=2)
            
            return indices


class Discriminator(nn.Module):
    """
    Discriminator model for Improved WGAN-GP.
    """
    def __init__(self, seq_len=150, vocab_size=4, hidden_dim=256):
        """
        Initialize the discriminator.
        
        Args:
            seq_len (int): Length of the input sequences.
            vocab_size (int): Size of the vocabulary (4 for DNA: A, C, G, T).
            hidden_dim (int): Dimension of the hidden state.
        """
        super(Discriminator, self).__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Linear(vocab_size, hidden_dim)
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=5, padding=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 4 * seq_len, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, 1)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim * 2)
        self.ln3 = nn.LayerNorm(hidden_dim * 4)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input sequences of shape (batch_size, seq_len, vocab_size).
            
        Returns:
            torch.Tensor: Discriminator output of shape (batch_size, 1).
        """
        # Embedding
        x = self.embedding(x)
        
        # Transpose for convolutional layers (batch_size, hidden_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Convolutional layers
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = self.ln1(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)
        
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = self.ln2(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(x)
        
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = self.ln3(x.transpose(1, 2)).transpose(1, 2)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def save_sequences_to_fasta(sequences, output_file):
    """
    Save generated sequences to a FASTA file.
    
    Args:
        sequences (torch.Tensor): Indices of nucleotides [batch_size, seq_len]
        output_file (str): Path to save the FASTA file
    """
    # Convert to numpy array
    indices = sequences.cpu().numpy()
    
    # Map indices to nucleotides
    nucleotides = ['A', 'C', 'G', 'T']
    
    # Create FASTA content
    fasta_content = ""
    for i, seq_indices in enumerate(indices):
        seq_str = ''.join([nucleotides[idx] for idx in seq_indices])
        fasta_content += f">improved_wgan_{i+1}\n{seq_str}\n"
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write(fasta_content)
    
    print(f"Saved {len(indices)} sequences to {output_file}")
