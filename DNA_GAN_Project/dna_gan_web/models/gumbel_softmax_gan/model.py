"""
Simplified Gumbel-Softmax GAN model for DNA sequence generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Generator(nn.Module):
    """
    Generator model for Gumbel-Softmax GAN.
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
            num_layers=1,
            batch_first=True
        )
        
        # Output layer
        self.output = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, z, temperature=1.0, hard=False):
        """
        Forward pass.
        
        Args:
            z (torch.Tensor): Input noise tensor of shape (batch_size, noise_dim).
            temperature (float): Temperature parameter for Gumbel-Softmax.
            hard (bool): Whether to use hard or soft Gumbel-Softmax.
            
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
        
        # Apply Gumbel-Softmax
        if hard:
            # Hard Gumbel-Softmax (one-hot)
            _, indices = torch.max(logits, dim=2)
            one_hot = torch.zeros_like(logits).scatter_(2, indices.unsqueeze(2), 1.0)
            return one_hot
        else:
            # Soft Gumbel-Softmax
            return F.gumbel_softmax(logits, tau=temperature, hard=False, dim=2)
    
    def generate(self, num_sequences, device, temperature=1.0, hard=True):
        """
        Generate DNA sequences.
        
        Args:
            num_sequences (int): Number of sequences to generate.
            device (torch.device): Device to use for generation.
            temperature (float): Temperature parameter for Gumbel-Softmax.
            hard (bool): Whether to use hard or soft Gumbel-Softmax.
            
        Returns:
            torch.Tensor: Generated sequences of shape (num_sequences, seq_len, vocab_size).
        """
        with torch.no_grad():
            # Generate random noise
            z = torch.randn(num_sequences, self.noise_dim).to(device)
            
            # Generate sequences
            sequences = self.forward(z, temperature, hard)
            
            return sequences


class Discriminator(nn.Module):
    """
    Discriminator model for Gumbel-Softmax GAN.
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
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(vocab_size, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * seq_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
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
        # Transpose for convolutional layers (batch_size, vocab_size, seq_len)
        x = x.transpose(1, 2)
        
        # Convolutional layers
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = self.dropout(x)
        x = F.leaky_relu(self.conv3(x), 0.2)
        
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
        sequences (torch.Tensor): One-hot encoded sequences [batch_size, seq_len, vocab_size]
        output_file (str): Path to save the FASTA file
    """
    # Convert one-hot to indices
    if sequences.dim() == 3:
        indices = torch.argmax(sequences, dim=2).cpu().numpy()
    else:
        indices = sequences.cpu().numpy()
    
    # Map indices to nucleotides
    nucleotides = ['A', 'C', 'G', 'T']
    
    # Create FASTA content
    fasta_content = ""
    for i, seq_indices in enumerate(indices):
        seq_str = ''.join([nucleotides[idx] for idx in seq_indices])
        fasta_content += f">gumbel_softmax_{i+1}\n{seq_str}\n"
    
    # Save to file
    with open(output_file, 'w') as f:
        f.write(fasta_content)
    
    print(f"Saved {len(indices)} sequences to {output_file}")
