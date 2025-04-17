"""
DNA Sequence GAN Model

This module implements a Generative Adversarial Network (GAN) for generating synthetic DNA sequences using PyTorch.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Tuple, Union, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from data_loader import DNADataLoader
from dna_utils import one_hot_decode, DNA_ALPHABET

class Generator(nn.Module):
    """
    Generator model for DNA sequence generation.
    """
    def __init__(self, latent_dim: int, sequence_length: int, num_classes: int):
        """
        Initialize the Generator model.

        Args:
            latent_dim (int): Dimension of the latent space.
            sequence_length (int): Length of DNA sequences to generate.
            num_classes (int): Number of nucleotide classes (typically 4 for DNA).
        """
        super(Generator, self).__init__()

        self.sequence_length = sequence_length
        self.num_classes = num_classes

        # Main network
        self.main = nn.Sequential(
            # Input layer
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),

            # Hidden layer
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),

            # Output layer
            nn.Linear(512, sequence_length * num_classes),
        )

        # Softmax to get probability distribution over nucleotides
        self.softmax = nn.Softmax(dim=2)

    def forward(self, z):
        """
        Forward pass of the generator.

        Args:
            z (torch.Tensor): Random noise tensor of shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Generated DNA sequences as one-hot encodings.
        """
        # Pass through main network
        x = self.main(z)

        # Reshape to (batch_size, sequence_length, num_classes)
        x = x.view(-1, self.sequence_length, self.num_classes)

        # Apply softmax to get probability distribution
        x = self.softmax(x)

        return x


class Discriminator(nn.Module):
    """
    Discriminator model for DNA sequence classification.
    """
    def __init__(self, sequence_length: int, num_classes: int):
        """
        Initialize the Discriminator model.

        Args:
            sequence_length (int): Length of DNA sequences.
            num_classes (int): Number of nucleotide classes (typically 4 for DNA).
        """
        super(Discriminator, self).__init__()

        # Convolutional layers
        self.conv = nn.Sequential(
            nn.Conv1d(num_classes, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(128 * sequence_length, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of the discriminator.

        Args:
            x (torch.Tensor): DNA sequences as one-hot encodings of shape (batch_size, sequence_length, num_classes).

        Returns:
            torch.Tensor: Probability that each sequence is real.
        """
        # Transpose to (batch_size, num_classes, sequence_length) for Conv1D
        x = x.transpose(1, 2)

        # Pass through convolutional layers
        x = self.conv(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Pass through fully connected layers
        x = self.fc(x)

        return x


class DNAGAN:
    """
    Generative Adversarial Network for DNA sequence generation using PyTorch.
    """

    def __init__(self,
                 sequence_length: int = 100,
                 batch_size: int = 32,
                 latent_dim: int = 100,
                 checkpoint_dir: str = 'models',
                 device: str = None):
        """
        Initialize the DNA GAN model.

        Args:
            sequence_length (int): Length of DNA sequences to generate.
            batch_size (int): Batch size for training.
            latent_dim (int): Dimension of the latent space.
            checkpoint_dir (str): Directory to save model checkpoints.
            device (str): Device to run the model on ('cuda' or 'cpu'). If None, uses CUDA if available.
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.checkpoint_dir = checkpoint_dir
        self.num_classes = len(DNA_ALPHABET)  # A, C, G, T

        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create data loader
        self.data_loader = DNADataLoader(sequence_length=sequence_length, batch_size=batch_size)

        # Build models
        self.generator = Generator(latent_dim, sequence_length, self.num_classes).to(self.device)
        self.discriminator = Discriminator(sequence_length, self.num_classes).to(self.device)

        # Define optimizers
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

        # Define loss function
        self.loss_fn = nn.BCELoss()

        # Training history
        self.history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'epochs': 0
        }

        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _train_step(self, real_sequences: torch.Tensor) -> Tuple[float, float]:
        """
        Perform a single training step.

        Args:
            real_sequences (torch.Tensor): Batch of real DNA sequences.

        Returns:
            Tuple[float, float]: Discriminator loss and generator loss.
        """
        batch_size = real_sequences.size(0)

        # Create labels for real and fake sequences
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)

        # -----------------
        # Train Discriminator
        # -----------------

        # Zero the gradients
        self.discriminator_optimizer.zero_grad()

        # Forward pass with real sequences
        real_output = self.discriminator(real_sequences)
        d_loss_real = self.loss_fn(real_output, real_labels)

        # Generate fake sequences
        noise = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_sequences = self.generator(noise)

        # Forward pass with fake sequences
        fake_output = self.discriminator(fake_sequences.detach())  # Detach to avoid training generator
        d_loss_fake = self.loss_fn(fake_output, fake_labels)

        # Combine losses and update discriminator
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.discriminator_optimizer.step()

        # -----------------
        # Train Generator
        # -----------------

        # Zero the gradients
        self.generator_optimizer.zero_grad()

        # Forward pass with fake sequences through discriminator
        fake_output = self.discriminator(fake_sequences)  # Now use the connected graph

        # Generator wants discriminator to think its outputs are real
        g_loss = self.loss_fn(fake_output, real_labels)
        g_loss.backward()
        self.generator_optimizer.step()

        return d_loss.item(), g_loss.item()

    def load_data_from_dataset(self, dataset: TensorDataset) -> None:
        """
        Load data from a PyTorch dataset.

        Args:
            dataset (TensorDataset): PyTorch dataset containing encoded sequences.
        """
        self.dataset = dataset
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )
        print(f"Loaded dataset with {len(dataset)} samples")

    def load_data(self, file_path: str, file_type: str = 'auto') -> None:
        """
        Load DNA sequence data from a file.

        Args:
            file_path (str): Path to the sequence file.
            file_type (str): Type of file ('fasta', 'csv', or 'auto' to detect from extension).
        """
        try:
            # Load and preprocess data
            self.data_loader.load_and_prepare(file_path, file_type)
            print(f"Loaded data from {file_path}")

            # Create PyTorch dataset and dataloader
            tensor_data = torch.tensor(self.data_loader.encoded_sequences, dtype=torch.float32)
            self.dataset = TensorDataset(tensor_data)
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True
            )
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            print("Generating dummy data instead...")
            self.data_loader.generate_dummy_data(num_sequences=100)
            self.data_loader.preprocess()

            # Create PyTorch dataset and dataloader
            tensor_data = torch.tensor(self.data_loader.encoded_sequences, dtype=torch.float32)
            self.dataset = TensorDataset(tensor_data)
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True
            )

    def train(self, epochs: int = 100, save_interval: int = 10, verbose: bool = True) -> Dict:
        """
        Train the GAN model.

        Args:
            epochs (int): Number of epochs to train.
            save_interval (int): Interval (in epochs) to save checkpoints.
            verbose (bool): Whether to print progress.

        Returns:
            Dict: Training history.
        """
        if not hasattr(self, 'dataloader') or self.dataloader is None:
            raise ValueError("No dataset available. Call load_data() first.")

        # Set models to training mode
        self.generator.train()
        self.discriminator.train()

        start_epoch = self.history['epochs']
        total_epochs = start_epoch + epochs

        for epoch in range(start_epoch + 1, total_epochs + 1):
            start_time = time.time()

            # Initialize metrics for this epoch
            epoch_disc_loss = 0.0
            epoch_gen_loss = 0.0
            num_batches = 0

            # Training loop
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch}/{total_epochs}", disable=not verbose)
            for batch in progress_bar:
                # Move batch to device
                real_sequences = batch[0].to(self.device)

                # Train models
                disc_loss, gen_loss = self._train_step(real_sequences)

                # Update metrics
                epoch_disc_loss += disc_loss
                epoch_gen_loss += gen_loss
                num_batches += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'd_loss': f"{disc_loss:.4f}",
                    'g_loss': f"{gen_loss:.4f}"
                })

            # Calculate average losses
            epoch_disc_loss /= num_batches
            epoch_gen_loss /= num_batches

            # Update history
            self.history['discriminator_loss'].append(epoch_disc_loss)
            self.history['generator_loss'].append(epoch_gen_loss)
            self.history['epochs'] = epoch

            # Print progress
            if verbose:
                print(f"Epoch {epoch}/{total_epochs} - "
                      f"Discriminator Loss: {epoch_disc_loss:.4f}, "
                      f"Generator Loss: {epoch_gen_loss:.4f}, "
                      f"Time: {time.time() - start_time:.2f}s")

            # Save checkpoint
            if epoch % save_interval == 0:
                self.save_checkpoint(epoch)

        return self.history

    def generate(self, num_sequences: int = 10) -> List[str]:
        """
        Generate synthetic DNA sequences.

        Args:
            num_sequences (int): Number of sequences to generate.

        Returns:
            List[str]: List of generated DNA sequences.
        """
        # Set model to evaluation mode
        self.generator.eval()

        # Generate random noise
        noise = torch.randn(num_sequences, self.latent_dim, device=self.device)

        # Generate sequences
        with torch.no_grad():
            generated_sequences = self.generator(noise)

        # Convert to DNA sequences
        dna_sequences = []
        for seq in generated_sequences.cpu().numpy():
            dna_sequences.append(one_hot_decode(seq))

        # Set model back to training mode
        self.generator.train()

        return dna_sequences

    def save_checkpoint(self, epoch: int) -> None:
        """
        Save model checkpoint.

        Args:
            epoch (int): Current epoch.
        """
        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Save generator
        generator_path = os.path.join(self.checkpoint_dir, f"generator_epoch_{epoch}.pt")
        torch.save({
            'model_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.generator_optimizer.state_dict(),
        }, generator_path)

        # Save discriminator
        discriminator_path = os.path.join(self.checkpoint_dir, f"discriminator_epoch_{epoch}.pt")
        torch.save({
            'model_state_dict': self.discriminator.state_dict(),
            'optimizer_state_dict': self.discriminator_optimizer.state_dict(),
        }, discriminator_path)

        # Save history
        history_path = os.path.join(self.checkpoint_dir, "training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.history, f)

        print(f"Checkpoint saved at epoch {epoch}")

    def load_checkpoint(self, epoch: Optional[int] = None) -> None:
        """
        Load model checkpoint.

        Args:
            epoch (Optional[int]): Epoch to load. If None, loads the latest checkpoint.
        """
        if epoch is None:
            # Find the latest checkpoint
            generator_checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.startswith("generator_epoch_")]
            if not generator_checkpoints:
                print("No checkpoints found.")
                return

            # Extract epoch numbers
            epochs = [int(f.split("_")[-1].split(".")[0]) for f in generator_checkpoints]
            epoch = max(epochs)

        # Load generator
        generator_path = os.path.join(self.checkpoint_dir, f"generator_epoch_{epoch}.pt")
        if os.path.exists(generator_path):
            checkpoint = torch.load(generator_path, map_location=self.device)
            self.generator.load_state_dict(checkpoint['model_state_dict'])
            self.generator_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Generator loaded from {generator_path}")
        else:
            print(f"Generator checkpoint not found: {generator_path}")

        # Load discriminator
        discriminator_path = os.path.join(self.checkpoint_dir, f"discriminator_epoch_{epoch}.pt")
        if os.path.exists(discriminator_path):
            checkpoint = torch.load(discriminator_path, map_location=self.device)
            self.discriminator.load_state_dict(checkpoint['model_state_dict'])
            self.discriminator_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Discriminator loaded from {discriminator_path}")
        else:
            print(f"Discriminator checkpoint not found: {discriminator_path}")

        # Load history
        history_path = os.path.join(self.checkpoint_dir, "training_history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.history = json.load(f)
            print(f"Training history loaded from {history_path}")
        else:
            print(f"Training history not found: {history_path}")

    def plot_training_history(self) -> plt.Figure:
        """
        Plot training history.

        Returns:
            plt.Figure: Matplotlib figure with training history.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        epochs = range(1, len(self.history['generator_loss']) + 1)
        ax.plot(epochs, self.history['generator_loss'], 'b-', label='Generator Loss')
        ax.plot(epochs, self.history['discriminator_loss'], 'r-', label='Discriminator Loss')

        ax.set_title('Training History')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True)

        return fig
