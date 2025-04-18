"""
Training utilities for DNA sequence generation using Gumbel-Softmax GAN.

This module provides functions for training the GAN model and evaluating its performance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score

from .models import Generator, CNNDiscriminator, LSTMDiscriminator
from .metrics import calculate_gc_content, calculate_morans_i, calculate_diversity


def train_gan(generator, discriminator, data_loader, num_epochs=500,
              noise_dim=100, batch_size=64, device='cuda',
              lr_g=1e-4, lr_d=1e-4, checkpoint_dir='checkpoints',
              log_interval=10, save_interval=10, temperature=1.0,
              start_epoch=0, history=None):
    """
    Train the GAN model.

    Args:
        generator (nn.Module): Generator model.
        discriminator (nn.Module): Discriminator model.
        data_loader (DataLoader): DataLoader for real DNA sequences.
        num_epochs (int): Number of training epochs.
        noise_dim (int): Dimension of the input noise vector.
        batch_size (int): Batch size.
        device (str): Device to use for training ('cuda' or 'cpu').
        lr_g (float): Learning rate for the generator.
        lr_d (float): Learning rate for the discriminator.
        checkpoint_dir (str): Directory to save checkpoints.
        log_interval (int): Interval for logging training progress.
        save_interval (int): Interval for saving checkpoints.
        temperature (float): Temperature parameter for Gumbel-Softmax.
        start_epoch (int): Epoch to start training from (for resuming training).
        history (dict): Training history (for resuming training).

    Returns:
        dict: Dictionary containing training history.
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(checkpoint_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )

    # Set up optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

    # Loss function
    criterion = nn.BCELoss()

    # Training history
    if history is None:
        history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'discriminator_accuracy': [],
            'gc_content': [],
            'morans_i': [],
            'diversity': []
        }

    # Training loop
    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_start_time = time.time()

        # Initialize metrics for this epoch
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_d_acc = 0.0
        epoch_gc_content = 0.0
        epoch_morans_i = 0.0
        epoch_diversity = 0.0

        # Progress bar
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, real_sequences in enumerate(pbar):
            # Move data to device
            real_sequences = real_sequences.to(device)
            batch_size = real_sequences.size(0)

            # Create labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_d.zero_grad()

            # Train with real sequences
            real_outputs = discriminator(real_sequences)
            d_loss_real = criterion(real_outputs, real_labels)

            # Train with fake sequences
            noise = torch.randn(batch_size, noise_dim).to(device)
            fake_sequences = generator(noise, temperature=temperature, hard=False)
            fake_outputs = discriminator(fake_sequences.detach())
            d_loss_fake = criterion(fake_outputs, fake_labels)

            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            # Calculate discriminator accuracy
            real_preds = (real_outputs > 0.5).float()
            fake_preds = (fake_outputs < 0.5).float()
            d_acc = (torch.sum(real_preds) + torch.sum(fake_preds)) / (2 * batch_size)

            # ---------------------
            # Train Generator
            # ---------------------
            optimizer_g.zero_grad()

            # Generate fake sequences
            fake_sequences = generator(noise, temperature=temperature, hard=False)
            fake_outputs = discriminator(fake_sequences)

            # Generator loss
            g_loss = criterion(fake_outputs, real_labels)
            g_loss.backward()
            optimizer_g.step()

            # Update metrics
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_d_acc += d_acc.item()

            # Update progress bar
            pbar.set_postfix({
                'g_loss': g_loss.item(),
                'd_loss': d_loss.item(),
                'd_acc': d_acc.item()
            })

        # Calculate average metrics for this epoch
        epoch_g_loss /= len(data_loader)
        epoch_d_loss /= len(data_loader)
        epoch_d_acc /= len(data_loader)

        # Generate sequences for evaluation
        with torch.no_grad():
            eval_noise = torch.randn(100, noise_dim).to(device)
            eval_sequences = generator(eval_noise, temperature=1.0, hard=True)

            # Calculate biological metrics
            epoch_gc_content = calculate_gc_content(eval_sequences)
            epoch_morans_i = calculate_morans_i(eval_sequences)
            epoch_diversity = calculate_diversity(eval_sequences)

        # Update history
        history['generator_loss'].append(epoch_g_loss)
        history['discriminator_loss'].append(epoch_d_loss)
        history['discriminator_accuracy'].append(epoch_d_acc)
        history['gc_content'].append(epoch_gc_content)
        history['morans_i'].append(epoch_morans_i)
        history['diversity'].append(epoch_diversity)

        # Log progress
        epoch_time = time.time() - epoch_start_time
        if (epoch + 1) % log_interval == 0:
            logging.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Time: {epoch_time:.2f}s | "
                f"G Loss: {epoch_g_loss:.4f} | "
                f"D Loss: {epoch_d_loss:.4f} | "
                f"D Acc: {epoch_d_acc:.4f} | "
                f"GC: {epoch_gc_content:.4f} | "
                f"Moran's I: {epoch_morans_i:.4f} | "
                f"Diversity: {epoch_diversity:.4f}"
            )

        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
                'g_loss': epoch_g_loss,
                'd_loss': epoch_d_loss,
                'd_acc': epoch_d_acc,
                'gc_content': epoch_gc_content,
                'morans_i': epoch_morans_i,
                'diversity': epoch_diversity,
                'history': history
            }, checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")

    # Save final model
    final_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save({
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'history': history
    }, final_path)
    logging.info(f"Final model saved to {final_path}")

    return history


def plot_training_history(history, save_path=None):
    """
    Plot training history.

    Args:
        history (dict): Dictionary containing training history.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(15, 10))

    # Plot losses
    plt.subplot(2, 2, 1)
    plt.plot(history['generator_loss'], label='Generator Loss')
    plt.plot(history['discriminator_loss'], label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator and Discriminator Loss')
    plt.legend()
    plt.grid(True)

    # Plot discriminator accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['discriminator_accuracy'], label='Discriminator Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Discriminator Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot GC content
    plt.subplot(2, 2, 3)
    plt.plot(history['gc_content'], label='GC Content')
    plt.xlabel('Epoch')
    plt.ylabel('GC Content')
    plt.title('GC Content')
    plt.legend()
    plt.grid(True)

    # Plot Moran's I and diversity
    plt.subplot(2, 2, 4)
    plt.plot(history['morans_i'], label="Moran's I")
    plt.plot(history['diversity'], label='Diversity')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title("Moran's I and Diversity")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")

    plt.show()


def load_checkpoint(checkpoint_path, generator, discriminator, device):
    """
    Load a checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint.
        generator (nn.Module): Generator model.
        discriminator (nn.Module): Discriminator model.
        device (str): Device to load the models on.

    Returns:
        tuple: Tuple containing the loaded generator, discriminator, and history.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    generator.to(device)
    discriminator.to(device)

    return generator, discriminator, checkpoint.get('history', {})
