"""
Training utilities for DNA sequence generation using Wasserstein GAN with gradient penalty.

This module provides functions for training the improved GAN model and evaluating its performance.
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

from .models import ImprovedGenerator, ImprovedDiscriminator
from .metrics import calculate_gc_content, calculate_morans_i, calculate_diversity


def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    """
    Compute gradient penalty for WGAN-GP.

    Args:
        discriminator (nn.Module): Discriminator model.
        real_samples (torch.Tensor): Real DNA sequences.
        fake_samples (torch.Tensor): Generated DNA sequences.
        device (torch.device): Device to use for computation.

    Returns:
        torch.Tensor: Gradient penalty.
    """
    # Random weight for interpolation
    alpha = torch.rand(real_samples.size(0), 1, 1).to(device)

    # Interpolated samples
    interpolated = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)

    # Discriminator output for interpolated samples
    d_interpolated = discriminator(interpolated)

    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Compute gradient penalty
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()

    return gradient_penalty


def train_wgan_gp(generator, discriminator, data_loader, num_epochs=500,
                 noise_dim=100, batch_size=64, device='cuda',
                 lr_g=1e-4, lr_d=1e-4, checkpoint_dir='checkpoints',
                 log_interval=10, save_interval=10, temperature=1.0,
                 start_epoch=0, history=None, n_critic=5, lambda_gp=10):
    """
    Train the WGAN-GP model.

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
        n_critic (int): Number of discriminator updates per generator update.
        lambda_gp (float): Weight for gradient penalty.

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

    # Training history
    if history is None:
        history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'wasserstein_distance': [],
            'gradient_penalty': [],
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
        epoch_wasserstein_distance = 0.0
        epoch_gradient_penalty = 0.0
        epoch_gc_content = 0.0
        epoch_morans_i = 0.0
        epoch_diversity = 0.0

        # Progress bar
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{start_epoch + num_epochs}")

        for batch_idx, real_sequences in enumerate(pbar):
            # Move data to device
            real_sequences = real_sequences.to(device)
            batch_size = real_sequences.size(0)

            # ---------------------
            # Train Discriminator
            # ---------------------
            for _ in range(n_critic):
                optimizer_d.zero_grad()

                # Train with real sequences
                real_validity = discriminator(real_sequences)

                # Train with fake sequences
                noise = torch.randn(batch_size, noise_dim).to(device)
                fake_sequences = generator(noise, temperature=temperature, hard=False)
                fake_validity = discriminator(fake_sequences.detach())

                # Compute Wasserstein distance
                wasserstein_distance = real_validity.mean() - fake_validity.mean()

                # Compute gradient penalty
                gradient_penalty = compute_gradient_penalty(
                    discriminator, real_sequences, fake_sequences.detach(), device
                )

                # Total discriminator loss
                d_loss = -wasserstein_distance + lambda_gp * gradient_penalty
                d_loss.backward()
                optimizer_d.step()

                # Update metrics
                epoch_d_loss += d_loss.item()
                epoch_wasserstein_distance += wasserstein_distance.item()
                epoch_gradient_penalty += gradient_penalty.item()

            # ---------------------
            # Train Generator
            # ---------------------
            optimizer_g.zero_grad()

            # Generate fake sequences
            fake_sequences = generator(noise, temperature=temperature, hard=False)
            fake_validity = discriminator(fake_sequences)

            # Generator loss
            g_loss = -fake_validity.mean()
            g_loss.backward()
            optimizer_g.step()

            # Update metrics
            epoch_g_loss += g_loss.item()

            # Update progress bar
            pbar.set_postfix({
                'g_loss': g_loss.item(),
                'd_loss': d_loss.item(),
                'w_dist': wasserstein_distance.item()
            })

        # Calculate average metrics for this epoch
        num_batches = len(data_loader)
        epoch_g_loss /= num_batches
        epoch_d_loss /= (num_batches * n_critic)
        epoch_wasserstein_distance /= (num_batches * n_critic)
        epoch_gradient_penalty /= (num_batches * n_critic)

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
        history['wasserstein_distance'].append(epoch_wasserstein_distance)
        history['gradient_penalty'].append(epoch_gradient_penalty)
        history['gc_content'].append(epoch_gc_content)
        history['morans_i'].append(epoch_morans_i)
        history['diversity'].append(epoch_diversity)

        # Log progress
        epoch_time = time.time() - epoch_start_time
        if (epoch + 1) % log_interval == 0:
            logging.info(
                f"Epoch {epoch+1}/{start_epoch + num_epochs} | "
                f"Time: {epoch_time:.2f}s | "
                f"G Loss: {epoch_g_loss:.4f} | "
                f"D Loss: {epoch_d_loss:.4f} | "
                f"W Dist: {epoch_wasserstein_distance:.4f} | "
                f"GP: {epoch_gradient_penalty:.4f} | "
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
                'wasserstein_distance': epoch_wasserstein_distance,
                'gradient_penalty': epoch_gradient_penalty,
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
    plt.figure(figsize=(15, 12))

    # Plot losses
    plt.subplot(3, 2, 1)
    plt.plot(history['generator_loss'], label='Generator Loss')
    plt.plot(history['discriminator_loss'], label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator and Discriminator Loss')
    plt.legend()
    plt.grid(True)

    # Plot Wasserstein distance
    plt.subplot(3, 2, 2)
    plt.plot(history['wasserstein_distance'], label='Wasserstein Distance')
    plt.xlabel('Epoch')
    plt.ylabel('Distance')
    plt.title('Wasserstein Distance')
    plt.legend()
    plt.grid(True)

    # Plot gradient penalty
    plt.subplot(3, 2, 3)
    plt.plot(history['gradient_penalty'], label='Gradient Penalty')
    plt.xlabel('Epoch')
    plt.ylabel('Penalty')
    plt.title('Gradient Penalty')
    plt.legend()
    plt.grid(True)

    # Plot GC content
    plt.subplot(3, 2, 4)
    plt.plot(history['gc_content'], label='GC Content')
    plt.xlabel('Epoch')
    plt.ylabel('GC Content')
    plt.title('GC Content')
    plt.legend()
    plt.grid(True)

    # Plot Moran's I
    plt.subplot(3, 2, 5)
    plt.plot(history['morans_i'], label="Moran's I")
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title("Moran's I")
    plt.legend()
    plt.grid(True)

    # Plot diversity
    plt.subplot(3, 2, 6)
    plt.plot(history['diversity'], label='Diversity')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Diversity')
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
