"""
DNA Sequence GAN Web Interface

This module provides a web interface for visualizing DNA sequences and GAN results.
"""

import os
import json
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for
import plotly
import pandas as pd
from werkzeug.utils import secure_filename

from data_loader import DNADataLoader
from gan_model import DNAGAN
from visualization import DNAVisualizer
from evaluation import DNAEvaluator
from dna_utils import pad_sequences

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'data'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store data and model
data_loader = None
gan_model = None
real_sequences = []
synthetic_sequences = []
evaluation_results = {}

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload."""
    global data_loader, real_sequences

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Initialize data loader
        sequence_length = request.form.get('sequence_length', type=int)
        data_loader = DNADataLoader(sequence_length=sequence_length)

        # Load data
        try:
            if filename.endswith(('.fasta', '.fa')):
                real_sequences = data_loader.load_fasta(file_path)
            elif filename.endswith('.csv'):
                real_sequences = data_loader.load_csv(file_path)
            else:
                return jsonify({'error': 'Unsupported file format'}), 400

            # Preprocess data
            data_loader.preprocess()

            return jsonify({
                'success': True,
                'message': f'Loaded {len(real_sequences)} sequences',
                'sequence_count': len(real_sequences),
                'sequence_length': data_loader.sequence_length
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    return jsonify({'error': 'Failed to upload file'}), 500

@app.route('/generate_dummy', methods=['POST'])
def generate_dummy():
    """Generate dummy data for testing."""
    global data_loader, real_sequences

    num_sequences = request.form.get('num_sequences', 100, type=int)
    min_length = request.form.get('min_length', 50, type=int)
    max_length = request.form.get('max_length', 100, type=int)

    # Initialize data loader
    sequence_length = request.form.get('sequence_length', None, type=int)
    data_loader = DNADataLoader(sequence_length=sequence_length)

    # Generate dummy data
    real_sequences = data_loader.generate_dummy_data(
        num_sequences=num_sequences,
        min_length=min_length,
        max_length=max_length
    )

    # Preprocess data
    data_loader.preprocess()

    return jsonify({
        'success': True,
        'message': f'Generated {len(real_sequences)} dummy sequences',
        'sequence_count': len(real_sequences),
        'sequence_length': data_loader.sequence_length
    })

@app.route('/initialize_gan', methods=['POST'])
def initialize_gan():
    """Initialize the GAN model."""
    global gan_model, data_loader

    if data_loader is None:
        return jsonify({'error': 'No data loaded. Please upload data first.'}), 400

    # Get parameters
    batch_size = request.form.get('batch_size', 32, type=int)
    latent_dim = request.form.get('latent_dim', 100, type=int)

    # Get device preference
    device = request.form.get('device', None)

    # Initialize GAN
    gan_model = DNAGAN(
        sequence_length=data_loader.sequence_length,
        batch_size=batch_size,
        latent_dim=latent_dim,
        device=device
    )

    # Create dataset and load it into the model
    dataset = data_loader.create_dataset()
    gan_model.load_data_from_dataset(dataset)

    return jsonify({
        'success': True,
        'message': 'GAN model initialized',
        'sequence_length': data_loader.sequence_length,
        'batch_size': batch_size,
        'latent_dim': latent_dim,
        'device': gan_model.device
    })

@app.route('/train_gan', methods=['POST'])
def train_gan():
    """Train the GAN model."""
    global gan_model

    if gan_model is None:
        return jsonify({'error': 'GAN model not initialized. Please initialize the model first.'}), 400

    # Get parameters
    epochs = request.form.get('epochs', 10, type=int)
    save_interval = request.form.get('save_interval', 5, type=int)

    # Train the model
    try:
        history = gan_model.train(epochs=epochs, save_interval=save_interval)

        return jsonify({
            'success': True,
            'message': f'Trained for {epochs} epochs',
            'history': {
                'generator_loss': [float(x) for x in history['generator_loss']],
                'discriminator_loss': [float(x) for x in history['discriminator_loss']],
                'epochs': history['epochs']
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/load_checkpoint', methods=['POST'])
def load_checkpoint():
    """Load a model checkpoint."""
    global gan_model

    if gan_model is None:
        return jsonify({'error': 'GAN model not initialized. Please initialize the model first.'}), 400

    # Get parameters
    epoch = request.form.get('epoch', None, type=int)

    # Load checkpoint
    try:
        gan_model.load_checkpoint(epoch)

        return jsonify({
            'success': True,
            'message': f'Loaded checkpoint from epoch {epoch if epoch else "latest"}',
            'history': {
                'generator_loss': [float(x) for x in gan_model.history['generator_loss']],
                'discriminator_loss': [float(x) for x in gan_model.history['discriminator_loss']],
                'epochs': gan_model.history['epochs']
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_sequences', methods=['POST'])
def generate_sequences():
    """Generate synthetic sequences."""
    global gan_model, synthetic_sequences

    if gan_model is None:
        return jsonify({'error': 'GAN model not initialized. Please initialize the model first.'}), 400

    # Get parameters
    num_sequences = request.form.get('num_sequences', 10, type=int)

    # Generate sequences
    try:
        synthetic_sequences = gan_model.generate(num_sequences=num_sequences)

        return jsonify({
            'success': True,
            'message': f'Generated {len(synthetic_sequences)} sequences',
            'sequence_count': len(synthetic_sequences),
            'sequences': synthetic_sequences[:10]  # Send first 10 for preview
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate():
    """Evaluate synthetic sequences."""
    global real_sequences, synthetic_sequences, evaluation_results

    if not real_sequences:
        return jsonify({'error': 'No real sequences available. Please upload data first.'}), 400

    if not synthetic_sequences:
        return jsonify({'error': 'No synthetic sequences available. Please generate sequences first.'}), 400

    # Ensure sequences have the same length
    if len(set(len(seq) for seq in real_sequences)) > 1 or len(set(len(seq) for seq in synthetic_sequences)) > 1:
        # Pad sequences to the maximum length
        max_length = max(max(len(seq) for seq in real_sequences), max(len(seq) for seq in synthetic_sequences))
        real_sequences_padded = pad_sequences(real_sequences, max_length)
        synthetic_sequences_padded = pad_sequences(synthetic_sequences, max_length)
    else:
        real_sequences_padded = real_sequences
        synthetic_sequences_padded = synthetic_sequences

    # Evaluate sequences
    try:
        evaluation_results = DNAEvaluator.comprehensive_evaluation(
            real_sequences_padded, synthetic_sequences_padded
        )

        # Generate report
        report = DNAEvaluator.generate_report(evaluation_results)

        return jsonify({
            'success': True,
            'message': 'Evaluation completed',
            'report': report,
            'results': evaluation_results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/visualize_sequences', methods=['GET'])
def visualize_sequences():
    """Visualize real and synthetic sequences."""
    global real_sequences, synthetic_sequences

    if not real_sequences:
        return jsonify({'error': 'No real sequences available. Please upload data first.'}), 400

    # Create visualizations
    try:
        # Limit to first 10 sequences for visualization
        real_preview = real_sequences[:10]

        # Create sequence viewer
        seq_viewer = DNAVisualizer.plotly_sequence_viewer(
            real_preview,
            title="Real DNA Sequences"
        )

        # Create k-mer distribution
        kmer_dist = DNAVisualizer.plotly_kmer_distribution(
            real_sequences,
            k=3,
            title="3-mer Distribution in Real Sequences"
        )

        # Create GC content distribution
        gc_dist = DNAVisualizer.plotly_gc_content_comparison(
            real_sequences,
            real_sequences,  # Using same data for both to avoid errors if no synthetic data
            title="GC Content in Real Sequences"
        )

        # Convert to JSON
        seq_viewer_json = json.dumps(seq_viewer, cls=plotly.utils.PlotlyJSONEncoder)
        kmer_dist_json = json.dumps(kmer_dist, cls=plotly.utils.PlotlyJSONEncoder)
        gc_dist_json = json.dumps(gc_dist, cls=plotly.utils.PlotlyJSONEncoder)

        return jsonify({
            'success': True,
            'seq_viewer': seq_viewer_json,
            'kmer_dist': kmer_dist_json,
            'gc_dist': gc_dist_json,
            'has_synthetic': len(synthetic_sequences) > 0
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/visualize_comparison', methods=['GET'])
def visualize_comparison():
    """Visualize comparison between real and synthetic sequences."""
    global real_sequences, synthetic_sequences

    if not real_sequences:
        return jsonify({'error': 'No real sequences available. Please upload data first.'}), 400

    if not synthetic_sequences:
        return jsonify({'error': 'No synthetic sequences available. Please generate sequences first.'}), 400

    # Create visualizations
    try:
        # Limit to first 5 sequences of each for visualization
        real_preview = real_sequences[:5]
        synth_preview = synthetic_sequences[:5]

        # Create sequence viewers
        real_viewer = DNAVisualizer.plotly_sequence_viewer(
            real_preview,
            title="Real DNA Sequences"
        )

        synth_viewer = DNAVisualizer.plotly_sequence_viewer(
            synth_preview,
            title="Synthetic DNA Sequences"
        )

        # Create k-mer distribution comparison
        kmer_comp = DNAVisualizer.plotly_compare_kmer_distributions(
            real_sequences,
            synthetic_sequences,
            k=3,
            title="3-mer Distribution Comparison"
        )

        # Create GC content comparison
        gc_comp = DNAVisualizer.plotly_gc_content_comparison(
            real_sequences,
            synthetic_sequences,
            title="GC Content Comparison"
        )

        # Convert to JSON
        real_viewer_json = json.dumps(real_viewer, cls=plotly.utils.PlotlyJSONEncoder)
        synth_viewer_json = json.dumps(synth_viewer, cls=plotly.utils.PlotlyJSONEncoder)
        kmer_comp_json = json.dumps(kmer_comp, cls=plotly.utils.PlotlyJSONEncoder)
        gc_comp_json = json.dumps(gc_comp, cls=plotly.utils.PlotlyJSONEncoder)

        return jsonify({
            'success': True,
            'real_viewer': real_viewer_json,
            'synth_viewer': synth_viewer_json,
            'kmer_comp': kmer_comp_json,
            'gc_comp': gc_comp_json
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/training_history', methods=['GET'])
def training_history():
    """Get training history."""
    global gan_model

    if gan_model is None:
        return jsonify({'error': 'GAN model not initialized. Please initialize the model first.'}), 400

    # Create visualization
    try:
        history_plot = DNAVisualizer.plotly_training_history(
            gan_model.history,
            title="GAN Training History"
        )

        # Convert to JSON
        history_json = json.dumps(history_plot, cls=plotly.utils.PlotlyJSONEncoder)

        return jsonify({
            'success': True,
            'history_plot': history_json,
            'history_data': {
                'generator_loss': [float(x) for x in gan_model.history['generator_loss']],
                'discriminator_loss': [float(x) for x in gan_model.history['discriminator_loss']],
                'epochs': gan_model.history['epochs']
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_sequences', methods=['GET'])
def download_sequences():
    """Download synthetic sequences as CSV."""
    global synthetic_sequences

    if not synthetic_sequences:
        return jsonify({'error': 'No synthetic sequences available. Please generate sequences first.'}), 400

    try:
        # Create DataFrame
        df = pd.DataFrame({
            'sequence_id': range(len(synthetic_sequences)),
            'sequence': synthetic_sequences
        })

        # Save to CSV
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], 'synthetic_sequences.csv')
        df.to_csv(csv_path, index=False)

        return jsonify({
            'success': True,
            'download_url': url_for('static', filename='data/synthetic_sequences.csv')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
