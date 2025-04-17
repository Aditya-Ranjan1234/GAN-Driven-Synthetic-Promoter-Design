"""
DNA Sequence GAN - Streamlit Web Application

This is the main file for the Streamlit web application for DNA Sequence GAN.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from PIL import Image
import sys
import time

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from data_loader import DNADataLoader
from gan_model import DNAGAN
from visualization import DNAVisualizer
from evaluation import DNAEvaluator
from dna_utils import pad_sequences

# Set page configuration
st.set_page_config(
    page_title="DNA Sequence GAN",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = None
if 'gan_model' not in st.session_state:
    st.session_state.gan_model = None
if 'real_sequences' not in st.session_state:
    st.session_state.real_sequences = []
if 'synthetic_sequences' not in st.session_state:
    st.session_state.synthetic_sequences = []
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = {}
if 'training_history' not in st.session_state:
    st.session_state.training_history = {}

# Create a sidebar for navigation
st.sidebar.title("DNA Sequence GAN")
st.sidebar.image("web_app/static/dna_icon.png", width=100)

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Data Management", "Model Training", "Sequence Generation", "Evaluation", "Visualization"]
)

# Display status in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Status")
data_status = "✅ Loaded" if st.session_state.data_loader is not None else "❌ Not loaded"
model_status = "✅ Initialized" if st.session_state.gan_model is not None else "❌ Not initialized"
training_status = "✅ Trained" if st.session_state.gan_model is not None and len(st.session_state.gan_model.history['generator_loss']) > 0 else "❌ Not trained"
generation_status = "✅ Generated" if len(st.session_state.synthetic_sequences) > 0 else "❌ Not generated"

st.sidebar.markdown(f"**Data:** {data_status}")
st.sidebar.markdown(f"**Model:** {model_status}")
st.sidebar.markdown(f"**Training:** {training_status}")
st.sidebar.markdown(f"**Generation:** {generation_status}")

# Home page
if page == "Home":
    st.title("DNA Sequence GAN")
    st.markdown("### Generate synthetic DNA sequences using Generative Adversarial Networks")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        This application allows you to:
        
        1. **Load DNA sequence data** from FASTA or CSV files
        2. **Train a GAN model** to generate synthetic DNA sequences
        3. **Generate synthetic sequences** that mimic real biological sequences
        4. **Evaluate the quality** of synthetic sequences
        5. **Visualize and compare** real and synthetic sequences
        
        Use the sidebar to navigate between different sections of the application.
        """)
        
        st.markdown("### Getting Started")
        st.markdown("""
        1. Go to the **Data Management** page to load or generate data
        2. Initialize and train the model in the **Model Training** page
        3. Generate synthetic sequences in the **Sequence Generation** page
        4. Evaluate the results in the **Evaluation** page
        5. Create visualizations in the **Visualization** page
        """)
    
    with col2:
        st.image("web_app/static/dna_gan.png", caption="DNA Sequence GAN")
        
        st.markdown("### About")
        st.markdown("""
        This application uses PyTorch to implement a Generative Adversarial Network (GAN) for DNA sequence generation.
        
        The GAN consists of two neural networks:
        - A **Generator** that creates synthetic DNA sequences
        - A **Discriminator** that tries to distinguish between real and synthetic sequences
        
        Through adversarial training, the generator learns to produce increasingly realistic DNA sequences.
        """)

# Data Management page
elif page == "Data Management":
    st.title("Data Management")
    
    tab1, tab2 = st.tabs(["Upload Data", "Generate Dummy Data"])
    
    with tab1:
        st.header("Upload DNA Sequence Data")
        
        uploaded_file = st.file_uploader("Upload FASTA or CSV file", type=["fasta", "fa", "csv"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            sequence_length = st.number_input("Sequence Length (leave blank for auto)", min_value=10, value=None)
            
        with col2:
            batch_size = st.number_input("Batch Size", min_value=1, value=32)
        
        if st.button("Upload and Process"):
            if uploaded_file is not None:
                # Save uploaded file temporarily
                with open(os.path.join("data", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Create data loader
                st.session_state.data_loader = DNADataLoader(sequence_length=sequence_length, batch_size=batch_size)
                
                # Load data
                try:
                    if uploaded_file.name.endswith((".fasta", ".fa")):
                        st.session_state.real_sequences = st.session_state.data_loader.load_fasta(os.path.join("data", uploaded_file.name))
                    elif uploaded_file.name.endswith(".csv"):
                        st.session_state.real_sequences = st.session_state.data_loader.load_csv(os.path.join("data", uploaded_file.name))
                    
                    # Preprocess data
                    st.session_state.data_loader.preprocess()
                    
                    st.success(f"Successfully loaded {len(st.session_state.real_sequences)} sequences")
                    
                    # Display sample sequences
                    st.subheader("Sample Sequences")
                    for i, seq in enumerate(st.session_state.real_sequences[:5]):
                        st.code(f"Sequence {i+1}: {seq}")
                
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
            else:
                st.error("Please upload a file")
    
    with tab2:
        st.header("Generate Dummy Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_sequences = st.number_input("Number of Sequences", min_value=1, value=100)
            
        with col2:
            min_length = st.number_input("Minimum Length", min_value=10, value=50)
            
        with col3:
            max_length = st.number_input("Maximum Length", min_value=10, value=100)
        
        sequence_length_dummy = st.number_input("Fixed Sequence Length (leave blank for auto)", min_value=10, value=None)
        batch_size_dummy = st.number_input("Batch Size for Dummy Data", min_value=1, value=32)
        
        if st.button("Generate Dummy Data"):
            # Create data loader
            st.session_state.data_loader = DNADataLoader(sequence_length=sequence_length_dummy, batch_size=batch_size_dummy)
            
            # Generate dummy data
            st.session_state.real_sequences = st.session_state.data_loader.generate_dummy_data(
                num_sequences=num_sequences,
                min_length=min_length,
                max_length=max_length
            )
            
            # Preprocess data
            st.session_state.data_loader.preprocess()
            
            st.success(f"Successfully generated {len(st.session_state.real_sequences)} dummy sequences")
            
            # Display sample sequences
            st.subheader("Sample Sequences")
            for i, seq in enumerate(st.session_state.real_sequences[:5]):
                st.code(f"Sequence {i+1}: {seq}")
    
    # Display data statistics if data is loaded
    if st.session_state.data_loader is not None and len(st.session_state.real_sequences) > 0:
        st.markdown("---")
        st.header("Data Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Number of Sequences", len(st.session_state.real_sequences))
            
        with col2:
            if st.session_state.data_loader.sequence_length is not None:
                st.metric("Sequence Length", st.session_state.data_loader.sequence_length)
            else:
                lengths = [len(seq) for seq in st.session_state.real_sequences]
                st.metric("Average Sequence Length", round(sum(lengths) / len(lengths), 2))
            
        with col3:
            st.metric("Batch Size", st.session_state.data_loader.batch_size)
        
        # Display sequence length distribution
        if len(set(len(seq) for seq in st.session_state.real_sequences)) > 1:
            st.subheader("Sequence Length Distribution")
            lengths = [len(seq) for seq in st.session_state.real_sequences]
            fig = go.Figure(data=[go.Histogram(x=lengths, nbinsx=20)])
            fig.update_layout(
                title="Sequence Length Distribution",
                xaxis_title="Length",
                yaxis_title="Count"
            )
            st.plotly_chart(fig)
        
        # Display nucleotide composition
        st.subheader("Nucleotide Composition")
        all_nucleotides = ''.join(st.session_state.real_sequences)
        nucleotide_counts = {
            'A': all_nucleotides.count('A'),
            'C': all_nucleotides.count('C'),
            'G': all_nucleotides.count('G'),
            'T': all_nucleotides.count('T')
        }
        
        fig = go.Figure(data=[go.Pie(
            labels=list(nucleotide_counts.keys()),
            values=list(nucleotide_counts.values()),
            hole=.3,
            marker_colors=['green', 'blue', 'orange', 'red']
        )])
        fig.update_layout(title="Nucleotide Composition")
        st.plotly_chart(fig)

# Model Training page
elif page == "Model Training":
    st.title("Model Training")
    
    if st.session_state.data_loader is None or len(st.session_state.real_sequences) == 0:
        st.warning("Please load data first in the Data Management page")
    else:
        tab1, tab2, tab3 = st.tabs(["Initialize Model", "Train Model", "Load Checkpoint"])
        
        with tab1:
            st.header("Initialize GAN Model")
            
            col1, col2 = st.columns(2)
            
            with col1:
                latent_dim = st.number_input("Latent Dimension", min_value=10, value=100)
                
            with col2:
                device = st.selectbox("Device", ["cuda" if torch.cuda.is_available() else "cpu", "cpu"])
            
            if st.button("Initialize Model"):
                # Initialize GAN
                st.session_state.gan_model = DNAGAN(
                    sequence_length=st.session_state.data_loader.sequence_length,
                    batch_size=st.session_state.data_loader.batch_size,
                    latent_dim=latent_dim,
                    device=device
                )
                
                # Create dataset and load it into the model
                dataset = st.session_state.data_loader.create_dataset()
                st.session_state.gan_model.load_data_from_dataset(dataset)
                
                st.success("Successfully initialized GAN model")
                
                # Display model architecture
                st.subheader("Generator Architecture")
                st.code(str(st.session_state.gan_model.generator))
                
                st.subheader("Discriminator Architecture")
                st.code(str(st.session_state.gan_model.discriminator))
        
        with tab2:
            st.header("Train GAN Model")
            
            if st.session_state.gan_model is None:
                st.warning("Please initialize the model first")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    epochs = st.number_input("Number of Epochs", min_value=1, value=20)
                    
                with col2:
                    save_interval = st.number_input("Save Interval (epochs)", min_value=1, value=5)
                
                if st.button("Train Model"):
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Create placeholders for loss plots
                    loss_plot = st.empty()
                    
                    # Training history
                    history = {
                        'generator_loss': [],
                        'discriminator_loss': [],
                        'epochs': []
                    }
                    
                    # Train the model
                    start_epoch = st.session_state.gan_model.history['epochs']
                    total_epochs = start_epoch + epochs
                    
                    # Set models to training mode
                    st.session_state.gan_model.generator.train()
                    st.session_state.gan_model.discriminator.train()
                    
                    for epoch in range(start_epoch + 1, total_epochs + 1):
                        start_time = time.time()
                        
                        # Initialize metrics for this epoch
                        epoch_disc_loss = 0.0
                        epoch_gen_loss = 0.0
                        num_batches = 0
                        
                        # Training loop
                        for batch in st.session_state.gan_model.dataloader:
                            # Move batch to device
                            real_sequences = batch[0].to(st.session_state.gan_model.device)
                            
                            # Train models
                            disc_loss, gen_loss = st.session_state.gan_model._train_step(real_sequences)
                            
                            # Update metrics
                            epoch_disc_loss += disc_loss
                            epoch_gen_loss += gen_loss
                            num_batches += 1
                        
                        # Calculate average losses
                        epoch_disc_loss /= num_batches
                        epoch_gen_loss /= num_batches
                        
                        # Update history
                        st.session_state.gan_model.history['discriminator_loss'].append(epoch_disc_loss)
                        st.session_state.gan_model.history['generator_loss'].append(epoch_gen_loss)
                        st.session_state.gan_model.history['epochs'] = epoch
                        
                        # Update local history for plotting
                        history['generator_loss'].append(epoch_gen_loss)
                        history['discriminator_loss'].append(epoch_disc_loss)
                        history['epochs'].append(epoch)
                        
                        # Update progress
                        progress = (epoch - start_epoch) / epochs
                        progress_bar.progress(progress)
                        status_text.text(f"Epoch {epoch}/{total_epochs} - Discriminator Loss: {epoch_disc_loss:.4f}, Generator Loss: {epoch_gen_loss:.4f}, Time: {time.time() - start_time:.2f}s")
                        
                        # Update loss plot
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=history['epochs'],
                            y=history['generator_loss'],
                            mode='lines',
                            name='Generator Loss'
                        ))
                        fig.add_trace(go.Scatter(
                            x=history['epochs'],
                            y=history['discriminator_loss'],
                            mode='lines',
                            name='Discriminator Loss'
                        ))
                        fig.update_layout(
                            title="Training Loss",
                            xaxis_title="Epoch",
                            yaxis_title="Loss",
                            legend_title="Loss Type"
                        )
                        loss_plot.plotly_chart(fig)
                        
                        # Save checkpoint
                        if epoch % save_interval == 0:
                            st.session_state.gan_model.save_checkpoint(epoch)
                    
                    # Complete progress bar
                    progress_bar.progress(1.0)
                    status_text.text("Training completed!")
                    
                    st.success(f"Successfully trained model for {epochs} epochs")
        
        with tab3:
            st.header("Load Model Checkpoint")
            
            if st.session_state.gan_model is None:
                st.warning("Please initialize the model first")
            else:
                # Get available checkpoints
                checkpoint_dir = st.session_state.gan_model.checkpoint_dir
                if os.path.exists(checkpoint_dir):
                    generator_checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("generator_epoch_")]
                    if generator_checkpoints:
                        # Extract epoch numbers
                        epochs = [int(f.split("_")[-1].split(".")[0]) for f in generator_checkpoints]
                        
                        # Create selectbox for epochs
                        selected_epoch = st.selectbox(
                            "Select Checkpoint Epoch",
                            sorted(epochs, reverse=True),
                            format_func=lambda x: f"Epoch {x}"
                        )
                        
                        if st.button("Load Checkpoint"):
                            st.session_state.gan_model.load_checkpoint(epoch=selected_epoch)
                            st.success(f"Successfully loaded checkpoint from epoch {selected_epoch}")
                            
                            # Display training history
                            st.subheader("Training History")
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=list(range(1, len(st.session_state.gan_model.history['generator_loss']) + 1)),
                                y=st.session_state.gan_model.history['generator_loss'],
                                mode='lines',
                                name='Generator Loss'
                            ))
                            fig.add_trace(go.Scatter(
                                x=list(range(1, len(st.session_state.gan_model.history['discriminator_loss']) + 1)),
                                y=st.session_state.gan_model.history['discriminator_loss'],
                                mode='lines',
                                name='Discriminator Loss'
                            ))
                            fig.update_layout(
                                title="Training Loss",
                                xaxis_title="Epoch",
                                yaxis_title="Loss",
                                legend_title="Loss Type"
                            )
                            st.plotly_chart(fig)
                    else:
                        st.warning("No checkpoints found")
                else:
                    st.warning(f"Checkpoint directory {checkpoint_dir} does not exist")

# Sequence Generation page
elif page == "Sequence Generation":
    st.title("Sequence Generation")
    
    if st.session_state.gan_model is None:
        st.warning("Please initialize and train the model first")
    else:
        st.header("Generate Synthetic DNA Sequences")
        
        num_sequences = st.number_input("Number of Sequences to Generate", min_value=1, value=10)
        
        if st.button("Generate Sequences"):
            with st.spinner("Generating sequences..."):
                # Generate sequences
                st.session_state.synthetic_sequences = st.session_state.gan_model.generate(num_sequences=num_sequences)
                
                st.success(f"Successfully generated {len(st.session_state.synthetic_sequences)} sequences")
        
        # Display generated sequences if available
        if len(st.session_state.synthetic_sequences) > 0:
            st.subheader("Generated Sequences")
            
            # Display sample sequences
            for i, seq in enumerate(st.session_state.synthetic_sequences[:10]):
                st.code(f"Sequence {i+1}: {seq}")
            
            # Download button
            df = pd.DataFrame({
                'sequence_id': range(len(st.session_state.synthetic_sequences)),
                'sequence': st.session_state.synthetic_sequences
            })
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Sequences as CSV",
                data=csv,
                file_name="synthetic_sequences.csv",
                mime="text/csv"
            )
            
            # Display sequence statistics
            st.markdown("---")
            st.header("Sequence Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Number of Sequences", len(st.session_state.synthetic_sequences))
                
            with col2:
                lengths = [len(seq) for seq in st.session_state.synthetic_sequences]
                st.metric("Average Sequence Length", round(sum(lengths) / len(lengths), 2))
            
            # Display nucleotide composition
            st.subheader("Nucleotide Composition")
            all_nucleotides = ''.join(st.session_state.synthetic_sequences)
            nucleotide_counts = {
                'A': all_nucleotides.count('A'),
                'C': all_nucleotides.count('C'),
                'G': all_nucleotides.count('G'),
                'T': all_nucleotides.count('T')
            }
            
            fig = go.Figure(data=[go.Pie(
                labels=list(nucleotide_counts.keys()),
                values=list(nucleotide_counts.values()),
                hole=.3,
                marker_colors=['green', 'blue', 'orange', 'red']
            )])
            fig.update_layout(title="Nucleotide Composition")
            st.plotly_chart(fig)

# Evaluation page
elif page == "Evaluation":
    st.title("Sequence Evaluation")
    
    if len(st.session_state.real_sequences) == 0:
        st.warning("Please load real sequences first")
    elif len(st.session_state.synthetic_sequences) == 0:
        st.warning("Please generate synthetic sequences first")
    else:
        st.header("Evaluate Synthetic Sequences")
        
        if st.button("Run Evaluation"):
            with st.spinner("Evaluating sequences..."):
                # Ensure sequences have the same length
                if len(set(len(seq) for seq in st.session_state.real_sequences)) > 1 or len(set(len(seq) for seq in st.session_state.synthetic_sequences)) > 1:
                    max_length = max(max(len(seq) for seq in st.session_state.real_sequences), max(len(seq) for seq in st.session_state.synthetic_sequences))
                    real_sequences_padded = pad_sequences(st.session_state.real_sequences, max_length)
                    synthetic_sequences_padded = pad_sequences(st.session_state.synthetic_sequences, max_length)
                    st.info(f"Padded sequences to uniform length of {max_length}")
                else:
                    real_sequences_padded = st.session_state.real_sequences
                    synthetic_sequences_padded = st.session_state.synthetic_sequences
                
                # Comprehensive evaluation
                st.session_state.evaluation_results = DNAEvaluator.comprehensive_evaluation(
                    real_sequences_padded, synthetic_sequences_padded
                )
                
                # Generate report
                report = DNAEvaluator.generate_report(st.session_state.evaluation_results)
                
                st.success("Evaluation completed")
                
                # Display report
                st.markdown("---")
                st.header("Evaluation Report")
                st.markdown(report)
                
                # Download button for report
                st.download_button(
                    label="Download Report as Markdown",
                    data=report,
                    file_name="evaluation_report.md",
                    mime="text/markdown"
                )
        
        # Display previous evaluation results if available
        elif st.session_state.evaluation_results:
            # Generate report
            report = DNAEvaluator.generate_report(st.session_state.evaluation_results)
            
            # Display report
            st.markdown("---")
            st.header("Evaluation Report")
            st.markdown(report)
            
            # Download button for report
            st.download_button(
                label="Download Report as Markdown",
                data=report,
                file_name="evaluation_report.md",
                mime="text/markdown"
            )

# Visualization page
elif page == "Visualization":
    st.title("Sequence Visualization")
    
    if len(st.session_state.real_sequences) == 0:
        st.warning("Please load real sequences first")
    else:
        tab1, tab2, tab3 = st.tabs(["Sequence Visualization", "Distribution Comparison", "Training History"])
        
        with tab1:
            st.header("DNA Sequence Visualization")
            
            # Ensure sequences have the same length for visualization
            if len(st.session_state.real_sequences) > 0:
                if len(set(len(seq) for seq in st.session_state.real_sequences)) > 1:
                    max_length = max(len(seq) for seq in st.session_state.real_sequences)
                    real_sequences_padded = pad_sequences(st.session_state.real_sequences, max_length)
                else:
                    real_sequences_padded = st.session_state.real_sequences
                
                # Real sequences logo
                st.subheader("Real DNA Sequence Logo")
                fig = DNAVisualizer.sequence_logo(
                    real_sequences_padded[:10], 
                    title="Real DNA Sequence Logo"
                )
                st.pyplot(fig)
            
            # Synthetic sequences logo
            if len(st.session_state.synthetic_sequences) > 0:
                if len(set(len(seq) for seq in st.session_state.synthetic_sequences)) > 1:
                    max_length = max(len(seq) for seq in st.session_state.synthetic_sequences)
                    synthetic_sequences_padded = pad_sequences(st.session_state.synthetic_sequences, max_length)
                else:
                    synthetic_sequences_padded = st.session_state.synthetic_sequences
                
                st.subheader("Synthetic DNA Sequence Logo")
                fig = DNAVisualizer.sequence_logo(
                    synthetic_sequences_padded[:10], 
                    title="Synthetic DNA Sequence Logo"
                )
                st.pyplot(fig)
                
                # Interactive sequence viewer
                st.subheader("Interactive Sequence Viewer")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Real Sequences**")
                    fig = DNAVisualizer.plotly_sequence_viewer(
                        real_sequences_padded[:5], 
                        title="Real DNA Sequences"
                    )
                    st.plotly_chart(fig)
                
                with col2:
                    st.markdown("**Synthetic Sequences**")
                    fig = DNAVisualizer.plotly_sequence_viewer(
                        synthetic_sequences_padded[:5], 
                        title="Synthetic DNA Sequences"
                    )
                    st.plotly_chart(fig)
        
        with tab2:
            st.header("Distribution Comparison")
            
            if len(st.session_state.synthetic_sequences) == 0:
                st.warning("Please generate synthetic sequences first")
            else:
                # Ensure sequences have the same length for comparison
                if len(set(len(seq) for seq in st.session_state.real_sequences)) > 1 or len(set(len(seq) for seq in st.session_state.synthetic_sequences)) > 1:
                    max_length = max(max(len(seq) for seq in st.session_state.real_sequences), max(len(seq) for seq in st.session_state.synthetic_sequences))
                    real_sequences_padded = pad_sequences(st.session_state.real_sequences, max_length)
                    synthetic_sequences_padded = pad_sequences(st.session_state.synthetic_sequences, max_length)
                else:
                    real_sequences_padded = st.session_state.real_sequences
                    synthetic_sequences_padded = st.session_state.synthetic_sequences
                
                # K-mer distribution comparison
                st.subheader("K-mer Distribution Comparison")
                
                k_value = st.slider("K-mer Length", min_value=2, max_value=6, value=3)
                
                fig = DNAVisualizer.plotly_compare_kmer_distributions(
                    real_sequences_padded, 
                    synthetic_sequences_padded, 
                    k=k_value, 
                    title=f"{k_value}-mer Distribution Comparison"
                )
                st.plotly_chart(fig)
                
                # GC content comparison
                st.subheader("GC Content Comparison")
                
                fig = DNAVisualizer.plotly_gc_content_comparison(
                    real_sequences_padded, 
                    synthetic_sequences_padded, 
                    title="GC Content Comparison"
                )
                st.plotly_chart(fig)
        
        with tab3:
            st.header("Training History")
            
            if st.session_state.gan_model is None:
                st.warning("Please train the model first")
            elif len(st.session_state.gan_model.history['generator_loss']) == 0:
                st.warning("No training history available")
            else:
                # Display training history
                fig = DNAVisualizer.plotly_training_history(
                    st.session_state.gan_model.history, 
                    title="GAN Training History"
                )
                st.plotly_chart(fig)

# Add footer
st.markdown("---")
st.markdown("DNA Sequence GAN | Created with Streamlit")
