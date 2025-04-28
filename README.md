# DNA GAN Project

This directory contains the main code for the DNA GAN project.

## Directory Structure

- models/: GAN model implementations
  - gumbel_softmax_gan/: Gumbel-Softmax GAN implementation
  - improved_wgan/: Improved WGAN-GP implementation
- web_app/: Streamlit web interface for visualizing and comparing DNA sequences
- utils/: Utility functions for data processing and analysis
- evaluation/: Metrics for evaluating generated sequences
- isualization/: Tools for visualizing DNA sequence properties
- docs/: Project documentation

## Running the Web Interface

To run the Streamlit web interface:

`ash
cd web_app
streamlit run templates/simplified_app.py
`
