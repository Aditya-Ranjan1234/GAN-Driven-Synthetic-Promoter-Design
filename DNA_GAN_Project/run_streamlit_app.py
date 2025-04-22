"""
Script to run the Streamlit app for DNA sequence generation visualization.
"""

import os
import sys
import subprocess
import argparse


def main():
    """
    Main function to run the Streamlit app.
    """
    parser = argparse.ArgumentParser(description='Run the DNA sequence generation visualization app')
    
    parser.add_argument('--port', type=int, default=8501,
                        help='Port to run the Streamlit app on')
    parser.add_argument('--prepare-data', action='store_true',
                        help='Prepare dummy data for visualization')
    
    args = parser.parse_args()
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change to the project root directory
    os.chdir(script_dir)
    
    # Prepare dummy data if requested
    if args.prepare_data:
        print("Preparing dummy data for visualization...")
        
        # Run the data preparation script
        sys.path.append(script_dir)
        from utils.data_preparation import (
            generate_dummy_sequences,
            generate_dummy_history
        )
        
        # Create directories if they don't exist
        os.makedirs("data", exist_ok=True)
        os.makedirs("models/gumbel_softmax_gan/checkpoints", exist_ok=True)
        os.makedirs("models/improved_wgan/checkpoints/improved_dna_gan", exist_ok=True)
        
        # Generate dummy data
        generate_dummy_sequences("data/clean_all_dna_sequences.fasta", model_type='original')
        generate_dummy_sequences("data/gumbel_generated_sequences.fasta", model_type='gumbel')
        generate_dummy_sequences("data/improved_generated_sequences.fasta", model_type='improved')
        
        # Generate dummy training history
        generate_dummy_history("models/gumbel_softmax_gan/checkpoints/training_history.json", model_type='gumbel')
        generate_dummy_history("models/improved_wgan/checkpoints/improved_dna_gan/training_history.json", model_type='improved')
        
        print("Dummy data preparation complete.")
    
    # Run the Streamlit app
    print(f"Starting Streamlit app on port {args.port}...")
    
    streamlit_app_path = os.path.join(script_dir, "visualization", "streamlit_app.py")
    
    try:
        subprocess.run([
            "streamlit", "run", streamlit_app_path,
            "--server.port", str(args.port)
        ])
    except FileNotFoundError:
        print("Error: Streamlit not found. Please install it with 'pip install streamlit'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
