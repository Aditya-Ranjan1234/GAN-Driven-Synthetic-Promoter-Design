"""
Script to run the DNA sequence comparison Streamlit app.
"""

import os
import sys
import subprocess
import argparse


def main():
    """
    Main function to run the DNA sequence comparison Streamlit app.
    """
    parser = argparse.ArgumentParser(description='Run the DNA sequence comparison app')

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
        from utils.data_preparation import generate_dummy_sequences

        # Create directories if they don't exist
        os.makedirs("data", exist_ok=True)

        # Check if original data file exists
        if os.path.exists("data/seq_download.pl.fasta"):
            # Try to extract FASTA content using the improved script
            try:
                from utils.improved_extract_fasta import improved_extract_fasta
                success = improved_extract_fasta("data/seq_download.pl.fasta", "data/clean_seq_download.fasta")
                if success:
                    print("Successfully extracted FASTA content from HTML file.")
                else:
                    raise Exception("Extraction failed")
            except Exception as e:
                print(f"Error extracting FASTA content: {e}")
                print("Generating sample FASTA file instead...")
                from utils.generate_sample_fasta import generate_sample_fasta
                generate_sample_fasta("data/clean_seq_download.fasta")
        else:
            print("Warning: Original data file not found. Generating sample FASTA file instead.")
            from utils.generate_sample_fasta import generate_sample_fasta
            generate_sample_fasta("data/clean_seq_download.fasta")

        # Generate dummy data for generated sequences if they don't exist
        if not os.path.exists("data/gumbel_generated_sequences.fasta"):
            generate_dummy_sequences("data/gumbel_generated_sequences.fasta", model_type='gumbel')

        if not os.path.exists("data/improved_generated_sequences.fasta"):
            generate_dummy_sequences("data/improved_generated_sequences.fasta", model_type='improved')

        print("Dummy data preparation complete.")

    # Run the Streamlit app
    print(f"Starting Streamlit app on port {args.port}...")

    streamlit_app_path = os.path.join(script_dir, "visualization", "streamlit_app_comparison.py")

    try:
        # Add the current directory to the Python path
        os.environ["PYTHONPATH"] = os.getcwd()
        subprocess.run([
            "streamlit", "run", streamlit_app_path,
            "--server.port", str(args.port)
        ])
    except FileNotFoundError:
        print("Error: Streamlit not found. Please install it with 'pip install streamlit'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
