"""
Setup script for the dna_gan package.
"""

from setuptools import setup, find_packages

setup(
    name="dna_gan",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.48.0",
        "scikit-learn>=0.23.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="DNA sequence generation using Gumbel-Softmax GAN",
    keywords="dna, gan, deep learning, bioinformatics",
    url="https://github.com/yourusername/dna-gan",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.6",
)
