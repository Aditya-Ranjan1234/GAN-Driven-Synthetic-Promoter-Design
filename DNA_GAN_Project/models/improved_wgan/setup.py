"""
Setup script for the ImprovedDNAGAN package.
"""

from setuptools import setup, find_packages

setup(
    name="ImprovedDNAGAN",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.5",
        "matplotlib>=3.4.3",
        "tqdm>=4.62.3",
        "scikit-learn>=1.0.1",
        "biopython>=1.79",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Improved DNA sequence generation using Wasserstein GAN with gradient penalty",
    keywords="DNA, GAN, deep learning, bioinformatics",
    url="https://github.com/yourusername/improved-dna-gan",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
)
