# DNA Sequence Conversion Tools

This folder contains scripts and files for converting DNA sequences from HTML/FASTA format to CSV and properly formatted FASTA files.

## Files

### Input Files
- `seq_download.pl.fasta` - The original FASTA file downloaded from the source (HTML file with FASTA data)

### Output Files
- `dna_sequences.csv` - DNA sequences in CSV format
- `dna_sequences.fasta` - DNA sequences in FASTA format
- `formatted_sequences.fasta` - Properly formatted FASTA file for loading into bioinformatics tools
- `clean_dna_sequences.fasta` - Cleaned FASTA file with properly formatted sequences
- `final_dna_sequences.fasta` - Final FASTA file with properly formatted sequences
- `all_dna_sequences.fasta` - All sequences extracted from the original file
- `clean_all_dna_sequences.fasta` - All sequences properly formatted for loading into bioinformatics tools

### Scripts
- `html_to_csv_converter.py` - Initial script to convert HTML to CSV
- `html_to_fasta_converter.py` - Initial script to convert HTML to FASTA
- `convert_to_csv.py` - Script to convert FASTA to CSV
- `extract_sequences.py` - Script to extract sequences using Biopython
- `extract_sequences_simple.py` - Simplified script to extract sequences
- `extract_dna_sequences.py` - Script to extract DNA sequences
- `extract_dna_sequences_final.py` - Final version of the extraction script
- `extract_dna_sequences_clean.py` - Script to clean and extract DNA sequences
- `extract_dna_sequences_complete.py` - Complete script to extract and save in both CSV and FASTA formats
- `format_fasta.py` - Script to properly format FASTA file for loading
- `validate_fasta.py` - Script to validate FASTA files
- `final_fasta_cleaner.py` - Script to clean and format FASTA files
- `fix_fasta.py` - Script to fix issues in FASTA files
- `extract_all_sequences.py` - Script to extract all sequences from the original file
- `clean_all_sequences.py` - Script to clean and format all sequences

## Usage

### To extract a few sequences and convert to a properly formatted FASTA file:

```bash
python format_fasta.py
```

This will read the `seq_download.pl.fasta` file and create a properly formatted `formatted_sequences.fasta` file.

### To extract all sequences from the original file:

```bash
python extract_all_sequences.py
```

This will extract all sequences from the original file and save them to `all_dna_sequences.fasta`.

### To clean and format all sequences:

```bash
python clean_all_sequences.py
```

This will read the `all_dna_sequences.fasta` file and create a properly formatted `clean_all_dna_sequences.fasta` file with all 4765 sequences.

## FASTA Format

The FASTA format consists of:
- A header line starting with '>' followed by a sequence identifier and optional description
- The sequence data on subsequent lines, typically wrapped at 60-80 characters per line

Example:
```
>EP17001 (+) Pv snRNA U1; range -499 to 100.
NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN
...
```

## CSV Format

The CSV format includes columns for:
- `sequence_id` - The identifier of the sequence
- `description` - The description of the sequence
- `sequence` - The DNA sequence data
