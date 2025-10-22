# My XL Ulasan Preprocessing

This project contains a script to preprocess the "Ulasan My XL 1000 Data Labelled" dataset for text classification.

## Setup

1.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The entire preprocessing pipeline is contained in a single script: `src/preprocessing.py`. To run the pipeline, execute the following command:

```bash
python3 src/preprocessing.py
```

This script will:
1.  Load and inspect the raw data.
2.  Clean the text data (lowercasing, removing special characters, etc.).
3.  Tokenize the cleaned text, pad the sequences, and encode the labels.
4.  Split the data into training, validation, and test sets.
5.  Run quality assurance assertions to verify the integrity of the data.
6.  Save all the processed data and artifacts (`tokenizer.pkl`, `label_encoder.pkl`, etc.) to the `data/processed` directory.

## Data Report

A detailed report of the preprocessing steps and dataset statistics can be found in `data_report.md`.
