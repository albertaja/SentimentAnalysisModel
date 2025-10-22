# Data Preprocessing Report

## Overview

This report details the preprocessing pipeline for the "Ulasan My XL 1000 Data Labelled" dataset. The goal of this process was to clean and prepare the text data for a deep learning text classification task.

## Preprocessing Steps

The following preprocessing steps were applied to the raw review text:

1.  **Lowercasing:** All text was converted to lowercase.
2.  **Special Character Removal:** HTML tags, URLs, punctuation, and extra whitespace were removed.
3.  **Slang Normalization:** Common Indonesian slang and abbreviations were replaced with their standard forms using a manually curated dictionary.
4.  **Stopword Removal:** Common Indonesian stopwords were removed using the `Sastrawi` library.
5.  **Stemming:** Words were reduced to their root form using the `Sastrawi` stemmer.

## Tokenization and Sequence Preparation

-   **Tokenization:** The cleaned text was tokenized using the Keras `Tokenizer`.
-   **Vocabulary Size:** 4046
-   **Sequence Padding:** Sequences were padded to a length of 26, which was the median review length.

## Label Encoding

The `Sentimen` column, containing the class labels, was encoded into integer form using `sklearn.preprocessing.LabelEncoder`. The class mapping is as follows:

-   `-1` -> `0`
-   `0` -> `1`
-   `1` -> `2`

## Data Splits

The data was split into training, validation, and test sets with a 70/15/15 stratified split.

-   **Training Set:** 700 samples
-   **Validation Set:** 150 samples
-   **Test Set:** 150 samples

## Saved Artifacts

The following artifacts have been saved to the `data/processed` directory:

-   `cleaned_data.csv`: The preprocessed text data.
-   `tokenizer.pkl`: The Keras Tokenizer object.
-   `label_encoder.pkl`: The scikit-learn LabelEncoder object.
-   `X_padded.npy`: The padded sequences of tokenized text.
-   `y_encoded.npy`: The encoded labels.
-   `X_train.npy`, `y_train.npy`: Training data.
-   `X_val.npy`, `y_val.npy`: Validation data.
-   `X_test.npy`, `y_test.npy`: Test data.
