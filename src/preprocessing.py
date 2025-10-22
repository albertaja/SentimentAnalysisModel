# This script performs the full preprocessing pipeline for the "Ulasan My XL 1000 Data Labelled" dataset.
# It includes data inspection, text cleaning, tokenization, sequence padding, label encoding, data splitting, and quality assurance assertions.

import pandas as pd
import re
import string
import numpy as np
from collections import Counter
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

# --- 1. Load and Inspect the Dataset ---
print("--- 1. Loading and Inspecting Dataset ---")
df = pd.read_csv('data/raw/Ulasan My XL 1000 Data Labelled.csv')
print("Dataset Info:")
df.info()
print("\nSample Rows:")
print(df.head())
print("\nClass Distribution (Sentimen):")
print(df['Sentimen'].value_counts())
df = df.dropna(subset=['Ulasan'])  # Drop rows with missing reviews

# --- 2. Text Preprocessing ---
print("\n--- 2. Preprocessing Text ---")

# Define preprocessing functions
def lowercase(text):
    """Converts text to lowercase."""
    return text.lower()

def remove_special_characters(text):
    """Removes HTML tags, URLs, punctuation, and extra spaces."""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_slang(text):
    """Normalizes common Indonesian slang and abbreviations."""
    slang_dict = {
        'yg': 'yang', 'ga': 'tidak', 'gak': 'tidak', 'gk': 'tidak', 'gaada': 'tidak ada',
        'aja': 'saja', 'sm': 'sama', 'udah': 'sudah', 'utk': 'untuk', 'bgt': 'banget',
        'jd': 'jadi', 'jg': 'juga', 'lg': 'lagi', 'dgn': 'dengan', 'dr': 'dari',
        'klo': 'kalau', 'knp': 'kenapa', 'trs': 'terus', 'sdh': 'sudah'
    }
    words = text.split()
    reformed = [slang_dict[word] if word in slang_dict else word for word in words]
    return " ".join(reformed)

factory = StopWordRemoverFactory()
stopword_remover = factory.create_stop_word_remover()
def remove_stopwords(text):
    """Removes Indonesian stopwords."""
    return stopword_remover.remove(text)

stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()
def stem_text(text):
    """Stems Indonesian words to their root form."""
    return stemmer.stem(text)

def get_most_common_tokens(texts, n=20):
    """Calculates the most common tokens in a list of texts."""
    all_text = " ".join(texts)
    tokens = word_tokenize(all_text)
    return Counter(tokens).most_common(n)

def preprocess_text(text):
    """Applies the full text preprocessing pipeline."""
    text = lowercase(text)
    text = remove_special_characters(text)
    text = normalize_slang(text)
    text = remove_stopwords(text)
    text = stem_text(text)
    return text

# Report most common tokens before cleaning
print("\nMost common tokens BEFORE cleaning:")
print(get_most_common_tokens(df['Ulasan']))

# Apply the preprocessing pipeline
df['cleaned_ulasan'] = df['Ulasan'].apply(preprocess_text)
df.to_csv('data/processed/cleaned_data.csv', index=False)

# Report most common tokens after cleaning
print("\nMost common tokens AFTER cleaning:")
print(get_most_common_tokens(df['cleaned_ulasan']))
print("\nText preprocessing complete.")

# --- 3. Tokenization and Sequence Preparation ---
print("\n--- 3. Tokenizing and Preparing Sequences ---")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['cleaned_ulasan'])
sequences = tokenizer.texts_to_sequences(df['cleaned_ulasan'])

review_lengths = [len(x) for x in sequences]
max_len = int(np.median(review_lengths))
X = pad_sequences(sequences, maxlen=max_len)
print(f"Padded sequences to length: {max_len}")

# --- 4. Label Encoding ---
print("\n--- 4. Encoding Labels ---")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Sentimen'])
print("Labels encoded.")

# --- 5. Train/Validation/Test Split ---
print("\n--- 5. Splitting Data ---")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
print("Data splitting complete.")

# --- 6. Quality Assurance Assertions ---
print("\n--- 6. Running Quality Assurance Assertions ---")
# Assert that there are no missing values in the cleaned text
assert df['cleaned_ulasan'].isnull().sum() == 0, "Missing values found after cleaning."
# Assert that the splits have the correct proportions
assert len(X_train) / len(X) == 0.7, "Train split proportion is incorrect."
assert len(X_val) / len(X) == 0.15, "Validation split proportion is incorrect."
assert len(X_test) / len(X) == 0.15, "Test split proportion is incorrect."
# Assert that the label distribution is similar across splits (stratification check)
original_dist = np.bincount(y) / len(y)
train_dist = np.bincount(y_train) / len(y_train)
val_dist = np.bincount(y_val) / len(y_val)
test_dist = np.bincount(y_test) / len(y_test)
assert np.allclose(original_dist, train_dist, atol=0.01), "Train set label distribution is skewed."
assert np.allclose(original_dist, val_dist, atol=0.01), "Validation set label distribution is skewed."
assert np.allclose(original_dist, test_dist, atol=0.01), "Test set label distribution is skewed."
print("All assertions passed.")

# --- 7. Export Artifacts ---
print("\n--- 7. Exporting Artifacts ---")
with open('data/processed/tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('data/processed/label_encoder.pkl', 'wb') as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

np.save('data/processed/X_train.npy', X_train)
np.save('data/processed/y_train.npy', y_train)
np.save('data/processed/X_val.npy', X_val)
np.save('data/processed/y_val.npy', y_val)
np.save('data/processed/X_test.npy', X_test)
np.save('data/processed/y_test.npy', y_test)
print("Artifacts exported successfully.")

# --- Summary ---
print("\n--- Preprocessing Summary ---")
print(f"Vocabulary size: {len(tokenizer.word_index)}")
print(f"Max sequence length: {max_len}")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print("---------------------------\n")
