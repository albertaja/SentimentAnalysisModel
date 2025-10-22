# Inference Script for LSTM + Transformer Sentiment Analysis Model
# This script loads the trained model and provides sentiment prediction functionality

import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class SentimentAnalyzer:
    """
    Indonesian Sentiment Analysis using LSTM + Transformer Hybrid Model
    
    This class provides an easy-to-use interface for sentiment prediction
    on Indonesian text using the trained hybrid model.
    """
    
    def __init__(self, model_path='models/lstm_transformer_sentiment_model.h5',
                 tokenizer_path='data/processed/tokenizer.pkl',
                 label_encoder_path='data/processed/label_encoder.pkl'):
        """
        Initialize the sentiment analyzer with trained model and preprocessing artifacts.
        
        Args:
            model_path (str): Path to the trained model file
            tokenizer_path (str): Path to the fitted tokenizer
            label_encoder_path (str): Path to the label encoder
        """
        
        print("Loading Indonesian Sentiment Analysis Model...")
        
        # Load model
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"✓ Model loaded from {model_path}")
        except Exception as e:
            raise FileNotFoundError(f"Could not load model from {model_path}: {e}")
        
        # Load tokenizer
        try:
            with open(tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            print(f"✓ Tokenizer loaded from {tokenizer_path}")
        except Exception as e:
            raise FileNotFoundError(f"Could not load tokenizer from {tokenizer_path}: {e}")
        
        # Load label encoder
        try:
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print(f"✓ Label encoder loaded from {label_encoder_path}")
        except Exception as e:
            raise FileNotFoundError(f"Could not load label encoder from {label_encoder_path}: {e}")
        
        # Initialize text preprocessing components
        self._initialize_preprocessing()
        
        # Model parameters
        self.max_length = self.model.input_shape[1]
        self.vocab_size = len(self.tokenizer.word_index) + 1
        
        print(f"✓ Model ready! Max sequence length: {self.max_length}")
        print(f"✓ Vocabulary size: {self.vocab_size}")
        print(f"✓ Classes: {self.label_encoder.classes_}")
        print("\n=== Sentiment Analysis Ready ===")
    
    def _initialize_preprocessing(self):
        """Initialize Indonesian text preprocessing components."""
        
        # Slang dictionary for Indonesian text normalization
        self.slang_dict = {
            'yg': 'yang', 'ga': 'tidak', 'gak': 'tidak', 'gk': 'tidak', 'gaada': 'tidak ada',
            'aja': 'saja', 'sm': 'sama', 'udah': 'sudah', 'utk': 'untuk', 'bgt': 'banget',
            'jd': 'jadi', 'jg': 'juga', 'lg': 'lagi', 'dgn': 'dengan', 'dr': 'dari',
            'klo': 'kalau', 'knp': 'kenapa', 'trs': 'terus', 'sdh': 'sudah'
        }
        
        # Initialize Sastrawi components
        factory = StopWordRemoverFactory()
        self.stopword_remover = factory.create_stop_word_remover()
        
        stemmer_factory = StemmerFactory()
        self.stemmer = stemmer_factory.create_stemmer()
        
        print("✓ Indonesian text preprocessing initialized")
    
    def preprocess_text(self, text):
        """
        Preprocess Indonesian text using the same pipeline as training.
        
        Args:
            text (str): Raw Indonesian text
            
        Returns:
            str: Preprocessed text
        """
        
        if not isinstance(text, str):
            text = str(text)
        
        # 1. Lowercasing
        text = text.lower()
        
        # 2. Remove special characters, HTML, URLs
        text = re.sub(r'<.*?>', '', text)  # HTML tags
        text = re.sub(r'http\S+', '', text)  # URLs
        text = text.translate(str.maketrans('', '', string.punctuation))  # Punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Extra whitespace
        
        # 3. Normalize slang
        words = text.split()
        reformed = [self.slang_dict[word] if word in self.slang_dict else word for word in words]
        text = " ".join(reformed)
        
        # 4. Remove stopwords
        text = self.stopword_remover.remove(text)
        
        # 5. Stemming
        text = self.stemmer.stem(text)
        
        return text
    
    def predict_sentiment(self, text, return_probabilities=False):
        """
        Predict sentiment for a single text.
        
        Args:
            text (str): Input text in Indonesian
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            dict: Prediction results containing sentiment, confidence, and optionally probabilities
        """
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Tokenize and convert to sequence
        sequence = self.tokenizer.texts_to_sequences([processed_text])
        
        # Pad sequence to model input length
        padded_sequence = pad_sequences(sequence, maxlen=self.max_length)
        
        # Get model prediction
        prediction_probs = self.model.predict(padded_sequence, verbose=0)[0]
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(prediction_probs)
        confidence = float(np.max(prediction_probs))
        
        # Decode to original label
        sentiment = self.label_encoder.inverse_transform([predicted_class_idx])[0]
        
        # Prepare result
        result = {
            'text': text,
            'processed_text': processed_text,
            'sentiment': sentiment,
            'confidence': confidence,
            'predicted_class': int(predicted_class_idx)
        }
        
        if return_probabilities:
            class_probs = {}
            for i, class_name in enumerate(self.label_encoder.classes_):
                class_probs[f'prob_class_{class_name}'] = float(prediction_probs[i])
            result['probabilities'] = class_probs
        
        return result
    
    def predict_batch(self, texts, return_probabilities=False):
        """
        Predict sentiment for multiple texts.
        
        Args:
            texts (list): List of Indonesian texts
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            list: List of prediction dictionaries
        """
        
        results = []
        
        for text in texts:
            try:
                result = self.predict_sentiment(text, return_probabilities)
                results.append(result)
            except Exception as e:
                # Handle individual text errors gracefully
                error_result = {
                    'text': text,
                    'error': str(e),
                    'sentiment': None,
                    'confidence': 0.0
                }
                results.append(error_result)
        
        return results
    
    def analyze_csv(self, csv_path, text_column, output_path=None):
        """
        Analyze sentiment for texts in a CSV file.
        
        Args:
            csv_path (str): Path to input CSV file
            text_column (str): Name of the column containing text
            output_path (str): Path to save results (optional)
            
        Returns:
            pd.DataFrame: DataFrame with original data plus sentiment predictions
        """
        
        print(f"Loading CSV from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV. Available columns: {df.columns.tolist()}")
        
        print(f"Analyzing sentiment for {len(df)} texts...")
        
        # Get predictions for all texts
        texts = df[text_column].fillna('').astype(str).tolist()
        predictions = self.predict_batch(texts, return_probabilities=True)
        
        # Add predictions to dataframe
        df['predicted_sentiment'] = [p.get('sentiment', None) for p in predictions]
        df['sentiment_confidence'] = [p.get('confidence', 0.0) for p in predictions]
        df['processed_text'] = [p.get('processed_text', '') for p in predictions]
        
        # Add probability columns if available
        if predictions and 'probabilities' in predictions[0]:
            for class_name in self.label_encoder.classes_:
                col_name = f'prob_sentiment_{class_name}'
                df[col_name] = [p.get('probabilities', {}).get(f'prob_class_{class_name}', 0.0) for p in predictions]
        
        # Save results if output path provided
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")
        
        return df


def main():
    """Example usage of the SentimentAnalyzer class."""
    
    print("=== Indonesian Sentiment Analysis Demo ===")
    
    try:
        # Initialize analyzer
        analyzer = SentimentAnalyzer()
        
        # Example texts for testing
        test_texts = [
            "Paket internet XL sangat bagus dan murah!",
            "Sinyal XL lemah sekali, sangat mengecewakan",
            "Harga paket XL biasa saja, tidak terlalu mahal tapi juga tidak murah",
            "Aplikasi MyXL mudah digunakan dan fiturnya lengkap",
            "Pelayanan customer service XL buruk sekali"
        ]
        
        print("\n=== Single Text Predictions ===")
        
        for i, text in enumerate(test_texts, 1):
            result = analyzer.predict_sentiment(text, return_probabilities=True)
            
            print(f"\nExample {i}:")
            print(f"Text: {result['text']}")
            print(f"Processed: {result['processed_text']}")
            print(f"Sentiment: {result['sentiment']}")
            print(f"Confidence: {result['confidence']:.3f}")
            
            if 'probabilities' in result:
                print("Probabilities:")
                for prob_key, prob_val in result['probabilities'].items():
                    print(f"  {prob_key}: {prob_val:.3f}")
        
        print("\n=== Batch Prediction Example ===")
        
        batch_results = analyzer.predict_batch(test_texts[:3])
        
        for i, result in enumerate(batch_results, 1):
            print(f"Batch {i}: {result['sentiment']} (confidence: {result['confidence']:.3f})")
        
        print("\n=== Demo completed successfully! ===")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        print("\nMake sure you have:")
        print("1. Trained the model using src/train_model.py")
        print("2. All required files in the correct locations")
        print("3. Installed all dependencies from requirements.txt")


if __name__ == "__main__":
    main()