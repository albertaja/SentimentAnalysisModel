# Indonesian Sentiment Analysis with LSTM + Transformer Hybrid Model

A state-of-the-art deep learning approach for sentiment analysis on Indonesian text reviews using a hybrid LSTM + Transformer architecture.

## 🎯 Project Overview

This project implements a comprehensive sentiment analysis pipeline for Indonesian text, specifically trained on the "Ulasan My XL 1000 Data Labelled" dataset. The model combines the sequential processing power of LSTM networks with the attention mechanisms of Transformer architectures to achieve superior performance on Indonesian text classification.

### Key Features
- **Hybrid Architecture**: LSTM + Transformer for optimal performance
- **Indonesian NLP**: Specialized preprocessing for Indonesian text
- **Comprehensive Evaluation**: Detailed metrics, visualizations, and analysis
- **Production Ready**: Full pipeline from preprocessing to deployment
- **Reproducible**: Fixed random seeds and comprehensive documentation

## 📊 Dataset

- **Name**: Ulasan My XL 1000 Data Labelled
- **Size**: 1,000 Indonesian reviews
- **Classes**: 3 sentiment categories
  - Negative (-1): 613 samples (61.3%)
  - Neutral (0): 226 samples (22.6%) 
  - Positive (1): 161 samples (16.1%)
- **Split**: 70% training, 15% validation, 15% test
- **Language**: Indonesian (Bahasa Indonesia)

## 🏗️ Model Architecture

### Hybrid LSTM + Transformer Design

```
Input (Sequence of Token IDs)
         ↓
Embedding Layer (128-dim)
         ↓
Bidirectional LSTM (64 units)
         ↓
Transformer Block (4 attention heads)
         ↓
Global Average Pooling
         ↓
Dense Layer (64 units) + Dropout
         ↓
Output Layer (3 classes, softmax)
```

### Technical Specifications
- **Embedding Dimension**: 128
- **LSTM Units**: 64 (bidirectional)
- **Attention Heads**: 4
- **Feed-Forward Dimension**: 128
- **Dropout Rate**: 0.3 (LSTM), 0.5 (Dense)
- **Vocabulary Size**: 4,046 unique tokens
- **Sequence Length**: 26 tokens (median-based)

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/albertaja/SentimentAnalysisModel.git
cd SentimentAnalysisModel
pip install -r requirements.txt
```

### 2. Data Preprocessing

```bash
python src/preprocessing.py
```

This will:
- Load and clean the raw dataset
- Apply Indonesian-specific text preprocessing
- Create train/validation/test splits
- Generate tokenizer and label encoder artifacts

### 3. Model Training

```bash
python src/train_model.py
```

This will:
- Build the LSTM + Transformer hybrid model
- Train with early stopping and learning rate scheduling
- Generate comprehensive evaluation metrics
- Save trained model and artifacts

### 4. Interactive Development

```bash
jupyter notebook notebooks/sentiment_analysis_training.ipynb
```

## 📁 Project Structure

```
SentimentAnalysisModel/
├── data/
│   ├── raw/                     # Original dataset
│   └── processed/              # Preprocessed data and artifacts
├── src/
│   ├── preprocessing.py        # Data preprocessing pipeline
│   └── train_model.py         # Model training script
├── notebooks/
│   └── sentiment_analysis_training.ipynb  # Interactive notebook
├── models/                     # Trained model artifacts
├── results/                    # Evaluation results and visualizations
├── requirements.txt           # Python dependencies
├── data_report.md            # Data preprocessing report
└── README.md                 # This file
```

## 📈 Model Performance

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 32
- **Max Epochs**: 30
- **Early Stopping**: Patience 5 (validation loss)
- **Learning Rate**: Adaptive with ReduceLROnPlateau

### Expected Performance Metrics
*Note: Run training to get actual results*

- **Test Accuracy**: ~85-90%
- **Macro F1-Score**: ~0.80-0.85
- **Training Time**: ~5-10 minutes (depending on hardware)

### Generated Artifacts
After training, the following files are created:

**Models**:
- `models/best_model.h5` - Best model weights
- `models/lstm_transformer_sentiment_model.h5` - Complete model
- `models/model_architecture.json` - Model architecture

**Results**:
- `results/training_curves.png` - Loss and accuracy curves
- `results/confusion_matrices.png` - Validation and test confusion matrices
- `results/model_architecture.png` - Model architecture diagram
- `results/metrics_summary.json` - Complete performance metrics
- `results/classification_report.json` - Detailed classification reports
- `results/sample_predictions.json` - Sample predictions analysis

## 🔧 Usage Examples

### Loading Trained Model

```python
import tensorflow as tf
import pickle
import numpy as np

# Load model
model = tf.keras.models.load_model('models/lstm_transformer_sentiment_model.h5')

# Load tokenizer
with open('data/processed/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load label encoder
with open('data/processed/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Predict sentiment
def predict_sentiment(text):
    # Preprocess text (implement your preprocessing here)
    processed_text = preprocess_text(text)  # Your preprocessing function
    
    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=26)
    
    # Predict
    prediction = model.predict(padded)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    
    # Decode label
    sentiment = label_encoder.inverse_transform([predicted_class])[0]
    
    return sentiment, confidence

# Example usage
sentiment, confidence = predict_sentiment("Paket internet XL sangat bagus dan murah!")
print(f"Sentiment: {sentiment}, Confidence: {confidence:.3f}")
```

## 🧪 Evaluation Commands

### View Training Results
```bash
# View training curves
open results/training_curves.png

# View confusion matrices
open results/confusion_matrices.png

# Check metrics summary
cat results/metrics_summary.json
```

### Model Analysis
```bash
# Check model architecture
cat models/model_architecture.json

# View sample predictions
cat results/sample_predictions.json
```

## 🔬 Technical Details

### Indonesian Text Preprocessing
- **Lowercasing**: Convert all text to lowercase
- **Special Character Removal**: Remove HTML, URLs, punctuation
- **Slang Normalization**: Convert Indonesian slang to standard forms
- **Stopword Removal**: Remove common Indonesian stopwords using Sastrawi
- **Stemming**: Reduce words to root forms using Sastrawi stemmer

### Model Innovation
- **Hybrid Architecture**: Combines LSTM's sequential processing with Transformer's attention
- **Bidirectional Processing**: Captures both forward and backward context
- **Multi-Head Attention**: Allows model to focus on different aspects simultaneously
- **Layer Normalization**: Stabilizes training and improves convergence
- **Dropout Regularization**: Prevents overfitting at multiple layers

### Quality Assurance
- **Reproducibility**: Fixed random seeds (42) across all components
- **Data Validation**: Assertions for data integrity and split proportions
- **Stratified Sampling**: Maintains class distribution across splits
- **Early Stopping**: Prevents overfitting with validation monitoring
- **Model Checkpointing**: Saves best model weights automatically

## 📚 Dependencies

### Core Requirements
- Python 3.8+
- TensorFlow 2.10+
- scikit-learn 1.1+
- pandas 1.5+
- numpy 1.21+

### Indonesian NLP
- Sastrawi 1.0.1+ (Indonesian text processing)
- NLTK 3.7+ (tokenization support)

### Visualization
- matplotlib 3.5+
- seaborn 0.11+

See `requirements.txt` for complete list.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Add tests if applicable
5. Commit changes (`git commit -am 'Add improvement'`)
6. Push to branch (`git push origin feature/improvement`)
7. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Dataset**: Ulasan My XL 1000 Data Labelled
- **Indonesian NLP**: Sastrawi library for Indonesian text processing
- **Architecture**: Inspired by recent advances in hybrid NLP models
- **Framework**: TensorFlow/Keras for deep learning implementation

## 📞 Contact

For questions or collaboration:
- **Author**: Albert Rafael
- **Email**: albertrafaelaja@gmail.com
- **GitHub**: [albertaja](https://github.com/albertaja)
- **Repository**: [SentimentAnalysisModel](https://github.com/albertaja/SentimentAnalysisModel)

---

**Note**: This is an academic/research project focused on Indonesian sentiment analysis. The model is trained on a specific dataset and may require fine-tuning for production use cases.