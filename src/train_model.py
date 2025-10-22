# LSTM + Transformer Hybrid Model for Sentiment Analysis
# This script trains a state-of-the-art deep learning model on Indonesian text reviews

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.utils import plot_model, to_categorical
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# Create necessary directories
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('notebooks', exist_ok=True)

print("=== LSTM + Transformer Hybrid Model Training ===")
print(f"TensorFlow version: {tf.__version__}")
print(f"Random seed set to: {RANDOM_SEED}")

# --- 1. Load and Prepare Data ---
print("\n--- 1. Loading Preprocessed Data ---")

# Load train/validation/test splits
X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')
X_val = np.load('data/processed/X_val.npy')
y_val = np.load('data/processed/y_val.npy')
X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')

# Load tokenizer and label encoder
with open('data/processed/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('data/processed/label_encoder.pkl', 'rb') as handle:
    label_encoder = pickle.load(handle)

# Load cleaned data for sample content display
cleaned_df = pd.read_csv('data/processed/cleaned_data.csv')

# Print basic data statistics
print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Sequence length: {X_train.shape[1]}")
print(f"Vocabulary size: {len(tokenizer.word_index)}")
print(f"Number of classes: {len(label_encoder.classes_)}")
print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# Class distribution
unique, counts = np.unique(y_train, return_counts=True)
print(f"Training class distribution: {dict(zip(unique, counts))}")

# Sample content
print("\nSample preprocessed text (first 3 reviews):")
for i in range(3):
    print(f"Review {i+1}: {cleaned_df['cleaned_ulasan'].iloc[i][:100]}...")
    print(f"Sentiment: {cleaned_df['Sentimen'].iloc[i]} -> Encoded: {y_train[i]}")

# --- 2. Model Architecture: LSTM + Transformer Hybrid ---
print("\n--- 2. Building LSTM + Transformer Hybrid Model ---")

@keras.saving.register_keras_serializable(package="Custom", name="TransformerBlock")
class TransformerBlock(layers.Layer):
    """Custom Transformer Encoder Block"""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.dropout_rate,
        })
        return config

def create_lstm_transformer_model(vocab_size, max_length, embed_dim=128, lstm_units=64, 
                                 num_heads=4, ff_dim=128, num_classes=3):
    """Create LSTM + Transformer Hybrid Model"""
    
    # Input layer
    inputs = layers.Input(shape=(max_length,))
    
    # Embedding layer
    embedding = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(inputs)
    
    # Bidirectional LSTM layer (outputs sequence)
    lstm_out = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)
    )(embedding)
    
    # Transformer encoder block expects features dimension = embedding of BiLSTM (2*lstm_units)
    proj = layers.Dense(embed_dim * 2)(lstm_out)
    transformer_out = TransformerBlock(embed_dim*2, num_heads, ff_dim)(proj)
    
    # Global average pooling to reduce sequence dimension
    pooled = layers.GlobalAveragePooling1D()(transformer_out)
    
    # Dense layers with dropout
    dense1 = layers.Dense(64, activation='relu')(pooled)
    dropout = layers.Dropout(0.5)(dense1)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(dropout)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Model parameters
VOCAB_SIZE = len(tokenizer.word_index) + 1  # +1 for padding token
MAX_LENGTH = X_train.shape[1]
NUM_CLASSES = len(label_encoder.classes_)
EMBED_DIM = 128
LSTM_UNITS = 64
NUM_HEADS = 4
FF_DIM = 128

# Create model
model = create_lstm_transformer_model(
    vocab_size=VOCAB_SIZE,
    max_length=MAX_LENGTH,
    embed_dim=EMBED_DIM,
    lstm_units=LSTM_UNITS,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    num_classes=NUM_CLASSES
)

# Model summary
print("Model Architecture:")
model.summary()

# Save model architecture plot
try:
    plot_model(model, to_file='results/model_architecture.png', show_shapes=True, show_layer_names=True)
    print("Model architecture diagram saved to results/model_architecture.png")
except Exception as e:
    print(f"Could not save model plot: {e}")

# --- 3. Training Configuration ---
print("\n--- 3. Training Configuration ---")

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Using sparse since labels are integers
    metrics=['accuracy']
)

# Training parameters
BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 5

print("Optimizer: Adam")
print("Loss function: sparse_categorical_crossentropy")
print(f"Batch size: {BATCH_SIZE}")
print(f"Max epochs: {EPOCHS}")
print(f"Early stopping patience: {PATIENCE}")

# --- 4. Callbacks ---
print("\n--- 4. Setting up Callbacks ---")

# Early stopping
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    restore_best_weights=True,
    verbose=1
)

# Model checkpointing (Keras 3: full model must end with .keras)
model_checkpoint = callbacks.ModelCheckpoint(
    'models/best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# Learning rate scheduler
lr_scheduler = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

callback_list = [early_stopping, model_checkpoint, lr_scheduler]

# --- 5. Model Training ---
print("\n--- 5. Training Model ---")

start_time = datetime.now()
print(f"Training started at: {start_time}")

# Train the model
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=callback_list,
    verbose=1
)

end_time = datetime.now()
training_duration = end_time - start_time
print(f"Training completed at: {end_time}")
print(f"Training duration: {training_duration}")

# Save training history
with open('results/training_history.json', 'w') as f:
    # Convert numpy values to Python native types for JSON serialization
    history_dict = {key: [float(val) for val in values] for key, values in history.history.items()}
    json.dump(history_dict, f, indent=2)

print("Training history saved to results/training_history.json")

# --- 6. Training Visualization ---
print("\n--- 6. Creating Training Visualizations ---")

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Loss curves
ax1.plot(history.history['loss'], label='Training Loss', marker='o')
ax1.plot(history.history['val_loss'], label='Validation Loss', marker='s')
ax1.set_title('Model Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# Accuracy curves
ax2.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
ax2.set_title('Model Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("Training curves saved to results/training_curves.png")

# --- 7. Model Evaluation ---
print("\n--- 7. Model Evaluation ---")

# Load best model (full model saved as .keras)
best_model = tf.keras.models.load_model('models/best_model.keras')
print("Best full model loaded for evaluation")

# Evaluate on validation and test sets
val_loss, val_accuracy = best_model.evaluate(X_val, y_val, verbose=0)
test_loss, test_accuracy = best_model.evaluate(X_test, y_test, verbose=0)

print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
print(f"Test - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")

# Get predictions
y_val_pred = best_model.predict(X_val, verbose=0)
y_test_pred = best_model.predict(X_test, verbose=0)

# Convert predictions to class labels
y_val_pred_classes = np.argmax(y_val_pred, axis=1)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)

print("Predictions generated for validation and test sets")

# --- 8. Detailed Metrics & Analysis ---
print("\n--- 8. Detailed Metrics & Analysis ---")

# Classification reports
print("=== VALIDATION SET CLASSIFICATION REPORT ===")
val_report = classification_report(y_val, y_val_pred_classes, 
                                 target_names=[f'Class_{i}' for i in range(NUM_CLASSES)], 
                                 output_dict=True)
print(classification_report(y_val, y_val_pred_classes, 
                          target_names=[f'Class_{i}' for i in range(NUM_CLASSES)]))

print("\n=== TEST SET CLASSIFICATION REPORT ===")
test_report = classification_report(y_test, y_test_pred_classes, 
                                  target_names=[f'Class_{i}' for i in range(NUM_CLASSES)], 
                                  output_dict=True)
print(classification_report(y_test, y_test_pred_classes, 
                          target_names=[f'Class_{i}' for i in range(NUM_CLASSES)]))

# Save classification reports
with open('results/validation_classification_report.json', 'w') as f:
    json.dump(val_report, f, indent=2)
with open('results/test_classification_report.json', 'w') as f:
    json.dump(test_report, f, indent=2)

# Confusion matrices
val_cm = confusion_matrix(y_val, y_val_pred_classes)
test_cm = confusion_matrix(y_test, y_test_pred_classes)

# Plot confusion matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Validation confusion matrix
sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=[f'Pred_{i}' for i in range(NUM_CLASSES)],
            yticklabels=[f'True_{i}' for i in range(NUM_CLASSES)])
ax1.set_title('Validation Set Confusion Matrix')
ax1.set_xlabel('Predicted Label')
ax1.set_ylabel('True Label')

# Test confusion matrix
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=[f'Pred_{i}' for i in range(NUM_CLASSES)],
            yticklabels=[f'True_{i}' for i in range(NUM_CLASSES)])
ax2.set_title('Test Set Confusion Matrix')
ax2.set_xlabel('Predicted Label')
ax2.set_ylabel('True Label')

plt.tight_layout()
plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print("Confusion matrices saved to results/confusion_matrices.png")

# Overall metrics summary
metrics_summary = {
    'model_info': {
        'architecture': 'LSTM + Transformer Hybrid',
        'vocab_size': VOCAB_SIZE,
        'max_length': MAX_LENGTH,
        'embed_dim': EMBED_DIM,
        'lstm_units': LSTM_UNITS,
        'num_heads': NUM_HEADS,
        'ff_dim': FF_DIM,
        'num_classes': NUM_CLASSES
    },
    'training_info': {
        'batch_size': BATCH_SIZE,
        'epochs_trained': len(history.history['loss']),
        'training_duration_seconds': training_duration.total_seconds(),
        'early_stopping_patience': PATIENCE
    },
    'performance': {
        'validation': {
            'loss': float(val_loss),
            'accuracy': float(val_accuracy),
            'macro_f1': float(val_report['macro avg']['f1-score']),
            'weighted_f1': float(val_report['weighted avg']['f1-score'])
        },
        'test': {
            'loss': float(test_loss),
            'accuracy': float(test_accuracy),
            'macro_f1': float(test_report['macro avg']['f1-score']),
            'weighted_f1': float(test_report['weighted avg']['f1-score'])
        }
    }
}

# Save metrics summary
with open('results/metrics_summary.json', 'w') as f:
    json.dump(metrics_summary, f, indent=2)

print("Metrics summary saved to results/metrics_summary.json")

# --- 9. Sample Predictions Analysis ---
print("\n--- 9. Sample Predictions Analysis ---")

# Get sample predictions with original text
sample_indices = np.random.choice(len(X_test), 10, replace=False)
sample_predictions = []

for idx in sample_indices:
    original_idx = len(X_train) + len(X_val) + idx  # Adjust for original dataframe index
    if original_idx < len(cleaned_df):
        sample_pred = {
            'index': int(idx),
            'original_text': cleaned_df['Ulasan'].iloc[original_idx][:200] + "...",
            'cleaned_text': cleaned_df['cleaned_ulasan'].iloc[original_idx][:100] + "...",
            'true_label': int(y_test[idx]),
            'predicted_label': int(y_test_pred_classes[idx]),
            'prediction_confidence': float(np.max(y_test_pred[idx])),
            'correct_prediction': bool(y_test[idx] == y_test_pred_classes[idx])
        }
        sample_predictions.append(sample_pred)

# Save sample predictions
with open('results/sample_predictions.json', 'w') as f:
    json.dump(sample_predictions, f, indent=2, ensure_ascii=False)

# --- 10. Save Final Model ---
print("\n--- 10. Saving Final Model ---")

# Save complete final model in Keras format
model.save('models/lstm_transformer_sentiment_model.keras')
print("Complete model saved to models/lstm_transformer_sentiment_model.keras")

# Save model architecture as JSON
model_json = model.to_json()
with open('models/model_architecture.json', 'w') as f:
    json.dump(model_json, f, indent=2)

print("Model architecture saved to models/model_architecture.json")

# --- 11. Final Summary ---
print("\n" + "="*60)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"Final Test Accuracy: {test_accuracy:.4f}")
print(f"Final Test F1-Score (Macro): {test_report['macro avg']['f1-score']:.4f}")
print(f"Training Duration: {training_duration}")
print(f"Model saved to: models/lstm_transformer_sentiment_model.keras")
print(f"Results saved to: results/ directory")
print("="*60)

print("\nAll artifacts have been generated successfully!")
print("Next steps: Update README.md and push to GitHub repository.")