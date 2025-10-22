# LSTM + Transformer Hybrid Model for Sentiment Analysis (Balanced Training)
# This script trains a deep learning model on Indonesian text reviews with class imbalance handling

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
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.saving import register_keras_serializable
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

print("=== LSTM + Transformer Hybrid Model Training (Balanced) ===")
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

# Basic stats
print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test.shape}")
print(f"Sequence length: {X_train.shape[1]}")
print(f"Vocabulary size: {len(tokenizer.word_index)}")
print(f"Number of classes: {len(label_encoder.classes_)}")
print(f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# Class distribution and weights
classes = np.unique(y_train)
class_counts = {int(c): int((y_train == c).sum()) for c in classes}
print(f"Training class distribution: {class_counts}")

# Compute class weights (balanced)
class_weight_vals = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight = {int(c): float(w) for c, w in zip(classes, class_weight_vals)}
print(f"Computed class weights: {class_weight}")

# --- 2. Model Architecture: LSTM + Transformer Hybrid ---
print("\n--- 2. Building LSTM + Transformer Hybrid Model ---")

@register_keras_serializable(package="Custom", name="TransformerBlock")
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
    inputs = layers.Input(shape=(max_length,))
    x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(inputs)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, dropout=0.4, recurrent_dropout=0.2))(x)
    x = layers.Dense(embed_dim * 2)(x)
    x = TransformerBlock(embed_dim*2, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

VOCAB_SIZE = len(tokenizer.word_index) + 1
MAX_LENGTH = X_train.shape[1]
NUM_CLASSES = len(label_encoder.classes_)

model = create_lstm_transformer_model(VOCAB_SIZE, MAX_LENGTH, embed_dim=128, lstm_units=64, num_heads=4, ff_dim=128, num_classes=NUM_CLASSES)
print("Model Architecture:")
model.summary()

# --- 3. Training Configuration ---
print("\n--- 3. Training Configuration ---")
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

BATCH_SIZE = 32
EPOCHS = 40
PATIENCE = 6

print("Optimizer: Adam (1e-3)")
print("Loss: sparse_categorical_crossentropy (with class_weight)")
print(f"Batch size: {BATCH_SIZE}, Max epochs: {EPOCHS}, Patience: {PATIENCE}")

# --- 4. Callbacks ---
print("\n--- 4. Setting up Callbacks ---")
early_stopping = callbacks.EarlyStopping(monitor='val_macro_f1' if False else 'val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1)
model_ckpt = callbacks.ModelCheckpoint('models/best_balanced_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

callback_list = [early_stopping, model_ckpt, reduce_lr]

# --- 5. Training ---
print("\n--- 5. Training (with class weights) ---")
start_time = datetime.now()
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    class_weight=class_weight,
                    callbacks=callback_list,
                    verbose=1)
end_time = datetime.now()
print(f"Training duration: {end_time - start_time}")

# Save history
with open('results/training_history_balanced.json', 'w') as f:
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    json.dump(history_dict, f, indent=2)

# --- 6. Evaluation ---
print("\n--- 6. Evaluation ---")
best_model = tf.keras.models.load_model('models/best_balanced_model.keras')
val_loss, val_acc = best_model.evaluate(X_val, y_val, verbose=0)
test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)
print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
print(f"Test - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

y_val_pred = best_model.predict(X_val, verbose=0)
y_test_pred = best_model.predict(X_test, verbose=0)

y_val_pred_classes = np.argmax(y_val_pred, axis=1)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)

val_report = classification_report(y_val, y_val_pred_classes, target_names=[f'Class_{i}' for i in range(NUM_CLASSES)], output_dict=True)
test_report = classification_report(y_test, y_test_pred_classes, target_names=[f'Class_{i}' for i in range(NUM_CLASSES)], output_dict=True)

with open('results/validation_classification_report_balanced.json', 'w') as f:
    json.dump(val_report, f, indent=2)
with open('results/test_classification_report_balanced.json', 'w') as f:
    json.dump(test_report, f, indent=2)

print("Balanced classification reports saved.")

# Confusion matrices
val_cm = confusion_matrix(y_val, y_val_pred_classes)
test_cm = confusion_matrix(y_test, y_test_pred_classes)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', ax=ax1, xticklabels=[f'Pred_{i}' for i in range(NUM_CLASSES)], yticklabels=[f'True_{i}' for i in range(NUM_CLASSES)])
ax1.set_title('Validation Confusion Matrix (Balanced)')
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', ax=ax2, xticklabels=[f'Pred_{i}' for i in range(NUM_CLASSES)], yticklabels=[f'True_{i}' for i in range(NUM_CLASSES)])
ax2.set_title('Test Confusion Matrix (Balanced)')
plt.tight_layout()
plt.savefig('results/confusion_matrices_balanced.png', dpi=300, bbox_inches='tight')
plt.close()
print("Balanced confusion matrices saved.")

# Save final balanced model
best_model.save('models/lstm_transformer_sentiment_model_balanced.keras')
print("Final balanced model saved to models/lstm_transformer_sentiment_model_balanced.keras")
