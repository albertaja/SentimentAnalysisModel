# LSTM-only Sentiment Classifier (with class balancing options)
# Simple, efficient baseline without Transformer; supports oversampling and focal/class-weighted loss

import os
import json
import pickle
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=== LSTM-only Sentiment Classifier ===")

# --- 1) Load data ---
X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')
X_val = np.load('data/processed/X_val.npy')
y_val = np.load('data/processed/y_val.npy')
X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')

with open('data/processed/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('data/processed/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

VOCAB_SIZE = len(tokenizer.word_index) + 1
MAX_LENGTH = X_train.shape[1]
NUM_CLASSES = len(label_encoder.classes_)
print(f"Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}, Vocab {VOCAB_SIZE}, MaxLen {MAX_LENGTH}")

# --- 2) Optional: class weights (balanced) ---
classes = np.unique(y_train)
cls_w = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
CLASS_WEIGHT = {int(c): float(w) for c, w in zip(classes, cls_w)}
print(f"Class weights: {CLASS_WEIGHT}")

# --- 3) Model (LSTM-only) ---

def build_lstm_model(vocab_size, max_len, num_classes, embed_dim=128, lstm_units=128, dropout=0.5):
    inp = layers.Input(shape=(max_len,))
    x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(inp)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False, dropout=0.4, recurrent_dropout=0.2))(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inp, out)

model = build_lstm_model(VOCAB_SIZE, MAX_LENGTH, NUM_CLASSES, embed_dim=128, lstm_units=128, dropout=0.5)
model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

# --- 4) Callbacks ---
ckpt_path = 'models/best_lstm.keras'
ckpt = callbacks.ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, verbose=1)
early = callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
reduce = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

# --- 5) Train ---
print("\n--- Training LSTM-only ---")
start = datetime.now()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=40,
    batch_size=32,
    class_weight=CLASS_WEIGHT,  # balances classes
    callbacks=[ckpt, early, reduce],
    verbose=1
)
end = datetime.now()
print(f"Training time: {end - start}")

with open('results/training_history_lstm.json', 'w') as f:
    json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

# --- 6) Evaluate ---
best = tf.keras.models.load_model(ckpt_path)
val_loss, val_acc = best.evaluate(X_val, y_val, verbose=0)
test_loss, test_acc = best.evaluate(X_test, y_test, verbose=0)
print(f"Validation: loss={val_loss:.4f}, acc={val_acc:.4f}")
print(f"Test: loss={test_loss:.4f}, acc={test_acc:.4f}")

y_val_prob = best.predict(X_val, verbose=0)
y_test_prob = best.predict(X_test, verbose=0)
y_val_pred = np.argmax(y_val_prob, axis=1)
y_test_pred = np.argmax(y_test_prob, axis=1)

val_report = classification_report(y_val, y_val_pred, target_names=[f'Class_{i}' for i in range(NUM_CLASSES)], output_dict=True)
test_report = classification_report(y_test, y_test_pred, target_names=[f'Class_{i}' for i in range(NUM_CLASSES)], output_dict=True)

with open('results/validation_classification_report_lstm.json', 'w') as f:
    json.dump(val_report, f, indent=2)
with open('results/test_classification_report_lstm.json', 'w') as f:
    json.dump(test_report, f, indent=2)

# Confusion matrices
val_cm = confusion_matrix(y_val, y_val_pred)
test_cm = confusion_matrix(y_test, y_test_pred)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', ax=ax1, xticklabels=[f'Pred_{i}' for i in range(NUM_CLASSES)], yticklabels=[f'True_{i}' for i in range(NUM_CLASSES)])
ax1.set_title('Validation Confusion Matrix (LSTM)')
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', ax=ax2, xticklabels=[f'Pred_{i}' for i in range(NUM_CLASSES)], yticklabels=[f'True_{i}' for i in range(NUM_CLASSES)])
ax2.set_title('Test Confusion Matrix (LSTM)')
plt.tight_layout()
plt.savefig('results/confusion_matrices_lstm.png', dpi=300, bbox_inches='tight')
plt.close()

# Save final LSTM model
best.save('models/lstm_only_sentiment_model.keras')
print("Saved LSTM-only model to models/lstm_only_sentiment_model.keras")
