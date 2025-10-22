# LSTM + Transformer Hybrid with Focal Loss and Macro-F1 Monitoring
# Improves minority-class performance via focal loss, oversampling, and macro-F1 early stopping

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils import resample
from tensorflow.keras.saving import register_keras_serializable
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("=== LSTM+Transformer with Focal Loss (Balanced) ===")

# --- 1. Load data ---
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

print(f"Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}, Classes {NUM_CLASSES}")

# --- 2. Optional oversampling for minority classes ---
print("\n--- Oversampling minority classes ---")
Xy = np.hstack([X_train, y_train.reshape(-1, 1)])
Xy_0 = Xy[Xy[:, -1] == 0]
Xy_1 = Xy[Xy[:, -1] == 1]
Xy_2 = Xy[Xy[:, -1] == 2]
max_n = max(len(Xy_0), len(Xy_1), len(Xy_2))
Xy_0_up = resample(Xy_0, replace=True, n_samples=max_n, random_state=RANDOM_SEED)
Xy_1_up = resample(Xy_1, replace=True, n_samples=max_n, random_state=RANDOM_SEED)
Xy_2_up = resample(Xy_2, replace=True, n_samples=max_n, random_state=RANDOM_SEED)
Xy_up = np.vstack([Xy_0_up, Xy_1_up, Xy_2_up])
np.random.shuffle(Xy_up)
X_train_bal = Xy_up[:, :-1].astype(np.int32)
y_train_bal = Xy_up[:, -1].astype(np.int32)
print(f"Balanced train shape: {X_train_bal.shape}, class counts: {dict(zip(*np.unique(y_train_bal, return_counts=True)))}")

# --- 3. Model ---
@register_keras_serializable(package="Custom", name="TransformerBlock")
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim)
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.do1 = layers.Dropout(rate)
        self.do2 = layers.Dropout(rate)
    def call(self, x, training=None):
        h = self.att(x, x, training=training)
        h = self.do1(h, training=training)
        x = self.norm1(x + h)
        h = self.ffn(x, training=training)
        h = self.do2(h, training=training)
        return self.norm2(x + h)
    def get_config(self):
        cfg = super().get_config()
        cfg.update({'embed_dim': self.embed_dim})
        return cfg

def build_model(vocab_size, max_len, num_classes, embed_dim=128, lstm_units=64, heads=2, ff_dim=128):
    inp = layers.Input(shape=(max_len,))
    x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(inp)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True, dropout=0.5, recurrent_dropout=0.2))(x)
    x = layers.Dense(embed_dim*2)(x)
    x = TransformerBlock(embed_dim*2, heads, ff_dim, rate=0.2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.6)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.6)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inp, out)

model = build_model(VOCAB_SIZE, MAX_LENGTH, NUM_CLASSES)
print(model.summary())

# --- 4. Focal loss (alpha from class distribution, gamma=2) ---
counts = np.bincount(y_train)
alpha_raw = 1.0 / np.maximum(counts, 1)
alpha = alpha_raw / alpha_raw.sum()
alpha = tf.constant(alpha, dtype=tf.float32)

def focal_loss(y_true, y_pred, gamma=2.0, alpha_vec=alpha):
    y_true = tf.cast(y_true, tf.int32)
    y_true_oh = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    ce = -tf.reduce_sum(y_true_oh * tf.math.log(y_pred), axis=-1)
    pt = tf.reduce_sum(y_true_oh * y_pred, axis=-1)
    alpha_w = tf.reduce_sum(y_true_oh * alpha_vec, axis=-1)
    fl = alpha_w * tf.pow(1.0 - pt, gamma) * ce
    return tf.reduce_mean(fl)

# --- 5. Compile ---
opt = keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=opt, loss=focal_loss, metrics=['accuracy'])

# --- 6. Macro-F1 callback for validation ---
class MacroF1Callback(callbacks.Callback):
    def __init__(self, x_val, y_val):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.best_f1 = -1
        self.history = []
    def on_epoch_end(self, epoch, logs=None):
        y_prob = self.model.predict(self.x_val, verbose=0)
        y_pred = np.argmax(y_prob, axis=1)
        f1 = f1_score(self.y_val, y_pred, average='macro', zero_division=0)
        logs = logs or {}
        logs['val_macro_f1'] = f1
        self.history.append(float(f1))
        print(f"Epoch {epoch+1}: val_macro_f1={f1:.4f}")
        # Save best model by macro F1
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.model.save('models/best_focal_model.keras')
            print(f"Saved best model (macro F1 improved to {f1:.4f})")

f1_cb = MacroF1Callback(X_val, y_val)

early = callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
reduce = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

# --- 7. Train ---
print("\n--- Training (oversampled + focal loss) ---")
hist = model.fit(
    X_train_bal, y_train_bal,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early, reduce, f1_cb],
    verbose=1
)

with open('results/training_history_focal.json', 'w') as f:
    json.dump({k: [float(v) for v in vals] for k, vals in hist.history.items()}, f, indent=2)

# --- 8. Evaluate best focal model by macro F1 ---
best = tf.keras.models.load_model('models/best_focal_model.keras', custom_objects={'focal_loss': focal_loss, 'TransformerBlock': TransformerBlock})
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

with open('results/validation_classification_report_focal.json', 'w') as f:
    json.dump(val_report, f, indent=2)
with open('results/test_classification_report_focal.json', 'w') as f:
    json.dump(test_report, f, indent=2)

# Confusion matrices
val_cm = confusion_matrix(y_val, y_val_pred)
test_cm = confusion_matrix(y_test, y_test_pred)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues', ax=ax1, xticklabels=[f'Pred_{i}' for i in range(NUM_CLASSES)], yticklabels=[f'True_{i}' for i in range(NUM_CLASSES)])
ax1.set_title('Validation Confusion Matrix (Focal)')
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', ax=ax2, xticklabels=[f'Pred_{i}' for i in range(NUM_CLASSES)], yticklabels=[f'True_{i}' for i in range(NUM_CLASSES)])
ax2.set_title('Test Confusion Matrix (Focal)')
plt.tight_layout()
plt.savefig('results/confusion_matrices_focal.png', dpi=300, bbox_inches='tight')
plt.close()

# Save final model
best.save('models/lstm_transformer_sentiment_model_focal.keras')
print("Saved focal-loss model to models/lstm_transformer_sentiment_model_focal.keras")
