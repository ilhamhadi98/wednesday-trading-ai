from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from .config import MODEL_PATH, OUTPUT_DIR, SCALER_PATH, Settings, ensure_output_dir


META_PATH = OUTPUT_DIR / "model_meta.json"


@dataclass
class ModelArtifacts:
    model: tf.keras.Model
    scaler: StandardScaler
    feature_cols: list[str]
    lookback: int


def set_seed(seed: int = 42) -> None:
    tf.keras.utils.set_random_seed(seed)


def build_lstm_model(input_shape: tuple[int, int], learning_rate: float) -> tf.keras.Model:
    """Arsitektur BiLSTM + Self-Attention yang lebih robust untuk loss minimisation.

    Perbaikan vs versi lama:
    - L2 regularization pada LSTM & Dense untuk mencegah overfitting
    - Self-Attention layer: model belajar fokus ke bar yang paling relevan
    - BatchNormalization untuk stabilitas training
    - Dropout lebih agresif (0.35) untuk generalisasi lebih baik
    - Dense layer tambahan sebelum output untuk kapasitas representasi
    """
    inputs = layers.Input(shape=input_shape, name="sequence_input")

    # ── Bidirectional LSTM pertama ─────────────────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(
            96,
            return_sequences=True,
            kernel_regularizer=l2(1e-4),
            recurrent_regularizer=l2(1e-4),
        ),
        name="bilstm_1",
    )(inputs)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(0.30, name="drop_1")(x)

    # ── Bidirectional LSTM kedua ───────────────────────────────────────────
    x = layers.Bidirectional(
        layers.LSTM(
            48,
            return_sequences=True,
            kernel_regularizer=l2(1e-4),
        ),
        name="bilstm_2",
    )(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(0.30, name="drop_2")(x)

    # ── Self-Attention ─────────────────────────────────────────────────────
    # Beri bobot lebih tinggi ke bar yang paling informatif
    attn_scores = layers.Dense(1, activation="tanh", name="attn_score")(x)   # (B, T, 1)
    attn_weights = layers.Softmax(axis=1, name="attn_weights")(attn_scores)   # (B, T, 1)
    context = layers.Multiply(name="attn_context")([x, attn_weights])         # (B, T, 96)
    context = layers.Lambda(
        lambda t: tf.reduce_sum(t, axis=1), name="attn_sum"
    )(context)                                                                 # (B, 96)

    # ── Head ──────────────────────────────────────────────────────────────
    x = layers.Dense(64, activation="relu", kernel_regularizer=l2(1e-4), name="dense_1")(context)
    x = layers.BatchNormalization(name="bn_3")(x)
    x = layers.Dropout(0.25, name="drop_3")(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=l2(1e-4), name="dense_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="TradingLSTM_Attention")
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),  # gradient clipping
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    settings: Settings,
) -> tuple[tf.keras.Model, dict]:
    set_seed(42)
    model = build_lstm_model(
        input_shape=(x_train.shape[1], x_train.shape[2]),
        learning_rate=settings.learning_rate,
    )

    from .config import MODEL_PATH
    import os
    if os.path.exists(MODEL_PATH):
        try:
            # We try to load weights even if it's a full model file, 
            # as it's the most flexible way to transfer learned state to a new architecture
            model.load_weights(str(MODEL_PATH))
            print(f"[OK] [Modeling] Loaded existing weights to continue learning!")
        except Exception as e:
            # If load_weights fails (e.g. it's a full model file that load_weights doesn't like), 
            # we try to load the whole model
            try:
                temp_model = tf.keras.models.load_model(str(MODEL_PATH), safe_mode=False)
                model.set_weights(temp_model.get_weights())
                print(f"[OK] [Modeling] Loaded weights from full model file!")
            except Exception as e2:
                print(f"[WARN] [Modeling] Could not load weights/model: {e2}. Starting fresh.")

    # Hitung class_weight untuk menangani ketidakseimbangan BUY vs SELL
    n_total = len(y_train)
    n_pos = max(int(y_train.sum()), 1)
    n_neg = max(n_total - n_pos, 1)
    class_weight = {
        0: n_total / (2 * n_neg),
        1: n_total / (2 * n_pos),
    }

    callbacks = [
        EarlyStopping(
            monitor="val_auc",
            patience=8,          # kesabaran lebih untuk model yang lebih kompleks
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.4,
            patience=4,
            min_lr=5e-6,
            verbose=1,
        ),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=settings.epochs,
        batch_size=settings.batch_size,
        verbose=1,
        callbacks=callbacks,
        class_weight=class_weight,
    )
    
    # Simpan FULL model agar load_model() berhasil di live trading
    try:
        model.save(str(MODEL_PATH))
    except Exception as e:
        print(f"[WARN] [Modeling] Failed to save updated model: {e}")
        
    return model, history.history


def predict_proba(model: tf.keras.Model, x: np.ndarray) -> np.ndarray:
    preds = model.predict(x, verbose=0).reshape(-1)
    return preds


def save_artifacts(
    model: tf.keras.Model,
    scaler: StandardScaler,
    feature_cols: list[str],
    lookback: int,
) -> None:
    ensure_output_dir()
    model.save(MODEL_PATH)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"feature_cols": feature_cols, "lookback": lookback}, f, indent=2)


def load_artifacts() -> ModelArtifacts:
    if not MODEL_PATH.exists() or not SCALER_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            "Artifact model belum ditemukan. Jalankan training dulu (train_model.py)."
        )
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    try:
        model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)
    except Exception as e:
        print(f"[WARN] [Modeling] Failed to load full model structure (likely Lambda shape inference error). Rebuilding and loading weights...")
        from .config import load_settings
        settings = load_settings()
        input_shape = (meta["lookback"], len(meta["feature_cols"]))
        model = build_lstm_model(input_shape, settings.learning_rate)
        model.load_weights(MODEL_PATH)

    return ModelArtifacts(
        model=model,
        scaler=scaler,
        feature_cols=meta["feature_cols"],
        lookback=meta["lookback"],
    )


def save_training_history(history: dict, path: Path | None = None) -> Path:
    ensure_output_dir()
    out = path or OUTPUT_DIR / "training_history.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    return out
