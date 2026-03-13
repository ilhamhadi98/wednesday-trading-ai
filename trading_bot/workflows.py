from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .config import Settings
from .data_pipeline import build_multitimeframe_frame, build_multitimeframe_frame_h1_h4, make_sequences
from .modeling import predict_proba, train_model
from .mt5_client import MT5Client


@dataclass
class TrainValidationSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray


# Peta: timeframe → jumlah bar default & prefix kolom
_TF_BARS_DEFAULTS: dict[str, int] = {
    "M1": 100_000,
    "M5": 50_000,
    "M15": 135_000,
    "M30": 60_000,
    "H1": 10_000,
    "H4": 5_000,
    "D1": 2_000,
}
_TF_PREFIX: dict[str, str] = {
    "M1": "m1_",
    "M5": "m5_",
    "M15": "m15_",
    "M30": "m30_",
    "H1": "h1_",
    "H4": "h4_",
    "D1": "d1_",
}


def fetch_feature_frame(
    client: MT5Client,
    settings: Settings,
    symbol: str | None = None,
    fast_tf: str | None = None,
    slow_tf: str | None = None,
    bars_fast: int | None = None,
    bars_slow: int | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    active_symbol = symbol or settings.symbol
    tf_fast = fast_tf or settings.backtest_fast_timeframe
    tf_slow = slow_tf or settings.backtest_slow_timeframe

    # Jumlah bar: gunakan argumen eksplisit → lalu cek setting yang sesuai → lalu default peta
    if bars_fast is None:
        if tf_fast == "M15":
            bars_fast = settings.bars_m15
        elif tf_fast == "H1":
            bars_fast = settings.bars_h1
        else:
            bars_fast = _TF_BARS_DEFAULTS.get(tf_fast, 10_000)
    if bars_slow is None:
        if tf_slow == "H4":
            bars_slow = settings.bars_h4
        elif tf_slow == "H1":
            bars_slow = settings.bars_h1
        else:
            bars_slow = _TF_BARS_DEFAULTS.get(tf_slow, 5_000)

    fast_prefix = _TF_PREFIX.get(tf_fast, tf_fast.lower() + "_")
    slow_prefix = _TF_PREFIX.get(tf_slow, tf_slow.lower() + "_")

    df_fast = client.copy_rates(active_symbol, tf_fast, bars_fast)
    df_slow = client.copy_rates(active_symbol, tf_slow, bars_slow)
    frame, feature_cols = build_multitimeframe_frame(
        fast_df=df_fast,
        slow_df=df_slow,
        fast_prefix=fast_prefix,
        slow_prefix=slow_prefix,
    )
    return frame, feature_cols


def split_train_val(
    x: np.ndarray, y: np.ndarray, train_ratio: float = 0.8
) -> TrainValidationSplit:
    split_idx = int(len(x) * train_ratio)
    return TrainValidationSplit(
        x_train=x[:split_idx],
        y_train=y[:split_idx],
        x_val=x[split_idx:],
        y_val=y[split_idx:],
    )


def compute_binary_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, probs))
    else:
        out["auc"] = None
    return out


def train_once(
    frame: pd.DataFrame, feature_cols: list[str], settings: Settings
):
    dataset, scaler = make_sequences(
        frame=frame,
        feature_cols=feature_cols,
        lookback=settings.lookback,
        scaler=None,
        fit_scaler=True,
    )
    split = split_train_val(dataset.x, dataset.y, 0.8)
    model, history = train_model(
        x_train=split.x_train,
        y_train=split.y_train,
        x_val=split.x_val,
        y_val=split.y_val,
        settings=settings,
    )
    probs = predict_proba(model, split.x_val)
    metrics = compute_binary_metrics(split.y_val, probs, threshold=0.5)
    return model, scaler, history, metrics, dataset


def walk_forward_probabilities(
    frame: pd.DataFrame,
    feature_cols: list[str],
    settings: Settings,
    start_ratio: float = 0.7,
) -> tuple[pd.Index, np.ndarray]:
    n = len(frame)
    # Data training minimal = lookback*5 + 100 sequences agar model punya cukup contoh
    min_train_samples = settings.lookback * 5 + 100
    start_idx = max(settings.lookback + min_train_samples, int(n * start_ratio))
    start_idx = min(start_idx, n - 1)

    test_bars = n - start_idx
    print(f"[Walk-Forward] Total frame: {n:,} | Start idx: {start_idx:,} | Test bars: {test_bars:,}")
    print(f"[Walk-Forward] Retrain setiap {settings.retrain_every_bars} bar | Min train: {min_train_samples}")

    if test_bars < 10:
        print("[Walk-Forward] PERINGATAN: Test bars terlalu sedikit (<10)! "
              "Tambah BARS_H1/BARS_M15 di .env")
        return pd.Index([]), np.array([], dtype=np.float32)

    probs = []
    times = []
    model = None
    scaler = None
    last_train_idx = -1
    retrain_count = 0
    log_every = max(1, test_bars // 10)

    for i in range(start_idx, n):
        progress = i - start_idx
        if progress % log_every == 0:
            pct = progress / test_bars * 100
            print(f"[Walk-Forward] {pct:5.1f}% ({progress:,}/{test_bars:,}) | "
                  f"Retrains: {retrain_count} | Sinyal: {len(probs)}")

        needs_retrain = model is None or (i - last_train_idx) >= settings.retrain_every_bars
        if needs_retrain:
            train_frame = frame.iloc[:i].copy()
            train_data, scaler = make_sequences(
                frame=train_frame,
                feature_cols=feature_cols,
                lookback=settings.lookback,
                scaler=None,
                fit_scaler=True,
            )
            if len(train_data.x) < min_train_samples:
                print(f"[Walk-Forward] Skip retrain bar {i}: data tidak cukup "
                      f"({len(train_data.x)} < {min_train_samples})")
                continue
            split = split_train_val(train_data.x, train_data.y, train_ratio=0.85)
            model, _ = train_model(
                x_train=split.x_train,
                y_train=split.y_train,
                x_val=split.x_val,
                y_val=split.y_val,
                settings=settings,
            )
            last_train_idx = i
            retrain_count += 1

        if model is None or scaler is None:
            continue

        window = frame.iloc[i - settings.lookback : i].copy()
        window_features = window[feature_cols].values
        if np.isnan(window_features).any():
            continue
        window_scaled = scaler.transform(window_features)
        x_now = np.expand_dims(window_scaled, axis=0).astype(np.float32)
        prob = float(predict_proba(model, x_now)[0])
        probs.append(prob)
        times.append(frame.index[i])

    print(f"[Walk-Forward] Selesai. Total sinyal: {len(probs):,} dari {test_bars:,} bar")
    if len(probs) > 0:
        arr = np.array(probs, dtype=np.float32)
        buy_sigs = int((arr >= settings.prediction_buy_threshold).sum())
        sell_sigs = int((arr <= settings.prediction_sell_threshold).sum())
        print(f"[Walk-Forward] Distribusi: BUY={buy_sigs} | SELL={sell_sigs} | "
              f"HOLD={len(probs) - buy_sigs - sell_sigs}")
    return pd.Index(times), np.array(probs, dtype=np.float32)
