from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# Indikator teknikal
# ─────────────────────────────────────────────────────────────────────────────

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1 / period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-10)
    return 100 - (100 / (1 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Kembalikan (macd_line, signal_line, histogram)."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger(series: pd.Series, period: int = 20, n_std: float = 2.0):
    """Kembalikan (upper, middle, lower, %B)."""
    middle = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = middle + n_std * std
    lower = middle - n_std * std
    pct_b = (series - lower) / (upper - lower + 1e-10)
    return upper, middle, lower, pct_b


def _stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    """Kembalikan (%K, %D)."""
    lowest_low = df["low"].rolling(k_period).min()
    highest_high = df["high"].rolling(k_period).max()
    pct_k = (df["close"] - lowest_low) / (highest_high - lowest_low + 1e-10) * 100
    pct_d = pct_k.rolling(d_period).mean()
    return pct_k, pct_d


def _cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    mean_typical = typical.rolling(period).mean()
    mean_dev = typical.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
    return (typical - mean_typical) / (0.015 * mean_dev + 1e-10)


def _williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Williams %R."""
    highest_high = df["high"].rolling(period).max()
    lowest_low = df["low"].rolling(period).min()
    return -100 * (highest_high - df["close"]) / (highest_high - lowest_low + 1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering utama
# ─────────────────────────────────────────────────────────────────────────────

def add_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("time").reset_index(drop=True)

    close = out["close"]

    # ── Price action ───────────────────────────────────────────────────────
    out[f"{prefix}ret_1"] = close.pct_change(1)
    out[f"{prefix}ret_4"] = close.pct_change(4)
    out[f"{prefix}ret_8"] = close.pct_change(8)
    out[f"{prefix}ret_16"] = close.pct_change(16)
    out[f"{prefix}range"] = (out["high"] - out["low"]) / (close + 1e-10)
    out[f"{prefix}body"] = (close - out["open"]) / (out["open"] + 1e-10)
    # Wick dominance: apakah wick atas atau bawah lebih dominan
    wick_up = out["high"] - close.clip(lower=out["open"])
    wick_dn = close.clip(upper=out["open"]) - out["low"]
    out[f"{prefix}wick_ratio"] = (wick_up - wick_dn) / (out["high"] - out["low"] + 1e-10)

    # ── Moving averages ────────────────────────────────────────────────────
    out[f"{prefix}sma_10"] = close.rolling(10).mean()
    out[f"{prefix}sma_20"] = close.rolling(20).mean()
    out[f"{prefix}sma_50"] = close.rolling(50).mean()
    out[f"{prefix}ema_9"] = close.ewm(span=9, adjust=False).mean()
    out[f"{prefix}ema_20"] = close.ewm(span=20, adjust=False).mean()
    # SMA crossover ratio (MA cepat / MA lambat − 1)
    out[f"{prefix}sma_cross"] = out[f"{prefix}sma_10"] / (out[f"{prefix}sma_50"] + 1e-10) - 1

    # ── Oscillators ────────────────────────────────────────────────────────
    out[f"{prefix}rsi_7"] = _rsi(close, 7)
    out[f"{prefix}rsi_14"] = _rsi(close, 14)
    out[f"{prefix}rsi_21"] = _rsi(close, 21)

    stoch_k, stoch_d = _stochastic(out)
    out[f"{prefix}stoch_k"] = stoch_k
    out[f"{prefix}stoch_d"] = stoch_d
    out[f"{prefix}stoch_diff"] = stoch_k - stoch_d

    out[f"{prefix}cci"] = _cci(out)
    out[f"{prefix}willr"] = _williams_r(out)

    # ── MACD ───────────────────────────────────────────────────────────────
    macd_line, macd_sig, macd_hist = _macd(close)
    out[f"{prefix}macd"] = macd_line
    out[f"{prefix}macd_sig"] = macd_sig
    out[f"{prefix}macd_hist"] = macd_hist

    # ── Volatilitas ────────────────────────────────────────────────────────
    out[f"{prefix}atr_14"] = _atr(out, 14)
    # ATR relatif terhadap close (normalisasi)
    out[f"{prefix}atr_pct"] = out[f"{prefix}atr_14"] / (close + 1e-10)

    boll_up, boll_mid, boll_lo, boll_pctb = _bollinger(close)
    out[f"{prefix}boll_upper"] = boll_up
    out[f"{prefix}boll_lower"] = boll_lo
    out[f"{prefix}boll_pctb"] = boll_pctb
    out[f"{prefix}boll_width"] = (boll_up - boll_lo) / (boll_mid + 1e-10)

    # ── Volume ─────────────────────────────────────────────────────────────
    out[f"{prefix}volume_z"] = (
        (out["volume"] - out["volume"].rolling(30).mean())
        / (out["volume"].rolling(30).std() + 1e-10)
    )
    # Volume MA ratio
    out[f"{prefix}volume_ma_ratio"] = out["volume"] / (out["volume"].rolling(10).mean() + 1e-10)

    out = out.rename(
        columns={
            "open": f"{prefix}open",
            "high": f"{prefix}high",
            "low": f"{prefix}low",
            "close": f"{prefix}close",
            "volume": f"{prefix}volume",
            "spread": f"{prefix}spread",
        }
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Multi-timeframe frame builder (fleksibel: bisa H1+H4 atau M15+H1)
# ─────────────────────────────────────────────────────────────────────────────

def build_multitimeframe_frame(
    fast_df: pd.DataFrame,     # timeframe cepat (contoh M15 atau H1)
    slow_df: pd.DataFrame,     # timeframe lambat (contoh H1 atau H4)
    fast_prefix: str = "h1_",  # prefix kolom timeframe cepat
    slow_prefix: str = "h4_",  # prefix kolom timeframe lambat
) -> tuple[pd.DataFrame, list[str]]:
    fast = add_features(fast_df, fast_prefix).set_index("time")
    slow = add_features(slow_df, slow_prefix).set_index("time")

    slow_aligned = slow.reindex(fast.index, method="ffill")
    full = pd.concat([fast, slow_aligned], axis=1)

    # MT5 biasanya mengikutkan candle yang sedang berjalan di bar terakhir.
    # Buang bar ini agar label training/backtest memakai candle sudah close.
    if len(full) > 2:
        full = full.iloc[:-1].copy()

    full["future_ret_1"] = full[f"{fast_prefix}close"].shift(-1) / full[f"{fast_prefix}close"] - 1.0
    full["target"] = (full["future_ret_1"] > 0).astype(int)
    full = full.dropna().copy()

    # Simpan close/spread untuk backtest eksekusi.
    full["exec_close"] = full[f"{fast_prefix}close"]
    full["exec_spread"] = full[f"{fast_prefix}spread"]

    ignore_cols = {
        "target",
        "future_ret_1",
        "exec_close",
        "exec_spread",
    }
    feature_cols = [
        col
        for col in full.columns
        if col not in ignore_cols and pd.api.types.is_numeric_dtype(full[col])
    ]
    return full, feature_cols


# Alias backward-compatible untuk kode lama yang masih memakai H1/H4
def build_multitimeframe_frame_h1_h4(
    h1_df: pd.DataFrame, h4_df: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    return build_multitimeframe_frame(
        fast_df=h1_df,
        slow_df=h4_df,
        fast_prefix="h1_",
        slow_prefix="h4_",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sequence dataset
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SequenceDataset:
    x: np.ndarray
    y: np.ndarray
    timestamps: pd.Index
    close: np.ndarray
    spread: np.ndarray


def make_sequences(
    frame: pd.DataFrame,
    feature_cols: list[str],
    lookback: int,
    scaler: StandardScaler | None = None,
    fit_scaler: bool = True,
) -> tuple[SequenceDataset, StandardScaler]:
    work = frame.copy()
    features = work[feature_cols].values

    if scaler is None:
        scaler = StandardScaler()
        fit_scaler = True

    if fit_scaler:
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)

    x, y, times, closes, spreads = [], [], [], [], []
    target = work["target"].values.astype(np.float32)
    close = work["exec_close"].values.astype(np.float64)
    spread = work["exec_spread"].values.astype(np.float64)

    for i in range(lookback, len(work)):
        x.append(features[i - lookback : i, :])
        y.append(target[i])
        times.append(work.index[i])
        closes.append(close[i])
        spreads.append(spread[i])

    dataset = SequenceDataset(
        x=np.array(x, dtype=np.float32),
        y=np.array(y, dtype=np.float32),
        timestamps=pd.Index(times),
        close=np.array(closes, dtype=np.float64),
        spread=np.array(spreads, dtype=np.float64),
    )
    return dataset, scaler
