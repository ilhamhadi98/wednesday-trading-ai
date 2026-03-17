"""
Adaptive Retraining Module
==========================
Melakukan training ulang model secara otomatis pada pair tertentu
jika consecutive losses >= N (default: 5).

Fitur:
- Per-symbol model: setiap pair memiliki model sendiri di outputs/model_{SYMBOL}.keras
- Berjalan di background thread agar tidak memblokir live trading pair lain
- Menggunakan data 3 bulan terakhir yang diambil langsung dari MT5
- Setelah training selesai, model baru dipakai langsung untuk pair tersebut
"""
from __future__ import annotations

import json
import pickle
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    import MetaTrader5 as mt5
except ImportError as exc:
    raise RuntimeError("MetaTrader5 belum terinstall.") from exc

from .config import Settings, OUTPUT_DIR
from .mt5_client import MT5Client
from .modeling import build_lstm_model, ModelArtifacts, save_training_history
from .workflows import fetch_feature_frame, train_once
from .adaptive_sizing import analyze_performance


# Set untuk melacak pair yang sedang dalam proses retraining (hindari duplikat)
_retraining_in_progress: set[str] = set()
_lock = threading.Lock()


def get_symbol_model_paths(symbol: str) -> tuple[Path, Path, Path]:
    """Kembalikan path model, scaler, dan meta khusus untuk symbol ini."""
    safe = symbol.replace("/", "_").upper()
    return (
        OUTPUT_DIR / f"model_{safe}.keras",
        OUTPUT_DIR / f"scaler_{safe}.pkl",
        OUTPUT_DIR / f"model_meta_{safe}.json",
    )


def load_symbol_artifacts(symbol: str) -> ModelArtifacts | None:
    """
    Muat model khusus untuk symbol ini jika tersedia.
    Returns None jika belum ada (gunakan model global sebagai fallback).
    """
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler

    model_path, scaler_path, meta_path = get_symbol_model_paths(symbol)
    if not (model_path.exists() and scaler_path.exists() and meta_path.exists()):
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        try:
            model = tf.keras.models.load_model(str(model_path), safe_mode=False)
        except Exception:
            from .config import load_settings
            s = load_settings()
            input_shape = (meta["lookback"], len(meta["feature_cols"]))
            model = build_lstm_model(input_shape, s.learning_rate)
            model.load_weights(str(model_path))

        print(f"[RETRAIN] Loaded per-symbol model for {symbol}", flush=True)
        return ModelArtifacts(
            model=model,
            scaler=scaler,
            feature_cols=meta["feature_cols"],
            lookback=meta["lookback"],
        )
    except Exception as e:
        print(f"[RETRAIN] Gagal load model untuk {symbol}: {e}", flush=True)
        return None


def _do_retrain(symbol: str, client: MT5Client, settings: Settings) -> None:
    """
    Fungsi retraining yang berjalan di thread terpisah.
    Ambil 3 bulan data, latih model, simpan per-symbol.
    """
    try:
        print(f"\n[RETRAIN] Mulai training ulang {symbol} (3 bulan terakhir)...", flush=True)

        # Hitung jumlah bar untuk 3 bulan
        # M15: 3 bulan × 30 hari × 24 jam × 4 bar/jam = ~8640 bar (tambah buffer 20%)
        # H1: 3 bulan × 30 hari × 24 jam = ~2160 bar
        bars_m15 = int(3 * 30 * 24 * 4 * 1.2) + 500  # ~10808
        bars_h1 = int(3 * 30 * 24 * 1.2) + 200       # ~2792

        # Batasi agar tidak terlalu besar / timeout MT5
        bars_m15 = min(bars_m15, 50_000)
        bars_h1 = min(bars_h1, 10_000)

        print(
            f"[RETRAIN] {symbol}: mengambil {bars_m15} bar M15 dan {bars_h1} bar H1...",
            flush=True,
        )

        frame, feature_cols = fetch_feature_frame(
            client,
            settings,
            symbol=symbol,
            fast_tf=settings.live_fast_timeframe,
            slow_tf=settings.live_slow_timeframe,
            bars_fast=bars_m15,
            bars_slow=bars_h1,
        )

        if len(frame) < settings.min_bars_required:
            print(
                f"[RETRAIN] GAGAL: Data {symbol} tidak cukup ({len(frame)} bar). Dibatalkan.",
                flush=True,
            )
            return

        print(f"[RETRAIN] {symbol}: {len(frame)} bar diterima. Mulai training...", flush=True)

        model, scaler, history, metrics, _ = train_once(
            frame=frame,
            feature_cols=feature_cols,
            settings=settings,
        )

        # Simpan model per-symbol
        model_path, scaler_path, meta_path = get_symbol_model_paths(symbol)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        model.save(str(model_path))
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"feature_cols": feature_cols, "lookback": settings.lookback}, f, indent=2)

        save_training_history(
            history,
            path=OUTPUT_DIR / f"training_history_{symbol.replace('/', '_')}.json",
        )

        print(
            f"[RETRAIN] SELESAI: {symbol} selesai dilatih ulang! "
            f"AUC={metrics.get('auc', 'N/A'):.4f} | "
            f"F1={metrics.get('f1', 'N/A'):.4f} | "
            f"Accuracy={metrics.get('accuracy', 'N/A'):.4f}",
            flush=True,
        )
        print(f"[RETRAIN] Model baru disimpan: {model_path}", flush=True)

    except Exception as e:
        print(f"[RETRAIN] ERROR saat retrain {symbol}: {e}", flush=True)
    finally:
        with _lock:
            _retraining_in_progress.discard(symbol)


def check_and_trigger_retrain(
    symbol: str,
    client: MT5Client,
    settings: Settings,
    consecutive_loss_threshold: int = 5,
    n_trades: int = 20,
    days_back: int = 30,
) -> bool:
    """
    Cek apakah pair ini perlu dilatih ulang berdasarkan consecutive losses.
    
    Args:
        symbol: simbol pair yang akan dicek
        client: MT5Client yang sudah terkoneksi
        settings: pengaturan bot
        consecutive_loss_threshold: jumlah loss berturut untuk trigger (default: 5)
        
    Returns:
        True jika retraining dimulai, False jika tidak perlu
    """
    with _lock:
        if symbol in _retraining_in_progress:
            print(
                f"[RETRAIN] {symbol} sudah dalam proses retraining, skip.",
                flush=True,
            )
            return False

    perf = analyze_performance(symbol=symbol, n_trades=n_trades, days_back=days_back)

    if perf.consecutive_losses < consecutive_loss_threshold:
        return False

    print(
        f"\n[RETRAIN] TRIGGER! {symbol} mengalami {perf.consecutive_losses} loss berturut-turut "
        f"(threshold={consecutive_loss_threshold}). Memulai retraining otomatis di background...",
        flush=True,
    )

    with _lock:
        _retraining_in_progress.add(symbol)

    thread = threading.Thread(
        target=_do_retrain,
        args=(symbol, client, settings),
        daemon=True,
        name=f"retrain-{symbol}",
    )
    thread.start()
    return True


def is_retraining(symbol: str) -> bool:
    """Cek apakah symbol sedang dalam proses retraining."""
    with _lock:
        return symbol in _retraining_in_progress
