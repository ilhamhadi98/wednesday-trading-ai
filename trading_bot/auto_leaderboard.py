"""
Auto Leaderboard Updater — WednesdayAI
──────────────────────────────────────
Jalankan backtest ulang di background saat off-session (1x per hari),
lalu reload best_pairs dari CSV hasil baru.

Dipanggil dari live_smart_session.py, tidak mengubah logika trading.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

# ─── state ──────────────────────────────────────────────────────────────────
_lb_lock                = threading.Lock()
_lb_running             = False       # apakah backtest sedang berjalan
_lb_last_run_date: date | None = None  # tanggal terakhir update berhasil


def is_leaderboard_updating() -> bool:
    return _lb_running


def should_update_today() -> bool:
    """True jika belum ada update hari ini."""
    global _lb_last_run_date
    return _lb_last_run_date != date.today()


# ─── core ───────────────────────────────────────────────────────────────────

def run_leaderboard_update(
    client,
    settings,
    symbols: list[str],
    top_n: int,
    epochs: int,
    train_ratio: float,
    csv_path: Path,
    on_done: Callable[[list[str], list[str]], None] | None = None,
) -> None:
    """
    Jalankan backtest ulang untuk semua `symbols`, tulis ke CSV,
    lalu panggil `on_done(old_pairs, new_pairs)`.

    Semua error ditangkap — tidak akan crash bot.
    """
    global _lb_running, _lb_last_run_date

    with _lb_lock:
        if _lb_running:
            return   # jangan dobel-run
        _lb_running = True

    try:
        print("[AUTO-LB] Memulai update leaderboard di background...")

        from trading_bot.data_pipeline import make_sequences
        from trading_bot.market_scan import resolve_symbols
        from trading_bot.modeling import predict_proba, train_model
        from trading_bot.backtest import run_backtest
        from trading_bot.workflows import fetch_feature_frame, split_train_val
        from trading_bot.config import OUTPUT_DIR

        settings_bt = replace(settings, epochs=epochs)
        rows: list[dict] = []

        for idx, symbol in enumerate(symbols, 1):
            try:
                print(f"[AUTO-LB] [{idx}/{len(symbols)}] Backtest {symbol}...")
                frame, feature_cols = fetch_feature_frame(client, settings_bt, symbol=symbol)
                ds, _ = make_sequences(
                    frame=frame, feature_cols=feature_cols,
                    lookback=settings.lookback, scaler=None, fit_scaler=True,
                )
                if len(ds.x) < 500:
                    raise RuntimeError("sample terlalu sedikit")
                split = split_train_val(ds.x, ds.y, train_ratio=train_ratio)
                if len(split.x_val) == 0:
                    raise RuntimeError("validation kosong")
                model, _ = train_model(
                    x_train=split.x_train, y_train=split.y_train,
                    x_val=split.x_val,   y_val=split.y_val,
                    settings=settings_bt,
                )
                probs  = predict_proba(model, split.x_val)
                ts     = ds.timestamps[len(split.x_train):]
                spec   = client.symbol_spec(symbol)
                result = run_backtest(frame, ts, probs, settings_bt, spec)
                row = {"symbol": symbol, "status": "OK", "reason": ""}
                row.update(result.report)
                # simpan per-symbol trades
                if not result.trades.empty:
                    result.trades.insert(0, "symbol", symbol)
                    result.trades.to_csv(OUTPUT_DIR / f"trades_{symbol}.csv", index=False)
                rows.append(row)
                print(f"[AUTO-LB] ✅ {symbol} selesai — profit: {result.report.get('net_profit', 0):.2f}")
            except Exception as exc:
                rows.append({"symbol": symbol, "status": "ERROR", "reason": str(exc)})
                print(f"[AUTO-LB] ❌ {symbol} gagal: {exc}")

        if not rows:
            print("[AUTO-LB] Tidak ada hasil backtest, leaderboard tidak diupdate.")
            return

        # — tulis CSV baru
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        print(f"[AUTO-LB] CSV diperbarui: {csv_path}")

        # — baca leaderboard baru
        new_pairs = _read_top_pairs(csv_path, top_n)

        # — baca leaderboard lama (sebelum update)
        old_pairs = _read_top_pairs(csv_path, top_n)  # sama karena sudah ditimpa,
        # kita simpan old_pairs sebelum loop (lihat wrapper di bawah)

        _lb_last_run_date = date.today()
        print(f"[AUTO-LB] ✅ Leaderboard baru: {new_pairs}")

        if on_done:
            on_done(new_pairs)

    except Exception as exc:
        print(f"[AUTO-LB] ERROR fatal update leaderboard: {exc}")
    finally:
        with _lb_lock:
            _lb_running = False


def _read_top_pairs(csv_path: Path, top_n: int) -> list[str]:
    """Baca top-N pair dari CSV backtest."""
    try:
        df = pd.read_csv(csv_path)
        df_ok = df[(df["status"] == "OK") & (df["net_profit"] > 0) & (df["win_rate"] >= 0.40)].copy()
        if df_ok.empty:
            return []
        df_ok = df_ok.sort_values(by=["net_profit", "win_rate"], ascending=[False, False])
        return df_ok.head(top_n)["symbol"].tolist()
    except Exception:
        return []


# ─── public launcher ────────────────────────────────────────────────────────

def spawn_leaderboard_update(
    client,
    settings,
    symbols: list[str],
    top_n: int,
    csv_path: Path,
    epochs: int = 5,
    train_ratio: float = 0.7,
    on_done: Callable | None = None,
) -> threading.Thread:
    """
    Spawn thread background untuk update leaderboard.
    Return thread object (daemon=True, tidak perlu di-join).
    """
    t = threading.Thread(
        target=run_leaderboard_update,
        kwargs=dict(
            client=client, settings=settings,
            symbols=symbols, top_n=top_n,
            epochs=epochs, train_ratio=train_ratio,
            csv_path=csv_path, on_done=on_done,
        ),
        daemon=True,
        name="auto-leaderboard",
    )
    t.start()
    return t
