"""run_backtest.py – Backtest utama dengan dukungan timeframe M15 + 46 bulan data.

Penggunaan:
    # Backtest walk-forward M15 + H1, data 46 bulan, modal 100
    python run_backtest.py --mode walk_forward --fast-tf M15 --slow-tf H1 --months 46

    # Backtest klasik M15 + H1
    python run_backtest.py --mode train_test --fast-tf M15 --slow-tf H1 --months 46

    # Backtest default H1 + H4 (semua data dari server)
    python run_backtest.py --mode walk_forward
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone, timedelta

from trading_bot.backtest import run_backtest, save_backtest_result
from trading_bot.config import OUTPUT_DIR, load_settings
from trading_bot.data_pipeline import make_sequences
from trading_bot.modeling import predict_proba, train_model
from trading_bot.mt5_client import MT5Client
from trading_bot.workflows import fetch_feature_frame, split_train_val, walk_forward_probabilities


def _bars_for_months(months: int, timeframe: str) -> int:
    """Hitung jumlah bar yang dibutuhkan untuk N bulan pada timeframe tertentu.
    Asumsi: market buka ~22 jam/hari, 5 hari/minggu.
    """
    minutes_per_tf = {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30,
        "H1": 60, "H4": 240, "D1": 1440,
    }
    tf_min = minutes_per_tf.get(timeframe.upper(), 60)
    # 22 jam x 60 menit x 5 hari / 7 hari * 30.44 hari/bulan * N bulan
    bars = int((22 * 60 / tf_min) * (5 / 7) * 30.44 * months)
    # Tambah buffer 20% untuk hari libur, dll.
    return int(bars * 1.20) + 500


def run_train_test_backtest(client: MT5Client, fast_tf: str, slow_tf: str, months: int | None = None):
    settings = load_settings()

    if months is not None:
        bars_fast = _bars_for_months(months, fast_tf)
        bars_slow = _bars_for_months(months, slow_tf)
        print(f"[Backtest] Mode: train_test | {fast_tf}+{slow_tf} | {months} bulan")
        print(f"[Backtest] Jumlah bar: {fast_tf}={bars_fast:,}, {slow_tf}={bars_slow:,}")
    else:
        bars_fast = None
        bars_slow = None
        print(f"[Backtest] Mode: train_test | {fast_tf}+{slow_tf} | bars dari .env")

    frame, feature_cols = fetch_feature_frame(
        client, settings,
        fast_tf=fast_tf, slow_tf=slow_tf,
        bars_fast=bars_fast, bars_slow=bars_slow,
    )
    print(f"[Backtest] Total baris data: {len(frame):,}")

    ds, scaler = make_sequences(frame, feature_cols, settings.lookback, scaler=None, fit_scaler=True)
    split = split_train_val(ds.x, ds.y, train_ratio=0.7)
    print(f"[Backtest] Train: {len(split.x_train):,} | Val/Test: {len(split.x_val):,}")
    model, hist = train_model(
        x_train=split.x_train,
        y_train=split.y_train,
        x_val=split.x_val,
        y_val=split.y_val,
        settings=settings,
    )
    # Laporan metrik training
    best_auc = max(hist.get("val_auc", [0]))
    print(f"[Backtest] Best val_AUC: {best_auc:.4f}")
    probs = predict_proba(model, split.x_val)
    buy_count = int((probs >= settings.prediction_buy_threshold).sum())
    sell_count = int((probs <= settings.prediction_sell_threshold).sum())
    print(f"[Backtest] Sinyal: BUY={buy_count} | SELL={sell_count} | "
          f"HOLD={len(probs)-buy_count-sell_count}")
    ts = ds.timestamps[len(split.x_train):]
    spec = client.symbol_spec(settings.symbol)
    result = run_backtest(frame, ts, probs, settings, spec)
    return result


def run_walk_forward_backtest(client: MT5Client, fast_tf: str, slow_tf: str, months: int):
    settings = load_settings()

    bars_fast = _bars_for_months(months, fast_tf)
    bars_slow = _bars_for_months(months, slow_tf)

    print(f"[Backtest] Mode: walk_forward | {fast_tf}+{slow_tf} | {months} bulan")
    print(f"[Backtest] Jumlah bar: {fast_tf}={bars_fast:,}, {slow_tf}={bars_slow:,}")

    frame, feature_cols = fetch_feature_frame(
        client, settings,
        fast_tf=fast_tf, slow_tf=slow_tf,
        bars_fast=bars_fast, bars_slow=bars_slow,
    )
    print(f"[Backtest] Total baris data: {len(frame):,}")

    times, probs = walk_forward_probabilities(
        frame=frame,
        feature_cols=feature_cols,
        settings=settings,
        start_ratio=0.7,
    )
    spec = client.symbol_spec(settings.symbol)
    result = run_backtest(frame, times, probs, settings, spec)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest strategi deep learning di data MT5.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["walk_forward", "train_test"],
        default="walk_forward",
        help="walk_forward = self-learning retrain berkala | train_test = train sekali",
    )
    parser.add_argument(
        "--fast-tf",
        default=None,
        help="Timeframe cepat (M5, M15, M30, H1). Default: dari .env (BACKTEST_FAST_TF)",
    )
    parser.add_argument(
        "--slow-tf",
        default=None,
        help="Timeframe lambat (H1, H4, D1). Default: dari .env (BACKTEST_SLOW_TF)",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=None,
        help="Jumlah bulan data historis yang diambil (contoh: 46). "
             "Jika tidak diisi, memakai BARS_H1/BARS_H4/BARS_M15 dari .env.",
    )
    args = parser.parse_args()

    settings = load_settings()
    fast_tf = (args.fast_tf or settings.backtest_fast_timeframe).upper()
    slow_tf = (args.slow_tf or settings.backtest_slow_timeframe).upper()
    months = args.months

    print("=" * 60)
    print(f"  AUTO TRADING BOT – BACKTEST")
    print(f"  Timeframe: {fast_tf} + {slow_tf}")
    print(f"  Data     : {'Semua bar di .env' if months is None else str(months) + ' bulan'}")
    print(f"  Modal    : {settings.initial_balance:,.2f}")
    print(f"  Simbol   : {settings.symbol}")
    print(f"  Risk     : {settings.risk_profile}")
    print(f"  SL/TP    : {settings.stop_loss_pips}/{settings.take_profit_pips} pip")
    print(f"  Trailing : {'ON' if settings.use_trailing_stop else 'OFF'} "
          f"({settings.trailing_stop_pips} pip)")
    print(f"  CB limit : {settings.max_drawdown_circuit_breaker * 100:.0f}%")
    print("=" * 60)

    client = MT5Client(settings)

    try:
        client.connect()
        if months is not None:
            if args.mode == "walk_forward":
                result = run_walk_forward_backtest(client, fast_tf, slow_tf, months)
            else:
                result = run_train_test_backtest(client, fast_tf, slow_tf, months)
        else:
            # Jalankan dengan jumlah bar dari env (perilaku lama)
            from trading_bot.workflows import fetch_feature_frame, walk_forward_probabilities
            from trading_bot.data_pipeline import make_sequences
            from trading_bot.modeling import predict_proba, train_model

            frame, feature_cols = fetch_feature_frame(client, settings, fast_tf=fast_tf, slow_tf=slow_tf)
            if args.mode == "walk_forward":
                times, probs = walk_forward_probabilities(frame, feature_cols, settings)
                spec = client.symbol_spec(settings.symbol)
                result = run_backtest(frame, times, probs, settings, spec)
            else:
                ds, _ = make_sequences(frame, feature_cols, settings.lookback)
                split = split_train_val(ds.x, ds.y, 0.7)
                model, _ = train_model(split.x_train, split.y_train, split.x_val, split.y_val, settings)
                probs = predict_proba(model, split.x_val)
                ts = ds.timestamps[len(split.x_train):]
                spec = client.symbol_spec(settings.symbol)
                result = run_backtest(frame, ts, probs, settings, spec)

        report_path, trades_path = save_backtest_result(result)
        summary_path = OUTPUT_DIR / "backtest_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"{'=' * 50}\n")
            f.write(f"  BACKTEST REPORT  ({fast_tf}+{slow_tf})\n")
            f.write(f"{'=' * 50}\n")
            for k, v in result.report.items():
                f.write(f"  {k:<35}: {v}\n")

        print("\n✅ Backtest selesai!")
        print(json.dumps(result.report, indent=2))
        print(f"\n📄 Report : {report_path}")
        print(f"📊 Trades : {trades_path}")
        print(f"📝 Summary: {summary_path}")
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
