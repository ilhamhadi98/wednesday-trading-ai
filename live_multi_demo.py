from __future__ import annotations

import argparse
import time
from dataclasses import replace

import pandas as pd

from trading_bot.config import load_settings
from trading_bot.execution import DemoExecutor
from trading_bot.market_scan import scan_opportunities
from trading_bot.modeling import load_artifacts
from trading_bot.mt5_client import MT5Client


def signal_to_int(label: str) -> int:
    if label == "BUY":
        return 1
    if label == "SELL":
        return -1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live demo multi-symbol: scan peluang banyak simbol lalu eksekusi top sinyal."
    )
    parser.add_argument("--top", type=int, default=0)
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--all-symbols", action="store_true")
    args = parser.parse_args()

    settings = load_settings()

    # ── Model global: TIDAK pernah ditimpa/diubah oleh retraining ─────────────
    global_artifacts = load_artifacts()

    explicit_symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    last_seen_bar: dict[str, pd.Timestamp] = {}
    max_active = args.top if args.top > 0 else settings.max_active_pairs
    max_active = min(max_active, settings.max_active_pairs)

    # Set pair yang sudah pernah di-trigger training (hindari retrain berulang)
    first_time_trained: set[str] = set()

    # Lazy import adaptive modules
    from trading_bot.adaptive_retrain import (
        check_and_trigger_retrain,
        load_symbol_artifacts,
        is_retraining,
        _do_retrain,
        _retraining_in_progress,
        _lock,
    )
    import threading

    def get_artifacts_for_symbol(symbol: str):
        """
        Kembalikan model terbaik untuk pair ini:
        - Jika ada model per-symbol (hasil retraining) → pakai itu
        - Jika tidak ada → fallback ke model global (aman, tidak berubah)
        """
        sym_arts = load_symbol_artifacts(symbol)
        return sym_arts if sym_arts is not None else global_artifacts

    client = MT5Client(settings)
    try:
        client.connect()
        print(
            "[LIVE-MULTI] start. "
            f"mode={settings.symbol_discovery_mode} max_symbols={settings.scan_max_symbols} "
            f"max_active={max_active} pair_only={not args.all_symbols} execute={args.execute} "
            f"risk={settings.risk_profile} tf={settings.live_fast_timeframe}/{settings.live_slow_timeframe}",
            flush=True,
        )
        print("[LIVE-MULTI] tekan CTRL+C untuk berhenti.", flush=True)
        print("[LIVE-MULTI] Adaptive retraining AKTIF (trigger: 5 loss berturut per pair).", flush=True)

        while True:
            df = scan_opportunities(
                client=client,
                settings=settings,
                artifacts=global_artifacts,
                symbols=explicit_symbols if explicit_symbols else None,
                pair_only=not args.all_symbols,
                fast_tf=settings.live_fast_timeframe,
                slow_tf=settings.live_slow_timeframe,
            )

            if df.empty:
                print("[LIVE-MULTI] tidak ada data scan.", flush=True)
                if args.once:
                    break
                time.sleep(settings.poll_seconds)
                continue

            df.to_json("outputs/live_signals.json", orient="records", indent=2)

            # -- Cek trigger retraining per pair (background thread, non-blocking) --
            for sym in df["symbol"].tolist():
                if is_retraining(sym):
                    continue  # sudah berjalan, skip

                sym_arts = load_symbol_artifacts(sym)

                # Kasus 1: Pair BARU yang belum pernah punya model khusus
                if sym_arts is None and sym not in first_time_trained:
                    print(
                        f"[LIVE-MULTI] [BARU] {sym} belum ada model khusus. "
                        "Memulai initial training di background...",
                        flush=True,
                    )
                    with _lock:
                        _retraining_in_progress.add(sym)
                    t = threading.Thread(
                        target=_do_retrain,
                        args=(sym, client, settings),
                        daemon=True,
                        name=f"init-train-{sym}",
                    )
                    t.start()
                    first_time_trained.add(sym)
                    continue  # sudah trigger initial, skip cek loss streak

                # Kasus 2: Pair yang sudah ada model, cek streak loss
                triggered = check_and_trigger_retrain(
                    symbol=sym,
                    client=client,
                    settings=settings,
                    consecutive_loss_threshold=5,
                )
                if triggered:
                    print(
                        f"[LIVE-MULTI] [RETRAIN] {sym} sedang dilatih ulang di background. "
                        "Pair lain TIDAK terpengaruh.",
                        flush=True,
                    )


            # ── Re-evaluasi sinyal menggunakan model terbaik per pair ─────────
            for idx, row in df.iterrows():
                sym = str(row["symbol"])
                if row["status"] != "OK":
                    continue
                sym_arts = get_artifacts_for_symbol(sym)
                if sym_arts is not global_artifacts:
                    # Pair ini sudah memiliki model khusus hasil retraining
                    try:
                        from trading_bot.market_scan import evaluate_symbol
                        updated = evaluate_symbol(
                            client=client,
                            settings=settings,
                            artifacts=sym_arts,
                            symbol=sym,
                            fast_tf=settings.live_fast_timeframe,
                            slow_tf=settings.live_slow_timeframe,
                        )
                        df.at[idx, "signal"] = updated.signal
                        df.at[idx, "probability"] = updated.probability
                        df.at[idx, "confidence"] = updated.confidence
                    except Exception:
                        pass  # fallback tetap pakai sinyal dari global model

            actionable = df[
                (df["status"] == "OK") & (df["signal"].isin(["BUY", "SELL"]))
            ].copy()
            if actionable.empty:
                print("[LIVE-MULTI] tidak ada sinyal BUY/SELL saat ini.", flush=True)
                if args.once:
                    break
                time.sleep(settings.poll_seconds)
                continue

            actionable = actionable.head(max_active)
            active_positions = [
                p for p in client.positions_all() if p.magic == settings.magic_number
            ]
            active_symbols = {p.symbol for p in active_positions}
            if len(active_symbols) > max_active:
                print(
                    f"[LIVE-MULTI] peringatan: posisi aktif {len(active_symbols)} > limit {max_active}.",
                    flush=True,
                )

            for _, row in actionable.iterrows():
                symbol = str(row["symbol"])
                bar_time = pd.Timestamp(row["bar_time"])
                if symbol in last_seen_bar and bar_time <= last_seen_bar[symbol]:
                    continue

                signal = signal_to_int(str(row["signal"]))
                symbol_settings = replace(settings, symbol=symbol)

                # Tampilkan tag model yang dipakai untuk pair ini
                arts = get_artifacts_for_symbol(symbol)
                model_tag = f"per-symbol" if arts is not global_artifacts else "global"

                if args.execute:
                    if signal != 0 and symbol not in active_symbols and len(active_symbols) >= max_active:
                        print(
                            f"[LIVE-MULTI] {symbol} skip: limit {max_active} posisi aktif sudah penuh.",
                            flush=True,
                        )
                        last_seen_bar[symbol] = bar_time
                        continue

                    spec = client.symbol_spec(symbol)
                    executor = DemoExecutor(client, symbol_settings, spec)
                    action = executor.handle_signal(signal)
                    print(
                        f"[LIVE-MULTI] symbol={symbol} bar={bar_time} "
                        f"signal={row['signal']} prob={row['probability']:.4f} "
                        f"model=[{model_tag}] ok={action.ok} msg={action.message}",
                        flush=True,
                    )
                    if action.ok and "open" in action.message.lower():
                        active_symbols.add(symbol)
                else:
                    print(
                        f"[LIVE-MULTI][DRY-RUN] symbol={symbol} bar={bar_time} "
                        f"signal={row['signal']} prob={row['probability']:.4f} "
                        f"model=[{model_tag}] aksi tidak dikirim",
                        flush=True,
                    )
                last_seen_bar[symbol] = bar_time

            if args.once:
                break
            time.sleep(settings.poll_seconds)

    except KeyboardInterrupt:
        print("[LIVE-MULTI] dihentikan user.", flush=True)
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
