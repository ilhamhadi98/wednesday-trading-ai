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
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Jumlah simbol teratas (BUY/SELL) yang dieksekusi tiap siklus.",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Daftar simbol dipisah koma. Kosong = pakai discovery mode.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Jalankan 1 siklus scan+eksekusi lalu selesai.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Aktifkan pengiriman order ke akun demo. Default hanya scan (dry-run).",
    )
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Jika aktif, scan semua simbol (bukan pair saja). Default pair-only.",
    )
    args = parser.parse_args()

    settings = load_settings()
    artifacts = load_artifacts()
    explicit_symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    last_seen_bar: dict[str, pd.Timestamp] = {}
    max_active = args.top if args.top > 0 else settings.max_active_pairs
    max_active = min(max_active, settings.max_active_pairs)

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

        while True:
            df = scan_opportunities(
                client=client,
                settings=settings,
                artifacts=artifacts,
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
                    f"[LIVE-MULTI] peringatan: posisi aktif saat ini {len(active_symbols)} > limit {max_active}. "
                    "Tidak akan menambah posisi baru sampai jumlah turun.",
                    flush=True,
                )
            for _, row in actionable.iterrows():
                symbol = str(row["symbol"])
                bar_time = pd.Timestamp(row["bar_time"])
                if symbol in last_seen_bar and bar_time <= last_seen_bar[symbol]:
                    continue

                signal = signal_to_int(str(row["signal"]))
                symbol_settings = replace(settings, symbol=symbol)
                if args.execute:
                    if signal != 0 and symbol not in active_symbols and len(active_symbols) >= max_active:
                        print(
                            f"[LIVE-MULTI] symbol={symbol} dilewati karena limit posisi aktif {max_active} sudah penuh.",
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
                        f"ok={action.ok} msg={action.message}",
                        flush=True,
                    )
                    if action.ok and "open" in action.message.lower():
                        active_symbols.add(symbol)
                else:
                    print(
                        f"[LIVE-MULTI][DRY-RUN] symbol={symbol} bar={bar_time} "
                        f"signal={row['signal']} prob={row['probability']:.4f} "
                        "aksi tidak dikirim",
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
