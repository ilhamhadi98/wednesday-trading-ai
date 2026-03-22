import argparse
import time
import threading
from dataclasses import replace
import pandas as pd
from datetime import datetime, timezone

# ── Telegram notifications (non-blocking, aman jika gagal) ──────────────────
try:
    from trading_bot.telegram_notify import notify_signal, notify_trade_close
    _TG_ENABLED = True
except Exception:
    _TG_ENABLED = False
    def notify_signal(*a, **kw): pass
    def notify_trade_close(*a, **kw): pass

from trading_bot.config import load_settings, OUTPUT_DIR
from trading_bot.execution import DemoExecutor
from trading_bot.market_scan import scan_opportunities
from trading_bot.modeling import load_artifacts
from trading_bot.mt5_client import MT5Client
from trading_bot.adaptive_retrain import (
    load_symbol_artifacts,
    _do_retrain,
    is_retraining,
    _retraining_in_progress,
    _lock
)

def write_live_state(session: str, monitoring: list, logs: str):
    import json
    try:
        from trading_bot.adaptive_retrain import _retraining_in_progress
        with open("outputs/live_state.json", "w") as f:
            json.dump({
                "session": session,
                "monitoring": monitoring,
                "training": list(_retraining_in_progress),
                "logs": [logs]
            }, f)
    except: pass
    
def signal_to_int(label: str) -> int:
    return 1 if label == "BUY" else -1 if label == "SELL" else 0

def get_leaderboard_pairs(csv_path, top_n=10):
    try:
        df = pd.read_csv(csv_path)
        # Filter pair yang profitabel dan aman
        df_ok = df[(df["status"] == "OK") & (df["net_profit"] > 0) & (df["win_rate"] >= 0.40)].copy()
        if df_ok.empty:
            return []
        # Urutkan berdasarkan profit paling mutlak
        df_ok = df_ok.sort_values(by=["net_profit", "win_rate"], ascending=[False, False])
        return df_ok.head(top_n)["symbol"].tolist()
    except Exception as e:
        print(f"[ERROR] Gagal memuat leaderboard: {e}. Harap jalankan run_multi_backtest.py terlebih dahulu.")
        return []

def is_trading_session() -> bool:
    """Mengizinkan trading hanya pada sesi London & New York (08:00 UTC - 21:00 UTC)"""
    now_utc = datetime.now(timezone.utc).hour
    return 8 <= now_utc < 21

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=10, help="Ambil Top N pair dari Leaderboard")
    parser.add_argument("--execute", action="store_true", help="Kirim order ke MT5")
    args = parser.parse_args()

    settings = load_settings()
    global_artifacts = load_artifacts()
    client = MT5Client(settings)

    # Membaca Leaderboard
    csv_path = OUTPUT_DIR / "multi_backtest_results.csv"
    best_pairs = get_leaderboard_pairs(csv_path, top_n=args.top)
    
    if not best_pairs:
        print("[SMART-LIVE] Leaderboard kosong. Mengeksekusi fallback major pairs.")
        best_pairs = ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY", "USDCAD"]
    
    # Memaksa XAUUSD dan XAGUSD masuk jika diminta user (apapun yang terjadi)
    for gold_silver in ["XAUUSD", "XAGUSD"]:
        if gold_silver not in best_pairs:
            best_pairs.append(gold_silver)

    print("\n=======================================================")
    print(" 🌟 SMART SESSION & LEADERBOARD AI TRADER 🌟")
    print("=======================================================")
    print(f"[DAFTAR PAIR PILIHAN]: {', '.join(best_pairs)}")
    print(f"[LIMIT POSISI]: {settings.max_active_pairs} Pair bersamaan")
    print(f"[SESI TRADING]: 08:00 UTC - 21:00 UTC (London & New York)")
    print("=======================================================\n")

    last_seen_bar        = {}
    first_time_trained   = set()
    # Tracking posisi terbuka untuk deteksi close
    _open_positions_meta: dict = {}   # ticket -> {symbol, signal, entry, sl, tp, lot, entry_time, price_data}

    try:
        client.connect()
        while True:
            # Skenario 1: Di Luar Jam Trading (Sesi Asia / Sepi)
            if not is_trading_session():
                msg = f"[{datetime.now(timezone.utc).strftime('%H:%M UTC')}] Di luar sesi aktif. AI masuk mode TRAINING & MONITORING..."
                print(msg)
                write_live_state("Off-Session (Training)", best_pairs, msg)
                
                # Biarkan AI berlatih (Transfer Learning) pada pair andalannya agar besok lebih pintar
                for sym in best_pairs:
                    if sym not in first_time_trained and not is_retraining(sym):
                        print(f" ⚙️ [TRAINING OFF-SESSION] Mempersiapkan model per-symbol untuk {sym}...")
                        with _lock:
                            _retraining_in_progress.add(sym)
                        t = threading.Thread(
                            target=_do_retrain,
                            args=(sym, client, settings),
                            daemon=True,
                            name=f"train-off-{sym}"
                        )
                        t.start()
                        first_time_trained.add(sym)
                        time.sleep(5)  # Beri jeda antar spawn thread
                
                # Tidur lama karena pasar sedang sepi
                time.sleep(15 * 60)
                continue

            # Skenario 2: Di Dalam Jam Trading (Sesi London & NY)
            msg = f"[{datetime.now(timezone.utc).strftime('%H:%M UTC')}] Market Sesion Aktif! AI melakukan scanning ketat..."
            print(msg)
            write_live_state("Active-Session (Trading)", best_pairs, msg)
            
            df = scan_opportunities(
                client=client,
                settings=settings,
                artifacts=global_artifacts,
                symbols=best_pairs,
                pair_only=False, # Karena kita sudah menyediakan eksplisit list terbaik
                fast_tf=settings.live_fast_timeframe,
                slow_tf=settings.live_slow_timeframe,
            )

            if df.empty:
                print(" -> Tidak ada sinyal memadai saat ini.")
                time.sleep(settings.poll_seconds)
                continue

            # Mengganti prediksi global dengan prediksi Per-Symbol (Hasil Transfer Learning Off-Session)
            for idx, row in df.iterrows():
                sym = str(row["symbol"])
                if row["status"] != "OK":
                    continue
                sym_arts = load_symbol_artifacts(sym)
                if sym_arts:
                    # Model pintar ditemukan
                    try:
                        from trading_bot.market_scan import evaluate_symbol
                        updated = evaluate_symbol(client, settings, sym_arts, sym, settings.live_fast_timeframe, settings.live_slow_timeframe)
                        df.at[idx, "signal"] = updated.signal
                        df.at[idx, "probability"] = updated.probability
                        df.at[idx, "confidence"] = updated.confidence
                    except Exception:
                        pass
            
            actionable = df[(df["status"] == "OK") & (df["signal"].isin(["BUY", "SELL"]))].copy()
            if actionable.empty:
                print(" -> Semua sinyal ditahan (HOLD).")
                time.sleep(settings.poll_seconds)
                continue

            active_positions = [p for p in client.positions_all() if p.magic == settings.magic_number]
            active_symbols   = {p.symbol for p in active_positions}
            active_tickets   = {p.ticket for p in active_positions}

            # ── Deteksi posisi yang baru CLOSE (ada di meta tapi tidak lagi di MT5) ──
            if _TG_ENABLED and args.execute:
                for ticket, meta in list(_open_positions_meta.items()):
                    if ticket not in active_tickets:
                        try:
                            # Ambil data harga untuk chart
                            raw_df = client.get_ohlcv(
                                meta["symbol"], settings.live_fast_timeframe, n=120
                            ) if hasattr(client, "get_ohlcv") else None
                            acct    = client.account_info()
                            balance = acct.balance if acct else 0.0
                            # Estimasi close price dari harga current (tidak ada history deal di sini)
                            sym_info = client.symbol_spec(meta["symbol"])
                            close_p  = sym_info.bid if meta["signal"] == "BUY" else sym_info.ask
                            pnl_est  = (close_p - meta["entry"]) * (1 if meta["signal"] == "BUY" else -1) \
                                       * meta["lot"] * (sym_info.trade_contract_size if hasattr(sym_info, "trade_contract_size") else 100000)
                            notify_trade_close(
                                symbol=meta["symbol"], signal=meta["signal"],
                                entry_price=meta["entry"], close_price=close_p,
                                sl=meta["sl"], tp=meta["tp"], lot=meta["lot"],
                                pnl=pnl_est, balance=balance, reason="CLOSED",
                                entry_time=meta["entry_time"],
                                close_time=datetime.now(timezone.utc),
                                price_data=raw_df,
                            )
                        except Exception as _ex:
                            print(f"[TG] close-notify error: {_ex}")
                        del _open_positions_meta[ticket]

            for _, row in actionable.iterrows():
                symbol = str(row["symbol"])
                bar_time = pd.Timestamp(row["bar_time"])
                if symbol in last_seen_bar and bar_time <= last_seen_bar[symbol]:
                    continue

                signal = signal_to_int(str(row["signal"]))
                symbol_settings = replace(settings, symbol=symbol)
                
                is_adaptive = "✅" if load_symbol_artifacts(symbol) else "❌"

                if args.execute:
                    if signal != 0 and symbol not in active_symbols and len(active_symbols) >= settings.max_active_pairs:
                        print(f"    ⚠️ {symbol} Skip: Kuota MAX {settings.max_active_pairs} posisi sudah penuh.")
                        last_seen_bar[symbol] = bar_time
                        continue

                    spec     = client.symbol_spec(symbol)
                    executor = DemoExecutor(client, symbol_settings, spec)
                    action   = executor.handle_signal(signal)
                    print(f"    🎯 {symbol} | Sinyal: {row['signal']} | Prob: {row['probability']:.2f} | Adaptive: {is_adaptive} | Aksi: {action.message}")

                    if action.ok and "open" in action.message.lower():
                        active_symbols.add(symbol)
                        # ── Telegram: kirim signal baru ──────────────────────
                        if _TG_ENABLED:
                            try:
                                acct    = client.account_info()
                                balance = acct.balance if acct else 0.0
                                # Cari ticket baru yang baru saja dibuka
                                new_pos = [p for p in client.positions_all()
                                           if p.symbol == symbol and p.magic == settings.magic_number]
                                if new_pos:
                                    p0 = new_pos[0]
                                    notify_signal(
                                        symbol=symbol,
                                        signal=str(row["signal"]),
                                        probability=float(row["probability"]),
                                        entry_price=p0.price_open,
                                        sl=p0.sl, tp=p0.tp, lot=p0.volume,
                                        balance=balance,
                                        adaptive=bool(load_symbol_artifacts(symbol)),
                                    )
                                    _open_positions_meta[p0.ticket] = {
                                        "symbol": symbol,
                                        "signal": str(row["signal"]),
                                        "entry":  p0.price_open,
                                        "sl": p0.sl, "tp": p0.tp,
                                        "lot": p0.volume,
                                        "entry_time": datetime.now(timezone.utc),
                                    }
                            except Exception as _ex:
                                print(f"[TG] signal-notify error: {_ex}")
                else:
                    print(f"    [DRY-RUN] {symbol} | Sinyal: {row['signal']} | Prob: {row['probability']:.2f} | Adaptive: {is_adaptive}")
                
                last_seen_bar[symbol] = bar_time

            write_live_state("Active-Session (Trading)", best_pairs, "Siklus scan selesai, menunggu poll berikutnya.")
            time.sleep(settings.poll_seconds)

    except KeyboardInterrupt:
        print("\n[SMART-LIVE] Sistem dihentikan oleh user.")
    finally:
        client.shutdown()

if __name__ == "__main__":
    main()
