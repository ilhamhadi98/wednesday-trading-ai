from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd

from .config import (
    BACKTEST_REPORT_PATH,
    BACKTEST_TRADES_PATH,
    Settings,
    ensure_output_dir,
)
from .mt5_client import SymbolSpec


@dataclass
class BacktestResult:
    report: dict
    trades: pd.DataFrame
    equity_curve: pd.DataFrame


def pip_size_from_digits(digits: int, point: float) -> float:
    return point * 10 if digits in {3, 5} else point


def _quantize_lot(lot: float, spec: SymbolSpec) -> float:
    lot = max(spec.volume_min, min(spec.volume_max, lot))
    steps = round((lot - spec.volume_min) / spec.volume_step)
    quantized = spec.volume_min + (steps * spec.volume_step)
    return round(max(spec.volume_min, quantized), 8)


def pnl_to_usd(amount_quote: float, symbol: str, current_price: float) -> float:
    if symbol.startswith("USD"):
        return amount_quote / current_price
    if symbol.endswith("USD"):
        return amount_quote
    if symbol.endswith("JPY"): return amount_quote / 150.0
    if symbol.endswith("CHF"): return amount_quote / 0.90
    if symbol.endswith("CAD"): return amount_quote / 1.35
    if symbol.endswith("GBP"): return amount_quote * 1.25
    if symbol.endswith("EUR"): return amount_quote * 1.08
    if symbol.endswith("AUD"): return amount_quote * 0.65
    return amount_quote

def usd_to_quote(amount_usd: float, symbol: str, current_price: float) -> float:
    if symbol.startswith("USD"):
        return amount_usd * current_price
    if symbol.endswith("USD"):
        return amount_usd
    if symbol.endswith("JPY"): return amount_usd * 150.0
    if symbol.endswith("CHF"): return amount_usd * 0.90
    if symbol.endswith("CAD"): return amount_usd * 1.35
    if symbol.endswith("GBP"): return amount_usd / 1.25
    if symbol.endswith("EUR"): return amount_usd / 1.08
    if symbol.endswith("AUD"): return amount_usd / 0.65
    return amount_usd


def compute_lot_size(balance: float, settings: Settings, spec: SymbolSpec,
                     current_price: float, sl_pips: Optional[float] = None) -> float:
    if not settings.use_risk_position_sizing:
        return _quantize_lot(settings.fixed_lot, spec)

    pip_sz = pip_size_from_digits(spec.digits, spec.point)
    effective_sl = sl_pips if sl_pips is not None else settings.stop_loss_pips
    sl_distance_price = effective_sl * pip_sz
    if sl_distance_price <= 0:
        return _quantize_lot(settings.fixed_lot, spec)

    risk_amount_usd = balance * settings.risk_per_trade
    risk_amount_quote = usd_to_quote(risk_amount_usd, spec.symbol, current_price)
    
    lot = risk_amount_quote / (sl_distance_price * spec.trade_contract_size + 1e-10)
    return _quantize_lot(lot, spec)


def _signal(prob: float, buy_th: float, sell_th: float) -> int:
    if prob >= buy_th:
        return 1
    if prob <= sell_th:
        return -1
    return 0


def _detect_high_low_cols(frame: pd.DataFrame) -> tuple[str, str]:
    """Deteksi kolom high/low secara otomatis untuk semua prefix timeframe."""
    for prefix in ("m15_", "m5_", "m1_", "h1_", "m30_", "h4_", "d1_"):
        h_col = f"{prefix}high"
        l_col = f"{prefix}low"
        if h_col in frame.columns and l_col in frame.columns:
            return h_col, l_col
    # fallback: cari kolom apapun yang mengandung 'high' dan 'low'
    h_candidates = [c for c in frame.columns if "high" in c]
    l_candidates = [c for c in frame.columns if "low" in c]
    if h_candidates and l_candidates:
        return h_candidates[0], l_candidates[0]
    return "exec_close", "exec_close"


def _detect_atr_col(frame: pd.DataFrame) -> Optional[str]:
    """Deteksi kolom ATR untuk dynamic SL/TP."""
    for prefix in ("m15_", "m5_", "m1_", "h1_", "m30_", "h4_", "d1_"):
        col = f"{prefix}atr_14"
        if col in frame.columns:
            return col
    return None


def run_backtest(
    frame: pd.DataFrame,
    timestamps: pd.Index,
    probs: np.ndarray,
    settings: Settings,
    symbol_spec: SymbolSpec,
    risk_manager: Any = None,
) -> BacktestResult:
    """Backtest engine dengan:
    - Deteksi kolom high/low otomatis (M15, H1, H4, dll)
    - ATR-based dynamic SL/TP: SL = 1.5× ATR, TP = 3× ATR (lebih adaptif)
    - Trailing Stop: geser SL mengikuti harga menguntungkan
    - Circuit Breaker: hentikan entry jika drawdown >= max_drawdown_circuit_breaker
    - Consecutive Loss Guard: pause entry setelah N kali kalah berturut-turut
    """
    if len(timestamps) != len(probs):
        raise ValueError("Panjang timestamps dan probs harus sama.")

    point = symbol_spec.point
    pip_size = pip_size_from_digits(symbol_spec.digits, point)

    use_trailing = getattr(settings, "use_trailing_stop", False)
    trailing_dist = getattr(settings, "trailing_stop_pips", 0.0) * pip_size
    circuit_limit = getattr(settings, "max_drawdown_circuit_breaker", 1.0)
    max_consec = getattr(settings, "max_consecutive_losses", 999)

    # Deteksi kolom high/low & ATR secara otomatis
    h_col, l_col = _detect_high_low_cols(frame)
    atr_col = _detect_atr_col(frame)

    print(f"[Backtest] Kolom High/Low: {h_col} / {l_col}")
    print(f"[Backtest] ATR dinamis   : {atr_col if atr_col else 'TIDAK (pakai pip tetap)'}")
    print(f"[Backtest] Total bars yg akan dievaluasi: {len(timestamps):,}")

    balance = settings.initial_balance
    equity = balance
    peak_equity = equity
    max_drawdown = 0.0

    position = None
    trades: list[dict] = []
    equity_rows: list[dict] = []

    consecutive_losses = 0
    circuit_tripped = False
    consec_paused = False
    consec_pause_until: int = -1

    signals_skipped_circuit = 0
    signals_skipped_consec = 0
    signals_total = 0

    for bar_i, (ts, prob) in enumerate(zip(timestamps, probs)):
        row = frame.loc[ts]
        close = float(row["exec_close"])
        high = float(row[h_col]) if h_col != "exec_close" else close
        low = float(row[l_col]) if l_col != "exec_close" else close
        spread_points = float(row["exec_spread"])
        spread_price = spread_points * point

        # ── Tentukan SL/TP: ATR-based jika tersedia, fallback ke pip tetap
        if atr_col and atr_col in row.index:
            atr_val = float(row[atr_col])
            if atr_val > 0 and not np.isnan(atr_val):
                # SL = 1.5× ATR agar tidak kena noise, TP = 3× ATR (R:R=1:2)
                sl_price_dist = max(atr_val * 1.5, settings.stop_loss_pips * pip_size)
                tp_price_dist = sl_price_dist * (settings.take_profit_pips / max(settings.stop_loss_pips, 1))
                sl_pips_eff = sl_price_dist / pip_size
            else:
                sl_price_dist = settings.stop_loss_pips * pip_size
                tp_price_dist = settings.take_profit_pips * pip_size
                sl_pips_eff = settings.stop_loss_pips
        else:
            sl_price_dist = settings.stop_loss_pips * pip_size
            tp_price_dist = settings.take_profit_pips * pip_size
            sl_pips_eff = settings.stop_loss_pips

        sig = _signal(
            prob,
            settings.prediction_buy_threshold,
            settings.prediction_sell_threshold,
        )
        if sig != 0:
            signals_total += 1

        # ── Cek circuit breaker & consecutive loss pause
        current_dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        
        if risk_manager is not None:
            is_allowed, rm_reason = risk_manager.is_trade_allowed(symbol_spec.symbol, current_dd, bar_i)
            circuit_tripped = not is_allowed and rm_reason == "CIRCUIT_BREAKER"
            consec_paused = not is_allowed and rm_reason == "CONSECUTIVE_LOSS_GUARD"
        else:
            circuit_tripped = current_dd >= circuit_limit
            if bar_i >= consec_pause_until:
                consec_paused = False

        # ── Kelola posisi yang berjalan ──────────────────────────────────────
        if position is not None:
            side = position["side"]

            if side == 1:  # BUY
                # Trailing stop: naikkan SL jika harga naik
                if use_trailing and trailing_dist > 0:
                    new_sl = high - trailing_dist
                    if new_sl > position["sl"]:
                        position["sl"] = new_sl

                if low <= position["sl"]:
                    exit_reason, exit_price = "SL", position["sl"]
                elif high >= position["tp"]:
                    exit_reason, exit_price = "TP", position["tp"]
                elif sig == -1:
                    exit_reason, exit_price = "REVERSE", close - spread_price / 2
                else:
                    exit_reason, exit_price = None, None
            else:  # SELL
                # Trailing stop: turunkan SL jika harga turun
                if use_trailing and trailing_dist > 0:
                    new_sl = low + trailing_dist
                    if new_sl < position["sl"]:
                        position["sl"] = new_sl

                if high >= position["sl"]:
                    exit_reason, exit_price = "SL", position["sl"]
                elif low <= position["tp"]:
                    exit_reason, exit_price = "TP", position["tp"]
                elif sig == 1:
                    exit_reason, exit_price = "REVERSE", close + spread_price / 2
                else:
                    exit_reason, exit_price = None, None

            if exit_price is not None:
                pnl_quote = (
                    (exit_price - position["entry_price"])
                    * side
                    * symbol_spec.trade_contract_size
                    * position["lot"]
                )
                pnl = pnl_to_usd(pnl_quote, symbol_spec.symbol, exit_price)
                balance += pnl
                equity = balance
                peak_equity = max(peak_equity, equity)
                dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
                max_drawdown = max(max_drawdown, dd)

                if risk_manager is not None:
                    risk_manager.register_trade_result(symbol_spec.symbol, pnl, bar_i)
                else:
                    if pnl <= 0:
                        consecutive_losses += 1
                        if consecutive_losses >= max_consec:
                            consec_paused = True
                            # Pause selama (2 × max_consec) bar ke depan — jeda "cooling off"
                            consec_pause_until = bar_i + max_consec * 2
                    else:
                        consecutive_losses = 0  # reset streak saat win

                trades.append({
                    "entry_time": position["entry_time"],
                    "exit_time": ts,
                    "side": "BUY" if side == 1 else "SELL",
                    "entry_price": position["entry_price"],
                    "exit_price": exit_price,
                    "sl": position["sl"],
                    "tp": position["tp"],
                    "lot": position["lot"],
                    "reason": exit_reason,
                    "pnl": pnl,
                    "balance_after": balance,
                    "prediction": position["prediction"],
                    "prob": prob,
                })
                position = None

        # ── Buka posisi baru ────────────────────────────────────────────────
        can_enter = (
            position is None
            and sig != 0
            and not circuit_tripped
            and not consec_paused
            and balance > 0
        )

        if sig != 0 and not can_enter:
            if circuit_tripped:
                signals_skipped_circuit += 1
            elif consec_paused:
                signals_skipped_consec += 1

        if can_enter:
            if risk_manager is not None:
                lot = risk_manager.compute_lot_size(balance, symbol_spec, current_price=close, sl_pips=sl_pips_eff)
            else:
                lot = compute_lot_size(balance, settings, symbol_spec, current_price=close, sl_pips=sl_pips_eff)

            if sig == 1:  # BUY
                entry_price = close + spread_price / 2
                sl = entry_price - sl_price_dist
                tp = entry_price + tp_price_dist
            else:          # SELL
                entry_price = close - spread_price / 2
                sl = entry_price + sl_price_dist
                tp = entry_price - tp_price_dist

            position = {
                "entry_time": ts,
                "entry_price": entry_price,
                "sl": sl,
                "tp": tp,
                "lot": lot,
                "side": sig,
                "prediction": float(prob),
            }

        equity_rows.append({
            "time": ts,
            "equity": equity,
            "balance": balance,
            "circuit_tripped": circuit_tripped,
            "consec_paused": consec_paused,
            "consecutive_losses": consecutive_losses,
        })

    # ── Tutup posisi yang masih buka di akhir data ──────────────────────────
    if position is not None:
        last_ts = timestamps[-1]
        row = frame.loc[last_ts]
        close = float(row["exec_close"])
        spread_pts = float(row["exec_spread"])
        spread_price = spread_pts * point
        ep = (
            close - spread_price / 2 if position["side"] == 1
            else close + spread_price / 2
        )
        pnl_quote = (
            (ep - position["entry_price"])
            * position["side"]
            * symbol_spec.trade_contract_size
            * position["lot"]
        )
        pnl = pnl_to_usd(pnl_quote, symbol_spec.symbol, ep)
        balance += pnl
        equity = balance
        peak_equity = max(peak_equity, equity)
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        max_drawdown = max(max_drawdown, dd)
        trades.append({
            "entry_time": position["entry_time"],
            "exit_time": last_ts,
            "side": "BUY" if position["side"] == 1 else "SELL",
            "entry_price": position["entry_price"],
            "exit_price": ep,
            "sl": position["sl"],
            "tp": position["tp"],
            "lot": position["lot"],
            "reason": "EOD",
            "pnl": pnl,
            "balance_after": balance,
            "prediction": position["prediction"],
            "prob": float(probs[-1]),
        })

    trades_df = pd.DataFrame(trades)
    equity_curve = pd.DataFrame(equity_rows)

    print(f"[Backtest] Sinyal total       : {signals_total}")
    print(f"[Backtest] Sinyal dieksekusi  : {len(trades_df)}")
    print(f"[Backtest] Skip circuit breaker: {signals_skipped_circuit}")
    print(f"[Backtest] Skip consec. pause : {signals_skipped_consec}")

    total_trades = len(trades_df)
    if total_trades > 0:
        wins = int((trades_df["pnl"] > 0).sum())
        losses = int((trades_df["pnl"] <= 0).sum())
        gross_profit = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum())
        gross_loss = float(-trades_df.loc[trades_df["pnl"] <= 0, "pnl"].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        net_profit = float(trades_df["pnl"].sum())
        win_rate = wins / total_trades
        avg_win = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].mean()) if wins > 0 else 0.0
        avg_loss = float(trades_df.loc[trades_df["pnl"] <= 0, "pnl"].mean()) if losses > 0 else 0.0
        pnl_arr = trades_df["pnl"].values
        sharpe = float(pnl_arr.mean() / (pnl_arr.std() + 1e-10) * np.sqrt(252)) if total_trades > 1 else 0.0
    else:
        wins = losses = 0
        gross_profit = gross_loss = net_profit = win_rate = avg_win = avg_loss = sharpe = 0.0
        profit_factor = 0.0

    recovery_factor = net_profit / (max_drawdown * settings.initial_balance + 1e-10)

    report = {
        "initial_balance": settings.initial_balance,
        "final_balance": round(balance, 4),
        "net_profit": round(net_profit, 4),
        "return_pct": round(((balance / settings.initial_balance) - 1) * 100, 2),
        "total_trades": total_trades,
        "signals_generated": signals_total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "gross_profit": round(gross_profit, 4),
        "gross_loss": round(gross_loss, 4),
        "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else "inf",
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "sharpe_ratio": round(sharpe, 4),
        "recovery_factor": round(recovery_factor, 4),
        "trailing_stop_enabled": use_trailing,
        "circuit_breaker_limit_pct": round(circuit_limit * 100, 1),
        "signals_skipped_circuit": signals_skipped_circuit,
        "signals_skipped_consec": signals_skipped_consec,
        "max_consecutive_losses_guard": max_consec,
        "risk_profile": settings.risk_profile,
        "stop_loss_pips": settings.stop_loss_pips,
        "take_profit_pips": settings.take_profit_pips,
        "risk_reward_ratio": round(
            settings.take_profit_pips / max(settings.stop_loss_pips, 1), 2
        ),
        "atr_dynamic_sl_tp": atr_col is not None,
        "high_low_col": h_col,
    }

    return BacktestResult(report=report, trades=trades_df, equity_curve=equity_curve)


def save_backtest_result(result: BacktestResult) -> tuple[Path, Path]:
    ensure_output_dir()
    with open(BACKTEST_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(result.report, f, indent=2, default=str)
    if not result.trades.empty:
        result.trades.to_csv(BACKTEST_TRADES_PATH, index=False)
    return BACKTEST_REPORT_PATH, BACKTEST_TRADES_PATH
