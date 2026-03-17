#!/usr/bin/env python
"""
run_llm_backtest.py  ─  Full LLM-Integrated Portfolio Backtest

Timeframes : M15 (fast) + H1 (slow)   [configurable via .env]
Period     : 3 months back (configurable)
Pipeline   : MT5/mock data → LSTM StrategistAI → LLM Orchestrator
              → gate (both must agree) → trade execution → HTML report

CLI Examples
------------
# Quick test — no MT5, no Ollama (mock everything)
python run_llm_backtest.py --symbols EURUSD,XAUUSD --months 3 --no-mt5 --dry-llm

# Real MT5 data, mock Ollama (fast)
python run_llm_backtest.py --symbols EURUSD,GBPUSD --months 3 --dry-llm

# Full real run (MT5 + Ollama must be running)
python run_llm_backtest.py --symbols EURUSD,GBPUSD,XAUUSD --months 3

Outputs (in outputs/ folder)
-----------------------------
llm_backtest_3m_report.json   ─ summary metrics
llm_backtest_3m_trades.csv    ─ every trade with LLM columns
llm_backtest_3m_equity.csv    ─ equity curve
llm_backtest_3m_plans.json    ─ detailed trading plan per trade
llm_backtest_3m_report.html   ─ rich interactive dashboard
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from trading_bot.backtest import (
    BacktestResult,
    compute_lot_size,
    pip_size_from_digits,
    pnl_to_usd,
)
from trading_bot.config import OUTPUT_DIR, ensure_output_dir, load_settings
from trading_bot.data_pipeline import build_multitimeframe_frame, make_sequences
from trading_bot.modeling import load_artifacts, predict_proba
from trading_bot.agents.llm_orchestrator import LLMOrchestrator

PROJECT_ROOT = Path(__file__).resolve().parent

# ─── Output paths ──────────────────────────────────────────────────────────────
TAG = "llm_backtest_3m"


def _out(name: str) -> Path:
    return OUTPUT_DIR / f"{TAG}_{name}"


MEMORY_FILE = OUTPUT_DIR / "agent_memory.json"


def _save_memory(recent_trades: list) -> None:
    """Save the recent trades buffer to a JSON file for persistence."""
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(recent_trades, f, indent=2)
    except Exception as e:
        print(f"[!] Failed to save memory: {e}")


def _load_memory() -> list:
    """Load the recent trades buffer from a JSON file."""
    if not MEMORY_FILE.exists():
        return []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[!] Failed to load memory: {e}")
        return []


# ─── Mock LLM for --dry-llm mode ───────────────────────────────────────────────

def _mock_llm_result(symbol: str, ml_signal: str, ml_prob: float) -> dict:
    """Deterministic mock LLM result for testing without Ollama."""
    rng = np.random.default_rng(abs(hash(symbol + str(round(ml_prob, 2)))) % (2 ** 31))
    # 75% chance LLM agrees with ML signal, 25% chance it says HOLD
    agree = rng.random() < 0.75
    if agree:
        llm_dec = ml_signal
    else:
        llm_dec = "HOLD"
    sentiments = ["BULLISH", "BEARISH", "NEUTRAL"]
    sent = "BULLISH" if ml_signal == "BUY" else "BEARISH" if ml_signal == "SELL" else "NEUTRAL"
    return {
        "symbol": symbol,
        "ml_signal": ml_signal,
        "ml_probability": ml_prob,
        "indicators": {},
        "news": {
            "sentiment": sent,
            "confidence": float(rng.uniform(0.55, 0.80)),
            "reasoning": "[MOCK] Simulated news sentiment.",
            "model": "mock",
            "error": None,
        },
        "technical": {
            "view": sent,
            "confidence": float(rng.uniform(0.55, 0.80)),
            "support": None,
            "resistance": None,
            "suggested_sl_pips": None,
            "suggested_tp_pips": None,
            "reasoning": "[MOCK] Simulated technical view.",
            "model": "mock",
            "error": None,
        },
        "decision": {
            "decision": llm_dec,
            "confidence": float(rng.uniform(0.50, 0.85)),
            "risk_reward_ratio": 1.5,
            "entry_rationale": f"[MOCK] {llm_dec} signal based on simulated analysis.",
            "risk_warning": "",
            "model": "mock",
            "error": None,
        },
        "llm_decision": llm_dec,
        "llm_confidence": float(rng.uniform(0.50, 0.85)),
        "elapsed_seconds": 0.0,
    }


# ─── TF helpers ────────────────────────────────────────────────────────────────

def _tf_minutes(tf: str) -> int:
    return {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30,
        "H1": 60, "H4": 240, "D1": 1440,
    }.get(tf.upper(), 60)


def _bars_needed(months: int, tf: str, extra_pct: float = 1.25) -> int:
    mins_per_bar = _tf_minutes(tf)
    trading_mins_per_day = 22 * 60
    bars_per_day = trading_mins_per_day / mins_per_bar * (5 / 7)
    return int(bars_per_day * 30.44 * months * extra_pct) + 500


# ─── Mock OHLCV ────────────────────────────────────────────────────────────────

def _mock_ohlcv(symbol: str, tf: str, months: int) -> pd.DataFrame:
    """Generate synthetic OHLCV data for --no-mt5 mode."""
    n = _bars_needed(months, tf)
    rng = np.random.default_rng(abs(hash(symbol + tf)) % (2 ** 31))
    base = {
        "EURUSD": 1.080, "GBPUSD": 1.270, "USDJPY": 149.5,
        "XAUUSD": 2320.0, "AUDUSD": 0.645, "USDCHF": 0.904,
        "BTCUSD": 65000.0,
    }.get(symbol, 1.0)
    prices = [base]
    vol = base * 0.0008
    for _ in range(n - 1):
        prices.append(max(prices[-1] + rng.normal(0, vol), base * 0.5))
    freq = f"{_tf_minutes(tf)}min"
    dates = pd.date_range(
        end=pd.Timestamp.now(tz="UTC"),
        periods=n,
        freq=freq,
        tz="UTC",
    )
    arr = np.array(prices)
    noise = np.abs(rng.normal(0, vol, n))
    df = pd.DataFrame({
        "time":        dates,
        "open":        arr,
        "high":        arr + noise * 1.2,
        "low":         arr - noise * 1.2,
        "close":       arr + rng.normal(0, vol * 0.3, n),
        "tick_volume": rng.integers(50, 5000, n).astype(float),
        "volume":      rng.integers(50, 5000, n).astype(float),
        "spread":      rng.integers(1, 30, n).astype(float),
        "real_volume": np.zeros(n),
    })
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"]  = df[["open", "low",  "close"]].min(axis=1)
    return df


# ─── Fetch from MT5 ────────────────────────────────────────────────────────────

def _fetch_mt5(client, symbol: str, tf: str, start_utc: datetime, end_utc: datetime,
               lookback_days: int) -> pd.DataFrame:
    from dateutil.relativedelta import relativedelta as rd  # noqa: PLC0415
    fetch_start = start_utc - rd(days=lookback_days)
    return client.copy_rates_range(symbol, tf, fetch_start, end_utc)


# ─── Signal generation (LSTM) ──────────────────────────────────────────────────

def _build_signals(
    df_fast: pd.DataFrame,
    df_slow: pd.DataFrame,
    artifacts,
    settings,
    start_utc: datetime,
    end_utc: datetime,
    fast_tf: str,
    slow_tf: str,
) -> tuple[pd.DataFrame, pd.Index, np.ndarray]:
    """Build feature frame, run LSTM, return (frame, timestamps, probs)."""
    frame, _ = build_multitimeframe_frame(
        df_fast, df_slow,
        fast_prefix=f"{fast_tf.lower()}_",
        slow_prefix=f"{slow_tf.lower()}_"
    )
    frame = frame[
        (frame.index >= pd.Timestamp(start_utc)) &
        (frame.index <= pd.Timestamp(end_utc))
    ].copy()

    if len(frame) < artifacts.lookback + 50:
        raise RuntimeError(f"Frame too short: {len(frame)} rows")

    missing = [c for c in artifacts.feature_cols if c not in frame.columns]
    if missing:
        raise RuntimeError(f"Missing features: {missing[:3]}")

    ds, _ = make_sequences(
        frame=frame,
        feature_cols=artifacts.feature_cols,
        lookback=artifacts.lookback,
        scaler=artifacts.scaler,
        fit_scaler=False,
    )
    probs = predict_proba(artifacts.model, ds.x)
    return frame, ds.timestamps, probs


# ─── LLM-Gated Backtest Engine ─────────────────────────────────────────────────

def run_llm_gated_backtest(
    symbol: str,
    frame: pd.DataFrame,
    timestamps: pd.Index,
    probs: np.ndarray,
    settings,
    spec,
    orchestrator: Optional[LLMOrchestrator],
    dry_llm: bool = False,
    initial_equity: float = 100.0,
    recent_trades_initial: list = None,
    persist_memory: bool = False,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Modified backtest engine that gates entries through the LLM.

    Returns (trades, equity_rows, trading_plans)
    """
    pip_size = pip_size_from_digits(spec.digits, spec.point)
    buy_th   = settings.prediction_buy_threshold
    sell_th  = settings.prediction_sell_threshold

    # Detect high/low columns
    h_col = next((f"{p}high" for p in ("m15_","m5_","h1_","m30_") if f"{p}high" in frame.columns), "exec_close")
    l_col = next((f"{p}low"  for p in ("m15_","m5_","h1_","m30_") if f"{p}low"  in frame.columns), "exec_close")
    atr_col = next((f"{p}atr_14" for p in ("m15_","m5_","h1_") if f"{p}atr_14" in frame.columns), None)

    # Backtest state
    balance = settings.initial_balance
    peak_equity = balance
    max_dd = 0.0
    position = None
    trade_id = 0
    consec_losses = 0
    pause_until = -1

    trades = []
    equity_rows = []
    trading_plans = []
    recent_trades = recent_trades_initial if recent_trades_initial is not None else []
    trade_id = 0

    print(f"  [Backtest] {symbol}: {len(timestamps):,} bars | H/L cols={h_col}/{l_col}")

    for bar_i, (ts, prob) in enumerate(zip(timestamps, probs)):
        row   = frame.loc[ts]
        close = float(row["exec_close"])
        high  = float(row[h_col]) if h_col != "exec_close" else close
        low   = float(row[l_col]) if l_col != "exec_close" else close
        spread_price = float(row.get("exec_spread", 0)) * spec.point

        # Dynamic SL/TP from ATR
        if atr_col and atr_col in row.index:
            atr = float(row[atr_col])
            if atr > 0 and not np.isnan(atr):
                sl_dist = max(atr * 1.5, settings.stop_loss_pips * pip_size)
                tp_dist = sl_dist * (settings.take_profit_pips / max(settings.stop_loss_pips, 1))
                sl_pips_eff = sl_dist / pip_size
            else:
                sl_dist = settings.stop_loss_pips * pip_size
                tp_dist = settings.take_profit_pips * pip_size
                sl_pips_eff = settings.stop_loss_pips
        else:
            sl_dist = settings.stop_loss_pips * pip_size
            tp_dist = settings.take_profit_pips * pip_size
            sl_pips_eff = settings.stop_loss_pips

        # Raw LSTM signal
        if prob >= buy_th:
            ml_sig = 1
            ml_label = "BUY"
        elif prob <= sell_th:
            ml_sig = -1
            ml_label = "SELL"
        else:
            ml_sig = 0
            ml_label = "HOLD"

        # ── Manage open position ────────────────────────────────────────────────
        if position is not None:
            side = position["side"]
            if side == 1:
                if low <= position["sl"]:
                    exit_r, exit_p = "SL", position["sl"]
                elif high >= position["tp"]:
                    exit_r, exit_p = "TP", position["tp"]
                elif ml_sig == -1:
                    exit_r, exit_p = "REVERSE", close - spread_price / 2
                else:
                    exit_r, exit_p = None, None
            else:
                if high >= position["sl"]:
                    exit_r, exit_p = "SL", position["sl"]
                elif low <= position["tp"]:
                    exit_r, exit_p = "TP", position["tp"]
                elif ml_sig == 1:
                    exit_r, exit_p = "REVERSE", close + spread_price / 2
                else:
                    exit_r, exit_p = None, None

            if exit_p is not None:
                pnl_q = (exit_p - position["entry_price"]) * side * spec.trade_contract_size * position["lot"]
                pnl   = pnl_to_usd(pnl_q, spec.symbol, exit_p)
                balance += pnl
                peak_equity = max(peak_equity, balance)
                dd = (peak_equity - balance) / peak_equity if peak_equity > 0 else 0.0
                max_dd = max(max_dd, dd)
                consec_losses = 0 if pnl > 0 else consec_losses + 1
                if consec_losses >= settings.max_consecutive_losses:
                    pause_until = bar_i + settings.max_consecutive_losses * 2

                # Update the plan with outcome
                plan = position.get("plan", {})
                plan["outcome"] = {
                    "exit_time": str(ts),
                    "exit_price": round(exit_p, 6),
                    "reason": exit_r,
                    "pnl": round(pnl, 4),
                    "balance_after": round(balance, 4),
                }
                trading_plans.append(plan)

                trades.append({
                    "symbol":         symbol,
                    "entry_time":     position["entry_time"],
                    "exit_time":      ts,
                    "side":           "BUY" if side == 1 else "SELL",
                    "entry_price":    position["entry_price"],
                    "exit_price":     exit_p,
                    "sl":             position["sl"],
                    "tp":             position["tp"],
                    "lot":            position["lot"],
                    "reason":         exit_r,
                    "pnl":            pnl,
                    "balance_after":  balance,
                    "probability":    position["probability"],
                    # LLM columns
                    "llm_decision":   position.get("llm_decision", ""),
                    "llm_confidence": position.get("llm_confidence", 0.0),
                    "llm_sentiment":  position.get("llm_sentiment", ""),
                    "llm_tech_view":  position.get("llm_tech_view", ""),
                    "llm_rationale":  position.get("llm_rationale", ""),
                    "llm_risk_warning": position.get("llm_risk_warning", ""),
                    "llm_filtered":   False,
                    "llm_elapsed_s":  position.get("llm_elapsed_s", 0.0),
                })
                # Add trade to memory buffer
                recent_trades.append({
                    "time": str(ts),
                    "side": "BUY" if side == 1 else "SELL",
                    "pnl_pips": round(pnl_q / (spec.trade_contract_size * position["lot"] * spec.point), 1) if position["lot"] > 0 else 0,
                    "pnl_usd": round(pnl, 2),
                    "reason": exit_r,
                    "rationale": position.get("llm_rationale", "")
                })
                
                if persist_memory:
                    _save_memory(recent_trades)
                
                position = None

        # ── Risk guards ─────────────────────────────────────────────────────────
        current_dd = (peak_equity - balance) / peak_equity if peak_equity > 0 else 0.0
        circuit = current_dd >= settings.max_drawdown_circuit_breaker
        paused  = bar_i < pause_until

        # ── Open new position ────────────────────────────────────────────────────
        if (ml_sig != 0 and position is None and not circuit and not paused and balance > 0):

            # ── LLM Gate ────────────────────────────────────────────────────────
            df_slice = frame.loc[:ts].tail(60)  # last 60 bars for context
            if "exec_close" in df_slice.columns:
                df_ohlcv = df_slice.rename(columns={
                    "exec_close": "close",
                    h_col + "_raw" if h_col != "exec_close" else h_col: "high",
                })
            else:
                df_ohlcv = df_slice.copy()
            # Ensure columns exist for orchestrator
            for col_map in [(h_col, "high"), (l_col, "low"), ("exec_close", "close")]:
                if col_map[0] in df_ohlcv.columns and col_map[1] not in df_ohlcv.columns:
                    df_ohlcv[col_map[1]] = df_ohlcv[col_map[0]]
            if "open" not in df_ohlcv.columns:
                df_ohlcv["open"] = df_ohlcv.get("close", df_ohlcv.iloc[:, 0])

            t_llm = time.time()
            if dry_llm or orchestrator is None:
                llm_result = _mock_llm_result(symbol, ml_label, float(prob))
            else:
                try:
                    llm_result = orchestrator.run(
                        symbol=symbol,
                        df=df_ohlcv,
                        ml_signal=ml_label,
                        ml_probability=float(prob),
                        portfolio_drawdown_pct=current_dd,
                        current_balance=balance,
                        recent_trades=recent_trades[-10:],  # Provide last 10 trades as memory
                    )
                except Exception as exc:
                    print(f"    [LLM] Error: {exc} - using mock")
                    llm_result = _mock_llm_result(symbol, ml_label, float(prob))
            llm_elapsed = round(time.time() - t_llm, 2)

            llm_dec_label = llm_result.get("llm_decision", "HOLD")
            llm_conf      = float(llm_result.get("llm_confidence", 0.5))
            llm_dec_int   = 1 if llm_dec_label == "BUY" else (-1 if llm_dec_label == "SELL" else 0)

            # Gate: LSTM and LLM must agree
            llm_agrees = (ml_sig == llm_dec_int) and llm_dec_int != 0

            if not llm_agrees:
                # Log as filtered trade
                trades.append({
                    "symbol":         symbol,
                    "entry_time":     ts,
                    "exit_time":      ts,
                    "side":           ml_label,
                    "entry_price":    close,
                    "exit_price":     close,
                    "sl":             0.0,
                    "tp":             0.0,
                    "lot":            0.0,
                    "reason":         "LLM_FILTERED",
                    "pnl":            0.0,
                    "balance_after":  balance,
                    "probability":    float(prob),
                    "llm_decision":   llm_dec_label,
                    "llm_confidence": llm_conf,
                    "llm_sentiment":  llm_result.get("news", {}).get("sentiment", ""),
                    "llm_tech_view":  llm_result.get("technical", {}).get("view", ""),
                    "llm_rationale":  llm_result.get("decision", {}).get("entry_rationale", ""),
                    "llm_risk_warning": llm_result.get("decision", {}).get("risk_warning", ""),
                    "llm_filtered":   True,
                    "llm_elapsed_s":  llm_elapsed,
                })
            else:
                # Execute the trade
                lot = compute_lot_size(balance, settings, spec, current_price=close, sl_pips=sl_pips_eff)

                # Enforce LSTM/ATR-based Dynamic SL/TP. 
                # Do NOT let LLM override distances to prevent tight stop-outs.
                sl_dist_use = sl_dist
                tp_dist_use = tp_dist

                if ml_sig == 1:
                    entry_p = close + spread_price / 2
                    sl_p    = entry_p - sl_dist_use
                    tp_p    = entry_p + tp_dist_use
                else:
                    entry_p = close - spread_price / 2
                    sl_p    = entry_p + sl_dist_use
                    tp_p    = entry_p - tp_dist_use

                trade_id += 1
                plan = {
                    "trade_id":   trade_id,
                    "symbol":     symbol,
                    "entry_time": str(ts),
                    "lstm_signal":      ml_label,
                    "lstm_probability": round(float(prob), 4),
                    "llm_news":     llm_result.get("news", {}),
                    "llm_tech":     llm_result.get("technical", {}),
                    "llm_decision": llm_result.get("decision", {}),
                    "execution": {
                        "entry_price": round(entry_p, 6),
                        "sl":          round(sl_p, 6),
                        "tp":          round(tp_p, 6),
                        "lot":         lot,
                        "sl_pips":     round(sl_dist_use / pip_size, 1),
                        "tp_pips":     round(tp_dist_use / pip_size, 1),
                    },
                    "outcome": None,  # filled on close
                }

                position = {
                    "entry_time":   str(ts),
                    "entry_price":  entry_p,
                    "sl":           sl_p,
                    "tp":           tp_p,
                    "lot":          lot,
                    "side":         ml_sig,
                    "probability":  float(prob),
                    "plan":         plan,
                    "llm_decision": llm_dec_label,
                    "llm_confidence": llm_conf,
                    "llm_sentiment": llm_result.get("news", {}).get("sentiment", ""),
                    "llm_tech_view": llm_result.get("technical", {}).get("view", ""),
                    "llm_rationale": llm_result.get("decision", {}).get("entry_rationale", ""),
                    "llm_risk_warning": llm_result.get("decision", {}).get("risk_warning", ""),
                    "llm_elapsed_s": llm_elapsed,
                }

        equity_rows.append({
            "time":             ts,
            "balance":          balance,
            "equity":           balance,
            "circuit_tripped":  circuit,
            "consec_paused":    paused,
        })

    # Close any remaining open position at end
    if position is not None and len(timestamps) > 0:
        last_ts  = timestamps[-1]
        last_row = frame.loc[last_ts]
        close    = float(last_row["exec_close"])
        spread_price = float(last_row.get("exec_spread", 0)) * spec.point
        side         = position["side"]
        exit_p   = (close - spread_price / 2) if side == 1 else (close + spread_price / 2)
        pnl_q    = (exit_p - position["entry_price"]) * side * spec.trade_contract_size * position["lot"]
        pnl      = pnl_to_usd(pnl_q, spec.symbol, exit_p)
        balance += pnl

        plan = position.get("plan", {})
        plan["outcome"] = {
            "exit_time":     str(last_ts),
            "exit_price":    round(exit_p, 6),
            "reason":        "EOD",
            "pnl":           round(pnl, 4),
            "balance_after": round(balance, 4),
        }
        trading_plans.append(plan)

        trades.append({
            "symbol":         symbol,
            "entry_time":     position["entry_time"],
            "exit_time":      last_ts,
            "side":           "BUY" if side == 1 else "SELL",
            "entry_price":    position["entry_price"],
            "exit_price":     exit_p,
            "sl":             position["sl"],
            "tp":             position["tp"],
            "lot":            position["lot"],
            "reason":         "EOD",
            "pnl":            pnl,
            "balance_after":  balance,
            "probability":    position["probability"],
            "llm_decision":   position.get("llm_decision", ""),
            "llm_confidence": position.get("llm_confidence", 0.0),
            "llm_sentiment":  position.get("llm_sentiment", ""),
            "llm_tech_view":  position.get("llm_tech_view", ""),
            "llm_rationale":  position.get("llm_rationale", ""),
            "llm_risk_warning": position.get("llm_risk_warning", ""),
            "llm_filtered":   False,
            "llm_elapsed_s":  position.get("llm_elapsed_s", 0.0),
        })

    return trades, equity_rows, trading_plans


# ─── Build Summary Report ───────────────────────────────────────────────────────

def _build_report(trades_df: pd.DataFrame, settings, symbol: str,
                  fast_tf: str, slow_tf: str, months: int,
                  start_utc: datetime, end_utc: datetime) -> dict:
    real = trades_df[trades_df["reason"] != "LLM_FILTERED"]
    filtered = trades_df[trades_df["reason"] == "LLM_FILTERED"]

    total  = len(real)
    wins   = int((real["pnl"] > 0).sum()) if total else 0
    losses = int((real["pnl"] <= 0).sum()) if total else 0
    gp     = float(real.loc[real["pnl"] > 0, "pnl"].sum()) if total else 0.0
    gl     = float(-real.loc[real["pnl"] <= 0, "pnl"].sum()) if total else 0.0
    pf     = (gp / gl) if gl > 0 else float("inf")
    net    = float(real["pnl"].sum()) if total else 0.0
    wr     = wins / total if total else 0.0
    final_bal = float(real["balance_after"].iloc[-1]) if total else settings.initial_balance
    avg_w  = float(real.loc[real["pnl"] > 0, "pnl"].mean()) if wins  else 0.0
    avg_l  = float(real.loc[real["pnl"] <= 0,"pnl"].mean()) if losses else 0.0

    # LLM accuracy: on executed trades, how confident was LLM?
    llm_avg_conf = float(real["llm_confidence"].mean()) if total else 0.0
    llm_filtered_count = len(filtered)

    # LLM agreement breakdown
    agreed_win_rate = 0.0
    if total > 0:
        agreed = real[real["llm_filtered"] == False]
        agreed_win_rate = float((agreed["pnl"] > 0).mean()) if len(agreed) else 0.0

    return {
        "symbol":             symbol,
        "backtest_fast_tf":   fast_tf,
        "backtest_slow_tf":   slow_tf,
        "period_months":      months,
        "period_start_utc":   str(start_utc),
        "period_end_utc":     str(end_utc),
        "initial_balance":    settings.initial_balance,
        "final_balance":      round(final_bal, 4),
        "net_profit":         round(net, 4),
        "return_pct":         round((final_bal / settings.initial_balance - 1) * 100, 2),
        "total_trades":       total,
        "wins":               wins,
        "losses":             losses,
        "win_rate":           round(wr, 4),
        "avg_win":            round(avg_w, 4),
        "avg_loss":           round(avg_l, 4),
        "gross_profit":       round(gp, 4),
        "gross_loss":         round(gl, 4),
        "profit_factor":      round(pf, 4) if pf != float("inf") else "inf",
        "llm_avg_confidence": round(llm_avg_conf, 3),
        "llm_filtered_trades": llm_filtered_count,
        "llm_agreed_win_rate": round(agreed_win_rate, 4),
        "llm_filter_rate_pct": round(
            llm_filtered_count / max(total + llm_filtered_count, 1) * 100, 1
        ),
    }


# ─── MockSpec for --no-mt5 mode ────────────────────────────────────────────────

class _MockSpec:
    def __init__(self, symbol: str):
        self.symbol            = symbol
        self.digits            = 5 if "JPY" not in symbol else 3
        self.point             = 0.00001 if "JPY" not in symbol else 0.001
        self.trade_contract_size = 100_000.0 if symbol not in {"XAUUSD", "BTCUSD"} else 100.0
        self.volume_min        = 0.01
        self.volume_max        = 100.0
        self.volume_step       = 0.01


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-Integrated Portfolio Backtest (M15+H1, 3 months)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--symbols", default="EURUSD,XAUUSD",
                        help="Comma-separated symbols (default: EURUSD,XAUUSD)")
    parser.add_argument("--months",  type=int, default=3,
                        help="Months of history (default: 3)")
    parser.add_argument("--no-mt5", action="store_true",
                        help="Use synthetic mock data (no MT5 needed)")
    parser.add_argument("--dry-llm", action="store_true",
                        help="Use mock LLM responses (no Ollama needed)")
    parser.add_argument("--fast-tf", default=None,
                        help="Fast timeframe override (default from .env: M15)")
    parser.add_argument("--slow-tf", default=None,
                        help="Slow timeframe override (default from .env: H1)")
    parser.add_argument("--fixed-lot", type=float, default=0.01)
    parser.add_argument("--open-report", action="store_true",
                        help="Auto-open HTML report in browser after generation")
    parser.add_argument("--persist", action="store_true",
                        help="Save recent trade history to disk after backtest")
    parser.add_argument("--load-memory", action="store_true",
                        help="Load recent trade history from disk at startup")
    args = parser.parse_args()

    ensure_output_dir()
    settings = load_settings()
    settings = replace(settings,
                       fixed_lot=args.fixed_lot,
                       use_risk_position_sizing=False)

    fast_tf = (args.fast_tf or settings.backtest_fast_timeframe or "M15").upper()
    slow_tf = (args.slow_tf or settings.backtest_slow_timeframe or "H1").upper()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    months  = args.months

    now_utc   = datetime.now(timezone.utc)
    start_utc = now_utc - relativedelta(months=months)

    print()
    print("=" * 65)
    print("  WEDNESDAY TRADING AI -- LLM-Integrated Backtest")
    print("=" * 65)
    print(f"  Symbols  : {', '.join(symbols)}")
    print(f"  Period   : {months} months ({start_utc.strftime('%Y-%m-%d')} - {now_utc.strftime('%Y-%m-%d')})")
    print(f"  Fast TF  : {fast_tf}")
    print(f"  Slow TF  : {slow_tf}")
    print(f"  MT5      : {'MOCK DATA' if args.no_mt5 else 'LIVE'}")
    print(f"  LLM      : {'MOCK (dry-llm)' if args.dry_llm else 'OLLAMA LIVE'}")
    print("=" * 65)
    print()

    # Load LSTM artifacts
    try:
        artifacts = load_artifacts()
        print(f"[OK] LSTM model loaded | lookback={artifacts.lookback} | features={len(artifacts.feature_cols)}")
    except Exception as exc:
        print(f"[!] Could not load LSTM artifacts: {exc}")
        print("    Run training first or use --dry-llm with --no-mt5 for pure mock mode.")
        return

    # Initialize LLM orchestrator
    orchestrator = None
    if not args.dry_llm:
        orchestrator = LLMOrchestrator(
            news_model=settings.ollama_news_model,
            tech_model=settings.ollama_tech_model,
            decision_model=settings.ollama_decision_model,
            base_url=settings.ollama_base_url,
        )
        print(f"[OK] LLM Orchestrator ready | news={settings.ollama_news_model} | tech={settings.ollama_tech_model}")
    else:
        print("[OK] LLM mode: MOCK (dry-llm)")

    # Connect to MT5
    client = None
    if not args.no_mt5:
        try:
            from trading_bot.mt5_client import MT5Client  # noqa: PLC0415
            client = MT5Client(settings)
            client.connect()
            print("[OK] MT5 connected\n")
        except Exception as exc:
            print(f"[!] MT5 connection failed: {exc}")
            print("    Falling back to mock data.\n")
            client = None

    all_trades:  list[pd.DataFrame]  = []
    all_equity:  list[pd.DataFrame]  = []
    all_plans:   list[dict]          = []
    all_reports: list[dict]          = []
    symbol_stats_rows: list[dict]    = []

    for sym in symbols:
        print("\n" + "=" * 63)
        print(f"  Processing: {sym}")
        print("=" * 63)

        try:
            # ── Get OHLCV data ──────────────────────────────────────────────
            lookback_days_fast = max(10, int(settings.lookback * _tf_minutes(fast_tf) / 1440) + 10)
            lookback_days_slow = max(20, int(settings.lookback * _tf_minutes(slow_tf) / 1440) + 20)

            if client:
                df_fast = _fetch_mt5(client, sym, fast_tf, start_utc, now_utc, lookback_days_fast)
                df_slow = _fetch_mt5(client, sym, slow_tf, start_utc, now_utc, lookback_days_slow)
            else:
                df_fast = _mock_ohlcv(sym, fast_tf, months + 1)
                df_slow = _mock_ohlcv(sym, slow_tf, months + 1)

            print(f"  [Data] {fast_tf}: {len(df_fast):,} bars | {slow_tf}: {len(df_slow):,} bars")

            # ── LSTM signals ────────────────────────────────────────────────
            frame, timestamps, probs = _build_signals(
                df_fast, df_slow, artifacts, settings, start_utc, now_utc, fast_tf, slow_tf
            )
            n_buy  = int((probs >= settings.prediction_buy_threshold).sum())
            n_sell = int((probs <= settings.prediction_sell_threshold).sum())
            print(f"  [LSTM] {len(timestamps):,} bars in window | BUY signals={n_buy} | SELL signals={n_sell}")

            # ── Spec ────────────────────────────────────────────────────────
            if client:
                spec = client.symbol_spec(sym)
            else:
                spec = _MockSpec(sym)

            # ── LLM-Gated backtest ──────────────────────────────────────────
            # Load cross-session memory if requested
            initial_memory = []
            if getattr(args, 'load_memory', False):
                initial_memory = _load_memory()
                if initial_memory:
                    print(f"  [Memory] Loaded {len(initial_memory)} trades from persistent storage")

            trades, equity_rows, plans = run_llm_gated_backtest(
                symbol=sym,
                frame=frame,
                timestamps=timestamps,
                probs=probs,
                settings=settings,
                spec=spec,
                orchestrator=orchestrator,
                dry_llm=args.dry_llm,
                initial_equity=settings.initial_balance,
                recent_trades_initial=initial_memory,
                persist_memory=getattr(args, 'persist', False)
            )

            trades_df  = pd.DataFrame(trades)
            equity_df  = pd.DataFrame(equity_rows)
            n_exec     = int((trades_df["reason"] != "LLM_FILTERED").sum()) if not trades_df.empty else 0
            n_filtered = int((trades_df["reason"] == "LLM_FILTERED").sum()) if not trades_df.empty else 0
            print(f"  [Result] Executed={n_exec} | LLM-filtered={n_filtered} | Plans={len(plans)}")

            if not trades_df.empty:
                real = trades_df[trades_df["reason"] != "LLM_FILTERED"]
                if not real.empty:
                    wr = float((real["pnl"] > 0).mean())
                    net = float(real["pnl"].sum())
                    print(f"  [PnL]   Net=${net:+.2f} | WinRate={wr*100:.1f}%")

            report = _build_report(trades_df, settings, sym, fast_tf, slow_tf,
                                   months, start_utc, now_utc)
            all_reports.append(report)
            all_trades.append(trades_df)
            all_equity.append(equity_df)
            all_plans.extend(plans)

            symbol_stats_rows.append({
                "symbol":     sym,
                "trades":     report["total_trades"],
                "wins":       report["wins"],
                "win_rate":   round(report["win_rate"] * 100, 1),
                "net_profit": round(report["net_profit"], 2),
                "llm_filtered": report["llm_filtered_trades"],
                "llm_avg_conf": round(report["llm_avg_confidence"] * 100, 1),
            })

        except Exception as exc:
            print(f"  [ERROR] {sym}: {exc}")
            import traceback; traceback.print_exc()
            continue

    if client:
        client.shutdown()

    if not all_trades:
        print("\n[!] No results to save.")
        return

    # ── Combine & save ──────────────────────────────────────────────────────────
    combined_trades = pd.concat(all_trades, ignore_index=True)
    combined_equity = pd.concat(all_equity, ignore_index=True)

    report_combined = {
        "symbols":          symbols,
        "backtest_fast_tf": fast_tf,
        "backtest_slow_tf": slow_tf,
        "period_months":    months,
        "period_start_utc": str(start_utc),
        "period_end_utc":   str(now_utc),
        "initial_balance":  settings.initial_balance,
        "per_symbol":       all_reports,
        "llm_mode":         "mock" if args.dry_llm else "ollama",
        "news_model":       settings.ollama_news_model,
        "tech_model":       settings.ollama_tech_model,
        "decision_model":   settings.ollama_decision_model,
    }

    _out("report.json").write_text(
        json.dumps(report_combined, indent=2, default=str), encoding="utf-8")
    combined_trades.to_csv(_out("trades.csv"), index=False)
    combined_equity.to_csv(_out("equity.csv"), index=False)
    _out("plans.json").write_text(
        json.dumps(all_plans, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(symbol_stats_rows).to_csv(_out("symbol_stats.csv"), index=False)

    # ── Generate HTML report ────────────────────────────────────────────────────
    print("\n[HTML] Generating dashboard...")
    try:
        from generate_llm_html_report import generate_llm_html  # noqa: PLC0415
        html_path = _out("report.html")
        generate_llm_html(
            report=report_combined,
            trades=combined_trades,
            equity=combined_equity,
            symbol_stats=pd.DataFrame(symbol_stats_rows),
            plans=all_plans,
            output_path=html_path,
        )
        print(f"[OK] Dashboard: {html_path.resolve()}")
        if args.open_report:
            import webbrowser  # noqa: PLC0415
            webbrowser.open(html_path.resolve().as_uri())
    except Exception as exc:
        print(f"[!] HTML generation failed: {exc}")

    # ── Final summary ───────────────────────────────────────────────────────────
    print("\n" + "-" * 63)
    print(f"  {'Symbol':<10} {'Trades':>7} {'WinRate':>9} {'NetPnL':>10} {'LLM_filter':>12}")
    print("-" * 63)
    for r in all_reports:
        print(f"  {r['symbol']:<10} {r['total_trades']:>7} "
              f"{r['win_rate']*100:>8.1f}% {r['net_profit']:>+10.2f} "
              f"{r['llm_filtered_trades']:>12}")
    print("-" * 63)
    print(f"\n  Reports saved to: {OUTPUT_DIR.resolve()}")
    print(f"  HTML Dashboard : {_out('report.html').resolve()}")


if __name__ == "__main__":
    main()
