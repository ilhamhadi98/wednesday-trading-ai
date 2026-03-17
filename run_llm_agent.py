#!/usr/bin/env python
"""
run_llm_agent.py — CLI entrypoint for the Ollama LLM multi-agent trading system.

Usage examples:

  # Dry-run with mock data (no MT5 required):
  python run_llm_agent.py --symbols EURUSD,XAUUSD --no-mt5 --dry-run

  # Live mode with MT5:
  python run_llm_agent.py --symbols EURUSD,GBPUSD

  # Custom models:
  python run_llm_agent.py --symbols EURUSD --news-model mistral:7b
      --tech-model qwen2.5:14b --decision-model deepseek-r1:14b

Output is saved to outputs/llm_decisions.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from trading_bot.config import OUTPUT_DIR, ensure_output_dir, load_settings
from trading_bot.agents.llm_orchestrator import LLMOrchestrator

PROJECT_ROOT = Path(__file__).resolve().parent
LLM_OUTPUT_PATH = OUTPUT_DIR / "llm_decisions.json"


# ──────────────────────────────────────────────────────────────────────────────
# Mock data helpers (used when --no-mt5 is set)
# ──────────────────────────────────────────────────────────────────────────────

def _mock_ohlcv(symbol: str, n: int = 200) -> pd.DataFrame:
    """Generate a synthetic OHLCV dataframe for testing without MT5."""
    rng = np.random.default_rng(abs(hash(symbol)) % (2**31))
    
    # Seed price based on symbol
    base_price = {
        "EURUSD": 1.08, "GBPUSD": 1.27, "USDJPY": 149.5,
        "XAUUSD": 2320.0, "AUDUSD": 0.645, "USDCHF": 0.904,
        "BTCUSD": 65000.0, "ETHUSD": 3200.0,
    }.get(symbol, 1.0)

    prices = [base_price]
    for _ in range(n - 1):
        change = rng.normal(0, base_price * 0.001)
        prices.append(max(prices[-1] + change, base_price * 0.5))

    dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="h")
    opens = prices
    highs = [p * (1 + abs(rng.normal(0, 0.0005))) for p in prices]
    lows  = [p * (1 - abs(rng.normal(0, 0.0005))) for p in prices]
    closes = [p * (1 + rng.normal(0, 0.0003)) for p in prices]
    volumes = rng.integers(100, 10000, size=n).tolist()

    return pd.DataFrame({
        "time": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "tick_volume": volumes,
    }).set_index("time")


def _mock_ml_signal(symbol: str) -> tuple[str, float]:
    """Return a deterministic mock ML signal for testing."""
    rng = np.random.default_rng(abs(hash(symbol + "ml")) % (2**31))
    choices = ["BUY", "SELL", "HOLD"]
    signal = choices[rng.integers(0, 3)]
    prob = float(rng.uniform(0.45, 0.75))
    return signal, prob


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ollama LLM Multi-Agent Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--symbols", type=str, default="EURUSD,XAUUSD",
        help="Comma-separated list of symbols to analyze (default: EURUSD,XAUUSD)"
    )
    parser.add_argument(
        "--no-mt5", action="store_true",
        help="Run with synthetic mock data instead of connecting to MT5"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Analyze and log decisions but do NOT execute any trades"
    )
    parser.add_argument(
        "--ollama-url", type=str, default=None,
        help="Ollama API base URL (default: from .env or http://localhost:11434)"
    )
    parser.add_argument(
        "--news-model", type=str, default=None,
        help="Ollama model for news agent (default: from .env or mistral:7b)"
    )
    parser.add_argument(
        "--tech-model", type=str, default=None,
        help="Ollama model for tech agent (default: from .env or qwen2.5:14b)"
    )
    parser.add_argument(
        "--decision-model", type=str, default=None,
        help="Ollama model for decision agent (default: from .env or deepseek-r1:14b)"
    )
    parser.add_argument(
        "--balance", type=float, default=None,
        help="Override account balance (used for risk context)"
    )
    args = parser.parse_args()

    ensure_output_dir()
    settings = load_settings()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    # Resolve model names: CLI arg > .env > default
    news_model     = args.news_model     or getattr(settings, "ollama_news_model", "mistral:7b")
    tech_model     = args.tech_model     or getattr(settings, "ollama_tech_model", "qwen2.5:14b")
    decision_model = args.decision_model or getattr(settings, "ollama_decision_model", "deepseek-r1:14b")
    base_url       = args.ollama_url     or getattr(settings, "ollama_base_url", "http://localhost:11434")
    balance        = args.balance        or settings.initial_balance

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║         WEDNESDAY TRADING AI — Ollama LLM Multi-Agent        ║
╠══════════════════════════════════════════════════════════════╣
║  News Agent    : {news_model:<43}║
║  Tech Agent    : {tech_model:<43}║
║  Decision Agent: {decision_model:<43}║
║  Ollama URL    : {base_url:<43}║
║  Symbols       : {', '.join(symbols):<43}║
║  Mode          : {'DRY-RUN (no execution)' if args.dry_run else 'LIVE (decisions logged)':<43}║
║  MT5           : {'MOCK DATA' if args.no_mt5 else 'CONNECTED':<43}║
╚══════════════════════════════════════════════════════════════╝
""")

    # Initialise the LLM orchestrator
    orchestrator = LLMOrchestrator(
        news_model=news_model,
        tech_model=tech_model,
        decision_model=decision_model,
        base_url=base_url,
    )

    # Optionally connect to MT5
    client = None
    if not args.no_mt5:
        try:
            from trading_bot.mt5_client import MT5Client  # noqa: PLC0415
            client = MT5Client(settings)
            client.connect()
            print("[MT5] ✅ Connected to MetaTrader 5\n")
        except Exception as exc:
            print(f"[MT5] ⚠️  Could not connect: {exc}")
            print("[MT5] Falling back to mock data mode.\n")
            client = None

    all_results = []

    for symbol in symbols:
        print(f"\n{'='*62}")
        print(f"  🔍 Analyzing symbol: {symbol}")
        print(f"{'='*62}")

        # Get OHLCV data
        if client:
            try:
                tf = settings.live_fast_timeframe
                df = client.copy_rates(symbol, tf, 200)
                ml_signal, ml_prob = "HOLD", 0.5
                # TODO: hook in existing LSTM model predictions here
            except Exception as exc:
                print(f"  [!] MT5 data fetch failed for {symbol}: {exc}. Using mock.")
                df = _mock_ohlcv(symbol)
                ml_signal, ml_prob = _mock_ml_signal(symbol)
        else:
            df = _mock_ohlcv(symbol)
            ml_signal, ml_prob = _mock_ml_signal(symbol)

        print(f"  ML Signal (LSTM): {ml_signal} | Probability: {ml_prob:.3f}")
        print()

        # Run the LLM pipeline
        try:
            result = orchestrator.run(
                symbol=symbol,
                df=df,
                ml_signal=ml_signal,
                ml_probability=ml_prob,
                current_balance=balance,
            )

            result["timestamp"] = datetime.now().isoformat()
            result["dry_run"] = args.dry_run

            all_results.append(result)

            # Summary banner
            decision_icon = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(
                result["llm_decision"], "⚪"
            )
            print(f"\n  {decision_icon} FINAL LLM DECISION: {result['llm_decision']} "
                  f"(confidence={result['llm_confidence']:.2f}) "
                  f"[{result['elapsed_seconds']}s]")

        except Exception as exc:
            print(f"  [ERROR] Pipeline failed for {symbol}: {exc}")
            all_results.append({"symbol": symbol, "error": str(exc),
                                 "timestamp": datetime.now().isoformat()})

    # Save all decisions to JSON
    LLM_OUTPUT_PATH.write_text(
        json.dumps(all_results, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n✅ Decisions saved to: {LLM_OUTPUT_PATH}")

    # Print summary table
    print("\n" + "─" * 62)
    print(f"  {'SYMBOL':<12} {'ML':>6} {'SENTIMENT':>10} {'TECH':>10} {'DECISION':>10}")
    print("─" * 62)
    for r in all_results:
        if "error" in r and r.get("decision") is None:
            print(f"  {r['symbol']:<12} {'ERROR':>6}")
            continue
        sym    = r.get("symbol", "?")
        ml_s   = r.get("ml_signal", "?")
        sent   = r.get("news", {}).get("sentiment", "?")[:8]
        tech_v = r.get("technical", {}).get("view", "?")[:8]
        dec    = r.get("llm_decision", "?")
        print(f"  {sym:<12} {ml_s:>6} {sent:>10} {tech_v:>10} {dec:>10}")
    print("─" * 62)

    if client:
        client.shutdown()


if __name__ == "__main__":
    main()
