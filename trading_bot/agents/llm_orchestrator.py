"""
LLM Orchestrator — coordinates the 3 Ollama LLM agents sequentially.

Pipeline:
  1. LLMNewsAgent    (mistral:7b)    → sentiment
  2. LLMTechAgent    (qwen2.5:14b)  → technical view
  3. LLMDecisionAgent(deepseek-r1:14b) → final BUY/SELL/HOLD

The orchestrator also computes basic technical indicators from raw OHLCV
so the tech agent always has numbers to reason about, even if callers
do not pre-compute them.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd
import numpy as np

from trading_bot.agents.llm_news_agent import LLMNewsAgent
from trading_bot.agents.llm_tech_agent import LLMTechAgent
from trading_bot.agents.llm_decision_agent import LLMDecisionAgent

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Indicator helpers (keep dependency-free — no pandas_ta needed)
# ──────────────────────────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / (loss + 1e-10)
    return float(100 - 100 / (1 + rs.iloc[-1]))


def _ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1])


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])


def _adx(df: pd.DataFrame, period: int = 14) -> float:
    high, low, prev_close = df["high"], df["low"], df["close"].shift(1)
    plus_dm = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-10)
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-10)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return float(dx.ewm(span=period, adjust=False).mean().iloc[-1])


def _bollinger(close: pd.Series, period: int = 20):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    return float((mid + 2 * std).iloc[-1]), float((mid - 2 * std).iloc[-1])


def _compute_indicators(df: pd.DataFrame) -> dict:
    """Compute all indicators from OHLCV dataframe."""
    close = df["close"]
    rsi = _rsi(close)
    macd_val, macd_sig = _macd(close)
    atr = _atr(df)
    adx = _adx(df)
    bb_upper, bb_lower = _bollinger(close)
    ema_fast = float(_ema(close, 20).iloc[-1])
    ema_slow = float(_ema(close, 50).iloc[-1])
    return {
        "rsi": rsi,
        "macd": macd_val,
        "macd_signal": macd_sig,
        "atr": atr,
        "adx": adx,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

class LLMOrchestrator:
    """
    Manages the 3-agent Ollama pipeline.

    Parameters
    ----------
    news_model : str
    tech_model : str
    decision_model : str
    base_url : str
        Shared Ollama API base URL.
    """

    def __init__(
        self,
        news_model: str = "mistral:7b",
        tech_model: str = "qwen2.5:14b",
        decision_model: str = "deepseek-r1:14b",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.news_agent = LLMNewsAgent(model=news_model, base_url=base_url)
        self.tech_agent = LLMTechAgent(model=tech_model, base_url=base_url)
        self.decision_agent = LLMDecisionAgent(model=decision_model, base_url=base_url)

    def run(
        self,
        symbol: str,
        df: pd.DataFrame,
        ml_signal: str = "HOLD",
        ml_probability: float = 0.5,
        portfolio_drawdown_pct: float = 0.0,
        current_balance: float = 100.0,
        open_positions: int = 0,
        recent_trades: list = None,
        extra_news: Optional[str] = None,
    ) -> dict:
        """
        Run the full 3-agent pipeline for a single symbol.

        Parameters
        ----------
        symbol : str
            Trading pair, e.g. "EURUSD".
        df : pd.DataFrame
            OHLCV dataframe with at least 50 rows and columns
            [open, high, low, close].
        ml_signal : str
            Signal produced by the existing LSTM model.
        ml_probability : float
            Confidence probability from LSTM model.
        portfolio_drawdown_pct : float
            Current portfolio drawdown (0.0–1.0).
        current_balance : float
            Current account balance in USD.
        open_positions : int
            Number of currently open positions.
        extra_news : str, optional
            Free-text news headlines to feed to the news agent.

        Returns
        -------
        dict
            Full pipeline result including all intermediate outputs.
        """
        t_start = time.time()
        close = df["close"]
        latest_close = float(close.iloc[-1])

        # Pre-compute 1d and 1w percentage changes
        pct_1d = 0.0
        pct_1w = 0.0
        if len(close) >= 2:
            pct_1d = (close.iloc[-1] / close.iloc[-2] - 1) * 100
        if len(close) >= 8:
            pct_1w = (close.iloc[-1] / close.iloc[-8] - 1) * 100

        # Compute technical indicators
        indicators = _compute_indicators(df)

        # ── Step 1: News/Sentiment Agent ──────────────────────────────────────
        print(f"  [LLM 1/3] 📰 NewsAgent (mistral:7b) analyzing {symbol}...")
        news_result = self.news_agent.analyze(
            symbol=symbol,
            latest_close=latest_close,
            pct_change_1d=pct_1d,
            pct_change_1w=pct_1w,
            rsi=indicators["rsi"],
            extra_news=extra_news,
        )
        print(
            f"           → Sentiment: {news_result['sentiment']} "
            f"(conf={news_result['confidence']:.2f}) | {news_result['reasoning']}"
        )

        # ── Step 2: Technical Analysis Agent ──────────────────────────────────
        print(f"  [LLM 2/3] 📊 TechAgent (qwen2.5:14b) analyzing {symbol}...")
        tech_result = self.tech_agent.analyze(
            symbol=symbol,
            df=df,
            ml_signal=ml_signal,
            ml_probability=ml_probability,
            **indicators,
        )
        print(
            f"           → View: {tech_result['view']} "
            f"(conf={tech_result['confidence']:.2f}) | {tech_result['reasoning']}"
        )

        # ── Step 3: Decision Agent ─────────────────────────────────────────────
        print(f"  [LLM 3/3] 🎯 DecisionAgent (deepseek-r1:14b) deciding for {symbol}...")
        decision_result = self.decision_agent.decide(
            symbol=symbol,
            ml_signal=ml_signal,
            ml_probability=ml_probability,
            sentiment=news_result["sentiment"],
            sentiment_confidence=news_result["confidence"],
            sentiment_reasoning=news_result.get("reasoning", ""),
            tech_view=tech_result["view"],
            tech_confidence=tech_result["confidence"],
            tech_reasoning=tech_result.get("reasoning", ""),
            suggested_sl_pips=tech_result.get("suggested_sl_pips"),
            suggested_tp_pips=tech_result.get("suggested_tp_pips"),
            portfolio_drawdown_pct=portfolio_drawdown_pct,
            current_balance=current_balance,
            open_positions=open_positions,
            recent_trades=recent_trades,
        )
        print(
            f"           → Decision: {decision_result['decision']} "
            f"(conf={decision_result['confidence']:.2f})\n"
            f"           Rationale: {decision_result['entry_rationale']}"
        )

        elapsed = time.time() - t_start
        return {
            "symbol": symbol,
            "ml_signal": ml_signal,
            "ml_probability": ml_probability,
            "indicators": indicators,
            "news": news_result,
            "technical": tech_result,
            "decision": decision_result,
            "llm_decision": decision_result["decision"],
            "llm_confidence": decision_result["confidence"],
            "elapsed_seconds": round(elapsed, 1),
        }
