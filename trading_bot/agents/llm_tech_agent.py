"""
LLM Technical Analysis Agent — powered by qwen2.5:14b via Ollama.

Role: Perform a deep technical analysis of a symbol using OHLCV data and
      computed indicators (RSI, MACD, ATR, ADX, Bollinger Bands).
      Returns structured analysis with a directional view and key levels.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen2.5:14b"

SYSTEM_PROMPT = """\
You are a technical analysis assistant evaluating market conditions for an existing algorithmic trade signal.
A quantitative ML model has generated a signal. Analyze the OHLCV and indicator data to provide a directional view and suggest Stop Loss / Take Profit levels.
Your primary role is to identify if there is obvious structural support for the ML signal or if there is a massive technical red flag (e.g., major resistance immediately ahead of a BUY).
Default to confirming the ML signal's direction unless the technicals are extremely opposed.

Return a JSON object ONLY — no extra text.

Required JSON format:
{
  "view": "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence": <float 0.5-1.0>,
  "support": <float or null>,
  "resistance": <float or null>,
  "suggested_sl_pips": <float or null>,
  "suggested_tp_pips": <float or null>,
  "reasoning": "<1-2 sentences explaining technical context>"
}
"""


def _format_candles(df: pd.DataFrame, n: int = 10) -> str:
    """Format the last N OHLCV candles into a compact string for the prompt."""
    recent = df.tail(n)[["open", "high", "low", "close"]].copy()
    recent = recent.round(5)
    lines = ["Recent OHLCV (oldest → newest):"]
    for idx, row in recent.iterrows():
        lines.append(
            f"  O={row['open']} H={row['high']} L={row['low']} C={row['close']}"
        )
    return "\n".join(lines)


def _format_indicators(
    rsi: Optional[float],
    macd: Optional[float],
    macd_signal: Optional[float],
    atr: Optional[float],
    adx: Optional[float],
    bb_upper: Optional[float],
    bb_lower: Optional[float],
    ema_fast: Optional[float],
    ema_slow: Optional[float],
) -> str:
    parts = ["Indicators:"]
    if rsi is not None:
        parts.append(f"  RSI(14)={rsi:.2f}")
    if macd is not None and macd_signal is not None:
        parts.append(f"  MACD={macd:.6f}  Signal={macd_signal:.6f}")
    if atr is not None:
        parts.append(f"  ATR(14)={atr:.6f}")
    if adx is not None:
        parts.append(f"  ADX(14)={adx:.2f}")
    if bb_upper is not None and bb_lower is not None:
        parts.append(f"  BB_upper={bb_upper:.5f}  BB_lower={bb_lower:.5f}")
    if ema_fast is not None and ema_slow is not None:
        parts.append(f"  EMA_fast={ema_fast:.5f}  EMA_slow={ema_slow:.5f}")
    return "\n".join(parts)


class LLMTechAgent:
    """
    Technical Analysis Agent using qwen2.5:14b.

    Parameters
    ----------
    model : str
        Ollama model tag, default "qwen2.5:14b".
    base_url : str
        Ollama API base URL.
    timeout : int
        Request timeout in seconds.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import ollama  # noqa: PLC0415
                self._client = ollama.Client(host=self.base_url)
            except ImportError:
                raise RuntimeError(
                    "ollama package not installed. Run: pip install ollama"
                )
        return self._client

    def analyze(
        self,
        symbol: str,
        df: pd.DataFrame,
        rsi: Optional[float] = None,
        macd: Optional[float] = None,
        macd_signal: Optional[float] = None,
        atr: Optional[float] = None,
        adx: Optional[float] = None,
        bb_upper: Optional[float] = None,
        bb_lower: Optional[float] = None,
        ema_fast: Optional[float] = None,
        ema_slow: Optional[float] = None,
        ml_signal: Optional[str] = None,
        ml_probability: Optional[float] = None,
    ) -> dict:
        """
        Perform technical analysis for a symbol.

        Parameters
        ----------
        symbol : str
        df : pd.DataFrame
            OHLCV dataframe with columns [open, high, low, close].
        ml_signal : str, optional
            Signal from the existing LSTM model ("BUY"/"SELL"/"HOLD").
        ml_probability : float, optional
            Probability from LSTM model.

        Returns
        -------
        dict with keys:
            view, confidence, support, resistance,
            suggested_sl_pips, suggested_tp_pips, reasoning, model, error
        """
        candle_text = _format_candles(df)
        indicator_text = _format_indicators(
            rsi, macd, macd_signal, atr, adx, bb_upper, bb_lower, ema_fast, ema_slow
        )

        ml_context = ""
        if ml_signal:
            ml_context = (
                f"\nExisting ML model signal: {ml_signal}"
                + (f" (prob={ml_probability:.3f})" if ml_probability else "")
            )

        user_content = (
            f"Symbol: {symbol}\n\n"
            + candle_text
            + "\n\n"
            + indicator_text
            + ml_context
        )

        try:
            client = self._get_client()
            response = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                options={"temperature": 0.1, "num_predict": 400},
            )
            raw = response["message"]["content"].strip()

            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            result = json.loads(raw)
            result.setdefault("view", "NEUTRAL")
            result.setdefault("confidence", 0.5)
            result.setdefault("support", None)
            result.setdefault("resistance", None)
            result.setdefault("suggested_sl_pips", None)
            result.setdefault("suggested_tp_pips", None)
            result.setdefault("reasoning", "")
            result["model"] = self.model
            result["error"] = None
            return result

        except Exception as exc:  # noqa: BLE001
            logger.warning("[TechAgent] Error calling %s: %s", self.model, exc)
            return {
                "view": "NEUTRAL",
                "confidence": 0.0,
                "support": None,
                "resistance": None,
                "suggested_sl_pips": None,
                "suggested_tp_pips": None,
                "reasoning": "LLM unavailable",
                "model": self.model,
                "error": str(exc),
            }
