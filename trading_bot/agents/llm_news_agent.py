"""
LLM News/Sentiment Agent — powered by mistral:7b via Ollama.

Role: Analyze the current market context for a given symbol and return a
      sentiment score (BULLISH / BEARISH / NEUTRAL) with a confidence level.

Since this system does not have a live news feed, the agent uses the
latest OHLCV price action summary as a proxy for market narrative.
When a real news-feed API (e.g. NewsAPI, EODHD) is available, replace
the `build_context` helper to inject actual headlines.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "mistral:7b"

SYSTEM_PROMPT = """\
You are a professional forex and crypto market sentiment analyst.
You will receive a short market context (symbol, recent price action, 
key technical levels) and must return a JSON object ONLY — no extra text.

Required JSON format:
{
  "sentiment": "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence": <float 0.0–1.0>,
  "reasoning": "<one short sentence>"
}
"""


def _build_context(
    symbol: str,
    latest_close: float,
    pct_change_1d: float,
    pct_change_1w: float,
    rsi: Optional[float] = None,
    extra_news: Optional[str] = None,
) -> str:
    """Build a compact natural-language context string for the LLM."""
    lines = [
        f"Symbol: {symbol}",
        f"Latest close price: {latest_close:.5f}",
        f"24h change: {pct_change_1d:+.2f}%",
        f"7-day change: {pct_change_1w:+.2f}%",
    ]
    if rsi is not None:
        lines.append(f"RSI(14): {rsi:.1f}")
    if extra_news:
        lines.append(f"News headlines: {extra_news}")
    return "\n".join(lines)


class LLMNewsAgent:
    """
    Sentiment/News Agent using mistral:7b.

    Parameters
    ----------
    model : str
        Ollama model tag, default "mistral:7b".
    base_url : str
        Ollama API base URL, default "http://localhost:11434".
    timeout : int
        Request timeout in seconds.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self._client = None  # lazy init

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
        latest_close: float,
        pct_change_1d: float = 0.0,
        pct_change_1w: float = 0.0,
        rsi: Optional[float] = None,
        extra_news: Optional[str] = None,
    ) -> dict:
        """
        Analyze market sentiment for a symbol.

        Returns
        -------
        dict with keys:
            sentiment  : "BULLISH" | "BEARISH" | "NEUTRAL"
            confidence : float 0.0–1.0
            reasoning  : str
            model      : str
            error      : str | None
        """
        context = _build_context(
            symbol, latest_close, pct_change_1d, pct_change_1w, rsi, extra_news
        )

        try:
            client = self._get_client()
            response = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
                options={"temperature": 0.1, "num_predict": 200},
            )
            raw = response["message"]["content"].strip()

            # Extract JSON even if model wraps it in markdown
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            result = json.loads(raw)
            result.setdefault("sentiment", "NEUTRAL")
            result.setdefault("confidence", 0.5)
            result.setdefault("reasoning", "")
            result["model"] = self.model
            result["error"] = None
            return result

        except Exception as exc:  # noqa: BLE001
            logger.warning("[NewsAgent] Error calling %s: %s", self.model, exc)
            return {
                "sentiment": "NEUTRAL",
                "confidence": 0.0,
                "reasoning": "LLM unavailable",
                "model": self.model,
                "error": str(exc),
            }
