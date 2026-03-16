"""
LLM Decision Agent — powered by deepseek-r1:14b via Ollama.

Role: The final decision maker. Receives the full context (ML signal +
      news sentiment + technical view) and produces the authoritative
      BUY / SELL / HOLD decision with detailed reasoning.

This is the most powerful model in the pipeline and is invoked last,
ensuring it has all available context before committing to a decision.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "deepseek-r1:14b"

SYSTEM_PROMPT = """\
You are an AI Risk Manager working alongside a highly profitable quantitative Machine Learning (LSTM) model. 
The ML model has already identified a high-probability trading setup. Your job is NOT to find a setup from scratch, but merely to look for SEVERE RED FLAGS that would invalidate the ML model's signal.

Your decision must respect:
1. Trust the ML signal: The ML model is highly accurate. Default to outputting the exact same decision as the ML signal (e.g., if ML says BUY, you must output BUY).
2. Override ONLY on major red flags: Override the ML model and choose HOLD only if there is a massive contradiction (e.g., highly bearish news during a BUY signal, or extreme resistance).
3. Do not demand perfect confluence. The ML model has already found the edge.
4. Adapt to Memory: Analyze the [SYSTEM MEMORY] block. If you are experiencing a recent losing streak (e.g., choppy sideways market), become STRICTER and more willing to veto weak signals. If you are on a winning streak, trust the ML model fully.

Return a JSON object ONLY — no extra text, no <think> blocks.

Required JSON format:
{
  "decision": "BUY" | "SELL" | "HOLD",
  "confidence": <float 0.8-1.0 if agreeing with ML, else lower>,
  "risk_reward_ratio": <float or null>,
  "entry_rationale": "<brief explanation, acknowledging the ML signal and recent system memory>",
  "risk_warning": "<any concerns or 'None' if clear>"
}
"""


def _build_decision_context(
    symbol: str,
    ml_signal: str,
    ml_probability: float,
    sentiment: str,
    sentiment_confidence: float,
    sentiment_reasoning: str,
    tech_view: str,
    tech_confidence: float,
    tech_reasoning: str,
    suggested_sl_pips: Optional[float],
    suggested_tp_pips: Optional[float],
    portfolio_drawdown_pct: float = 0.0,
    current_balance: float = 100.0,
    open_positions: int = 0,
    recent_trades: list = None,
) -> str:
    parts = [
        f"Symbol: {symbol}",
        f"\n--- ML Model Signal ---",
        f"Signal: {ml_signal}",
        f"Probability: {ml_probability:.3f}",
        f"\n--- News/Sentiment Agent (mistral:7b) ---",
        f"Sentiment: {sentiment} (confidence: {sentiment_confidence:.2f})",
        f"Reasoning: {sentiment_reasoning}",
        f"\n--- Technical Analysis Agent (qwen2.5:14b) ---",
        f"View: {tech_view} (confidence: {tech_confidence:.2f})",
        f"Reasoning: {tech_reasoning}",
    ]
    if suggested_sl_pips:
        parts.append(f"Suggested SL: {suggested_sl_pips} pips")
    if suggested_tp_pips:
        parts.append(f"Suggested TP: {suggested_tp_pips} pips")
    parts += [
        f"\n--- Portfolio State ---",
        f"Current balance: ${current_balance:.2f}",
        f"Current drawdown: {portfolio_drawdown_pct * 100:.1f}%",
        f"Open positions: {open_positions}",
    ]
    
    parts.append(f"\n--- [SYSTEM MEMORY] Recent Trades ---")
    if not recent_trades:
        parts.append("No recent trade history available yet.")
    else:
        wins = sum(1 for t in recent_trades if t.get("pnl_usd", 0) > 0)
        win_rate = (wins / len(recent_trades)) * 100
        parts.append(f"Recent Win Rate (last {len(recent_trades)} trades): {win_rate:.1f}%")
        parts.append("Latest actions:")
        for i, t in enumerate(recent_trades[-5:], 1):  # show max last 5 for brevity
            outcome = "WIN" if t.get("pnl_usd", 0) > 0 else "LOSS"
            parts.append(f" {i}. {t['side']} -> {outcome} ({t['pnl_pips']} pips). Old rationale: {t['rationale']}")
            
    return "\n".join(parts)


class LLMDecisionAgent:
    """
    Final Decision Agent using deepseek-r1:14b.

    Parameters
    ----------
    model : str
        Ollama model tag, default "deepseek-r1:14b".
    base_url : str
        Ollama API base URL.
    timeout : int
        Request timeout in seconds.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = "http://localhost:11434",
        timeout: int = 180,
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

    def decide(
        self,
        symbol: str,
        ml_signal: str,
        ml_probability: float,
        sentiment: str,
        sentiment_confidence: float,
        sentiment_reasoning: str,
        tech_view: str,
        tech_confidence: float,
        tech_reasoning: str,
        suggested_sl_pips: Optional[float] = None,
        suggested_tp_pips: Optional[float] = None,
        portfolio_drawdown_pct: float = 0.0,
        current_balance: float = 100.0,
        open_positions: int = 0,
        recent_trades: list = None,
    ) -> dict:
        """
        Make the final trading decision by synthesizing all agent inputs.

        Returns
        -------
        dict with keys:
            decision          : "BUY" | "SELL" | "HOLD"
            confidence        : float 0.0–1.0
            risk_reward_ratio : float | None
            entry_rationale   : str
            risk_warning      : str
            model             : str
            error             : str | None
        """
        context = _build_decision_context(
            symbol=symbol,
            ml_signal=ml_signal,
            ml_probability=ml_probability,
            sentiment=sentiment,
            sentiment_confidence=sentiment_confidence,
            sentiment_reasoning=sentiment_reasoning,
            tech_view=tech_view,
            tech_confidence=tech_confidence,
            tech_reasoning=tech_reasoning,
            suggested_sl_pips=suggested_sl_pips,
            suggested_tp_pips=suggested_tp_pips,
            portfolio_drawdown_pct=portfolio_drawdown_pct,
            current_balance=current_balance,
            open_positions=open_positions,
            recent_trades=recent_trades,
        )

        try:
            client = self._get_client()
            response = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
                options={"temperature": 0.0, "num_predict": 500},
            )
            raw = response["message"]["content"].strip()

            # deepseek-r1 sometimes wraps output in <think>...</think> blocks
            if "<think>" in raw and "</think>" in raw:
                raw = raw.split("</think>")[-1].strip()

            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            result = json.loads(raw)
            result.setdefault("decision", "HOLD")
            result.setdefault("confidence", 0.5)
            result.setdefault("risk_reward_ratio", None)
            result.setdefault("entry_rationale", "")
            result.setdefault("risk_warning", "")
            result["model"] = self.model
            result["error"] = None
            return result

        except Exception as exc:  # noqa: BLE001
            logger.warning("[DecisionAgent] Error calling %s: %s", self.model, exc)
            return {
                "decision": "HOLD",
                "confidence": 0.0,
                "risk_reward_ratio": None,
                "entry_rationale": "LLM unavailable — defaulting to HOLD",
                "risk_warning": str(exc),
                "model": self.model,
                "error": str(exc),
            }
