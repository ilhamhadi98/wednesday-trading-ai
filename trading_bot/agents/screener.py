import pandas as pd
import numpy as np
from typing import List
from trading_bot.mt5_client import MT5Client
from trading_bot.config import Settings

class MarketScreener:
    """
    Screener Agent (The Scout).
    Scans predefined markets to find the most tradeable pairs based on Volatility (ATR) and Trend Strength (ADX).
    """
    def __init__(self, client: MT5Client, settings: Settings):
        self.client = client
        self.settings = settings
        
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period + 1:
            return 0.0
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        return float(atr.iloc[-1])
        
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        if len(df) < period * 2:
            return 0.0
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm = plus_dm.apply(lambda x: x if x > 0 else 0)
        minus_dm = minus_dm.apply(lambda x: abs(x) if x < 0 else 0)
        
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=period, adjust=False).mean()
        return float(adx.iloc[-1])

    def screen_symbols(self, symbols: List[str], top_n: int = 5, timeframe: str = "H1") -> List[dict]:
        """
        Screen the given symbols to find the most active and trending pairs.
        Returns a sorted list of dictionaries with scores.
        """
        scores = []
        for symbol in symbols:
            try:
                # Fetch recent data for screening (e.g., last 200 bars)
                df = self.client.copy_rates(symbol, timeframe, 200)
                if df.empty or len(df) < 50:
                    continue
                
                # Calculate metrics
                atr = self._calculate_atr(df)
                adx = self._calculate_adx(df)
                
                # Handle volume
                avg_volume = 0.0
                if 'real_volume' in df.columns and df['real_volume'].sum() > 0:
                    avg_volume = float(df['real_volume'].mean())
                elif 'tick_volume' in df.columns:
                    avg_volume = float(df['tick_volume'].mean())
                
                # Normalize score
                # Strategy: Higher ADX means stronger trend. Higher ATR% means more volatility.
                close_price = df['close'].iloc[-1]
                atr_pct = (atr / close_price) * 100
                
                # Raw Score = Trend strength * Volatility factor
                raw_score = adx * atr_pct
                
                scores.append({
                    "symbol": symbol,
                    "score": round(raw_score, 4),
                    "adx": round(adx, 2),
                    "atr_pct": round(atr_pct, 4),
                    "avg_volume": round(avg_volume, 2)
                })
            except Exception as e:
                print(f"[Screener] Warning: Could not screen {symbol} - {e}")
                
        # Sort by highest score first
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_n]
