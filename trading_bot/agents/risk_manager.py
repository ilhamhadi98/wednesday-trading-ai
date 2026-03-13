import pandas as pd
from typing import Optional, Dict, Any, Tuple
from trading_bot.config import Settings
from trading_bot.mt5_client import SymbolSpec
from trading_bot.backtest import pnl_to_usd, usd_to_quote, pip_size_from_digits, _quantize_lot

class PortfolioRiskManager:
    """
    Risk Manager Agent (The Shield).
    Evaluates raw signals from the Strategist against capital preservation rules.
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        self.global_consecutive_losses = 0
        self.pair_consecutive_losses: Dict[str, int] = {}
        self.pair_pause_until_bar: Dict[str, int] = {}
        
    def compute_lot_size(self, balance: float, spec: SymbolSpec, current_price: float, sl_pips: Optional[float] = None) -> float:
        """
        Dynamically calculate position sizing based on account equity, risk percentage, and stop loss distance.
        """
        if not self.settings.use_risk_position_sizing:
            return _quantize_lot(self.settings.fixed_lot, spec)

        pip_sz = pip_size_from_digits(spec.digits, spec.point)
        effective_sl = sl_pips if sl_pips is not None else self.settings.stop_loss_pips
        sl_distance_price = effective_sl * pip_sz
        if sl_distance_price <= 0:
            return _quantize_lot(self.settings.fixed_lot, spec)

        risk_amount_usd = balance * self.settings.risk_per_trade
        risk_amount_quote = usd_to_quote(risk_amount_usd, spec.symbol, current_price)
        
        lot = risk_amount_quote / (sl_distance_price * spec.trade_contract_size + 1e-10)
        return _quantize_lot(lot, spec)

    def is_trade_allowed(self, symbol: str, current_drawdown_pct: float, current_bar_idx: int) -> Tuple[bool, str]:
        """
        Check if trading is allowed based on global circuit breaker and pair-specific consecutive losses.
        """
        circuit_limit = getattr(self.settings, "max_drawdown_circuit_breaker", 1.0)
        max_consec = getattr(self.settings, "max_consecutive_losses", 999)
        
        # 1. Global Drawdown Circuit Breaker
        if current_drawdown_pct >= circuit_limit:
            return False, "CIRCUIT_BREAKER"
            
        # 2. Re-activate paused pairs if cooler period has passed
        paused_until = self.pair_pause_until_bar.get(symbol, -1)
        if paused_until > 0 and current_bar_idx < paused_until:
            return False, "CONSECUTIVE_LOSS_GUARD"
        elif paused_until > 0 and current_bar_idx >= paused_until:
            # Cooler period ended, reset guard
            self.pair_consecutive_losses[symbol] = 0
            self.pair_pause_until_bar[symbol] = -1
            
        return True, "OK"
        
    def register_trade_result(self, symbol: str, pnl: float, current_bar_idx: int):
        """
        Update local risk states based on a closed trade. Identifies if the strategy is faltering on this pair.
        """
        max_consec = getattr(self.settings, "max_consecutive_losses", 999)
        
        if pnl <= 0:
            self.pair_consecutive_losses[symbol] = self.pair_consecutive_losses.get(symbol, 0) + 1
            self.global_consecutive_losses += 1
            
            # If hit limit, apply cooler period for the next (max_consec * 2) bars
            if self.pair_consecutive_losses[symbol] >= max_consec:
                self.pair_pause_until_bar[symbol] = current_bar_idx + (max_consec * 2)
        else:
            self.pair_consecutive_losses[symbol] = 0
            self.global_consecutive_losses = 0
            self.pair_pause_until_bar[symbol] = -1
