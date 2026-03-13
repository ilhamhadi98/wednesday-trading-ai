from __future__ import annotations

from dataclasses import dataclass

try:
    import MetaTrader5 as mt5
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "Package MetaTrader5 belum terinstall. Jalankan: pip install MetaTrader5"
    ) from exc

from .backtest import compute_lot_size, pip_size_from_digits
from .config import Settings
from .mt5_client import MT5Client, SymbolSpec


@dataclass
class TradeActionResult:
    ok: bool
    message: str


class DemoExecutor:
    def __init__(self, client: MT5Client, settings: Settings, spec: SymbolSpec):
        self.client = client
        self.settings = settings
        self.spec = spec

    def _round_price(self, value: float) -> float:
        return round(value, self.spec.digits)

    def _active_positions(self):
        positions = self.client.positions(self.settings.symbol)
        return [p for p in positions if p.magic == self.settings.magic_number]

    def close_position(self, position) -> TradeActionResult:
        tick = self.client.latest_tick(self.settings.symbol)
        side = mt5.ORDER_TYPE_SELL if position.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if side == mt5.ORDER_TYPE_SELL else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.settings.symbol,
            "volume": position.volume,
            "type": side,
            "position": position.ticket,
            "price": self._round_price(price),
            "deviation": self.settings.deviation,
            "magic": self.settings.magic_number,
            "comment": "dl-bot-close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None:
            return TradeActionResult(ok=False, message="order_send close None")
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return TradeActionResult(
                ok=False,
                message=f"close gagal, retcode={result.retcode}, comment={result.comment}",
            )
        return TradeActionResult(ok=True, message=f"close sukses ticket={position.ticket}")

    def open_position(self, signal: int) -> TradeActionResult:
        tick = self.client.latest_tick(self.settings.symbol)
        spread_points = (tick.ask - tick.bid) / self.spec.point
        if spread_points > self.settings.max_spread_points:
            return TradeActionResult(
                ok=False,
                message=f"spread {spread_points:.1f} > max {self.settings.max_spread_points}, skip",
            )

        account = self.client.account_info()
        lot = compute_lot_size(account.balance, self.settings, self.spec)
        pip_size = pip_size_from_digits(self.spec.digits, self.spec.point)

        if signal == 1:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
            sl = price - (self.settings.stop_loss_pips * pip_size)
            tp = price + (self.settings.take_profit_pips * pip_size)
            label = "BUY"
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
            sl = price + (self.settings.stop_loss_pips * pip_size)
            tp = price - (self.settings.take_profit_pips * pip_size)
            label = "SELL"

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.settings.symbol,
            "volume": lot,
            "type": order_type,
            "price": self._round_price(price),
            "sl": self._round_price(sl),
            "tp": self._round_price(tp),
            "deviation": self.settings.deviation,
            "magic": self.settings.magic_number,
            "comment": "dl-bot-open",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None:
            return TradeActionResult(ok=False, message=f"open {label} gagal: order_send None")
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return TradeActionResult(
                ok=False,
                message=f"open {label} gagal, retcode={result.retcode}, comment={result.comment}",
            )
        return TradeActionResult(ok=True, message=f"open {label} sukses ticket={result.order}")

    def handle_signal(self, signal: int) -> TradeActionResult:
        active = self._active_positions()

        if not active and signal == 0:
            return TradeActionResult(ok=True, message="tidak ada posisi, tidak ada sinyal")

        if active:
            pos = active[0]
            pos_side = 1 if pos.type == mt5.POSITION_TYPE_BUY else -1
            if signal == 0:
                return TradeActionResult(ok=True, message="posisi aktif dipertahankan")
            if signal == pos_side:
                return TradeActionResult(ok=True, message="posisi aktif sudah searah sinyal")

            close_res = self.close_position(pos)
            if not close_res.ok:
                return close_res
            open_res = self.open_position(signal)
            msg = f"{close_res.message}; {open_res.message}"
            return TradeActionResult(ok=open_res.ok, message=msg)

        return self.open_position(signal)
