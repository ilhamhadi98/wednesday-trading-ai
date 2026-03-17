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
from .adaptive_sizing import analyze_performance, compute_adaptive_lot


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

    def _get_filling_mode(self, symbol: str) -> int:
        """Deteksi filling mode yang didukung broker untuk simbol ini."""
        try:
            info = self.client.symbol_info(symbol)
            filling = getattr(info, "filling_mode", 0)
            # Bit 0 = FOK, Bit 1 = IOC, Bit 2 = RETURN
            if filling & 2:
                return mt5.ORDER_FILLING_IOC
            if filling & 1:
                return mt5.ORDER_FILLING_FOK
            if filling & 4:
                return mt5.ORDER_FILLING_RETURN
        except Exception:
            pass
        return mt5.ORDER_FILLING_IOC

    def _send_with_filling_fallback(self, request: dict) -> object:
        """Kirim order, fallback ke RETURN jika IOC/FOK ditolak broker."""
        filling_modes = [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN]
        last_result = None
        for mode in filling_modes:
            request["type_filling"] = mode
            result = mt5.order_send(request)
            if result is None:
                continue
            if result.retcode != 10030:  # 10030 = Unsupported filling mode
                return result
            last_result = result
        return last_result

    def open_position(self, signal: int) -> TradeActionResult:
        tick = self.client.latest_tick(self.settings.symbol)
        pip_size = pip_size_from_digits(self.spec.digits, self.spec.point)
        spread_points = (tick.ask - tick.bid) / self.spec.point
        spread_price = spread_points * self.spec.point

        # Cek spread vs batas config
        if spread_points > self.settings.max_spread_points:
            return TradeActionResult(
                ok=False,
                message=f"spread {spread_points:.1f} > max {self.settings.max_spread_points}, skip",
            )

        # Cek spread vs SL: jika spread > SL distance, pair ini tidak layak
        sl_price_dist = self.settings.stop_loss_pips * pip_size
        if spread_price >= sl_price_dist:
            return TradeActionResult(
                ok=False,
                message=f"spread {spread_price:.5f} >= sl_dist {sl_price_dist:.5f}, pair tidak layak",
            )

        # ── Harga entry ─────────────────────────────────────────────────────────
        if signal == 1:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
            label = "BUY"
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
            label = "SELL"

        # ── SL/TP dengan minimum stop level broker ─────────────────────────────
        try:
            sym_info = self.client.symbol_info(self.settings.symbol)
            min_stop_pts = int(getattr(sym_info, "trade_stops_level", 0))
            # Jarak minimum = max(stops_level, sl_pips, spread + 2pip buffer)
            min_dist = max(
                (min_stop_pts + 1) * self.spec.point,
                sl_price_dist,
                spread_price + 2 * pip_size,
            )
        except Exception:
            min_dist = sl_price_dist

        tp_dist = max(self.settings.take_profit_pips * pip_size, min_dist * 1.5)

        if signal == 1:
            sl = price - min_dist
            tp = price + tp_dist
        else:
            sl = price + min_dist
            tp = price - tp_dist

        # ── Lot size adaptif berdasarkan performa terkini ──────────────────────
        account = self.client.account_info()
        perf = analyze_performance(symbol=self.settings.symbol, n_trades=20, days_back=30)
        print(
            f"[ADAPTIVE] {self.settings.symbol} | last {perf.total_trades} trades | "
            f"WinRate={perf.win_rate*100:.0f}% | ConsecLoss={perf.consecutive_losses} | "
            f"PNL={perf.recent_pnl:.2f} | {perf.note}",
            flush=True,
        )
        lot = compute_adaptive_lot(
            balance=account.balance,
            settings=self.settings,
            spec=self.spec,
            current_price=price,
            perf=perf,
        )

        # ── Validasi margin tersedia sebelum kirim order ──────────────────────
        try:
            margin_needed = mt5.order_calc_margin(order_type, self.settings.symbol, lot, price)
            if margin_needed is not None and margin_needed > account.margin_free:
                return TradeActionResult(
                    ok=False,
                    message=f"margin tidak cukup: butuh {margin_needed:.2f}, tersedia {account.margin_free:.2f}",
                )
        except Exception:
            pass

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
        result = self._send_with_filling_fallback(request)
        if result is None:
            return TradeActionResult(ok=False, message=f"open {label} gagal: order_send None")
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return TradeActionResult(
                ok=False,
                message=f"open {label} gagal, retcode={result.retcode}, comment={result.comment}",
            )
        return TradeActionResult(ok=True, message=f"open {label} sukses ticket={result.order} lot={lot}")



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
