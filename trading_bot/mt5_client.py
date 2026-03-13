from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "Package MetaTrader5 belum terinstall. Jalankan: pip install MetaTrader5"
    ) from exc

from .config import Settings


TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


@dataclass
class SymbolSpec:
    symbol: str
    point: float
    digits: int
    volume_min: float
    volume_max: float
    volume_step: float
    trade_contract_size: float


class MT5Client:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.connected = False

    def connect(self) -> None:
        kwargs: dict[str, Any] = {}
        if self.settings.mt5_path:
            kwargs["path"] = self.settings.mt5_path

        if not mt5.initialize(**kwargs):
            code, msg = mt5.last_error()
            raise RuntimeError(f"Gagal initialize MT5: [{code}] {msg}")

        if self.settings.mt5_login:
            ok = mt5.login(
                login=self.settings.mt5_login,
                password=self.settings.mt5_password,
                server=self.settings.mt5_server,
            )
            if not ok:
                code, msg = mt5.last_error()
                mt5.shutdown()
                raise RuntimeError(f"Gagal login MT5: [{code}] {msg}")

        self.connected = True

    def shutdown(self) -> None:
        if self.connected:
            mt5.shutdown()
            self.connected = False

    def ensure_symbol(self, symbol: str) -> None:
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"Symbol {symbol} tidak ditemukan di MT5.")
        if not info.visible and not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Symbol {symbol} gagal di-select.")

    def symbol_spec(self, symbol: str) -> SymbolSpec:
        self.ensure_symbol(symbol)
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"Symbol {symbol} tidak ditemukan.")
        return SymbolSpec(
            symbol=symbol,
            point=info.point,
            digits=info.digits,
            volume_min=info.volume_min,
            volume_max=info.volume_max,
            volume_step=info.volume_step,
            trade_contract_size=info.trade_contract_size,
        )

    def symbol_info(self, symbol: str):
        self.ensure_symbol(symbol)
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"Symbol {symbol} tidak ditemukan.")
        return info

    def copy_rates(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        self.ensure_symbol(symbol)
        tf = TIMEFRAME_MAP[timeframe]
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            code, msg = mt5.last_error()
            raise RuntimeError(
                f"Gagal ambil data {symbol} {timeframe}: [{code}] {msg}"
            )
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(
            columns={
                "tick_volume": "volume",
                "real_volume": "real_volume",
                "spread": "spread",
            }
        )
        return df

    def copy_rates_range(
        self, symbol: str, timeframe: str, date_from: datetime, date_to: datetime
    ) -> pd.DataFrame:
        self.ensure_symbol(symbol)
        tf = TIMEFRAME_MAP[timeframe]
        rates = mt5.copy_rates_range(symbol, tf, date_from, date_to)
        if rates is None or len(rates) == 0:
            code, msg = mt5.last_error()
            raise RuntimeError(
                f"Gagal ambil data range {symbol} {timeframe}: [{code}] {msg}"
            )
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.rename(
            columns={
                "tick_volume": "volume",
                "real_volume": "real_volume",
                "spread": "spread",
            }
        )
        return df

    def list_symbols(
        self, visible_only: bool = True, tradable_only: bool = True
    ) -> list[str]:
        symbols = mt5.symbols_get()
        if symbols is None:
            code, msg = mt5.last_error()
            raise RuntimeError(f"Gagal ambil daftar simbol: [{code}] {msg}")

        out: list[str] = []
        for s in symbols:
            if visible_only and not s.visible:
                continue
            if tradable_only and getattr(s, "trade_mode", 0) == mt5.SYMBOL_TRADE_MODE_DISABLED:
                continue
            out.append(s.name)
        return sorted(set(out))

    def latest_tick(self, symbol: str):
        self.ensure_symbol(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"Gagal ambil tick untuk {symbol}.")
        return tick

    def positions(self, symbol: str):
        self.ensure_symbol(symbol)
        positions = mt5.positions_get(symbol=symbol)
        return positions or []

    def positions_all(self):
        positions = mt5.positions_get()
        return positions or []

    def account_info(self):
        info = mt5.account_info()
        if info is None:
            code, msg = mt5.last_error()
            raise RuntimeError(f"Gagal ambil account_info: [{code}] {msg}")
        return info

    @staticmethod
    def utc_now() -> datetime:
        return datetime.now(timezone.utc)
