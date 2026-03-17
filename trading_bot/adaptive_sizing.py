"""
Adaptive Position Sizer
=======================
Menyesuaikan ukuran lot secara dinamis berdasarkan performa trading terkini.

Logika:
- Ambil N trade terakhir dari history MT5
- Hitung consecutive losses & recent win rate
- Kurangi lot saat performa buruk, pulihkan saat performa membaik
- Minimum lot selalu = volume_min broker (tidak pernah di bawah itu)

Skema Pengurangan Lot:
  0    losses berturut-turut → 100% lot normal
  1    loss  berturut-turut → 60%  lot normal
  2    losses berturut-turut →  40%  lot normal
  3+   losses berturut-turut →  volume_min saja (mode ultra-konservatif)

Bonus Recovery: Setelah 2 win berturut-turut, lot dikembalikan ke level normal.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

try:
    import MetaTrader5 as mt5
except ImportError as exc:
    raise RuntimeError("MetaTrader5 belum terinstall.") from exc

from .mt5_client import SymbolSpec
from .config import Settings
from .backtest import compute_lot_size, pip_size_from_digits


@dataclass
class PerformanceSummary:
    total_trades: int
    wins: int
    losses: int
    consecutive_losses: int
    consecutive_wins: int
    win_rate: float
    recent_pnl: float
    lot_multiplier: float
    note: str


def _get_recent_trades(
    symbol: str | None = None,
    n_trades: int = 20,
    days_back: int = 30,
) -> list[dict]:
    """Ambil trade history terkini dari MT5, diurutkan terbaru duluan."""
    date_from = datetime.now(timezone.utc) - timedelta(days=days_back)
    date_to = datetime.now(timezone.utc)

    if symbol:
        deals = mt5.history_deals_get(date_from, date_to, group=f"*{symbol}*")
    else:
        deals = mt5.history_deals_get(date_from, date_to)

    if deals is None or len(deals) == 0:
        return []

    # Filter hanya deal penutup posisi (entry_type = 1 = DEAL_ENTRY_OUT)
    closed = [
        {
            "symbol": d.symbol,
            "profit": d.profit,
            "time": d.time,
            "type": d.type,
            "volume": d.volume,
            "price": d.price,
            "comment": d.comment,
        }
        for d in deals
        if d.entry == mt5.DEAL_ENTRY_OUT and d.profit != 0
    ]

    # Urutkan terbaru duluan, ambil N terakhir
    closed.sort(key=lambda x: x["time"], reverse=True)
    return closed[:n_trades]


def analyze_performance(
    symbol: str | None = None,
    n_trades: int = 20,
    days_back: int = 30,
) -> PerformanceSummary:
    """
    Analisa performa trading terkini dan tentukan multiplier lot.
    """
    trades = _get_recent_trades(symbol=symbol, n_trades=n_trades, days_back=days_back)

    if not trades:
        return PerformanceSummary(
            total_trades=0,
            wins=0,
            losses=0,
            consecutive_losses=0,
            consecutive_wins=0,
            win_rate=0.0,
            recent_pnl=0.0,
            lot_multiplier=1.0,
            note="Belum ada history trade. Gunakan lot normal.",
        )

    wins = sum(1 for t in trades if t["profit"] > 0)
    losses = sum(1 for t in trades if t["profit"] <= 0)
    total = len(trades)
    win_rate = wins / total if total > 0 else 0.0
    recent_pnl = sum(t["profit"] for t in trades)

    # Hitung consecutive losses (dari trade paling terbaru ke belakang)
    consecutive_losses = 0
    for t in trades:  # trades sudah diurut terbaru duluan
        if t["profit"] <= 0:
            consecutive_losses += 1
        else:
            break

    # Hitung consecutive wins
    consecutive_wins = 0
    for t in trades:
        if t["profit"] > 0:
            consecutive_wins += 1
        else:
            break

    # ── Tentukan multiplier lot ──────────────────────────────────────────────
    if consecutive_losses == 0:
        if consecutive_wins >= 2:
            multiplier = 1.0
            note = f"✅ {consecutive_wins} win berturut → lot penuh (100%)"
        else:
            multiplier = 1.0
            note = f"Performa netral → lot penuh (100%)"
    elif consecutive_losses == 1:
        multiplier = 0.6
        note = f"⚠️ 1 loss terakhir → lot dikurangi (60%)"
    elif consecutive_losses == 2:
        multiplier = 0.4
        note = f"⚠️ 2 loss berturut → lot dikurangi (40%)"
    else:
        multiplier = 0.0  # 0 = gunakan volume_min saja
        note = f"🛑 {consecutive_losses} loss berturut → mode ultra-konservatif (lot minimum)"

    return PerformanceSummary(
        total_trades=total,
        wins=wins,
        losses=losses,
        consecutive_losses=consecutive_losses,
        consecutive_wins=consecutive_wins,
        win_rate=win_rate,
        recent_pnl=recent_pnl,
        lot_multiplier=multiplier,
        note=note,
    )


def compute_adaptive_lot(
    balance: float,
    settings: Settings,
    spec: SymbolSpec,
    current_price: float,
    perf: Optional[PerformanceSummary] = None,
) -> float:
    """
    Hitung ukuran lot adaptif berdasarkan performa terkini.
    
    Returns:
        Lot size (float), selalu >= spec.volume_min
    """
    from .backtest import _quantize_lot

    if perf is None:
        perf = analyze_performance(symbol=spec.symbol)

    # Hitung lot dasar dari risk management
    try:
        base_lot = compute_lot_size(balance, settings, spec, current_price=current_price)
    except Exception:
        base_lot = spec.volume_min

    # Kalikan dengan multiplier performa
    if perf.lot_multiplier == 0.0:
        # Mode ultra-konservatif: hanya volume_min
        final_lot = spec.volume_min
    else:
        final_lot = base_lot * perf.lot_multiplier

    # Pastikan tidak di bawah volume_min
    final_lot = max(final_lot, spec.volume_min)
    return _quantize_lot(final_lot, spec)
