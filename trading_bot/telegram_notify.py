"""
Telegram Notification Module for WednesdayAI
─────────────────────────────────────────────
Kirim notifikasi ke Telegram tanpa mengubah algoritma/model utama.
- notify_signal()  : saat sinyal baru muncul
- notify_trade_close() : saat trade close + chart
"""

import io
import os
import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from datetime import datetime, timezone

BOT_TOKEN  = "8650275168:AAHDvXtnK0TbHGUDKDwQ93v8idJh37h_iTQ"
CHAT_ID    = "139137870"
BASE_URL   = f"https://api.telegram.org/bot{BOT_TOKEN}"


# ─── helpers ────────────────────────────────────────────────────────────────

def _send_message(text: str) -> None:
    try:
        requests.post(
            f"{BASE_URL}/sendMessage",
            data={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        print(f"[TG] sendMessage error: {e}")


def _send_photo(buf: io.BytesIO, caption: str) -> None:
    try:
        buf.seek(0)
        requests.post(
            f"{BASE_URL}/sendPhoto",
            data={"chat_id": CHAT_ID, "caption": caption, "parse_mode": "HTML"},
            files={"photo": ("chart.png", buf, "image/png")},
            timeout=20,
        )
    except Exception as e:
        print(f"[TG] sendPhoto error: {e}")


# ─── public API ─────────────────────────────────────────────────────────────

def notify_signal(
    symbol: str,
    signal: str,       # "BUY" / "SELL"
    probability: float,
    entry_price: float,
    sl: float,
    tp: float,
    lot: float,
    balance: float,
    adaptive: bool,
) -> None:
    """Kirim pesan teks saat sinyal baru dieksekusi."""
    direction = "🟢 BUY" if signal == "BUY" else "🔴 SELL"
    emoji_adaptive = "✅" if adaptive else "❌"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    text = (
        f"<b>⚡ SIGNAL BARU — {symbol}</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🕐 Waktu   : {now}\n"
        f"📊 Arah    : {direction}\n"
        f"🎯 Prob    : {probability:.2%}\n"
        f"💰 Entry   : {entry_price:.5f}\n"
        f"🛡️ SL      : {sl:.5f}\n"
        f"🏹 TP      : {tp:.5f}\n"
        f"📦 Lot     : {lot:.2f}\n"
        f"🏦 Balance : ${balance:,.2f}\n"
        f"🤖 Adaptive: {emoji_adaptive}\n"
        f"━━━━━━━━━━━━━━━━━━"
    )
    _send_message(text)


def notify_trade_close(
    symbol: str,
    signal: str,
    entry_price: float,
    close_price: float,
    sl: float,
    tp: float,
    lot: float,
    pnl: float,
    balance: float,
    reason: str,           # "TP" / "SL" / "MANUAL" / "REVERSE"
    entry_time: datetime,
    close_time: datetime,
    price_data: pd.DataFrame | None = None,   # DataFrame OHLC untuk chart
) -> None:
    """Kirim hasil trade close beserta chart equity."""
    pnl_sign = "🟢" if pnl >= 0 else "🔴"
    reason_map = {"TP": "✅ Take Profit", "SL": "❌ Stop Loss",
                  "MANUAL": "🔧 Manual", "REVERSE": "🔄 Reverse"}
    reason_str = reason_map.get(reason.upper(), reason)

    duration = close_time - entry_time
    hours, rem = divmod(int(duration.total_seconds()), 3600)
    minutes = rem // 60

    text = (
        f"<b>{pnl_sign} TRADE CLOSED — {symbol}</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📊 Arah    : {'🟢 BUY' if signal=='BUY' else '🔴 SELL'}\n"
        f"💰 Entry   : {entry_price:.5f}\n"
        f"🏁 Close   : {close_price:.5f}\n"
        f"🛡️ SL      : {sl:.5f}\n"
        f"🏹 TP      : {tp:.5f}\n"
        f"📦 Lot     : {lot:.2f}\n"
        f"⏱️ Durasi  : {hours}h {minutes}m\n"
        f"📋 Alasan  : {reason_str}\n"
        f"💵 PnL     : {pnl_sign} ${pnl:+.2f}\n"
        f"🏦 Balance : ${balance:,.2f}\n"
        f"━━━━━━━━━━━━━━━━━━"
    )

    if price_data is not None and not price_data.empty:
        buf = _build_chart(
            symbol=symbol,
            signal=signal,
            df=price_data,
            entry_price=entry_price,
            close_price=close_price,
            sl=sl,
            tp=tp,
            entry_time=entry_time,
            close_time=close_time,
            pnl=pnl,
        )
        _send_photo(buf, caption=text)
    else:
        _send_message(text)


# ─── chart builder ──────────────────────────────────────────────────────────

def _build_chart(
    symbol: str,
    signal: str,
    df: pd.DataFrame,
    entry_price: float,
    close_price: float,
    sl: float,
    tp: float,
    entry_time: datetime,
    close_time: datetime,
    pnl: float,
) -> io.BytesIO:
    """Buat candlestick chart dengan garis Entry / SL / TP."""
    # — ambil kolom fleksibel
    col_map = {}
    for want, candidates in {
        "open": ["open", "Open", "m15_open"],
        "high": ["high", "High", "m15_high"],
        "low":  ["low",  "Low",  "m15_low"],
        "close": ["close","Close","m15_close"],
        "time":  ["time", "Time", "datetime", "date"],
    }.items():
        for c in candidates:
            if c in df.columns:
                col_map[want] = c
                break

    required = ["open", "high", "low", "close"]
    if not all(k in col_map for k in required):
        # fallback: buat chart harga simple kalau OHLC tidak tersedia
        return _build_line_chart(symbol, signal, entry_price, close_price, sl, tp, pnl)

    # — filter antara entry_time dan close_time (±5 bar sebelum/sesudah)
    if "time" in col_map:
        df["_dt"] = pd.to_datetime(df[col_map["time"]])
        df_plot = df[(df["_dt"] >= entry_time) & (df["_dt"] <= close_time)].copy()
        if df_plot.empty:
            df_plot = df.tail(60).copy()
    else:
        df_plot = df.tail(60).copy()

    # max 80 candle agar tidak sesak
    df_plot = df_plot.tail(80).reset_index(drop=True)

    # — plot
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0f0f0f")
    ax.set_facecolor("#0f0f0f")

    xs = range(len(df_plot))
    for i, row in df_plot.iterrows():
        o = row[col_map["open"]]
        h = row[col_map["high"]]
        l = row[col_map["low"]]
        c = row[col_map["close"]]
        color = "#4ade80" if c >= o else "#f87171"
        ax.plot([i, i], [l, h], color=color, linewidth=0.8)
        ax.add_patch(plt.Rectangle((i - 0.3, min(o, c)), 0.6, abs(c - o),
                                   color=color, zorder=2))

    # — level lines
    ax.axhline(entry_price, color="#facc15", linewidth=1.5,
               linestyle="--", label=f"Entry {entry_price:.5f}", zorder=3)
    ax.axhline(sl, color="#f87171", linewidth=1.5,
               linestyle=":",  label=f"SL   {sl:.5f}",    zorder=3)
    ax.axhline(tp, color="#4ade80", linewidth=1.5,
               linestyle=":",  label=f"TP   {tp:.5f}",    zorder=3)
    ax.axhline(close_price, color="#c084fc", linewidth=1.5,
               linestyle="-.", label=f"Close {close_price:.5f}", zorder=3)

    # — shading
    color_band = "#4ade8033" if pnl >= 0 else "#f8717133"
    ax.axhspan(min(entry_price, close_price),
               max(entry_price, close_price),
               alpha=0.15, color="#4ade80" if pnl >= 0 else "#f87171")

    # — labels
    pnl_sign = "+" if pnl >= 0 else ""
    ax.set_title(
        f"{symbol}  |  {'BUY' if signal=='BUY' else 'SELL'}  |  PnL: {pnl_sign}${pnl:.2f}",
        color="white", fontsize=13, fontweight="bold", pad=12
    )
    ax.tick_params(colors="white", labelsize=8)
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.3,
              labelcolor="white", facecolor="#1a1a1a")
    ax.grid(axis="y", color="#2a2a2a", linewidth=0.5)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="#0f0f0f")
    plt.close(fig)
    return buf


def _build_line_chart(symbol, signal, entry, close_p, sl, tp, pnl):
    """Fallback chart tanpa data OHLC — hanya garis horizontal."""
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0f0f0f")
    ax.set_facecolor("#0f0f0f")
    for lvl, label, color, ls in [
        (entry,   f"Entry {entry:.5f}",   "#facc15", "--"),
        (sl,      f"SL {sl:.5f}",         "#f87171", ":"),
        (tp,      f"TP {tp:.5f}",         "#4ade80", ":"),
        (close_p, f"Close {close_p:.5f}", "#c084fc", "-."),
    ]:
        ax.axhline(lvl, linestyle=ls, color=color, linewidth=1.5, label=label)
    pnl_sign = "+" if pnl >= 0 else ""
    ax.set_title(f"{symbol}  {signal}  PnL: {pnl_sign}${pnl:.2f}",
                 color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.3,
              labelcolor="white", facecolor="#1a1a1a")
    ax.grid(axis="y", color="#2a2a2a", linewidth=0.5)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="#0f0f0f")
    plt.close(fig)
    return buf
