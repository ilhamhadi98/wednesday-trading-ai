from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import re

import numpy as np
import pandas as pd

from .backtest import pip_size_from_digits
from .config import Settings
from .modeling import ModelArtifacts, predict_proba
from .mt5_client import MT5Client
from .workflows import fetch_feature_frame


PAIR_RE = re.compile(r"^[A-Z]{6}$")
CCY_CODES = {
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "CHF",
    "CAD",
    "AUD",
    "NZD",
    "SEK",
    "NOK",
    "DKK",
    "PLN",
    "CZK",
    "HUF",
    "CNH",
    "CNY",
    "SGD",
    "HKD",
    "MXN",
    "BRL",
    "ZAR",
    "RUB",
    "TRY",
    "XAU",
    "XAG",
}
MAJOR_PAIRS = {
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCHF",
    "USDCAD",
    "AUDUSD",
    "NZDUSD",
}
G10_CCY = {"USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD", "SEK", "NOK", "DKK"}


@dataclass
class SymbolOpportunity:
    symbol: str
    status: str
    reason: str
    bar_time: pd.Timestamp | None
    probability: float | None
    signal: str
    confidence: float | None
    spread_points: float | None
    atr_pips: float | None


def signal_from_probability(prob: float, buy_th: float, sell_th: float) -> int:
    if prob >= buy_th:
        return 1
    if prob <= sell_th:
        return -1
    return 0


def signal_label(signal: int) -> str:
    if signal == 1:
        return "BUY"
    if signal == -1:
        return "SELL"
    return "HOLD"


def resolve_symbols(
    client: MT5Client, settings: Settings, explicit_symbols: Iterable[str] | None = None
) -> list[str]:
    if explicit_symbols:
        symbols = [s.strip().upper() for s in explicit_symbols if s and s.strip()]
        return sorted(set(symbols))

    mode = settings.symbol_discovery_mode.strip().upper()
    if mode == "LIST":
        symbols = settings.symbols
    elif mode == "ALL":
        symbols = client.list_symbols(visible_only=False, tradable_only=True)
    else:
        symbols = client.list_symbols(visible_only=True, tradable_only=True)

    symbols = sorted(set(s.strip().upper() for s in symbols if s and s.strip()))
    if settings.scan_max_symbols > 0:
        symbols = symbols[: settings.scan_max_symbols]
    return symbols


def is_pair_symbol(symbol_info) -> bool:
    name = str(symbol_info.name).upper()
    if not PAIR_RE.match(name):
        return False
    path = str(getattr(symbol_info, "path", "")).lower()
    if "forex" in path or "fx" in path:
        return True
    base, quote = name[:3], name[3:]
    return base in CCY_CODES and quote in CCY_CODES


def resolve_tradable_pairs(
    client: MT5Client, settings: Settings, explicit_symbols: Iterable[str] | None = None
) -> list[str]:
    if explicit_symbols:
        raw_symbols = [s.strip().upper() for s in explicit_symbols if s and s.strip()]
    else:
        mode = settings.symbol_discovery_mode.strip().upper()
        if mode == "LIST":
            raw_symbols = settings.symbols
        else:
            # Untuk pair scanning, pakai seluruh simbol tradable agar tidak bias alfabet/visible state.
            raw_symbols = client.list_symbols(visible_only=False, tradable_only=True)

    out = []
    info_map = {}
    for symbol in raw_symbols:
        try:
            info = client.symbol_info(symbol)
            info_map[symbol] = info
            if is_pair_symbol(info):
                out.append(symbol)
        except Exception:
            continue

    out = sorted(set(out))
    include_set = {x.strip().upper() for x in settings.pair_include if x.strip()}
    exclude_set = {x.strip().upper() for x in settings.pair_exclude if x.strip()}
    mode = settings.pair_filter_mode.strip().upper()

    def _pass_mode(symbol: str) -> bool:
        if include_set and mode in {"CUSTOM", "KUSTOM"}:
            return symbol in include_set
        if mode in {"MAJORS", "MAYOR"}:
            return symbol in MAJOR_PAIRS or symbol in include_set
        if mode in {"MAJORS_MINORS", "MAYOR_MINOR"}:
            base, quote = symbol[:3], symbol[3:]
            return (base in G10_CCY and quote in G10_CCY) or symbol in include_set
        return True

    filtered = []
    for symbol in out:
        if symbol in exclude_set:
            continue
        if not _pass_mode(symbol):
            continue
        info = info_map.get(symbol)
        if info is not None and float(getattr(info, "volume_min", 0.0)) > settings.pair_max_volume_min:
            continue
        filtered.append(symbol)

    filtered = sorted(set(filtered))
    if settings.scan_max_symbols > 0:
        filtered = filtered[: settings.scan_max_symbols]
    return filtered


def evaluate_symbol(
    client: MT5Client,
    settings: Settings,
    artifacts: ModelArtifacts,
    symbol: str,
    fast_tf: str | None = None,
    slow_tf: str | None = None,
) -> SymbolOpportunity:
    try:
        frame, _ = fetch_feature_frame(
            client,
            settings,
            symbol=symbol,
            fast_tf=fast_tf,
            slow_tf=slow_tf,
        )
        if len(frame) < max(settings.min_bars_required, artifacts.lookback + 10):
            return SymbolOpportunity(
                symbol=symbol,
                status="SKIP",
                reason="data_tidak_cukup",
                bar_time=None,
                probability=None,
                signal="HOLD",
                confidence=None,
                spread_points=None,
                atr_pips=None,
            )

        closed_frame = frame.iloc[:-1].copy()
        if len(closed_frame) < artifacts.lookback:
            return SymbolOpportunity(
                symbol=symbol,
                status="SKIP",
                reason="lookback_tidak_cukup",
                bar_time=None,
                probability=None,
                signal="HOLD",
                confidence=None,
                spread_points=None,
                atr_pips=None,
            )

        last_bar_time = pd.Timestamp(closed_frame.index[-1])
        now_utc = pd.Timestamp.now(tz="UTC")
        age_hours = (now_utc - last_bar_time).total_seconds() / 3600.0
        if age_hours > settings.max_signal_bar_age_hours:
            return SymbolOpportunity(
                symbol=symbol,
                status="SKIP",
                reason=f"bar_tua:{age_hours:.1f}h",
                bar_time=last_bar_time,
                probability=None,
                signal="HOLD",
                confidence=None,
                spread_points=None,
                atr_pips=None,
            )

        missing_cols = [c for c in artifacts.feature_cols if c not in closed_frame.columns]
        if missing_cols:
            return SymbolOpportunity(
                symbol=symbol,
                status="SKIP",
                reason=f"feature_hilang:{','.join(missing_cols[:3])}",
                bar_time=last_bar_time,
                probability=None,
                signal="HOLD",
                confidence=None,
                spread_points=None,
                atr_pips=None,
            )

        window = closed_frame.iloc[-artifacts.lookback :].copy()
        x = artifacts.scaler.transform(window[artifacts.feature_cols].values)
        x = np.expand_dims(x, axis=0).astype(np.float32)
        prob = float(predict_proba(artifacts.model, x)[0])
        signal = signal_from_probability(
            prob=prob,
            buy_th=settings.prediction_buy_threshold,
            sell_th=settings.prediction_sell_threshold,
        )

        tick = client.latest_tick(symbol)
        spec = client.symbol_spec(symbol)
        spread_points = (tick.ask - tick.bid) / spec.point
        pip_size = pip_size_from_digits(spec.digits, spec.point)
        atr_value = float(closed_frame.iloc[-1]["h1_atr_14"])
        atr_pips = atr_value / pip_size if pip_size > 0 else None
        confidence = abs(prob - 0.5)

        if spread_points > settings.max_spread_points:
            return SymbolOpportunity(
                symbol=symbol,
                status="SKIP",
                reason=f"spread_tinggi:{spread_points:.1f}",
                bar_time=last_bar_time,
                probability=prob,
                signal=signal_label(signal),
                confidence=confidence,
                spread_points=spread_points,
                atr_pips=atr_pips,
            )

        return SymbolOpportunity(
            symbol=symbol,
            status="OK",
            reason="",
            bar_time=last_bar_time,
            probability=prob,
            signal=signal_label(signal),
            confidence=confidence,
            spread_points=spread_points,
            atr_pips=atr_pips,
        )
    except Exception as exc:
        return SymbolOpportunity(
            symbol=symbol,
            status="ERROR",
            reason=str(exc),
            bar_time=None,
            probability=None,
            signal="HOLD",
            confidence=None,
            spread_points=None,
            atr_pips=None,
        )


def scan_opportunities(
    client: MT5Client,
    settings: Settings,
    artifacts: ModelArtifacts,
    symbols: Iterable[str] | None = None,
    pair_only: bool = False,
    fast_tf: str | None = None,
    slow_tf: str | None = None,
) -> pd.DataFrame:
    if pair_only:
        chosen_symbols = resolve_tradable_pairs(client, settings, symbols)
    else:
        chosen_symbols = resolve_symbols(client, settings, symbols)
    rows = []
    for symbol in chosen_symbols:
        opp = evaluate_symbol(
            client,
            settings,
            artifacts,
            symbol,
            fast_tf=fast_tf,
            slow_tf=slow_tf,
        )
        rows.append(
            {
                "symbol": opp.symbol,
                "status": opp.status,
                "reason": opp.reason,
                "bar_time": opp.bar_time,
                "signal": opp.signal,
                "probability": opp.probability,
                "confidence": opp.confidence,
                "spread_points": opp.spread_points,
                "atr_pips": opp.atr_pips,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df_ok = df[df["status"] == "OK"].copy()
    if not df_ok.empty:
        # Ranking: prioritaskan sinyal aktif (BUY/SELL) dengan confidence besar dan spread kecil.
        df_ok["signal_rank"] = df_ok["signal"].map({"BUY": 1, "SELL": 1, "HOLD": 0}).fillna(0)
        df_ok["score"] = (
            (df_ok["signal_rank"] * 2.0)
            + df_ok["confidence"].fillna(0.0)
            - (df_ok["spread_points"].fillna(0.0) * 0.001)
        )
        df_ok = df_ok.sort_values(["score", "confidence"], ascending=False).drop(
            columns=["signal_rank"]
        )
        df_other = df[df["status"] != "OK"].copy()
        return pd.concat([df_ok, df_other], ignore_index=True)

    return df
