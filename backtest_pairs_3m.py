from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from trading_bot.backtest import compute_lot_size, pip_size_from_digits
from trading_bot.config import OUTPUT_DIR, ensure_output_dir, load_settings
from trading_bot.data_pipeline import build_multitimeframe_frame, make_sequences
from trading_bot.market_scan import resolve_tradable_pairs
from trading_bot.modeling import load_artifacts, predict_proba
from trading_bot.mt5_client import MT5Client, SymbolSpec


@dataclass
class PairSignals:
    symbol: str
    spec: SymbolSpec
    signals: pd.DataFrame


def _tf_minutes(tf: str) -> int:
    m = tf.upper()
    table = {
        "M1": 1,
        "M5": 5,
        "M15": 15,
        "M30": 30,
        "H1": 60,
        "H4": 240,
        "D1": 1440,
    }
    return table.get(m, 60)


def discover_tradable_pairs(client: MT5Client, settings) -> list[str]:
    return resolve_tradable_pairs(
        client=client,
        settings=settings,
        explicit_symbols=None,
    )


def _build_pair_signals(
    client: MT5Client,
    settings,
    artifacts,
    symbol: str,
    start_utc: datetime,
    end_utc: datetime,
) -> PairSignals:
    fast_tf = settings.backtest_fast_timeframe
    slow_tf = settings.backtest_slow_timeframe

    fast_days = max(10, int((settings.lookback * _tf_minutes(fast_tf)) / 1440) + 10)
    slow_days = max(20, int((settings.lookback * _tf_minutes(slow_tf)) / 1440) + 20)
    h1_start = start_utc - relativedelta(days=fast_days)
    h4_start = start_utc - relativedelta(days=slow_days)

    h1 = client.copy_rates_range(symbol, fast_tf, h1_start, end_utc)
    h4 = client.copy_rates_range(symbol, slow_tf, h4_start, end_utc)
    frame, _ = build_multitimeframe_frame(h1, h4)
    frame = frame[(frame.index >= pd.Timestamp(start_utc)) & (frame.index <= pd.Timestamp(end_utc))].copy()

    if len(frame) < (artifacts.lookback + 80):
        raise RuntimeError("data frame terlalu pendek")

    missing_cols = [c for c in artifacts.feature_cols if c not in frame.columns]
    if missing_cols:
        raise RuntimeError(f"feature hilang: {','.join(missing_cols[:3])}")

    ds, _ = make_sequences(
        frame=frame,
        feature_cols=artifacts.feature_cols,
        lookback=artifacts.lookback,
        scaler=artifacts.scaler,
        fit_scaler=False,
    )
    if len(ds.x) < 60:
        raise RuntimeError("sequence terlalu sedikit")

    probs = predict_proba(artifacts.model, ds.x)
    signals = np.where(
        probs >= settings.prediction_buy_threshold,
        1,
        np.where(probs <= settings.prediction_sell_threshold, -1, 0),
    )
    conf = np.abs(probs - 0.5)

    candle_slice = frame.loc[ds.timestamps, ["h1_high", "h1_low"]]
    df = pd.DataFrame(
        {
            "time": ds.timestamps,
            "symbol": symbol,
            "probability": probs,
            "signal": signals,
            "confidence": conf,
            "close": ds.close,
            "spread_points": ds.spread,
            "high": candle_slice["h1_high"].values,
            "low": candle_slice["h1_low"].values,
        }
    )
    spec = client.symbol_spec(symbol)
    return PairSignals(symbol=symbol, spec=spec, signals=df)


def run_portfolio_backtest(
    signals_all: pd.DataFrame,
    specs: dict[str, SymbolSpec],
    settings,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    balance = settings.initial_balance
    peak_balance = balance
    max_drawdown = 0.0

    positions: dict[str, dict] = {}
    trades: list[dict] = []
    equity_rows: list[dict] = []

    last_row_by_symbol: dict[str, pd.Series] = {}
    grouped = signals_all.sort_values(["time", "confidence"], ascending=[True, False]).groupby("time")

    for ts, group in grouped:
        row_by_symbol = {row.symbol: row for row in group.itertuples(index=False)}

        for symbol, pos in list(positions.items()):
            row = row_by_symbol.get(symbol)
            if row is None:
                continue

            spec = specs[symbol]
            spread_price = float(row.spread_points) * spec.point
            close = float(row.close)
            high = float(row.high)
            low = float(row.low)
            signal = int(row.signal)
            exit_price = None
            reason = None

            if pos["side"] == 1:
                if low <= pos["sl"]:
                    exit_price, reason = pos["sl"], "SL"
                elif high >= pos["tp"]:
                    exit_price, reason = pos["tp"], "TP"
                elif signal == -1:
                    exit_price, reason = close - spread_price / 2.0, "REVERSE"
            else:
                if high >= pos["sl"]:
                    exit_price, reason = pos["sl"], "SL"
                elif low <= pos["tp"]:
                    exit_price, reason = pos["tp"], "TP"
                elif signal == 1:
                    exit_price, reason = close + spread_price / 2.0, "REVERSE"

            if exit_price is None:
                continue

            pnl = (
                (exit_price - pos["entry_price"])
                * pos["side"]
                * spec.trade_contract_size
                * pos["lot"]
            )
            balance += pnl
            peak_balance = max(peak_balance, balance)
            dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0.0
            max_drawdown = max(max_drawdown, dd)

            trades.append(
                {
                    "symbol": symbol,
                    "entry_time": pos["entry_time"],
                    "exit_time": ts,
                    "side": "BUY" if pos["side"] == 1 else "SELL",
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "sl": pos["sl"],
                    "tp": pos["tp"],
                    "lot": pos["lot"],
                    "reason": reason,
                    "pnl": pnl,
                    "balance_after": balance,
                    "probability": pos["probability"],
                }
            )
            del positions[symbol]

        slots = settings.max_active_pairs - len(positions)
        if slots > 0:
            candidates = []
            for row in group.itertuples(index=False):
                symbol = str(row.symbol)
                if symbol in positions:
                    continue
                signal = int(row.signal)
                if signal == 0:
                    continue
                if float(row.spread_points) > settings.max_spread_points:
                    continue
                candidates.append(row)

            candidates = sorted(candidates, key=lambda r: float(r.confidence), reverse=True)
            for row in candidates[:slots]:
                symbol = str(row.symbol)
                spec = specs[symbol]
                signal = int(row.signal)
                spread_price = float(row.spread_points) * spec.point
                close = float(row.close)
                pip_size = pip_size_from_digits(spec.digits, spec.point)
                lot = compute_lot_size(balance, settings, spec)

                if signal == 1:
                    entry_price = close + spread_price / 2.0
                    sl = entry_price - (settings.stop_loss_pips * pip_size)
                    tp = entry_price + (settings.take_profit_pips * pip_size)
                else:
                    entry_price = close - spread_price / 2.0
                    sl = entry_price + (settings.stop_loss_pips * pip_size)
                    tp = entry_price - (settings.take_profit_pips * pip_size)

                positions[symbol] = {
                    "entry_time": ts,
                    "entry_price": entry_price,
                    "sl": sl,
                    "tp": tp,
                    "lot": lot,
                    "side": signal,
                    "probability": float(row.probability),
                }

        for row in group.itertuples(index=False):
            last_row_by_symbol[str(row.symbol)] = row

        equity_rows.append(
            {
                "time": ts,
                "balance": balance,
                "active_positions": len(positions),
            }
        )

    for symbol, pos in list(positions.items()):
        row = last_row_by_symbol.get(symbol)
        if row is None:
            continue
        spec = specs[symbol]
        spread_price = float(row.spread_points) * spec.point
        close = float(row.close)
        exit_price = close - spread_price / 2.0 if pos["side"] == 1 else close + spread_price / 2.0
        pnl = (
            (exit_price - pos["entry_price"])
            * pos["side"]
            * spec.trade_contract_size
            * pos["lot"]
        )
        balance += pnl
        peak_balance = max(peak_balance, balance)
        dd = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0.0
        max_drawdown = max(max_drawdown, dd)
        trades.append(
            {
                "symbol": symbol,
                "entry_time": pos["entry_time"],
                "exit_time": row.time,
                "side": "BUY" if pos["side"] == 1 else "SELL",
                "entry_price": pos["entry_price"],
                "exit_price": exit_price,
                "sl": pos["sl"],
                "tp": pos["tp"],
                "lot": pos["lot"],
                "reason": "EOD",
                "pnl": pnl,
                "balance_after": balance,
                "probability": pos["probability"],
            }
        )

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)

    total_trades = len(trades_df)
    wins = int((trades_df["pnl"] > 0).sum()) if total_trades else 0
    losses = int((trades_df["pnl"] <= 0).sum()) if total_trades else 0
    gross_profit = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()) if total_trades else 0.0
    gross_loss = float(-trades_df.loc[trades_df["pnl"] <= 0, "pnl"].sum()) if total_trades else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
    net_profit = float(trades_df["pnl"].sum()) if total_trades else 0.0
    win_rate = (wins / total_trades) if total_trades else 0.0

    report = {
        "initial_balance": settings.initial_balance,
        "final_balance": balance,
        "net_profit": net_profit,
        "return_pct": ((balance / settings.initial_balance) - 1.0) * 100.0,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_drawdown * 100.0,
        "max_active_pairs": settings.max_active_pairs,
        "pairs_with_signals": int(signals_all["symbol"].nunique()) if not signals_all.empty else 0,
    }

    if total_trades > 0:
        symbol_stats = (
            trades_df.groupby("symbol")
            .agg(
                trades=("pnl", "count"),
                wins=("pnl", lambda x: int((x > 0).sum())),
                net_profit=("pnl", "sum"),
                avg_pnl=("pnl", "mean"),
            )
            .reset_index()
            .sort_values("net_profit", ascending=False)
        )
    else:
        symbol_stats = pd.DataFrame(columns=["symbol", "trades", "wins", "net_profit", "avg_pnl"])

    return report, trades_df, equity_df, symbol_stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest 3 bulan terakhir: scan semua pair tradable, entry BUY/SELL, max 3 pair aktif."
    )
    parser.add_argument(
        "--months",
        type=int,
        default=3,
        help="Periode backtest dalam bulan terakhir.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Batasi jumlah pair untuk test (0 = semua pair terbaca).",
    )
    parser.add_argument(
        "--pairs",
        type=str,
        default="",
        help="Daftar pair manual dipisah koma. Kosong = auto discover semua pair.",
    )
    parser.add_argument(
        "--risk-sizing",
        action="store_true",
        help="Gunakan risk-based lot sizing. Default pakai fixed lot agar lintas pair lebih stabil.",
    )
    parser.add_argument(
        "--fixed-lot",
        type=float,
        default=0.01,
        help="Lot tetap saat --risk-sizing tidak dipakai.",
    )
    args = parser.parse_args()

    ensure_output_dir()
    settings = load_settings()
    settings_bt = replace(
        settings,
        use_risk_position_sizing=args.risk_sizing,
        fixed_lot=args.fixed_lot,
    )
    artifacts = load_artifacts()

    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - relativedelta(months=args.months)

    explicit_pairs = [x.strip().upper() for x in args.pairs.split(",") if x.strip()]
    client = MT5Client(settings)
    pair_signals: list[PairSignals] = []
    skipped: list[dict] = []

    try:
        client.connect()
        if explicit_pairs:
            pairs = explicit_pairs
        else:
            pairs = discover_tradable_pairs(client, settings_bt)
        if args.max_pairs > 0:
            pairs = pairs[: args.max_pairs]

        print(
            f"Backtest portfolio dimulai. Pair ditemukan={len(pairs)} | periode={start_utc} s/d {now_utc}"
        )

        for idx, symbol in enumerate(pairs, start=1):
            try:
                print(f"[{idx}/{len(pairs)}] proses {symbol} ...")
                ps = _build_pair_signals(
                    client=client,
                    settings=settings_bt,
                    artifacts=artifacts,
                    symbol=symbol,
                    start_utc=start_utc,
                    end_utc=now_utc,
                )
                pair_signals.append(ps)
            except Exception as exc:
                skipped.append({"symbol": symbol, "reason": str(exc)})
    finally:
        client.shutdown()

    if not pair_signals:
        raise RuntimeError("Tidak ada pair yang valid untuk backtest.")

    all_signals = pd.concat([x.signals for x in pair_signals], ignore_index=True)
    specs = {x.symbol: x.spec for x in pair_signals}

    report, trades_df, equity_df, symbol_stats_df = run_portfolio_backtest(
        signals_all=all_signals,
        specs=specs,
        settings=settings_bt,
    )

    report["period_start_utc"] = str(start_utc)
    report["period_end_utc"] = str(now_utc)
    report["pairs_discovered"] = len(pairs)
    report["pairs_processed"] = len(pair_signals)
    report["pairs_skipped"] = len(skipped)
    report["lot_mode"] = "risk_sizing" if args.risk_sizing else "fixed_lot"
    report["fixed_lot"] = settings_bt.fixed_lot
    report["backtest_fast_tf"] = settings_bt.backtest_fast_timeframe
    report["backtest_slow_tf"] = settings_bt.backtest_slow_timeframe

    tag = f"{args.months}m"
    report_path = OUTPUT_DIR / f"portfolio_backtest_{tag}_report.json"
    trades_path = OUTPUT_DIR / f"portfolio_backtest_{tag}_trades.csv"
    equity_path = OUTPUT_DIR / f"portfolio_backtest_{tag}_equity.csv"
    symbols_path = OUTPUT_DIR / f"portfolio_backtest_{tag}_symbol_stats.csv"
    skipped_path = OUTPUT_DIR / f"portfolio_backtest_{tag}_skipped_pairs.csv"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    trades_df.to_csv(trades_path, index=False)
    equity_df.to_csv(equity_path, index=False)
    symbol_stats_df.to_csv(symbols_path, index=False)
    pd.DataFrame(skipped).to_csv(skipped_path, index=False)

    print("\nHasil backtest portfolio:")
    print(json.dumps(report, indent=2))
    print(f"Report: {report_path}")
    print(f"Trades: {trades_path}")
    print(f"Equity: {equity_path}")
    print(f"Per-symbol: {symbols_path}")
    print(f"Skipped pairs: {skipped_path}")


if __name__ == "__main__":
    main()
