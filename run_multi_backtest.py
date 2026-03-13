from __future__ import annotations

import argparse
import json
from dataclasses import replace

import pandas as pd

from trading_bot.backtest import run_backtest
from trading_bot.config import OUTPUT_DIR, ensure_output_dir, load_settings
from trading_bot.data_pipeline import make_sequences
from trading_bot.market_scan import resolve_symbols
from trading_bot.modeling import predict_proba, train_model
from trading_bot.mt5_client import MT5Client
from trading_bot.workflows import fetch_feature_frame, split_train_val


def backtest_one_symbol(
    client: MT5Client,
    symbol: str,
    settings,
    train_ratio: float,
) -> dict:
    frame, feature_cols = fetch_feature_frame(client, settings, symbol=symbol)
    ds, _ = make_sequences(
        frame=frame,
        feature_cols=feature_cols,
        lookback=settings.lookback,
        scaler=None,
        fit_scaler=True,
    )
    if len(ds.x) < 500:
        raise RuntimeError("sample sequence terlalu sedikit")

    split = split_train_val(ds.x, ds.y, train_ratio=train_ratio)
    if len(split.x_val) == 0:
        raise RuntimeError("validation kosong")

    model, _ = train_model(
        x_train=split.x_train,
        y_train=split.y_train,
        x_val=split.x_val,
        y_val=split.y_val,
        settings=settings,
    )
    probs = predict_proba(model, split.x_val)
    ts = ds.timestamps[len(split.x_train) :]
    spec = client.symbol_spec(symbol)
    result = run_backtest(frame, ts, probs, settings, spec)
    row = {"symbol": symbol}
    row.update(result.report)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch backtest lintas simbol untuk memilih market paling prospektif."
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Daftar simbol dipisah koma. Kosong = pakai discovery mode.",
    )
    parser.add_argument(
        "--max-symbols",
        type=int,
        default=12,
        help="Batasi jumlah simbol agar runtime terkontrol.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Epoch training per simbol saat batch backtest.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Rasio train vs validasi/backtest (0.5 - 0.9 disarankan).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Jumlah hasil teratas yang ditampilkan.",
    )
    args = parser.parse_args()

    ensure_output_dir()
    settings = load_settings()
    settings_bt = replace(settings, epochs=args.epochs)

    explicit_symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    client = MT5Client(settings)
    rows = []

    try:
        client.connect()
        symbols = resolve_symbols(
            client=client,
            settings=settings,
            explicit_symbols=explicit_symbols if explicit_symbols else None,
        )
        symbols = symbols[: args.max_symbols]

        print(f"Batch backtest dimulai. Total simbol: {len(symbols)}")
        for idx, symbol in enumerate(symbols, start=1):
            try:
                print(f"[{idx}/{len(symbols)}] backtest {symbol} ...")
                row = backtest_one_symbol(
                    client=client,
                    symbol=symbol,
                    settings=settings_bt,
                    train_ratio=args.train_ratio,
                )
                row["status"] = "OK"
                row["reason"] = ""
                rows.append(row)
            except Exception as exc:
                rows.append(
                    {
                        "symbol": symbol,
                        "status": "ERROR",
                        "reason": str(exc),
                    }
                )
            
            # Save progressively
            df = pd.DataFrame(rows)
            csv_path = OUTPUT_DIR / "multi_backtest_results.csv"
            json_path = OUTPUT_DIR / "multi_backtest_results.json"
            df.to_csv(csv_path, index=False)
            df.to_json(json_path, orient="records", indent=2)

            summary_path = OUTPUT_DIR / "multi_backtest_summary.json"
            summary = {
                "total_symbols": len(symbols),
                "ok_symbols": int((df["status"] == "OK").sum()),
                "error_symbols": int((df["status"] != "OK").sum()),
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
    finally:
        client.shutdown()

    df = pd.DataFrame(rows)
    if df.empty:
        print("Tidak ada hasil backtest.")
        return

    csv_path = OUTPUT_DIR / "multi_backtest_results.csv"
    json_path = OUTPUT_DIR / "multi_backtest_results.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    df_ok = df[df["status"] == "OK"].copy()
    if not df_ok.empty:
        ranked = df_ok.sort_values(
            ["return_pct", "profit_factor", "max_drawdown_pct"],
            ascending=[False, False, True],
        ).head(args.top)
        print()
        print("Top hasil batch backtest:")
        cols = [
            "symbol",
            "return_pct",
            "profit_factor",
            "max_drawdown_pct",
            "total_trades",
            "win_rate",
            "net_profit",
        ]
        print(ranked[cols].to_string(index=False))
    else:
        print("Semua simbol gagal dibacktest.")

    print()
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")

    summary_path = OUTPUT_DIR / "multi_backtest_summary.json"
    summary = {
        "total_symbols": int(len(df)),
        "ok_symbols": int((df["status"] == "OK").sum()),
        "error_symbols": int((df["status"] != "OK").sum()),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
