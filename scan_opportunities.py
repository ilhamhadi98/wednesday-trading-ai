from __future__ import annotations

import argparse
import json

from trading_bot.config import OUTPUT_DIR, ensure_output_dir, load_settings
from trading_bot.market_scan import scan_opportunities
from trading_bot.modeling import load_artifacts
from trading_bot.mt5_client import MT5Client


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan peluang trading lintas simbol MT5 (H1+H4)."
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="",
        help="Daftar simbol dipisah koma. Kosong = pakai mode discovery di .env",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Jumlah peluang teratas yang ditampilkan.",
    )
    parser.add_argument(
        "--include-hold",
        action="store_true",
        help="Tampilkan juga sinyal HOLD pada output terminal.",
    )
    parser.add_argument(
        "--all-symbols",
        action="store_true",
        help="Jika aktif, scan semua simbol (bukan pair saja). Default pair-only.",
    )
    args = parser.parse_args()

    settings = load_settings()
    ensure_output_dir()
    artifacts = load_artifacts()
    explicit_symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    client = MT5Client(settings)
    try:
        client.connect()
        df = scan_opportunities(
            client=client,
            settings=settings,
            artifacts=artifacts,
            symbols=explicit_symbols if explicit_symbols else None,
            pair_only=not args.all_symbols,
        )
    finally:
        client.shutdown()

    if df.empty:
        print("Tidak ada simbol yang bisa dianalisa.")
        return

    csv_path = OUTPUT_DIR / "market_opportunities.csv"
    json_path = OUTPUT_DIR / "market_opportunities.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2, date_format="iso")

    visible = df[df["status"] == "OK"].copy()
    if not args.include_hold:
        visible = visible[visible["signal"] != "HOLD"]
    visible = visible.head(args.top)

    print(f"Total simbol dianalisa: {len(df)}")
    print(f"Sinyal aktif (BUY/SELL) siap dipertimbangkan: {(df['signal'].isin(['BUY','SELL']) & (df['status'] == 'OK')).sum()}")
    print()
    print("Top peluang:")
    if visible.empty:
        print("Tidak ada sinyal BUY/SELL saat ini (hanya HOLD atau terfilter).")
    else:
        cols = [
            "symbol",
            "signal",
            "probability",
            "confidence",
            "spread_points",
            "atr_pips",
            "bar_time",
        ]
        print(visible[cols].to_string(index=False))

    print()
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")

    summary = {
        "total_symbols": int(len(df)),
        "ok_symbols": int((df["status"] == "OK").sum()),
        "buy_sell_signals": int(
            ((df["status"] == "OK") & (df["signal"].isin(["BUY", "SELL"]))).sum()
        ),
        "top_signals": (
            visible[
                [
                    "symbol",
                    "signal",
                    "probability",
                    "confidence",
                    "spread_points",
                    "atr_pips",
                    "bar_time",
                ]
            ].to_dict(orient="records")
            if not visible.empty
            else []
        ),
    }
    summary_path = OUTPUT_DIR / "market_scan_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
