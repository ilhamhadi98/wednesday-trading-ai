import argparse
import json
import pandas as pd
import time

from trading_bot.config import OUTPUT_DIR, MULTI_BACKTEST_TRADES_PATH, ensure_output_dir, load_settings
from trading_bot.mt5_client import MT5Client
from trading_bot.workflows import fetch_feature_frame
from trading_bot.backtest import run_backtest

from trading_bot.agents.screener import MarketScreener
from trading_bot.agents.strategist import StrategistAI
from trading_bot.agents.risk_manager import PortfolioRiskManager

def _bars_for_months(months: int, timeframe: str) -> int:
    minutes_per_tf = {
        "M1": 1, "M5": 5, "M15": 15, "M30": 30,
        "H1": 60, "H4": 240, "D1": 1440,
    }
    tf_min = minutes_per_tf.get(timeframe.upper(), 60)
    bars = int((22 * 60 / tf_min) * (5 / 7) * 30.44 * months)
    return int(bars * 1.20) + 500

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Trading System Orchestrator")
    parser.add_argument("--symbols", type=str, default="EURUSD,GBPUSD,AUDUSD,USDJPY,XAUUSD", help="Pool of symbols to screen")
    parser.add_argument("--top_n", type=int, default=5, help="Number of pairs the Screener should select")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--months", type=int, default=None, help="Number of months of historical data to fetch")
    args = parser.parse_args()
    
    ensure_output_dir()
    settings = load_settings()
    client = MT5Client(settings)
    
    try:
        client.connect()
        pool = [s.strip() for s in args.symbols.split(",") if s.strip()]
        
        # 1. Screener Agent
        print(f"\n[ORCHESTRATOR] 🚀 Waking up Screener Agent. Scanning {len(pool)} symbols...")
        screener = MarketScreener(client, settings)
        active_pairs = screener.screen_symbols(pool, top_n=args.top_n)
        
        if not active_pairs:
            print("[ORCHESTRATOR] ⚠️ Screener found no active pairs. Sleeping.")
            return
            
        print("\n[SCREENER] ✅ Selected Active Pairs:")
        for rank, p in enumerate(active_pairs, 1):
            print(f"  {rank}. {p['symbol']} (Score: {p['score']} | ADX: {p['adx']} | Semi-ATR: {p['atr_pct']}%)")
            
        selected_symbols = [p['symbol'] for p in active_pairs]
        
        # 2. Initialize Strategist and Risk Manager
        print("\n[ORCHESTRATOR] 🧠 Initializing Strategist (Transfer Learning Hub) and Risk Manager (Shield)...")
        strategist = StrategistAI(settings)
        risk_manager = PortfolioRiskManager(settings)
        
        results_rows = []
        all_trades = []
        
        # 3. Execution Loop
        start_time = time.time()
        for i, symbol in enumerate(selected_symbols, 1):
            print(f"\n=======================================================")
            print(f"[ORCHESTRATOR] 🔄 [{i}/{len(selected_symbols)}] Handing over {symbol} to Strategist & Risk Manager...")
            print(f"=======================================================")
            try:
                # Calculate bar counts if months are specified
                bars_fast = _bars_for_months(args.months, settings.backtest_fast_timeframe) if args.months else None
                bars_slow = _bars_for_months(args.months, settings.backtest_slow_timeframe) if args.months else None
                
                if args.months:
                    print(f"[*] Fetching ~{args.months} months of data ({bars_fast} fast bars, {bars_slow} slow bars) for {symbol}")
                    
                # Fetch Data
                frame, feature_cols = fetch_feature_frame(
                    client, settings, symbol=symbol,
                    bars_fast=bars_fast, bars_slow=bars_slow
                )
                
                # Strategist predicts and trains (Transfer Learning)
                probs, timestamps = strategist.analyze_and_predict(symbol, frame, feature_cols, args.train_ratio)
                
                if len(probs) == 0:
                    results_rows.append({"symbol": symbol, "status": "ERROR", "reason": "Not enough data/predictions"})
                    continue
                    
                # Backtest execution passing Risk Manager checks
                spec = client.symbol_spec(symbol)
                result = run_backtest(frame, timestamps, probs, settings, spec, risk_manager=risk_manager)
                
                row = {"symbol": symbol}
                row.update(result.report)
                row["status"] = "OK"
                row["reason"] = ""
                results_rows.append(row)
                
                print(f"[RISK MANAGER] 🛡️ {symbol} Report -> PNL: ${result.report.get('net_profit', 0):.2f} | WinRate: {result.report.get('win_rate', 0)*100:.1f}%")
                
                # Save progressively so dashboard updates live
                df = pd.DataFrame(results_rows)
                csv_path = OUTPUT_DIR / "multi_backtest_results.csv"
                json_path = OUTPUT_DIR / "multi_backtest_results.json"
                df.to_csv(csv_path, index=False)
                df.to_json(json_path, orient="records", indent=2)
                
                # Update and save all trades for frontend detail view
                if not result.trades.empty:
                    symbol_trades = result.trades.copy()
                    symbol_trades.insert(0, "symbol", symbol)
                    all_trades.append(symbol_trades)
                    pd.concat(all_trades).to_csv(MULTI_BACKTEST_TRADES_PATH, index=False)
                    pd.concat(all_trades).to_json(OUTPUT_DIR / "multi_backtest_trades.json", orient="records", indent=2)

                summary_path = OUTPUT_DIR / "multi_backtest_summary.json"
                summary = {
                    "total_symbols": len(selected_symbols),
                    "ok_symbols": int((df["status"] == "OK").sum()),
                    "error_symbols": int((df["status"] != "OK").sum()),
                }
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
                    
            except Exception as exc:
                print(f"[ORCHESTRATOR] ❌ Error on {symbol}: {exc}")
                results_rows.append({"symbol": symbol, "status": "ERROR", "reason": str(exc)})
                
        # 4. Save Final Report
        elapsed = time.time() - start_time
        print(f"\n[ORCHESTRATOR] 🎉 Multi-Agent cycle complete in {elapsed:.1f}s! Data handed to Next.js Dashboard.")
        
    finally:
        client.shutdown()

if __name__ == "__main__":
    main()
