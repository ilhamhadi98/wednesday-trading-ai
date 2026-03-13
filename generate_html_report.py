from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd


def _fmt_money(value: float) -> str:
    return f"{value:,.2f}"


def _fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def _build_streaks(results: pd.Series) -> tuple[int, int]:
    longest_win = 0
    longest_loss = 0
    cur_win = 0
    cur_loss = 0

    for value in results:
        if value > 0:
            cur_win += 1
            cur_loss = 0
            longest_win = max(longest_win, cur_win)
        else:
            cur_loss += 1
            cur_win = 0
            longest_loss = max(longest_loss, cur_loss)
    return longest_win, longest_loss


def _table_html(df: pd.DataFrame, index: bool = False, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    return df.to_html(
        index=index,
        border=0,
        classes="report-table",
        justify="left",
        escape=False,
    )


def generate_html(
    report: dict,
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    symbol_stats: pd.DataFrame,
    skipped_pairs: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    trades = trades.copy()
    equity = equity.copy()
    symbol_stats = symbol_stats.copy()
    skipped_pairs = skipped_pairs.copy()

    if not trades.empty:
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
        trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True, errors="coerce")
        trades = trades.sort_values("exit_time")
        trades["cum_pnl"] = trades["pnl"].cumsum()
        trades["result"] = trades["pnl"].apply(lambda x: "WIN" if x > 0 else "LOSS")
        trades["month"] = trades["exit_time"].dt.tz_convert("UTC").dt.strftime("%Y-%m")
    else:
        trades["cum_pnl"] = []
        trades["result"] = []
        trades["month"] = []

    if not equity.empty:
        equity["time"] = pd.to_datetime(equity["time"], utc=True, errors="coerce")
        equity = equity.sort_values("time")
        equity["peak_balance"] = equity["balance"].cummax()
        equity["drawdown_pct"] = (
            (equity["peak_balance"] - equity["balance"]) / equity["peak_balance"]
        ) * 100.0
    else:
        equity["drawdown_pct"] = []

    total_trades = int(len(trades))
    wins = int((trades["pnl"] > 0).sum()) if total_trades else 0
    losses = int((trades["pnl"] <= 0).sum()) if total_trades else 0
    avg_win = float(trades.loc[trades["pnl"] > 0, "pnl"].mean()) if wins else 0.0
    avg_loss = float(trades.loc[trades["pnl"] <= 0, "pnl"].mean()) if losses else 0.0
    expectancy = float(trades["pnl"].mean()) if total_trades else 0.0
    longest_win_streak, longest_loss_streak = _build_streaks(trades["pnl"]) if total_trades else (0, 0)

    reason_counts = (
        trades["reason"].value_counts().rename_axis("reason").reset_index(name="count")
        if total_trades
        else pd.DataFrame(columns=["reason", "count"])
    )
    side_counts = (
        trades["side"].value_counts().rename_axis("side").reset_index(name="count")
        if total_trades
        else pd.DataFrame(columns=["side", "count"])
    )
    monthly_pnl = (
        trades.groupby("month", as_index=False)["pnl"].sum().rename(columns={"pnl": "net_pnl"})
        if total_trades
        else pd.DataFrame(columns=["month", "net_pnl"])
    )

    top_wins = (
        trades.nlargest(12, "pnl")[
            ["symbol", "entry_time", "exit_time", "side", "reason", "pnl", "probability"]
        ]
        if total_trades
        else pd.DataFrame(columns=["symbol", "entry_time", "exit_time", "side", "reason", "pnl", "probability"])
    )
    top_losses = (
        trades.nsmallest(12, "pnl")[
            ["symbol", "entry_time", "exit_time", "side", "reason", "pnl", "probability"]
        ]
        if total_trades
        else pd.DataFrame(columns=["symbol", "entry_time", "exit_time", "side", "reason", "pnl", "probability"])
    )

    # Format output tables for readability.
    for tdf in [top_wins, top_losses]:
        if not tdf.empty:
            tdf["entry_time"] = tdf["entry_time"].dt.strftime("%Y-%m-%d %H:%M")
            tdf["exit_time"] = tdf["exit_time"].dt.strftime("%Y-%m-%d %H:%M")
            tdf["pnl"] = tdf["pnl"].map(lambda x: f"{x:,.2f}")
            tdf["probability"] = tdf["probability"].map(lambda x: f"{x:.4f}")

    display_trades = trades.copy()
    if not display_trades.empty:
        display_trades["entry_time"] = display_trades["entry_time"].dt.strftime("%Y-%m-%d %H:%M")
        display_trades["exit_time"] = display_trades["exit_time"].dt.strftime("%Y-%m-%d %H:%M")
        for col in ["entry_price", "exit_price", "sl", "tp", "lot", "pnl", "balance_after", "probability"]:
            if col in display_trades.columns:
                display_trades[col] = display_trades[col].map(lambda x: f"{x:,.5f}" if col not in {"pnl", "balance_after"} else f"{x:,.2f}")
        display_trades = display_trades[
            [
                "symbol",
                "entry_time",
                "exit_time",
                "side",
                "reason",
                "entry_price",
                "exit_price",
                "sl",
                "tp",
                "lot",
                "probability",
                "pnl",
                "balance_after",
            ]
        ]

    if not symbol_stats.empty:
        symbol_stats["net_profit"] = symbol_stats["net_profit"].map(lambda x: f"{x:,.2f}")
        symbol_stats["avg_pnl"] = symbol_stats["avg_pnl"].map(lambda x: f"{x:,.2f}")

    if not skipped_pairs.empty and "reason" in skipped_pairs.columns:
        skipped_pairs["reason"] = skipped_pairs["reason"].map(lambda x: html.escape(str(x)))

    report_meta = {
        "period_start_utc": report.get("period_start_utc", "-"),
        "period_end_utc": report.get("period_end_utc", "-"),
        "backtest_fast_tf": report.get("backtest_fast_tf", "-"),
        "backtest_slow_tf": report.get("backtest_slow_tf", "-"),
        "max_active_pairs": report.get("max_active_pairs", "-"),
        "pairs_discovered": report.get("pairs_discovered", "-"),
        "pairs_processed": report.get("pairs_processed", "-"),
        "pairs_skipped": report.get("pairs_skipped", "-"),
        "lot_mode": report.get("lot_mode", "-"),
        "fixed_lot": report.get("fixed_lot", "-"),
    }

    js_payload = {
        "equity_time": equity["time"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist() if not equity.empty else [],
        "equity_balance": equity["balance"].tolist() if "balance" in equity else [],
        "equity_drawdown": equity["drawdown_pct"].tolist() if "drawdown_pct" in equity else [],
        "trade_index": list(range(1, total_trades + 1)),
        "trade_pnl": trades["pnl"].tolist() if "pnl" in trades else [],
        "trade_cum_pnl": trades["cum_pnl"].tolist() if "cum_pnl" in trades else [],
        "win_loss_labels": ["WIN", "LOSS"],
        "win_loss_values": [wins, losses],
        "reason_labels": reason_counts["reason"].astype(str).tolist() if not reason_counts.empty else [],
        "reason_values": reason_counts["count"].tolist() if not reason_counts.empty else [],
        "monthly_labels": monthly_pnl["month"].astype(str).tolist() if not monthly_pnl.empty else [],
        "monthly_values": monthly_pnl["net_pnl"].tolist() if not monthly_pnl.empty else [],
        "symbol_labels": symbol_stats["symbol"].astype(str).tolist() if not symbol_stats.empty else [],
        "symbol_values": [
            float(str(v).replace(",", "")) if isinstance(v, str) else float(v)
            for v in (symbol_stats["net_profit"].tolist() if "net_profit" in symbol_stats else [])
        ],
    }

    summary_cards = [
        ("Return", _fmt_pct(float(report.get("return_pct", 0.0)))),
        ("Net Profit", _fmt_money(float(report.get("net_profit", 0.0)))),
        ("Profit Factor", f"{float(report.get('profit_factor', 0.0)):.2f}"),
        ("Max Drawdown", _fmt_pct(float(report.get("max_drawdown_pct", 0.0)))),
        ("Win Rate", _fmt_pct(float(report.get("win_rate", 0.0) * 100.0))),
        ("Total Trades", str(total_trades)),
        ("Average Win", _fmt_money(avg_win)),
        ("Average Loss", _fmt_money(avg_loss)),
        ("Expectancy/Trade", _fmt_money(expectancy)),
        ("Longest Win Streak", str(longest_win_streak)),
        ("Longest Loss Streak", str(longest_loss_streak)),
        ("Final Balance", _fmt_money(float(report.get("final_balance", 0.0)))),
    ]

    cards_html = "\n".join(
        f'<div class="card"><div class="label">{html.escape(k)}</div><div class="value">{html.escape(v)}</div></div>'
        for k, v in summary_cards
    )

    meta_html = "\n".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in report_meta.items()
    )

    html_output = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #111827;
      --panel2: #1f2937;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --accent: #22c55e;
      --danger: #ef4444;
      --warn: #f59e0b;
      --border: #374151;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: linear-gradient(160deg, #0b1220, #0f172a 40%, #111827);
      color: var(--text);
      font-family: "Segoe UI", Arial, sans-serif;
      line-height: 1.4;
    }}
    .container {{
      width: min(1500px, 96vw);
      margin: 24px auto 32px;
    }}
    .header {{
      background: rgba(17, 24, 39, 0.85);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px 20px;
      margin-bottom: 16px;
      backdrop-filter: blur(6px);
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 24px;
      letter-spacing: 0.2px;
    }}
    .sub {{
      color: var(--muted);
      font-size: 14px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}
    .card {{
      background: rgba(17, 24, 39, 0.9);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      min-height: 86px;
    }}
    .label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .6px;
      margin-bottom: 8px;
    }}
    .value {{
      font-size: 22px;
      font-weight: 700;
    }}
    .grid2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .panel {{
      background: rgba(17, 24, 39, 0.9);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px;
    }}
    .panel h3 {{
      margin: 4px 8px 8px;
      font-size: 15px;
      color: #d1d5db;
    }}
    .chart {{
      width: 100%;
      height: 360px;
    }}
    .chart.small {{
      height: 320px;
    }}
    .tables {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .report-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    .report-table th, .report-table td {{
      border: 1px solid var(--border);
      padding: 6px 8px;
      text-align: left;
      vertical-align: top;
      word-break: break-word;
    }}
    .report-table thead th {{
      background: var(--panel2);
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    .scroll {{
      max-height: 420px;
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 10px;
    }}
    .footnote {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
    }}
    @media (max-width: 1080px) {{
      .grid2, .tables {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>{html.escape(title)}</h1>
      <div class="sub">
        Period: {html.escape(str(report_meta["period_start_utc"]))} to {html.escape(str(report_meta["period_end_utc"]))}
      </div>
      <div class="sub">
        Backtest TF: {html.escape(str(report_meta["backtest_fast_tf"]))}/{html.escape(str(report_meta["backtest_slow_tf"]))}
        | Max Active Pairs: {html.escape(str(report_meta["max_active_pairs"]))}
        | Lot Mode: {html.escape(str(report_meta["lot_mode"]))}
      </div>
    </div>

    <div class="cards">{cards_html}</div>

    <div class="grid2">
      <div class="panel">
        <h3>Equity Curve and Drawdown</h3>
        <div id="equityChart" class="chart"></div>
      </div>
      <div class="panel">
        <h3>Cumulative PnL and Trade-by-Trade PnL</h3>
        <div id="pnlChart" class="chart"></div>
      </div>
    </div>

    <div class="grid2">
      <div class="panel">
        <h3>Win/Loss</h3>
        <div id="winLossChart" class="chart small"></div>
      </div>
      <div class="panel">
        <h3>Reason Breakdown</h3>
        <div id="reasonChart" class="chart small"></div>
      </div>
    </div>

    <div class="grid2">
      <div class="panel">
        <h3>Monthly Net PnL</h3>
        <div id="monthlyChart" class="chart small"></div>
      </div>
      <div class="panel">
        <h3>Net Profit by Symbol</h3>
        <div id="symbolChart" class="chart small"></div>
      </div>
    </div>

    <div class="tables">
      <div class="panel">
        <h3>Backtest Metadata</h3>
        <div class="scroll">{_table_html(pd.DataFrame(report_meta.items(), columns=["Key", "Value"]))}</div>
      </div>
      <div class="panel">
        <h3>Signal Side Distribution</h3>
        <div class="scroll">{_table_html(side_counts, index=False)}</div>
      </div>
    </div>

    <div class="tables">
      <div class="panel">
        <h3>Top Winning Trades</h3>
        <div class="scroll">{_table_html(top_wins, index=False)}</div>
      </div>
      <div class="panel">
        <h3>Top Losing Trades</h3>
        <div class="scroll">{_table_html(top_losses, index=False)}</div>
      </div>
    </div>

    <div class="panel" style="margin-bottom: 12px;">
      <h3>Per-Symbol Performance</h3>
      <div class="scroll">{_table_html(symbol_stats, index=False)}</div>
    </div>

    <div class="panel" style="margin-bottom: 12px;">
      <h3>Skipped Pairs</h3>
      <div class="scroll">{_table_html(skipped_pairs, index=False)}</div>
    </div>

    <div class="panel">
      <h3>Full Trade History</h3>
      <div class="scroll">{_table_html(display_trades, index=False)}</div>
      <div class="footnote">Tip: use browser search (Ctrl+F) to find symbol/time quickly.</div>
    </div>
  </div>

  <script>
    const payload = {json.dumps(js_payload)};

    const baseLayout = {{
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      font: {{color: '#e5e7eb'}},
      margin: {{l: 50, r: 20, t: 20, b: 45}},
      xaxis: {{gridcolor: '#374151'}},
      yaxis: {{gridcolor: '#374151'}},
      legend: {{orientation: 'h'}},
    }};

    Plotly.newPlot('equityChart', [
      {{
        x: payload.equity_time,
        y: payload.equity_balance,
        type: 'scatter',
        mode: 'lines',
        name: 'Balance',
        line: {{color: '#22c55e', width: 2}},
      }},
      {{
        x: payload.equity_time,
        y: payload.equity_drawdown,
        type: 'scatter',
        mode: 'lines',
        name: 'Drawdown %',
        yaxis: 'y2',
        line: {{color: '#ef4444', width: 1.6, dash: 'dot'}},
      }},
    ], {{
      ...baseLayout,
      yaxis: {{title: 'Balance', gridcolor: '#374151'}},
      yaxis2: {{
        title: 'Drawdown %',
        overlaying: 'y',
        side: 'right',
        gridcolor: 'rgba(0,0,0,0)',
      }},
    }}, {{displaylogo: false}});

    Plotly.newPlot('pnlChart', [
      {{
        x: payload.trade_index,
        y: payload.trade_pnl,
        type: 'bar',
        name: 'Trade PnL',
        marker: {{
          color: payload.trade_pnl.map(v => v >= 0 ? '#22c55e' : '#ef4444')
        }},
        opacity: 0.7,
      }},
      {{
        x: payload.trade_index,
        y: payload.trade_cum_pnl,
        type: 'scatter',
        mode: 'lines',
        name: 'Cumulative PnL',
        line: {{color: '#60a5fa', width: 2}},
        yaxis: 'y2',
      }},
    ], {{
      ...baseLayout,
      yaxis: {{title: 'Trade PnL', gridcolor: '#374151'}},
      yaxis2: {{
        title: 'Cumulative PnL',
        overlaying: 'y',
        side: 'right',
        gridcolor: 'rgba(0,0,0,0)',
      }},
      xaxis: {{title: 'Trade #', gridcolor: '#374151'}},
    }}, {{displaylogo: false}});

    Plotly.newPlot('winLossChart', [{{
      labels: payload.win_loss_labels,
      values: payload.win_loss_values,
      type: 'pie',
      hole: 0.45,
      marker: {{colors: ['#22c55e', '#ef4444']}},
      textinfo: 'label+percent+value',
    }}], {{
      ...baseLayout,
      margin: {{l: 20, r: 20, t: 20, b: 20}},
    }}, {{displaylogo: false}});

    Plotly.newPlot('reasonChart', [{{
      x: payload.reason_labels,
      y: payload.reason_values,
      type: 'bar',
      marker: {{color: '#f59e0b'}},
    }}], {{
      ...baseLayout,
      xaxis: {{title: 'Reason', gridcolor: '#374151'}},
      yaxis: {{title: 'Count', gridcolor: '#374151'}},
    }}, {{displaylogo: false}});

    Plotly.newPlot('monthlyChart', [{{
      x: payload.monthly_labels,
      y: payload.monthly_values,
      type: 'bar',
      marker: {{
        color: payload.monthly_values.map(v => v >= 0 ? '#22c55e' : '#ef4444')
      }},
    }}], {{
      ...baseLayout,
      xaxis: {{title: 'Month', gridcolor: '#374151'}},
      yaxis: {{title: 'Net PnL', gridcolor: '#374151'}},
    }}, {{displaylogo: false}});

    Plotly.newPlot('symbolChart', [{{
      x: payload.symbol_labels,
      y: payload.symbol_values,
      type: 'bar',
      marker: {{
        color: payload.symbol_values.map(v => v >= 0 ? '#22c55e' : '#ef4444')
      }},
    }}], {{
      ...baseLayout,
      xaxis: {{title: 'Symbol', gridcolor: '#374151'}},
      yaxis: {{title: 'Net Profit', gridcolor: '#374151'}},
    }}, {{displaylogo: false}});
  </script>
</body>
</html>
"""

    output_path.write_text(html_output, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate HTML performance report from backtest output files."
    )
    parser.add_argument(
        "--report-json",
        default="outputs/portfolio_backtest_3m_report.json",
        help="Path to report json file.",
    )
    parser.add_argument(
        "--trades-csv",
        default="outputs/portfolio_backtest_3m_trades.csv",
        help="Path to trades csv file.",
    )
    parser.add_argument(
        "--equity-csv",
        default="outputs/portfolio_backtest_3m_equity.csv",
        help="Path to equity csv file.",
    )
    parser.add_argument(
        "--symbol-stats-csv",
        default="outputs/portfolio_backtest_3m_symbol_stats.csv",
        help="Path to symbol stats csv file.",
    )
    parser.add_argument(
        "--skipped-csv",
        default="outputs/portfolio_backtest_3m_skipped_pairs.csv",
        help="Path to skipped pairs csv file.",
    )
    parser.add_argument(
        "--output",
        default="outputs/portfolio_backtest_3m_report.html",
        help="Output html path.",
    )
    parser.add_argument(
        "--title",
        default="Trading Backtest Report",
        help="Report title displayed in html.",
    )
    args = parser.parse_args()

    report_path = Path(args.report_json)
    trades_path = Path(args.trades_csv)
    equity_path = Path(args.equity_csv)
    symbol_stats_path = Path(args.symbol_stats_csv)
    skipped_path = Path(args.skipped_csv)
    output_path = Path(args.output)

    if not report_path.exists():
        raise FileNotFoundError(f"Report json not found: {report_path}")
    if not trades_path.exists():
        raise FileNotFoundError(f"Trades csv not found: {trades_path}")
    if not equity_path.exists():
        raise FileNotFoundError(f"Equity csv not found: {equity_path}")
    if not symbol_stats_path.exists():
        raise FileNotFoundError(f"Symbol stats csv not found: {symbol_stats_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    trades = pd.read_csv(trades_path)
    equity = pd.read_csv(equity_path)
    symbol_stats = pd.read_csv(symbol_stats_path)
    skipped_pairs = pd.read_csv(skipped_path) if skipped_path.exists() else pd.DataFrame()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_html(
        report=report,
        trades=trades,
        equity=equity,
        symbol_stats=symbol_stats,
        skipped_pairs=skipped_pairs,
        output_path=output_path,
        title=args.title,
    )
    print(f"HTML report generated: {output_path.resolve()}")


if __name__ == "__main__":
    main()
