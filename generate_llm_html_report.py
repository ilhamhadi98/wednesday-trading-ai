"""
generate_llm_html_report.py  ─  Enhanced HTML Dashboard for LLM-Integrated Backtest

Generates a rich single-file HTML report with:
  • Summary KPI cards (Return, WinRate, PnL, Drawdown, etc.)
  • Equity curve + drawdown chart
  • Trade PnL waterfall + cumulative
  • LLM accuracy panel (agreement rate, filtered trades)
  • LLM sentiment vs return correlation chart
  • Per-symbol performance bar chart
  • Monthly PnL breakdown
  • Win/Loss and Exit reason donut charts
  • Full trade table with LLM decision columns highlighted
  • Expandable trading plan detail per trade

Usage (standalone):
    python generate_llm_html_report.py
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path("outputs")
TAG        = "llm_backtest_3m"


def _fmt_money(v: float) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}{v:,.2f}"


def _fmt_pct(v: float) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.2f}%"


def _escape(v) -> str:
    return html.escape(str(v))


def _color(v: float) -> str:
    return "#22c55e" if v >= 0 else "#ef4444"


def _build_plans_html(plans: list[dict]) -> str:
    if not plans:
        return "<p style='color:#9ca3af;padding:12px'>No trading plans recorded.</p>"
    rows = []
    for p in plans[:200]:  # cap at 200 to keep file size reasonable
        outcome = p.get("outcome") or {}
        pnl     = outcome.get("pnl", 0)
        col     = _color(pnl)
        dec_obj = p.get("llm_decision", {})
        decision = dec_obj.get("decision", "") if isinstance(dec_obj, dict) else str(dec_obj)
        rationale = dec_obj.get("entry_rationale", "") if isinstance(dec_obj, dict) else ""
        risk_w    = dec_obj.get("risk_warning", "")  if isinstance(dec_obj, dict) else ""
        news     = p.get("llm_news", {}) or {}
        tech     = p.get("llm_tech", {}) or {}
        exec_    = p.get("execution", {}) or {}
        rows.append(f"""
  <details class="plan-row">
    <summary>
      <span class="plan-tid">#{p.get('trade_id','?')}</span>
      <span class="plan-sym">{_escape(p.get('symbol',''))}</span>
      <span class="plan-time">{_escape(str(p.get('entry_time',''))[:16])}</span>
      <span class="plan-lstm">{_escape(p.get('lstm_signal',''))}&nbsp;{_escape(str(p.get('lstm_probability',''))[:5])}</span>
      <span class="plan-llm" style="color:{('#22c55e' if decision=='BUY' else '#ef4444' if decision=='SELL' else '#f59e0b')}">{_escape(decision)}</span>
      <span class="plan-pnl" style="color:{col}">{_fmt_money(pnl)}</span>
      <span class="plan-reason">{_escape(outcome.get('reason',''))}</span>
    </summary>
    <div class="plan-body">
      <div class="plan-grid">
        <div><strong>🟢 Entry</strong><br>
          Price: {_escape(str(exec_.get('entry_price','')))} &nbsp;|&nbsp;
          SL: {_escape(str(exec_.get('sl','')))} ({_escape(str(exec_.get('sl_pips','')))} pip) &nbsp;|&nbsp;
          TP: {_escape(str(exec_.get('tp','')))} ({_escape(str(exec_.get('tp_pips','')))} pip) &nbsp;|&nbsp;
          Lot: {_escape(str(exec_.get('lot','')))}
        </div>
        <div><strong>📰 Sentiment (mistral:7b)</strong><br>
          {_escape(news.get('sentiment',''))} | conf={_escape(str(news.get('confidence',''))[:4])}<br>
          <em>{_escape(str(news.get('reasoning',''))[:120])}</em>
        </div>
        <div><strong>📊 Technical (qwen2.5:14b)</strong><br>
          View: {_escape(tech.get('view',''))} | conf={_escape(str(tech.get('confidence',''))[:4])}<br>
          <em>{_escape(str(tech.get('reasoning',''))[:120])}</em>
        </div>
        <div><strong>🎯 Decision (deepseek-r1:14b)</strong><br>
          <span style="color:{('#22c55e' if decision=='BUY' else '#ef4444' if decision=='SELL' else '#f59e0b')};font-weight:700">{_escape(decision)}</span><br>
          {_escape(str(rationale)[:200])}<br>
          {'⚠️ ' + _escape(str(risk_w)[:150]) if risk_w else ''}
        </div>
        <div><strong>📈 Outcome</strong><br>
          Exit: {_escape(str(outcome.get('exit_price','')))} | Reason: {_escape(outcome.get('reason',''))} |
          <span style="color:{col};font-weight:700">PnL: {_fmt_money(pnl)}</span>
        </div>
      </div>
    </div>
  </details>""")
    return "\n".join(rows)


def generate_llm_html(
    report: dict,
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    symbol_stats: pd.DataFrame,
    plans: list,
    output_path: Path,
) -> None:
    trades = trades.copy()
    equity = equity.copy()

    real     = trades[trades["reason"] != "LLM_FILTERED"].copy()
    filtered = trades[trades["reason"] == "LLM_FILTERED"].copy()

    # Datetime parsing
    for df_ in (real, filtered):
        for col in ("entry_time", "exit_time"):
            if col in df_.columns:
                df_[col] = pd.to_datetime(df_[col], utc=True, errors="coerce")
    if not equity.empty and "time" in equity.columns:
        equity["time"] = pd.to_datetime(equity["time"], utc=True, errors="coerce")
        equity = equity.sort_values("time")
        equity["peak"] = equity["balance"].cummax()
        equity["drawdown_pct"] = ((equity["peak"] - equity["balance"]) / equity["peak"]) * 100

    if not real.empty:
        real = real.sort_values("exit_time")
        real["cum_pnl"] = real["pnl"].cumsum()
        real["result"]  = real["pnl"].apply(lambda x: "WIN" if x > 0 else "LOSS")
        real["month"]   = real["exit_time"].dt.strftime("%Y-%m")

    # KPI metrics
    total    = len(real)
    wins     = int((real["pnl"] > 0).sum()) if total else 0
    losses   = total - wins
    net      = float(real["pnl"].sum()) if total else 0.0
    wr       = wins / total if total else 0.0
    gp       = float(real.loc[real["pnl"] > 0, "pnl"].sum()) if total else 0.0
    gl       = float(-real.loc[real["pnl"] <= 0, "pnl"].sum()) if total else 0.0
    pf       = gp / gl if gl > 0 else 0.0
    init_bal = float(report.get("initial_balance", 100))
    fin_bal  = float(real["balance_after"].iloc[-1]) if total else init_bal
    ret_pct  = (fin_bal / init_bal - 1) * 100
    avg_w    = float(real.loc[real["pnl"] > 0, "pnl"].mean()) if wins else 0.0
    avg_l    = float(real.loc[real["pnl"] <= 0, "pnl"].mean()) if losses else 0.0
    max_dd   = float(equity["drawdown_pct"].max()) if not equity.empty and "drawdown_pct" in equity.columns else 0.0
    llm_filtered_n = len(filtered)
    filter_rate    = llm_filtered_n / max(total + llm_filtered_n, 1) * 100
    llm_avg_conf   = float(real["llm_confidence"].mean() * 100) if total and "llm_confidence" in real else 0.0

    # Sentiment analysis
    sent_pnl = {"BULLISH": [], "BEARISH": [], "NEUTRAL": []}
    if total and "llm_sentiment" in real.columns:
        for s, g in real.groupby("llm_sentiment"):
            if s in sent_pnl:
                sent_pnl[s] = g["pnl"].tolist()

    # Chart data
    eq_times   = equity["time"].dt.strftime("%Y-%m-%d %H:%M").tolist() if not equity.empty else []
    eq_balance = equity["balance"].tolist() if not equity.empty else []
    eq_dd      = equity["drawdown_pct"].tolist() if "drawdown_pct" in equity.columns else []

    trade_pnl  = real["pnl"].tolist() if total else []
    trade_cum  = real["cum_pnl"].tolist() if total else []

    reason_vc  = real["reason"].value_counts() if total else pd.Series(dtype=int)
    monthly    = real.groupby("month")["pnl"].sum() if total else pd.Series(dtype=float)

    sym_labels = symbol_stats["symbol"].tolist() if not symbol_stats.empty else []
    sym_values = symbol_stats["net_profit"].tolist() if "net_profit" in symbol_stats.columns else []

    # LLM decision breakdown (incl. filtered)
    all_dec = trades["llm_decision"].value_counts() if not trades.empty and "llm_decision" in trades else pd.Series(dtype=int)

    # Trade table HTML
    display_cols = ["symbol", "entry_time", "exit_time", "side", "reason",
                    "entry_price", "exit_price", "sl", "tp", "lot",
                    "probability", "pnl", "balance_after",
                    "llm_decision", "llm_confidence", "llm_sentiment", "llm_tech_view"]
    disp = real[[c for c in display_cols if c in real.columns]].copy()
    if not disp.empty:
        for tc in ("entry_time", "exit_time"):
            if tc in disp:
                disp[tc] = disp[tc].dt.strftime("%Y-%m-%d %H:%M")
        for nc in ("entry_price", "exit_price", "sl", "tp"):
            if nc in disp:
                disp[nc] = disp[nc].map(lambda x: f"{x:.5f}")
        if "pnl" in disp:
            disp["pnl"] = disp["pnl"].map(lambda x: f"{x:+.2f}")
        if "balance_after" in disp:
            disp["balance_after"] = disp["balance_after"].map(lambda x: f"{x:.2f}")
        if "probability" in disp:
            disp["probability"] = disp["probability"].map(lambda x: f"{x:.3f}")
        if "llm_confidence" in disp:
            disp["llm_confidence"] = disp["llm_confidence"].map(lambda x: f"{x:.2f}")

    table_html = disp.to_html(index=False, border=0, classes="report-table",
                              justify="left", escape=False) if not disp.empty else "<p>No trades</p>"

    plans_html = _build_plans_html(plans)

    # Metadata
    fast_tf = report.get("backtest_fast_tf", "M15")
    slow_tf = report.get("backtest_slow_tf", "H1")
    period_s = report.get("period_start_utc", "")[:10]
    period_e = report.get("period_end_utc", "")[:10]
    llm_mode = report.get("llm_mode", "ollama")
    news_m   = report.get("news_model", "mistral:7b")
    tech_m   = report.get("tech_model", "qwen2.5:14b")
    dec_m    = report.get("decision_model", "deepseek-r1:14b")

    js = json.dumps({
        "eq_times": eq_times, "eq_balance": eq_balance, "eq_dd": eq_dd,
        "trade_pnl": trade_pnl, "trade_cum": trade_cum,
        "reason_labels": reason_vc.index.tolist(), "reason_values": reason_vc.tolist(),
        "monthly_labels": monthly.index.tolist(), "monthly_values": monthly.tolist(),
        "sym_labels": sym_labels, "sym_values": [float(v) for v in sym_values],
        "ll_labels": all_dec.index.tolist(), "ll_values": all_dec.tolist(),
        "wl_labels": ["WIN","LOSS"], "wl_values": [wins, losses],
        "sent_labels": list(sent_pnl.keys()),
        "sent_avg": [float(sum(v)/len(v)) if v else 0.0 for v in sent_pnl.values()],
    })

    kpi_cards = [
        ("Return",           _fmt_pct(ret_pct),     _color(ret_pct)),
        ("Net Profit",       _fmt_money(net),        _color(net)),
        ("Profit Factor",    f"{pf:.2f}",            _color(pf - 1)),
        ("Max Drawdown",     f"{max_dd:.2f}%",       "#ef4444"),
        ("Win Rate",         f"{wr*100:.1f}%",       _color(wr - 0.5)),
        ("Total Trades",     str(total),             "#60a5fa"),
        ("Avg Win",          _fmt_money(avg_w),      "#22c55e"),
        ("Avg Loss",         _fmt_money(avg_l),      "#ef4444"),
        ("Final Balance",    _fmt_money(fin_bal),    _color(fin_bal - init_bal)),
        ("LLM Avg Conf.",    f"{llm_avg_conf:.1f}%", "#a78bfa"),
        ("LLM Filtered",     str(llm_filtered_n),   "#f59e0b"),
        ("Filter Rate",      f"{filter_rate:.1f}%",  "#f59e0b"),
    ]

    cards_html = "\n".join(
        f'<div class="card">'
        f'<div class="label">{_escape(k)}</div>'
        f'<div class="value" style="color:{c}">{_escape(v)}</div>'
        f'</div>'
        for k, v, c in kpi_cards
    )

    html_out = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>LLM Trading Backtest — {fast_tf}/{slow_tf} | {period_s} to {period_e}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg:#0b1220; --panel:#111827; --panel2:#1f2937;
      --text:#e5e7eb; --muted:#9ca3af; --accent:#22c55e;
      --danger:#ef4444; --warn:#f59e0b; --llm:#a78bfa;
      --border:#374151;
    }}
    *{{box-sizing:border-box;margin:0;padding:0}}
    body{{background:linear-gradient(160deg,#080f1e,#0b1220 40%,#111827);
          color:var(--text);font-family:"Segoe UI",Arial,sans-serif;line-height:1.5}}
    .container{{width:min(1600px,96vw);margin:20px auto 40px}}
    /* Header */
    .header{{background:rgba(17,24,39,.9);border:1px solid var(--border);
             border-radius:16px;padding:20px 24px;margin-bottom:14px;
             backdrop-filter:blur(8px)}}
    .header h1{{font-size:26px;letter-spacing:.2px;margin-bottom:4px}}
    .header-meta{{display:flex;flex-wrap:wrap;gap:16px;margin-top:8px;font-size:13px;color:var(--muted)}}
    .badge{{background:rgba(167,139,250,.15);color:var(--llm);
            border:1px solid rgba(167,139,250,.35);border-radius:6px;
            padding:2px 8px;font-size:12px;font-weight:600}}
    /* Cards */
    .cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:10px;margin-bottom:14px}}
    .card{{background:rgba(17,24,39,.9);border:1px solid var(--border);
           border-radius:12px;padding:14px;min-height:80px}}
    .label{{color:var(--muted);font-size:11px;text-transform:uppercase;
            letter-spacing:.7px;margin-bottom:6px}}
    .value{{font-size:22px;font-weight:700}}
    /* Grid */
    .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px}}
    .grid3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:12px}}
    .panel{{background:rgba(17,24,39,.9);border:1px solid var(--border);
            border-radius:12px;padding:12px}}
    .panel h3{{font-size:14px;color:#d1d5db;margin-bottom:8px;padding:0 4px}}
    .chart{{width:100%;height:350px}}
    .chart.sm{{height:280px}}
    /* Table */
    .report-table{{width:100%;border-collapse:collapse;font-size:11px}}
    .report-table th,.report-table td{{border:1px solid var(--border);padding:5px 7px;
                                       text-align:left;word-break:break-word}}
    .report-table thead th{{background:var(--panel2);position:sticky;top:0;z-index:1}}
    .scroll{{max-height:420px;overflow:auto;border:1px solid var(--border);border-radius:10px}}
    /* Plans */
    .plans-container{{margin-bottom:12px}}
    details.plan-row{{border:1px solid var(--border);border-radius:8px;margin-bottom:6px;
                      background:rgba(17,24,39,.7);overflow:hidden}}
    details.plan-row summary{{display:grid;
      grid-template-columns:50px 90px 140px 100px 60px 80px 70px;
      gap:8px;align-items:center;padding:8px 12px;cursor:pointer;font-size:12px}}
    details.plan-row summary:hover{{background:rgba(255,255,255,.04)}}
    .plan-tid{{color:var(--muted);font-size:11px}}
    .plan-sym{{font-weight:700;color:#60a5fa}}
    .plan-time{{color:var(--muted);font-size:11px}}
    .plan-lstm{{color:#a78bfa}}
    .plan-pnl,.plan-llm{{font-weight:700}}
    .plan-reason{{color:var(--muted);font-size:11px}}
    .plan-body{{padding:12px 16px;border-top:1px solid var(--border);background:rgba(0,0,0,.2)}}
    .plan-grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:12px}}
    .plan-grid > div{{background:rgba(255,255,255,.03);border-radius:6px;padding:8px 10px}}
    /* Responsive */
    @media(max-width:1100px){{.grid2,.grid3{{grid-template-columns:1fr}}}}
    @media(max-width:700px){{.cards{{grid-template-columns:repeat(2,1fr)}}
      details.plan-row summary{{grid-template-columns:40px 70px 1fr 60px}}}}
  </style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>🤖 LLM-Integrated Trading Backtest Report</h1>
    <div class="header-meta">
      <span>📅 {_escape(period_s)} → {_escape(period_e)}</span>
      <span>📊 TF: <strong>{_escape(fast_tf)}/{_escape(slow_tf)}</strong></span>
      <span>💰 Initial: <strong>${init_bal:,.0f}</strong></span>
      <span class="badge">📰 {_escape(news_m)}</span>
      <span class="badge">📈 {_escape(tech_m)}</span>
      <span class="badge">🎯 {_escape(dec_m)}</span>
      <span style="color:{'#a78bfa' if llm_mode=='ollama' else '#f59e0b'}">
        LLM: {'🟢 LIVE' if llm_mode=='ollama' else '🟡 MOCK'}
      </span>
    </div>
  </div>

  <div class="cards">{cards_html}</div>

  <div class="grid2">
    <div class="panel"><h3>📈 Equity Curve &amp; Drawdown</h3><div id="eqChart" class="chart"></div></div>
    <div class="panel"><h3>💹 Trade PnL &amp; Cumulative</h3><div id="pnlChart" class="chart"></div></div>
  </div>

  <div class="grid3">
    <div class="panel"><h3>🤖 LLM Decision Distribution</h3><div id="llmDecChart" class="chart sm"></div></div>
    <div class="panel"><h3>✅ Win / Loss</h3><div id="wlChart" class="chart sm"></div></div>
    <div class="panel"><h3>🚪 Exit Reason</h3><div id="reasonChart" class="chart sm"></div></div>
  </div>

  <div class="grid3">
    <div class="panel"><h3>📰 Sentiment vs Avg PnL</h3><div id="sentChart" class="chart sm"></div></div>
    <div class="panel"><h3>📅 Monthly PnL</h3><div id="monthlyChart" class="chart sm"></div></div>
    <div class="panel"><h3>🏆 Per-Symbol Net Profit</h3><div id="symChart" class="chart sm"></div></div>
  </div>

  <div class="panel plans-container">
    <h3>📋 Trading Plans (LLM Reasoning per Trade — click to expand)</h3>
    <div style="font-size:11px;color:var(--muted);margin-bottom:8px;padding:0 4px">
      Columns: Trade# | Symbol | Entry Time | LSTM Signal | LLM Decision | PnL | Exit Reason
    </div>
    {plans_html}
  </div>

  <div class="panel" style="margin-bottom:12px">
    <h3>📂 Full Trade History (LLM-Executed Trades Only)</h3>
    <div class="scroll">{table_html}</div>
  </div>
</div>

<script>
const d = {js};

const base = {{
  paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
  font:{{color:'#e5e7eb',size:11}},
  margin:{{l:50,r:20,t:10,b:45}},
  xaxis:{{gridcolor:'#374151'}}, yaxis:{{gridcolor:'#374151'}},
  legend:{{orientation:'h',y:-0.15}},
}};

// Equity + Drawdown
Plotly.newPlot('eqChart', [
  {{x:d.eq_times,y:d.eq_balance,type:'scatter',mode:'lines',name:'Balance',
    line:{{color:'#22c55e',width:2}}}},
  {{x:d.eq_times,y:d.eq_dd,type:'scatter',mode:'lines',name:'Drawdown %',
    yaxis:'y2',line:{{color:'#ef4444',width:1.5,dash:'dot'}},fill:'tozeroy',
    fillcolor:'rgba(239,68,68,0.08)'}},
], {{...base,
  yaxis:{{title:'Balance ($)',gridcolor:'#374151'}},
  yaxis2:{{title:'Drawdown %',overlaying:'y',side:'right',gridcolor:'rgba(0,0,0,0)'}},
}}, {{displaylogo:false,responsive:true}});

// PnL waterfall
Plotly.newPlot('pnlChart', [
  {{x:d.trade_pnl.map((_,i)=>i+1), y:d.trade_pnl, type:'bar', name:'Trade PnL',
    marker:{{color:d.trade_pnl.map(v=>v>=0?'#22c55e':'#ef4444')}},opacity:0.8}},
  {{x:d.trade_cum.map((_,i)=>i+1), y:d.trade_cum, type:'scatter', mode:'lines',
    name:'Cum. PnL', yaxis:'y2', line:{{color:'#60a5fa',width:2}}}},
], {{...base,
  yaxis:{{title:'Trade PnL',gridcolor:'#374151'}},
  yaxis2:{{title:'Cumulative PnL',overlaying:'y',side:'right',gridcolor:'rgba(0,0,0,0)'}},
  xaxis:{{...base.xaxis,title:'Trade #'}},
}}, {{displaylogo:false,responsive:true}});

// LLM Decision donut
Plotly.newPlot('llmDecChart', [{{
  labels:d.ll_labels, values:d.ll_values, type:'pie', hole:0.45,
  marker:{{colors:d.ll_labels.map(l=>l==='BUY'?'#22c55e':l==='SELL'?'#ef4444':l==='HOLD'?'#f59e0b':'#6b7280')}},
  textinfo:'label+value',
}}], {{...base,margin:{{l:10,r:10,t:10,b:10}}}}, {{displaylogo:false,responsive:true}});

// Win/Loss donut
Plotly.newPlot('wlChart', [{{
  labels:d.wl_labels, values:d.wl_values, type:'pie', hole:0.45,
  marker:{{colors:['#22c55e','#ef4444']}}, textinfo:'label+percent+value',
}}], {{...base,margin:{{l:10,r:10,t:10,b:10}}}}, {{displaylogo:false,responsive:true}});

// Exit reason bar
Plotly.newPlot('reasonChart', [{{
  x:d.reason_labels, y:d.reason_values, type:'bar',
  marker:{{color:d.reason_labels.map(l=>l==='TP'?'#22c55e':l==='SL'?'#ef4444':'#f59e0b')}},
}}], {{...base,xaxis:{{...base.xaxis,title:'Reason'}},yaxis:{{...base.yaxis,title:'Count'}}}},
{{displaylogo:false,responsive:true}});

// Sentiment vs PnL
Plotly.newPlot('sentChart', [{{
  x:d.sent_labels, y:d.sent_avg, type:'bar',
  marker:{{color:d.sent_avg.map(v=>v>=0?'#22c55e':'#ef4444')}},
  text:d.sent_avg.map(v=>(v>=0?'+':'')+v.toFixed(3)), textposition:'outside',
}}], {{...base,xaxis:{{...base.xaxis,title:'LLM Sentiment'}},
  yaxis:{{...base.yaxis,title:'Avg PnL ($)'}}}}, {{displaylogo:false,responsive:true}});

// Monthly PnL
Plotly.newPlot('monthlyChart', [{{
  x:d.monthly_labels, y:d.monthly_values, type:'bar',
  marker:{{color:d.monthly_values.map(v=>v>=0?'#22c55e':'#ef4444')}},
}}], {{...base,xaxis:{{...base.xaxis,title:'Month'}},yaxis:{{...base.yaxis,title:'Net PnL'}}}},
{{displaylogo:false,responsive:true}});

// Per-symbol bar
Plotly.newPlot('symChart', [{{
  x:d.sym_labels, y:d.sym_values, type:'bar',
  marker:{{color:d.sym_values.map(v=>v>=0?'#22c55e':'#ef4444')}},
  text:d.sym_values.map(v=>(v>=0?'+':'')+v.toFixed(2)), textposition:'outside',
}}], {{...base,xaxis:{{...base.xaxis,title:'Symbol'}},yaxis:{{...base.yaxis,title:'Net Profit'}}}},
{{displaylogo:false,responsive:true}});
</script>
</body>
</html>"""

    output_path.write_text(html_out, encoding="utf-8")
    print(f"[HTML] Written: {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LLM backtest HTML dashboard")
    parser.add_argument("--report-json",    default=f"outputs/{TAG}_report.json")
    parser.add_argument("--trades-csv",     default=f"outputs/{TAG}_trades.csv")
    parser.add_argument("--equity-csv",     default=f"outputs/{TAG}_equity.csv")
    parser.add_argument("--symbol-stats",   default=f"outputs/{TAG}_symbol_stats.csv")
    parser.add_argument("--plans-json",     default=f"outputs/{TAG}_plans.json")
    parser.add_argument("--output",         default=f"outputs/{TAG}_report.html")
    args = parser.parse_args()

    report  = json.loads(Path(args.report_json).read_text("utf-8"))
    trades  = pd.read_csv(args.trades_csv)
    equity  = pd.read_csv(args.equity_csv)
    symstat = pd.read_csv(args.symbol_stats) if Path(args.symbol_stats).exists() else pd.DataFrame()
    plans   = json.loads(Path(args.plans_json).read_text("utf-8")) if Path(args.plans_json).exists() else []

    generate_llm_html(report, trades, equity, symstat, plans, Path(args.output))
    print(f"Report: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
