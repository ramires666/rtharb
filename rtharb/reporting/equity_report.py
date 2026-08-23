"""Build exact all-time, 2026 YTD, and August equity/drawdown reports."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from rtharb.config import AppConfig
from rtharb.data.loader import DataLoader
from rtharb.models.fair_value import FairValueModel
from rtharb.models.signals import SignalGenerator
from rtharb.backtest.engine import BacktestEngine
from rtharb.backtest.metrics import calculate_performance_metrics


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "equity_output"


def svg_chart(daily: pd.DataFrame, title: str, metrics, path: Path):
    width, height = 1600, 720
    x0, x1 = 100, 1540
    ey0, ey1 = 80, 455
    dy0, dy1 = 510, 655
    n = len(daily)
    xs = np.linspace(x0, x1, n)
    eq = daily.equity.to_numpy(float)
    lo, hi = float(eq.min()), float(eq.max())
    pad = max((hi-lo)*0.08, 25.0); lo -= pad; hi += pad
    ey = ey1 - (eq-lo)/(hi-lo)*(ey1-ey0)
    dd = daily.drawdown_pct.to_numpy(float)
    dd_hi = max(float(dd.max())*1.08, 0.1)
    dy = dy0 + dd/dd_hi*(dy1-dy0)
    eq_points = ' '.join(f'{x:.2f},{y:.2f}' for x,y in zip(xs,ey))
    dd_points = f'{x0},{dy0} ' + ' '.join(f'{x:.2f},{y:.2f}' for x,y in zip(xs,dy)) + f' {x1},{dy0}'
    ticks = np.unique(np.linspace(0,n-1,min(6,n),dtype=int))
    labels = ''.join(f'<text x="{xs[i]:.1f}" y="690" text-anchor="middle" fill="#8f9cae" font-size="13">{daily.index[i]}</text>' for i in ticks)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#0b0e14"/><text x="100" y="38" fill="#fff" font-size="25">{title}</text>
<text x="100" y="65" fill="#8f9cae" font-size="14">Net equity по закрытиям дней, старт $100,000 | Max DD по минутной equity {metrics.max_drawdown_pct:.2f}% (${metrics.max_drawdown_usd:,.2f})</text>
<rect x="{x0}" y="{ey0}" width="{x1-x0}" height="{ey1-ey0}" fill="#121722" stroke="#263041"/>
<line x1="{x0}" y1="{ey1-(100000-lo)/(hi-lo)*(ey1-ey0):.2f}" x2="{x1}" y2="{ey1-(100000-lo)/(hi-lo)*(ey1-ey0):.2f}" stroke="#667085" stroke-dasharray="6 5"/>
<polyline points="{eq_points}" fill="none" stroke="#36c5f0" stroke-width="2"/>
<text x="15" y="{ey0+8}" fill="#8f9cae" font-size="13">${hi:,.0f}</text><text x="15" y="{ey1}" fill="#8f9cae" font-size="13">${lo:,.0f}</text>
<text x="100" y="495" fill="#fff" font-size="16">Drawdown</text><rect x="{x0}" y="{dy0}" width="{x1-x0}" height="{dy1-dy0}" fill="#121722" stroke="#263041"/>
<polygon points="{dd_points}" fill="#ff5c5c" opacity="0.45"/><text x="20" y="{dy1}" fill="#ff8a8a" font-size="13">−{dd_hi:.2f}%</text>{labels}</svg>'''
    path.write_text(svg, encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    selected = json.loads((ROOT/"research_output"/"base_strategy_summary.json").read_text(encoding="utf-8"))["selected_parameters"]
    cfg = AppConfig.load(str(ROOT/"configs"/"default_config.yaml"))
    lead,target = DataLoader(cfg.cache_dir,"alpaca","sip").get_synchronized_pair("QQQ","NVDA")
    metrics = FairValueModel(selected["beta_mode"],selected["beta_days"],selected["window"],15).compute_intraday_metrics(lead,target)
    signals = SignalGenerator(z_entry=selected["z_entry"], reversal_delta=selected["hook_delta"],
        reversal_timeout_bars=selected["hook_timeout"], enable_extreme_entry_lockout=True,
        enable_extreme_emergency_exit=False, z_max_allowed=selected["z_lockout"], z_exit=selected["exit_band"],
        forced_close_time="15:55",min_session_warmup_bars=15).generate_signals(metrics)
    bt = BacktestEngine(cfg.backtest.initial_capital,cfg.backtest.position_size_usd,
        cfg.backtest.commission_per_share,cfg.backtest.slippage_pct,True).run(signals,"NVDA")
    results, trades = bt["df_results"], bt["trades_df"]
    periods = {"all_time":None,"2026_ytd":"2026-01-01","august_2026":"2026-08-01"}
    summaries = {}
    all_daily = []
    for key,start in periods.items():
        mask = np.ones(len(results),dtype=bool) if start is None else results.index >= pd.Timestamp(start,tz="America/New_York")
        loc = np.flatnonzero(mask); frame = results.iloc[loc].copy()
        base_actual = cfg.backtest.initial_capital if loc[0] == 0 else float(results.portfolio_equity.iloc[loc[0]-1])
        frame["portfolio_equity"] = cfg.backtest.initial_capital + frame.portfolio_equity - base_actual
        frame["portfolio_cash"] = cfg.backtest.initial_capital + frame.portfolio_cash - base_actual
        entry_times = pd.to_datetime(trades.entry_time, utc=True).dt.tz_convert("America/New_York")
        tmask = np.ones(len(trades),dtype=bool) if start is None else entry_times >= pd.Timestamp(start,tz="America/New_York")
        ptrades = trades.loc[tmask].copy()
        perf = calculate_performance_metrics(frame,ptrades,cfg.backtest.initial_capital)
        daily_eq = frame.groupby("session_date").portfolio_equity.last()
        with_start = pd.concat([pd.Series([cfg.backtest.initial_capital]),daily_eq.reset_index(drop=True)])
        running = with_start.cummax().iloc[1:].to_numpy()
        daily = pd.DataFrame({"date":[str(d) for d in daily_eq.index],"equity":daily_eq.to_numpy(),
                              "drawdown_pct":(running-daily_eq.to_numpy())/running*100}).set_index("date")
        daily.assign(period=key).reset_index().to_csv(OUT/f"daily_equity_{key}.csv",index=False)
        svg_chart(daily,{"all_time":"Equity — весь период","2026_ytd":"Equity — 2026 YTD","august_2026":"Equity — август 2026"}[key],perf,OUT/f"equity_{key}.svg")
        summaries[key] = asdict(perf)
        summaries[key]["gross_pnl_before_costs"] = float(ptrades["gross_pnl"].sum())
        summaries[key]["total_costs"] = float(ptrades["commission"].sum() + ptrades["slippage"].sum())
        summaries[key]["net_pnl_from_trade_log"] = float(ptrades["net_pnl"].sum())
        summaries[key]["pnl_reconciliation_error"] = float(perf.total_pnl - ptrades["net_pnl"].sum())
        summaries[key]["start"] = str(frame.index.min()); summaries[key]["end"] = str(frame.index.max())
        all_daily.append(daily.assign(period=key).reset_index())
    output = {"strategy_parameters":selected,"cost_model":{
        "position_size_usd":cfg.backtest.position_size_usd,
        "commission_per_share_per_side":cfg.backtest.commission_per_share,
        "round_trip_commission_per_share":2*cfg.backtest.commission_per_share,
        "slippage_per_execution_bps":cfg.backtest.slippage_pct*10000,
        "approx_round_trip_slippage_bps":2*cfg.backtest.slippage_pct*10000,
        "shares":"integer floor(position_size/effective_entry_price)"},"periods":summaries}
    (OUT/"equity_metrics.json").write_text(json.dumps(output,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    def row(k,label):
        m=summaries[k]; return f'<tr><td>{label}</td><td>{m["total_trades"]}</td><td>${m["gross_pnl_before_costs"]:,.2f}</td><td>${m["total_pnl"]:,.2f}<br>{m["total_return_pct"]:.2f}%</td><td>{m["sharpe_ratio"]:.2f}</td><td>{m["sortino_ratio"]:.2f}</td><td>{m["max_drawdown_pct"]:.2f}%<br>${m["max_drawdown_usd"]:,.2f}</td><td>{m["win_rate_pct"]:.1f}%</td><td>{m["profit_factor"]:.2f}</td><td>${m["total_commissions"]:,.2f}</td><td>${m["total_slippage"]:,.2f}</td><td>${m["total_costs"]:,.2f}</td></tr>'
    html=f'''<!doctype html><meta charset="utf-8"><title>Equity и drawdown</title><style>body{{font:16px system-ui;max-width:1500px;margin:auto;padding:28px;background:#0b0e14;color:#e6edf3}}table{{border-collapse:collapse;font-size:14px}}td,th{{padding:8px;border:1px solid #344054;text-align:right}}td:first-child,th:first-child{{text-align:left}}img{{width:100%}}</style><h1>Equity, drawdown и основные метрики</h1><p>Стратегия без stop-loss/time-stop. Август неполный: данные по 21.08.2026 включительно. Max DD рассчитан по минутной mark-to-market equity.</p><table><tr><th>Период</th><th>Сделки</th><th>Gross P&amp;L</th><th>Net P&amp;L / return</th><th>Sharpe</th><th>Sortino</th><th>Max DD</th><th>Win rate</th><th>PF</th><th>Комиссия</th><th>Slippage</th><th>Все издержки</th></tr>{row("all_time","Весь период")}{row("2026_ytd","2026 YTD")}{row("august_2026","Август 2026")}</table><h2>Издержки</h2><p>IBKR-модель: $0.0035 за акцию на каждой стороне ($0.007 round trip); slippage 2 bps на входе и 2 bps на выходе (примерно 4 bps round trip); позиция $20,000, целое число акций. Net P&amp;L уже после обеих издержек.</p><img src="equity_all_time.svg"><img src="equity_2026_ytd.svg"><img src="equity_august_2026.svg">'''
    (OUT/"EQUITY_REPORT.html").write_text(html,encoding="utf-8")
    print(json.dumps(output,ensure_ascii=False,indent=2,default=str))


if __name__=="__main__": main()
