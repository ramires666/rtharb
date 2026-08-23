"""One-year frozen VWAP-Z entry study with absolute-dollar NVDA brackets.

The entry stream is the already audited causal VWAP-Z / next-open cohort.
Only the exit is changed: an independent absolute stop and target, expressed
in NVDA dollars per share.  The convergence exit is deliberately disabled.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rtharb.config import AppConfig
from rtharb.data.loader import DataLoader
from rtharb.research.risk_reward import CAPITAL, COMMISSION, SIZE, SLIP, _raw_target


OUT = ROOT / "old" / "frozen_vwap_absolute" / "output" / "results"
COHORT_PATH = ROOT / "research_output" / "risk_reward" / "vwap_z_entry_cohort.csv"
START_DATE = pd.Timestamp("2025-08-22").date()
END_DATE = pd.Timestamp("2026-08-21").date()
DISTANCES = tuple(round(x * 0.25, 2) for x in range(1, 13))
TOP_DEVELOPMENT = 10


def _json_value(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def load_cohort() -> pd.DataFrame:
    cohort = pd.read_csv(COHORT_PATH)
    for column in ("entry_time", "exit_time"):
        cohort[column] = pd.to_datetime(cohort[column], format="mixed", utc=True).dt.tz_convert("America/New_York")
    cohort = cohort[(cohort.entry_time.dt.date >= START_DATE) & (cohort.entry_time.dt.date <= END_DATE)].copy()
    return cohort.sort_values("entry_time").reset_index(drop=True)


def one_trade(trade, raw, session_times, stop_usd: float, target_usd: float) -> dict:
    entry_ts = pd.Timestamp(trade.entry_time)
    base_exit_ts = pd.Timestamp(trade.exit_time)
    direction = 1 if str(trade.direction).upper() == "LONG" else -1
    if entry_ts not in raw:
        raise KeyError(f"Missing raw entry bar {entry_ts}")
    entry_ref = raw[entry_ts][0]
    entry_eff = entry_ref * (1 + SLIP if direction == 1 else 1 - SLIP)
    shares = math.floor(SIZE / entry_eff)
    if shares <= 0:
        raise ValueError(f"Invalid share count at {entry_ts}: {shares}")
    stop_price = entry_ref - stop_usd if direction == 1 else entry_ref + stop_usd
    target_price = entry_ref + target_usd if direction == 1 else entry_ref - target_usd
    timestamps = session_times[bisect.bisect_left(session_times, entry_ts):]
    if not timestamps:
        raise AssertionError(f"No raw bars from entry through EOD at {entry_ts}")
    eod_ts = session_times[-1]
    exit_ts, raw_exit, reason = eod_ts, raw[eod_ts][3], "FORCED_EOD"
    for ts in timestamps:
        op, hi, lo, _ = raw[ts]
        stop_hit = (op <= stop_price or lo <= stop_price) if direction == 1 else (op >= stop_price or hi >= stop_price)
        target_hit = hi >= target_price if direction == 1 else lo <= target_price
        # Conservative ambiguity rule: a same-bar stop always wins.
        if stop_hit:
            gap_through = op <= stop_price if direction == 1 else op >= stop_price
            raw_exit = op if gap_through else stop_price
            exit_ts, reason = ts, "STOP"
            break
        if target_hit:
            raw_exit, exit_ts, reason = target_price, ts, "TAKE_PROFIT_BRACKET"
            break
    exit_eff = raw_exit * (1 - SLIP if direction == 1 else 1 + SLIP)
    gross = direction * (raw_exit - entry_ref) * shares
    slippage = abs(entry_eff - entry_ref) * shares + abs(exit_eff - raw_exit) * shares
    commissions = 2 * shares * COMMISSION
    costs = slippage + commissions
    return {
        "entry_time": entry_ts,
        "base_convergence_exit_time_audit_only": base_exit_ts,
        "exit_time": exit_ts,
        "direction": "LONG" if direction == 1 else "SHORT",
        "entry_z": float(trade.entry_z),
        "entry_reference": entry_ref,
        "entry_price": entry_eff,
        "exit_reference": raw_exit,
        "exit_price": exit_eff,
        "shares": shares,
        "stop_usd_per_share": stop_usd,
        "target_usd_per_share": target_usd,
        "gross_risk_usd": stop_usd * shares,
        "gross_reward_usd": target_usd * shares,
        "risk_reward_ratio": target_usd / stop_usd,
        "stop_price": stop_price,
        "target_price": target_price,
        "exit_reason": reason,
        "gross_pnl": gross,
        "slippage": slippage,
        "commissions": commissions,
        "costs": costs,
        "net_pnl": gross - costs,
        "duration_bars": int((pd.Timestamp(exit_ts) - entry_ts).total_seconds() // 60),
    }


def metrics(trades: pd.DataFrame, candidate_count: int, skipped: int) -> dict:
    if trades.empty:
        return {
            "candidate_entries": candidate_count, "skipped_overlaps": skipped, "trades": 0,
            "gross_pnl": 0.0, "costs": 0.0, "net_pnl": 0.0, "win_rate_pct": 0.0,
            "profit_factor": 0.0, "net_sharpe": 0.0, "max_drawdown_usd": 0.0,
            "max_drawdown_pct": 0.0, "stops": 0, "targets": 0, "forced_eod": 0,
            "avg_net_trade": 0.0, "avg_gross_risk_usd": 0.0, "avg_gross_reward_usd": 0.0,
        }
    daily = trades.assign(day=pd.to_datetime(trades.exit_time).dt.date).groupby("day").net_pnl.sum()
    prior_equity = CAPITAL + daily.cumsum().shift(1).fillna(0.0)
    daily_returns = daily / prior_equity
    std = daily_returns.std(ddof=1)
    sharpe = float(np.sqrt(252) * daily_returns.mean() / std) if len(daily_returns) > 1 and std else 0.0
    wins = trades[trades.net_pnl > 0]
    losses = trades[trades.net_pnl <= 0]
    loss_total = float(losses.net_pnl.sum())
    pf = float(wins.net_pnl.sum() / abs(loss_total)) if loss_total else float("inf")
    equity = pd.concat([pd.Series([CAPITAL]), CAPITAL + trades.net_pnl.cumsum().reset_index(drop=True)], ignore_index=True)
    peaks = equity.cummax()
    dd = peaks - equity
    dd_pct = dd / peaks.replace(0.0, np.nan) * 100.0
    return {
        "candidate_entries": candidate_count,
        "skipped_overlaps": skipped,
        "trades": len(trades),
        "gross_pnl": float(trades.gross_pnl.sum()),
        "costs": float(trades.costs.sum()),
        "net_pnl": float(trades.net_pnl.sum()),
        "win_rate_pct": float((trades.net_pnl > 0).mean() * 100.0),
        "profit_factor": pf,
        "net_sharpe": sharpe,
        "max_drawdown_usd": float(dd.max()),
        "max_drawdown_pct": float(dd_pct.max()),
        "stops": int((trades.exit_reason == "STOP").sum()),
        "targets": int((trades.exit_reason == "TAKE_PROFIT_BRACKET").sum()),
        "forced_eod": int((trades.exit_reason == "FORCED_EOD").sum()),
        "avg_net_trade": float(trades.net_pnl.mean()),
        "avg_gross_risk_usd": float(trades.gross_risk_usd.mean()),
        "avg_gross_reward_usd": float(trades.gross_reward_usd.mean()),
    }


def evaluate(cohort: pd.DataFrame, raw: dict, session_map: dict, stop_usd: float, target_usd: float):
    rows: list[dict] = []
    skipped = 0
    next_available = None
    for trade in cohort.sort_values("entry_time").itertuples(index=False):
        entry_ts = pd.Timestamp(trade.entry_time)
        if next_available is not None and entry_ts <= next_available:
            skipped += 1
            continue
        result = one_trade(trade, raw, session_map[entry_ts.date()], stop_usd, target_usd)
        rows.append(result)
        next_available = pd.Timestamp(result["exit_time"])
    frame = pd.DataFrame(rows)
    result_metrics = metrics(frame, len(cohort), skipped)
    if result_metrics["trades"] + result_metrics["skipped_overlaps"] != len(cohort):
        raise AssertionError("Candidate entry reconciliation failed")
    if not frame.empty:
        if abs(frame.net_pnl.sum() - result_metrics["net_pnl"]) > 1e-8:
            raise AssertionError("Trade/metric net P&L mismatch")
        if abs(frame.costs.sum() - result_metrics["costs"]) > 1e-8:
            raise AssertionError("Trade/metric cost mismatch")
    return result_metrics, frame


def equity_frame(trades: pd.DataFrame) -> pd.DataFrame:
    initial_ts = pd.Timestamp(trades.entry_time.iloc[0]) - pd.Timedelta(minutes=1)
    out = pd.DataFrame({
        "timestamp": pd.concat([pd.Series([initial_ts]), trades.exit_time.reset_index(drop=True)], ignore_index=True),
        "trade_number": np.arange(len(trades) + 1),
        "net_pnl": pd.concat([pd.Series([0.0]), trades.net_pnl.reset_index(drop=True)], ignore_index=True),
    })
    out["equity"] = CAPITAL + out.net_pnl.cumsum()
    out["running_peak"] = out.equity.cummax()
    out["drawdown_usd"] = out.running_peak - out.equity
    out["drawdown_pct"] = out.drawdown_usd / out.running_peak * 100.0
    return out


def mark_to_market_equity(trades: pd.DataFrame, raw_times: list[pd.Timestamp], raw: dict) -> pd.DataFrame:
    """Minute-close equity with entry costs accrued immediately and exits fully realized."""
    ordered = trades.sort_values("entry_time").reset_index(drop=True)
    trade_ix = 0
    active = None
    realized_equity = CAPITAL
    rows = []
    for ts in raw_times:
        if active is not None and ts >= pd.Timestamp(active.exit_time):
            realized_equity += float(active.net_pnl)
            active = None
        if active is None and trade_ix < len(ordered) and ts == pd.Timestamp(ordered.iloc[trade_ix].entry_time):
            candidate = ordered.iloc[trade_ix]
            trade_ix += 1
            if ts >= pd.Timestamp(candidate.exit_time):
                realized_equity += float(candidate.net_pnl)
            else:
                active = candidate
        if active is None:
            value = realized_equity
        else:
            direction = 1 if str(active.direction).upper() == "LONG" else -1
            close = float(raw[ts][3])
            value = (realized_equity - float(active.commissions) / 2.0
                     + direction * (close - float(active.entry_price)) * int(active.shares))
        rows.append({"timestamp": ts, "equity": value})
    if active is not None or trade_ix != len(ordered):
        raise AssertionError("Not every selected trade was represented in minute MTM equity")
    out = pd.DataFrame(rows)
    out["running_peak"] = out.equity.cummax()
    out["drawdown_usd"] = out.running_peak - out.equity
    out["drawdown_pct"] = out.drawdown_usd / out.running_peak * 100.0
    return out


def write_html(summary: dict, equity: pd.DataFrame, finalists: pd.DataFrame) -> None:
    selected = summary["selected"]
    full = summary["selected_results"]["full"]
    hold = summary["selected_results"]["holdout"]
    mtm = summary["mark_to_market"]
    points = equity[["timestamp", "equity", "drawdown_usd", "drawdown_pct"]].copy()
    points.timestamp = points.timestamp.astype(str)
    payload = points.to_dict(orient="records")
    finalist_rows = finalists[["stop_usd", "target_usd", "dev_net_sharpe", "validation_net_sharpe", "robust_score", "validation_net_pnl"]].to_dict(orient="records")
    html = f"""<!doctype html><html lang=\"ru\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>VWAP-Z: абсолютные стоп и цель</title><style>
body{{margin:0;background:#0b1220;color:#e5edf7;font:14px system-ui,Segoe UI,sans-serif}}main{{max-width:1200px;margin:auto;padding:18px}}h1{{margin:0 0 6px}}.sub{{color:#9fb0c5}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:18px 0}}.card{{background:#121d2e;border:1px solid #26364d;border-radius:10px;padding:12px}}.v{{font-size:21px;font-weight:700;margin-top:4px}}canvas{{width:100%;height:310px;background:#0f1928;border:1px solid #26364d;border-radius:10px}}#dd{{height:150px;margin-top:10px}}table{{border-collapse:collapse;width:100%;margin-top:15px;background:#121d2e}}th,td{{padding:7px 9px;border-bottom:1px solid #26364d;text-align:right}}th:first-child,td:first-child{{text-align:left}}.note{{background:#16243a;border-left:4px solid #4fa3ff;padding:12px;margin:14px 0}}code{{color:#a8d5ff}}.tip{{position:fixed;display:none;pointer-events:none;background:#020617;border:1px solid #52719a;padding:7px;border-radius:6px;white-space:nowrap}}
</style></head><body><main><h1>VWAP-Z: абсолютные стоп и цель</h1><div class=\"sub\">Сырой SIP 1m, RTH {summary['period']['start']} — {summary['period']['end']}; frozen causal VWAP-Z entries, convergence exit выключен</div>
<div class=\"cards\"><div class=\"card\">Стоп / акция<div class=\"v\">${selected['stop_usd']:.2f}</div></div><div class=\"card\">Цель / акция<div class=\"v\">${selected['target_usd']:.2f}</div></div><div class=\"card\">Reward / Risk<div class=\"v\">{selected['target_usd']/selected['stop_usd']:.3f}R</div></div><div class=\"card\">Средний gross risk<div class=\"v\">${full['avg_gross_risk_usd']:,.0f}</div></div><div class=\"card\">Средний gross reward<div class=\"v\">${full['avg_gross_reward_usd']:,.0f}</div></div><div class=\"card\">Full net P&amp;L<div class=\"v\">${full['net_pnl']:,.2f}</div></div><div class=\"card\">MTM max drawdown<div class=\"v\">${mtm['max_drawdown_usd']:,.2f}<br><small>{mtm['max_drawdown_pct']:.2f}%</small></div></div><div class=\"card\">Holdout net P&amp;L<div class=\"v\">${hold['net_pnl']:,.2f}</div></div></div>
<div class=\"note\">Размер позиции $20,000. Риск в долларах сделки = <code>stop_usd × shares</code>, reward = <code>target_usd × shares</code>. Комиссия $0.0035/акция/сторона и slippage 2 bps на каждое исполнение вычтены. Stop-first на неоднозначной минуте; гэп через стоп исполняется по неблагоприятному open. Главный MDD рассчитан по минутной mark-to-market equity; MDD только по закрытым сделкам — ${full['max_drawdown_usd']:,.2f} / {full['max_drawdown_pct']:.2f}%.</div>
<h2>Минутная mark-to-market equity и drawdown</h2><canvas id=\"eq\"></canvas><canvas id=\"dd\"></canvas><div id=\"tip\" class=\"tip\"></div>
<h2>Финалисты development → validation</h2><table><thead><tr><th>Stop</th><th>Target</th><th>Dev Sharpe</th><th>Val Sharpe</th><th>Robust</th><th>Val net</th></tr></thead><tbody>{''.join(f'<tr><td>${r["stop_usd"]:.2f}</td><td>${r["target_usd"]:.2f}</td><td>{r["dev_net_sharpe"]:.3f}</td><td>{r["validation_net_sharpe"]:.3f}</td><td>{r["robust_score"]:.3f}</td><td>${r["validation_net_pnl"]:,.2f}</td></tr>' for r in finalist_rows)}</tbody></table>
<h2>Честная схема выбора</h2><p>Первая половина года — development; верхние {TOP_DEVELOPMENT} комбинаций переданы в validation. Выбор сделан по <code>min(dev Sharpe, validation Sharpe)</code>, holdout открыт ровно один раз после фиксации параметров. Full — только итоговый отчёт выбранной комбинации.</p>
</main><script>const data={json.dumps(payload, ensure_ascii=False)};const tip=document.getElementById('tip');
function chart(id,key,color,zero=false){{const c=document.getElementById(id),ctx=c.getContext('2d');function draw(){{const dpr=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*dpr;c.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);const vals=data.map(x=>x[key]),mn=zero?0:Math.min(...vals),mx=Math.max(...vals),pad=(mx-mn||1)*.08,lo=zero?0:mn-pad,hi=mx+pad;ctx.strokeStyle='#26364d';ctx.beginPath();for(let i=0;i<5;i++){{let y=15+(h-30)*i/4;ctx.moveTo(44,y);ctx.lineTo(w-10,y)}}ctx.stroke();ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();vals.forEach((v,i)=>{{let x=44+(w-54)*i/(vals.length-1),y=15+(h-30)*(hi-v)/(hi-lo);i?ctx.lineTo(x,y):ctx.moveTo(x,y)}});ctx.stroke();c.onmousemove=e=>{{let i=Math.max(0,Math.min(vals.length-1,Math.round((e.offsetX-44)/(w-54)*(vals.length-1))));tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';tip.textContent=data[i].timestamp+' | '+key+': '+Number(data[i][key]).toFixed(2)}};c.onmouseleave=()=>tip.style.display='none'}}draw();addEventListener('resize',draw)}}chart('eq','equity','#36d399');chart('dd','drawdown_usd','#fb7185',true);</script></body></html>"""
    (OUT / "REPORT.html").write_text(html, encoding="utf-8")


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = AppConfig.load(str(ROOT / "configs" / "default_config.yaml"))
    lead, target = DataLoader(cfg.cache_dir, "alpaca", "sip").get_synchronized_pair("QQQ", "NVDA")
    common = lead.index.intersection(target.index)
    session_days = sorted({d for d in common.date if START_DATE <= d <= END_DATE})
    if not session_days or session_days[0] != START_DATE or session_days[-1] != END_DATE:
        raise AssertionError(f"Requested completed-year endpoints unavailable: {session_days[:1]} .. {session_days[-1:]}")
    dev_end_ix, val_end_ix = len(session_days) // 2, len(session_days) * 3 // 4
    val_start, hold_start = session_days[dev_end_ix], session_days[val_end_ix]
    cohort = load_cohort()
    cohort_days = cohort.entry_time.dt.date
    cohorts = {
        "development": cohort[cohort_days < val_start],
        "validation": cohort[(cohort_days >= val_start) & (cohort_days < hold_start)],
        "holdout": cohort[cohort_days >= hold_start],
        "full": cohort,
    }
    raw = _raw_target(lead, target)
    raw_times = sorted(ts for ts in raw if START_DATE <= ts.date() <= END_DATE)
    session_map: dict = {}
    for ts in raw_times:
        session_map.setdefault(ts.date(), []).append(ts)

    # No holdout calls occur in the grid-development phase.
    grid_rows = []
    for stop_usd in DISTANCES:
        for target_usd in DISTANCES:
            dev_m, _ = evaluate(cohorts["development"], raw, session_map, stop_usd, target_usd)
            val_m, _ = evaluate(cohorts["validation"], raw, session_map, stop_usd, target_usd)
            row = {"stop_usd": stop_usd, "target_usd": target_usd, "rr": target_usd / stop_usd}
            row.update({f"development_{k}": v for k, v in dev_m.items()})
            row.update({f"validation_{k}": v for k, v in val_m.items()})
            grid_rows.append(row)
    grid = pd.DataFrame(grid_rows)
    ranked = grid.sort_values(["development_net_sharpe", "development_net_pnl"], ascending=False)
    top = ranked.head(TOP_DEVELOPMENT).copy()
    top["dev_net_sharpe"] = top.development_net_sharpe
    top["validation_net_sharpe"] = top.validation_net_sharpe
    top["validation_net_pnl"] = top.validation_net_pnl
    top["robust_score"] = np.minimum(top.dev_net_sharpe, top.validation_net_sharpe)
    finalists = top.sort_values(["robust_score", "validation_net_pnl"], ascending=False).reset_index(drop=True)
    chosen = finalists.iloc[0]
    selected = {"stop_usd": float(chosen.stop_usd), "target_usd": float(chosen.target_usd)}

    selected_results = {}
    selected_trades = {}
    for period, period_cohort in cohorts.items():
        result, trades = evaluate(period_cohort, raw, session_map, **selected)
        selected_results[period] = result
        selected_trades[period] = trades
        trades.to_csv(OUT / f"selected_{period}_trades.csv", index=False)

    split_rows = sum(len(selected_trades[p]) for p in ("development", "validation", "holdout"))
    split_net = sum(selected_results[p]["net_pnl"] for p in ("development", "validation", "holdout"))
    full_rows = len(selected_trades["full"])
    full_net = selected_results["full"]["net_pnl"]
    if split_rows != full_rows or abs(split_net - full_net) > 1e-8:
        raise AssertionError("Selected split/full trade reconciliation failed")

    closed_eq = equity_frame(selected_trades["full"])
    closed_eq.to_csv(OUT / "selected_full_closed_trade_equity.csv", index=False)
    mtm_eq = mark_to_market_equity(selected_trades["full"], raw_times, raw)
    mtm_eq.to_csv(OUT / "selected_full_equity.csv", index=False)
    mtm_peak_ix = int(mtm_eq.drawdown_usd.idxmax())
    mtm_stats = {
        "basis": "exact raw SIP 1-minute close mark-to-market",
        "bars": len(mtm_eq),
        "max_drawdown_usd": float(mtm_eq.loc[mtm_peak_ix, "drawdown_usd"]),
        "max_drawdown_pct": float(mtm_eq.loc[mtm_peak_ix, "drawdown_pct"]),
        "max_drawdown_time": pd.Timestamp(mtm_eq.loc[mtm_peak_ix, "timestamp"]),
        "final_equity": float(mtm_eq.equity.iloc[-1]),
    }
    # Preserve holdout purity while still making the wide grid self-contained:
    # holdout/full cells exist only for the single already-selected row.
    selected_mask = (grid.stop_usd == selected["stop_usd"]) & (grid.target_usd == selected["target_usd"])
    grid["selected"] = selected_mask
    for period in ("holdout", "full"):
        for key, value in selected_results[period].items():
            column = f"{period}_{key}"
            grid[column] = np.nan
            grid.loc[selected_mask, column] = value
    grid.to_csv(OUT / "full_grid.csv", index=False)
    finalists.to_csv(OUT / "finalists.csv", index=False)
    cohort.to_csv(OUT / "frozen_one_year_entry_cohort.csv", index=False)

    summary = {
        "study": "Frozen causal VWAP-Z entries with independent absolute NVDA brackets",
        "period": {"start": str(START_DATE), "end": str(END_DATE), "sessions": len(session_days)},
        "splits": {
            "development": {"sessions": dev_end_ix, "start": str(session_days[0]), "end": str(session_days[dev_end_ix - 1])},
            "validation": {"sessions": val_end_ix - dev_end_ix, "start": str(val_start), "end": str(session_days[val_end_ix - 1])},
            "holdout": {"sessions": len(session_days) - val_end_ix, "start": str(hold_start), "end": str(session_days[-1])},
        },
        "frozen_entry_cohort": {
            "path": str(COHORT_PATH.relative_to(ROOT)), "sha256": hashlib.sha256(COHORT_PATH.read_bytes()).hexdigest(),
            "one_year_candidates": len(cohort), "period_candidate_counts": {k: len(v) for k, v in cohorts.items()},
        },
        "grid": {"stop_usd": list(DISTANCES), "target_usd": list(DISTANCES), "combinations": len(grid), "top_development": TOP_DEVELOPMENT},
        "selection": {
            "method": f"top {TOP_DEVELOPMENT} development by net Sharpe/net P&L, then max min(development, validation Sharpe), validation net P&L tie-break; holdout once",
            "dev_net_sharpe": float(chosen.dev_net_sharpe), "validation_net_sharpe": float(chosen.validation_net_sharpe),
            "robust_score": float(chosen.robust_score),
            "holdout_opened_after_selection": True,
            "no_confirmed_edge": bool(float(chosen.robust_score) <= 0 or selected_results["holdout"]["net_pnl"] <= 0),
        },
        "selected": selected,
        "selected_results": selected_results,
        "execution": {
            "raw_data": "Alpaca SIP raw 1-minute OHLC, exact synchronized QQQ/NVDA RTH sessions",
            "entry": "frozen causal VWAP-Z signal, next bar open",
            "exit": "absolute NVDA stop/target; same-bar stop-first; gap-through stop at adverse open; otherwise final RTH close",
            "convergence_exit": False, "position_notional_usd": SIZE, "starting_capital_usd": CAPITAL,
            "commission_usd_per_share_per_side": COMMISSION, "slippage_fraction_per_execution": SLIP,
        },
        "mark_to_market": mtm_stats,
        "reconciliation": {
            "each_selected_period_candidates_accounted": all(selected_results[k]["trades"] + selected_results[k]["skipped_overlaps"] == len(v) for k, v in cohorts.items()),
            "selected_split_rows_equal_full": split_rows == full_rows,
            "selected_split_net_equal_full": abs(split_net - full_net) <= 1e-8,
            "equity_final_equal_capital_plus_net": abs(float(mtm_eq.equity.iloc[-1]) - (CAPITAL + full_net)) <= 1e-8,
            "trade_costs_equal_metrics": abs(float(selected_trades["full"].costs.sum()) - selected_results["full"]["costs"]) <= 1e-8,
            "trade_net_equal_metrics": abs(float(selected_trades["full"].net_pnl.sum()) - full_net) <= 1e-8,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_value), encoding="utf-8")
    write_html(summary, mtm_eq, finalists)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_value))


if __name__ == "__main__":
    main()
