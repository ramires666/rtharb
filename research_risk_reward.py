"""Frozen-entry cohort risk/reward study.

The base strategy supplies only the frozen entry cohort.  Each stop/reward
pair is then evaluated independently on raw NVDA 1m OHLC; the original
convergence exit is excluded and no re-entry is allowed after an early bracket
exit.
"""
from __future__ import annotations

import json
import math
import bisect
from pathlib import Path

import numpy as np
import pandas as pd

from rtharb.config import AppConfig
from rtharb.data.loader import DataLoader
from research_base_strategy import model_arrays, prepare_market, simulate
from research_vwap_strategy import vwap_arrays

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "research_output" / "risk_reward"
STOPS = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02]
RRS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
SLIP = 0.0002
COMMISSION = 0.0035
SIZE = 20_000.0
CAPITAL = 100_000.0


def _raw_target(lead: pd.DataFrame, target: pd.DataFrame):
    common = lead.index.intersection(target.index)
    t = target.loc[common]
    return {ts: (float(r.open), float(r.high), float(r.low), float(r.close))
            for ts, r in t.iterrows()}


def _one_trade(trade, raw, stop_pct, rr, raw_times):
    entry_ts = pd.Timestamp(trade.entry_time)
    # The frozen convergence exit is retained for audit only.  It is never an
    # exit candidate in this cohort study; brackets run until RTH session end.
    base_exit_ts = pd.Timestamp(trade.exit_time)
    direction = 1 if str(trade.direction).upper() == "LONG" else -1
    if entry_ts not in raw:
        raise KeyError(f"Missing raw entry bar {entry_ts}")
    entry_ref = raw[entry_ts][0]
    entry_eff = entry_ref * (1 + SLIP if direction == 1 else 1 - SLIP)
    shares = math.floor(SIZE / entry_eff)
    distance = entry_ref * stop_pct
    stop = entry_ref - distance if direction == 1 else entry_ref + distance
    target = entry_ref + distance * rr if direction == 1 else entry_ref - distance * rr
    session_times = raw_times
    timestamps = session_times[bisect.bisect_left(session_times, entry_ts):]
    eod_ts = session_times[-1]
    reason, raw_exit, exit_bar = "FORCED_EOD", raw[eod_ts][3], eod_ts
    for ts in timestamps:
        op, hi, lo, _ = raw[ts]
        # A stop hit on a gap fills at adverse open; otherwise at stop.
        stop_hit = (op <= stop or lo <= stop) if direction == 1 else (op >= stop or hi >= stop)
        target_hit = (hi >= target) if direction == 1 else (lo <= target)
        if stop_hit:
            raw_exit = op if (op <= stop if direction == 1 else op >= stop) else stop
            reason, exit_bar = "STOP", ts
            break
        if target_hit:
            raw_exit, reason, exit_bar = target, "TAKE_PROFIT_BRACKET", ts
            break
        # No convergence exit is allowed; continue to the session final bar.
    exit_eff = raw_exit * (1 - SLIP if direction == 1 else 1 + SLIP)
    gross = direction * (raw_exit - entry_ref) * shares
    slippage = abs(entry_eff - entry_ref) * shares + abs(exit_eff - raw_exit) * shares
    commissions = 2 * shares * COMMISSION
    net = gross - slippage - commissions
    return {
        "entry_time": entry_ts, "base_exit_time": base_exit_ts, "exit_time": exit_bar,
        "direction": "LONG" if direction == 1 else "SHORT", "entry_reference": entry_ref,
        "entry_price": entry_eff, "exit_price": exit_eff, "shares": shares,
        "stop_pct": stop_pct, "rr": rr, "stop_price": stop, "target_price": target,
        "exit_reason": reason, "gross_pnl": gross, "slippage": slippage,
        "commissions": commissions, "costs": slippage + commissions, "net_pnl": net,
        "duration_bars": int((pd.Timestamp(exit_bar) - entry_ts).total_seconds() // 60),
    }


def evaluate(cohort, raw, stop_pct, rr):
    raw_times = sorted(raw)
    session_map = {}
    for ts in raw_times:
        session_map.setdefault(ts.date(), []).append(ts)
    rows = []
    skipped = 0
    next_available = None
    # Fixed signal stream: do not admit a candidate while the prior bracket
    # remains open.  Later frozen entries become eligible after that exit.
    for trade in cohort.sort_values("entry_time").itertuples(index=False):
        entry_ts = pd.Timestamp(trade.entry_time)
        if next_available is not None and entry_ts <= next_available:
            skipped += 1
            continue
        result = _one_trade(trade, raw, stop_pct, rr, session_map[entry_ts.date()])
        rows.append(result)
        next_available = pd.Timestamp(result["exit_time"])
    df = pd.DataFrame(rows)
    if df.empty:
        return {"candidate_entries": len(cohort), "skipped_overlaps": skipped, "trades": 0, "gross_pnl": 0.0, "net_pnl": 0.0, "win_rate_pct": 0.0,
                "profit_factor": 0.0, "net_sharpe": 0.0, "stops": 0, "targets": 0,
            "forced_eod": 0}, df
    daily = df.assign(day=pd.to_datetime(df.exit_time).dt.date).groupby("day").net_pnl.sum()
    dr = daily / (CAPITAL + daily.cumsum().shift(1).fillna(0))
    sharpe = float(np.sqrt(252) * dr.mean() / dr.std(ddof=1)) if len(dr) > 1 and dr.std(ddof=1) else 0.0
    wins, losses = df[df.net_pnl > 0], df[df.net_pnl <= 0]
    pf = float(wins.net_pnl.sum() / abs(losses.net_pnl.sum())) if not losses.empty and losses.net_pnl.sum() else float("inf")
    return {"candidate_entries": len(cohort), "skipped_overlaps": skipped, "trades": len(df), "gross_pnl": float(df.gross_pnl.sum()), "net_pnl": float(df.net_pnl.sum()),
            "costs": float(df.costs.sum()), "win_rate_pct": float((df.net_pnl > 0).mean() * 100),
            "profit_factor": pf, "net_sharpe": sharpe,
            "stops": int((df.exit_reason == "STOP").sum()),
            "targets": int((df.exit_reason == "TAKE_PROFIT_BRACKET").sum()),
            "forced_eod": int((df.exit_reason == "FORCED_EOD").sum()),
            "avg_net_trade": float(df.net_pnl.mean())}, df


def run_rr_family(cohort, raw, session_days, prefix, out_dir):
    """Run the independent stop/TP grid for one frozen entry family."""
    d1, d2 = session_days[250], session_days[375]
    cohorts = {"development": cohort[pd.to_datetime(cohort.entry_time).dt.date < d1],
               "validation": cohort[(pd.to_datetime(cohort.entry_time).dt.date >= d1) & (pd.to_datetime(cohort.entry_time).dt.date < d2)],
               "holdout": cohort[pd.to_datetime(cohort.entry_time).dt.date >= d2], "full": cohort}
    grid = []
    for stop in STOPS:
        for rr in RRS:
            for period, c in cohorts.items():
                metrics, _ = evaluate(c, raw, stop, rr)
                grid.append({"stop_pct": stop, "rr": rr, "period": period, **metrics})
    grid_df = pd.DataFrame(grid)
    dev = grid_df[grid_df.period == "development"].sort_values(["net_sharpe", "net_pnl"], ascending=False)
    finalists = dev.head(10)[["stop_pct", "rr"]]
    val = grid_df.merge(finalists, on=["stop_pct", "rr"])
    dev_metrics = val[val.period == "development"][["stop_pct", "rr", "net_sharpe"]].rename(columns={"net_sharpe": "dev_net_sharpe"})
    val = val[val.period == "validation"].merge(dev_metrics, on=["stop_pct", "rr"])
    val["robust_score"] = np.minimum(val.net_sharpe, val.dev_net_sharpe)
    val = val.sort_values(["robust_score", "net_pnl"], ascending=False)
    selected = val.iloc[0]
    selected_row = {"stop_pct": float(selected.stop_pct), "rr": float(selected.rr)}
    grid_df.to_csv(out_dir / f"{prefix}_risk_reward_grid.csv", index=False)
    finalists.to_csv(out_dir / f"{prefix}_validation_candidates.csv", index=False)
    val.to_csv(out_dir / f"{prefix}_validation_selection.csv", index=False)
    cohort.to_csv(out_dir / f"{prefix}_entry_cohort.csv", index=False)
    selected_trades = {}
    for period, c in cohorts.items():
        m, t = evaluate(c, raw, selected_row["stop_pct"], selected_row["rr"])
        selected_trades[period] = m
        t.to_csv(out_dir / f"{prefix}_selected_{period}_trades.csv", index=False)
    reconciliation = {"candidate_entries": len(cohort), "grid_rows": len(grid_df),
                      "period_candidate_counts": {k: len(v) for k, v in cohorts.items()},
                      "selected_trade_counts": {k: v["trades"] for k, v in selected_trades.items()},
                      "cohort_accounted": all(v["trades"] + v["skipped_overlaps"] == len(c) for (k, v), c in zip(selected_trades.items(), cohorts.values())),
                      "overlapping_entries_skipped": True}
    return {"selected": selected_row,
            "selection": {"method": "top10 development, robust_score=min(dev_net_sharpe, validation_net_sharpe), tie net_pnl",
                           "dev_net_sharpe": float(selected.dev_net_sharpe), "validation_net_sharpe": float(selected.net_sharpe),
                           "robust_score": float(selected.robust_score),
                           "no_confirmed_edge": bool(float(selected.robust_score) <= 0 or selected_trades["holdout"]["net_pnl"] <= 0)},
            "selected_results": selected_trades, "reconciliation": reconciliation}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = AppConfig.load(str(ROOT / "configs" / "default_config.yaml"))
    lead, target = DataLoader(cfg.cache_dir, "alpaca", "sip").get_synchronized_pair("QQQ", "NVDA")
    raw = _raw_target(lead, target)
    base = prepare_market(lead, target)
    frozen = json.loads((ROOT / "research_output" / "base_strategy_summary.json").read_text(encoding="utf-8"))["selected_parameters"]
    arrays = model_arrays(base, frozen["beta_mode"], int(frozen["beta_days"]), int(frozen["window"]))
    p = {"beta_mode": frozen["beta_mode"], "beta_days": int(frozen["beta_days"]), "window": int(frozen["window"]),
         "z_entry": float(frozen["z_entry"]), "hook_delta": float(frozen["hook_delta"]),
         "hook_timeout": int(frozen["hook_timeout"]), "exit_band": float(frozen["exit_band"]),
         "z_lockout": frozen["z_lockout"], "direction": frozen["direction"]}
    base_result = simulate(arrays, p, collect=True)
    cohort = base_result["trades_df"].copy()
    # Match the frozen base study's session split, not trade-date quantiles.
    session_days = [pd.Timestamp(d).date() for d in arrays["unique_days"]]
    d1, d2 = session_days[250], session_days[375]
    cohorts = {"development": cohort[pd.to_datetime(cohort.entry_time).dt.date < d1],
               "validation": cohort[(pd.to_datetime(cohort.entry_time).dt.date >= d1) & (pd.to_datetime(cohort.entry_time).dt.date < d2)],
               "holdout": cohort[pd.to_datetime(cohort.entry_time).dt.date >= d2], "full": cohort}
    grid = []
    details = {}
    for stop in STOPS:
        for rr in RRS:
            key = f"s{stop:.4f}_rr{rr:g}"
            for period, c in cohorts.items():
                metrics, trades = evaluate(c, raw, stop, rr)
                row = {"stop_pct": stop, "rr": rr, "period": period, **metrics}
                grid.append(row)
                if period == "development":
                    details[key] = trades
    grid_df = pd.DataFrame(grid)
    dev = grid_df[grid_df.period == "development"].sort_values(["net_sharpe", "net_pnl"], ascending=False)
    finalists = dev.head(10)[["stop_pct", "rr"]]
    val = grid_df.merge(finalists, on=["stop_pct", "rr"])
    dev_metrics = val[val.period == "development"][["stop_pct", "rr", "net_sharpe"]].rename(columns={"net_sharpe": "dev_net_sharpe"})
    val = val[val.period == "validation"].merge(dev_metrics, on=["stop_pct", "rr"])
    val["robust_score"] = np.minimum(val.net_sharpe, val.dev_net_sharpe)
    val = val.sort_values(["robust_score", "net_pnl"], ascending=False)
    selected = val.iloc[0]
    selected_row = {"stop_pct": float(selected.stop_pct), "rr": float(selected.rr)}
    grid_df.to_csv(OUT / "risk_reward_grid.csv", index=False)
    finalists.to_csv(OUT / "validation_candidates.csv", index=False)
    cohort.to_csv(OUT / "base_entry_cohort.csv", index=False)
    selected_trades = {}
    for period, c in cohorts.items():
        m, t = evaluate(c, raw, selected_row["stop_pct"], selected_row["rr"])
        selected_trades[period] = m
        t.to_csv(OUT / f"selected_{period}_trades.csv", index=False)
    reconciliation = {"base_cohort_trades": len(cohort), "grid_rows": len(grid_df),
                      "period_trade_counts": {k: len(v) for k, v in cohorts.items()},
                      "selected_trade_counts": {k: v["trades"] for k, v in selected_trades.items()},
                      "cohort_accounted": all(v["trades"] + v["skipped_overlaps"] == len(c) for (k, v), c in zip(selected_trades.items(), cohorts.values())),
                      "overlapping_entries_skipped": True}
    no_confirmed_edge = bool(float(selected.robust_score) <= 0 or selected_trades["holdout"]["net_pnl"] <= 0)
    summary = {"frozen_parameters": frozen, "splits": {"development_sessions": 250, "validation_sessions": 125, "holdout_sessions": 126,
               "development_end": str(d1), "validation_end": str(d2)}, "selected": selected_row,
               "selection": {"method": "top10 development, robust_score=min(dev_net_sharpe, validation_net_sharpe), tie net_pnl",
                              "dev_net_sharpe": float(selected.dev_net_sharpe), "validation_net_sharpe": float(selected.net_sharpe),
                              "robust_score": float(selected.robust_score), "no_confirmed_edge": no_confirmed_edge},
               "selected_results": selected_trades, "reconciliation": reconciliation}
    vwap_summary_path = ROOT / "research_output" / "vwap_strategy" / "summary.json"
    vwap_selected = json.loads(vwap_summary_path.read_text(encoding="utf-8"))["selected"]
    vwap_a = vwap_arrays(lead, target, int(vwap_selected["beta_days"]), int(vwap_selected["window"]), int(vwap_selected["warmup_bars"]))
    vwap_p = {k: vwap_selected[k] for k in ["beta_days", "window", "z_entry", "hook_delta", "hook_timeout", "exit_band", "z_lockout", "direction"]}
    vwap_p.update(beta_mode="dynamic_rolling")
    vwap_cohort = simulate(vwap_a, vwap_p, collect=True)["trades_df"].copy()
    vwap_session_days = [pd.Timestamp(d).date() for d in vwap_a["unique_days"]]
    summary["vwap_z"] = {"frozen_parameters": vwap_selected,
                          "entry_source": "research_vwap_strategy.vwap_arrays + simulate(collect=True)",
                          **run_rr_family(vwap_cohort, raw, vwap_session_days, "vwap_z", OUT)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (OUT / "README.md").write_text("# Frozen-entry risk/reward cohort study\n\nClassic frozen Z/hook entries from `base_strategy_summary.json`; the original convergence exit is deliberately excluded. After each next-open entry, the only exits are stop, take-profit, or forced RTH session-end close (final raw 1m close). Brackets activate on the entry bar; same-bar stop wins over TP, gap-through stops fill at adverse open, and TP fills at its target. Candidate entries are consumed chronologically: if a bracket is still open, overlapping later frozen candidate events (including an event on the exit bar) are skipped; after early exit, only subsequent already-generated frozen events can enter. No new signals or re-entry are generated. Splits match the frozen 501-session study: 250 development, 125 validation, 126 holdout (boundaries 2025-08-22 and 2026-02-23). Top 10 development candidates are selected, then robust_score=min(development Sharpe, validation Sharpe) with net-PnL tie-break selects one. A non-positive robust score or non-positive holdout result is marked no_confirmed_edge. The evaluation function is reusable for a future VWAP-Z entry cohort.\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
