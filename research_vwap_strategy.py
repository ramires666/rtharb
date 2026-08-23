"""Causal session-VWAP fair-value research.

Hypothesis: during an RTH session, NVDA's deviation from a fair value built
from the *current cumulative* QQQ/NVDA VWAPs is mean reverting.  VWAP at bar
t uses only bars <= t; signals close at t and the imported simulator executes
at the next bar open.  No stop-loss or time-stop is used.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from rtharb.config import AppConfig
from rtharb.data.loader import DataLoader
from research_base_strategy import model_arrays, prepare_market, slice_arrays, simulate

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "research_output" / "vwap_strategy"


def _session_cum_vwap(frame: pd.DataFrame, common: pd.Index) -> np.ndarray:
    frame = frame.loc[common]
    # Typical-price VWAP; cumulative sums are causal and include the signal bar.
    typical = (frame["high"].to_numpy(float) + frame["low"].to_numpy(float) + frame["close"].to_numpy(float)) / 3.0
    volume = frame["volume"].to_numpy(float)
    day = common.date
    out = np.empty(len(common), dtype=float)
    for d in pd.unique(day):
        ix = np.flatnonzero(day == d)
        v = volume[ix]
        if np.any(v < 0):
            raise ValueError("Volume cannot be negative")
        cumulative_volume = np.cumsum(v)
        out[ix] = np.divide(
            np.cumsum(typical[ix] * v), cumulative_volume,
            out=np.full(len(ix), np.nan), where=cumulative_volume > 0,
        )
    return out


def vwap_arrays(lead: pd.DataFrame, target: pd.DataFrame, beta_days: int,
                window: int, warmup: int) -> dict:
    common = lead.index.intersection(target.index)
    lead, target = lead.loc[common], target.loc[common]
    base = prepare_market(lead, target)
    day = base["day"]
    n_days = len(base["unique_days"])
    lr = pd.Series(base["daily_lead_close"]).pct_change()
    tr = pd.Series(base["daily_target_close"]).pct_change()
    beta = (tr.rolling(beta_days, min_periods=beta_days).cov(lr) /
            lr.rolling(beta_days, min_periods=beta_days).var()).shift(1).clip(0.2, 4.0).fillna(1.5).to_numpy()
    lv = _session_cum_vwap(lead, common)
    tv = _session_cum_vwap(target, common)
    spread = target.close.to_numpy(float) / tv - 1.0 - beta[day] * (lead.close.to_numpy(float) / lv - 1.0)
    z = np.full(len(spread), np.nan)
    for start, end in zip(base["starts"], base["ends"]):
        x = spread[start:end + 1]
        count = np.minimum(np.arange(1, len(x) + 1), window)
        rolling_start = np.maximum(0, np.arange(len(x)) - window + 1)
        cs = np.r_[0.0, np.cumsum(x)]
        cs2 = np.r_[0.0, np.cumsum(x * x)]
        total = cs[np.arange(1, len(x) + 1)] - cs[rolling_start]
        total2 = cs2[np.arange(1, len(x) + 1)] - cs2[rolling_start]
        mean = total / count
        variance = np.divide(
            total2 - total * total / count, count - 1,
            out=np.full(len(x), np.nan), where=count > 1,
        )
        std = np.sqrt(np.maximum(variance, 0.0))
        values = np.divide(x - mean, std, out=np.full(len(x), np.nan), where=std > 1e-8)
        values[:warmup] = np.nan
        z[start:end + 1] = values
    fair = tv * (1.0 + beta[day] * (lead.close.to_numpy(float) / lv - 1.0))
    out = {k: v for k, v in base.items() if k not in {"r_lead", "r_target", "daily_lead_close", "daily_target_close", "starts", "ends"}}
    out["z"], out["abs_dev"] = z, target.close.to_numpy(float) - fair
    out["vwap_lead"], out["vwap_target"], out["fair_price"] = lv, tv, fair
    # Imported simulator has a fixed 15-bar gate. Shift its display bar so the
    # requested VWAP warm-up remains the only warm-up gate.
    out["bar"] = out["bar"] + 15 - warmup
    return out


def clean_result(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in {"trades_df", "daily_net", "daily_gross"}}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = AppConfig.load(str(ROOT / "configs" / "default_config.yaml"))
    lead, target = DataLoader(cfg.cache_dir, "alpaca", "sip").get_synchronized_pair("QQQ", "NVDA")
    n_days = len(pd.unique(lead.index.intersection(target.index).date))
    dev_end, val_end = n_days // 2, n_days * 3 // 4
    grid = []
    # Small staged grid: VWAP warm-up is the principal new hypothesis.
    # Stage 1 fixes the already-frozen 5-day beta and a 30-bar normalization
    # window while developing the new warm-up, Z, and hook dimensions. Stage 2
    # expands only ten finalists across 5/10/20-day betas and 15/30/60 windows.
    for warmup in [3, 5, 10, 15, 30]:
        for z_entry in [1.5, 2.0, 2.5]:
            for hook_delta, hook_timeout in [(0.0, 0), (0.10, 10), (0.15, 10), (0.25, 20)]:
                for lockout in [None, 4.0]:
                    grid.append({"warmup_bars": warmup, "window": 30, "beta_days": 5,
                                 "z_entry": z_entry, "hook_delta": hook_delta,
                                 "hook_timeout": hook_timeout, "exit_band": 0.0,
                                 "z_lockout": lockout, "direction": "both"})
    cache = {}
    def arrays(p):
        key = (p["beta_days"], p["warmup_bars"], p["window"])
        if key not in cache:
            cache[key] = vwap_arrays(lead, target, p["beta_days"], p["window"], p["warmup_bars"])
        return cache[key]
    def run_period(p, lo, hi, collect=False):
        a = slice_arrays(arrays(p), lo, hi)
        return simulate(a, {k: v for k, v in p.items() if k != "warmup_bars"}, collect=collect)

    dev = []
    for p in grid:
        r = run_period(p, 0, dev_end)
        dev.append({**p, **{k: r[k] for k in ["trades", "net_pnl", "net_return_pct", "net_sharpe", "net_sortino", "max_drawdown_pct", "win_rate_pct", "profit_factor"]}})
    dev_df = pd.DataFrame(dev).sort_values(["net_sharpe", "net_pnl"], ascending=False)
    dev_df.to_csv(OUT / "vwap_grid_dev.csv", index=False)
    finalists = dev_df[dev_df.trades >= 50].head(10)
    val = []
    for _, row in finalists.iterrows():
        for beta_days in [5, 10, 20]:
            for window in [15, 30, 60]:
                p = {k: row[k] for k in ["warmup_bars", "z_entry", "hook_delta", "hook_timeout", "exit_band", "z_lockout", "direction"]}
                p = {**p, "beta_days": beta_days, "window": window, "warmup_bars": int(p["warmup_bars"]), "hook_timeout": int(p["hook_timeout"]), "z_lockout": None if pd.isna(p["z_lockout"]) else p["z_lockout"]}
                dev_expanded = run_period(p, 0, dev_end)
                r = run_period(p, dev_end, val_end)
                val.append({**p, **{k: r[k] for k in ["trades", "net_pnl", "net_return_pct", "net_sharpe", "net_sortino", "max_drawdown_pct", "win_rate_pct", "profit_factor"]}, "dev_net_sharpe": dev_expanded["net_sharpe"]})
    val_df = pd.DataFrame(val)
    val_df["robust_score"] = np.minimum(val_df.net_sharpe, val_df.dev_net_sharpe)
    val_df.sort_values(["robust_score", "net_pnl"], ascending=False).to_csv(OUT / "vwap_finalists_validation.csv", index=False)
    chosen = val_df.sort_values(["robust_score", "net_pnl"], ascending=False).iloc[0]
    p = {k: chosen[k] for k in ["beta_days", "warmup_bars", "window", "z_entry", "hook_delta", "hook_timeout", "exit_band", "z_lockout", "direction"]}
    p = {**p, "beta_days": int(p["beta_days"]), "warmup_bars": int(p["warmup_bars"]), "window": int(p["window"]), "hook_timeout": int(p["hook_timeout"]), "z_lockout": None if pd.isna(p["z_lockout"]) else p["z_lockout"]}
    dev_r = run_period(p, 0, dev_end, True); val_r = run_period(p, dev_end, val_end, True); hold_r = run_period(p, val_end, n_days, True)
    hold_r["trades_df"].to_csv(OUT / "selected_holdout_trades.csv", index=False)
    baseline_params = json.loads((ROOT / "research_output" / "base_strategy_summary.json").read_text(encoding="utf-8"))["selected_parameters"]
    baseline_arrays = model_arrays(
        prepare_market(lead, target), baseline_params["beta_mode"],
        baseline_params["beta_days"], baseline_params["window"],
    )
    baseline = {
        "development": clean_result(simulate(slice_arrays(baseline_arrays, 0, dev_end), baseline_params)),
        "validation": clean_result(simulate(slice_arrays(baseline_arrays, dev_end, val_end), baseline_params)),
        "holdout": clean_result(simulate(slice_arrays(baseline_arrays, val_end, n_days), baseline_params)),
    }
    summary = {"hypothesis": "fair = cumulative session VWAP(NVDA) * (1 + beta * (QQQ/VWAP(QQQ)-1)); current-bar VWAP only; >=3 completed bars",
               "data": {"sessions": n_days, "dev": dev_end, "validation": val_end - dev_end, "holdout": n_days - val_end},
               "selected": p, "development": clean_result(dev_r), "validation": clean_result(val_r), "holdout": clean_result(hold_r),
               "baseline_parameters": baseline_params, "baseline": baseline,
               "tested_configurations": {"grid": len(grid), "validation": len(val_df)}}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT / "README.md").write_text("# Causal session-VWAP research\n\nVWAP is cumulative within each RTH session using typical price weighted by volume, and includes only bars through the current signal close. Fair value is `NVDA_VWAP * (1 + beta * (QQQ_close / QQQ_VWAP - 1))`. The first 3/5/10/15/30 completed bars are tested as warm-up. Signals use the existing hook, next-open execution, costs and forced EOD; no SL/time-stop. Development selects 10 signal finalists, expands them across beta/window choices, validation selects one, and holdout is evaluated once.\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
