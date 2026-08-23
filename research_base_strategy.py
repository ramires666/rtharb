"""Wide, staged research of the base intraday convergence hypothesis.

No stop-loss and no time-stop are used here. Parameters are developed on the
first 50% of sessions, selected on the next 25%, and evaluated once on the
last 25% holdout.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rtharb.config import AppConfig
from rtharb.data.loader import DataLoader


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "research_output"


def prepare_market(lead: pd.DataFrame, target: pd.DataFrame) -> dict[str, Any]:
    common = lead.index.intersection(target.index)
    lead, target = lead.loc[common], target.loc[common]
    dates = pd.Index(common.date)
    codes, unique_days = pd.factorize(dates, sort=False)
    times = common.hour.to_numpy() * 60 + common.minute.to_numpy()
    last = np.r_[codes[1:] != codes[:-1], True]
    penultimate = np.r_[last[1:], True]
    first = np.r_[True, codes[1:] != codes[:-1]]
    starts = np.flatnonzero(first); ends = np.flatnonzero(last)
    p0_lead = np.repeat(lead.close.to_numpy(float)[starts], ends - starts + 1)
    p0_target = np.repeat(target.close.to_numpy(float)[starts], ends - starts + 1)
    lead_close = lead.close.to_numpy(float); target_close = target.close.to_numpy(float)
    return {
        "timestamp": common.to_numpy(), "day": codes.astype(np.int32),
        "unique_days": unique_days, "time": times.astype(np.int16),
        "bar": np.concatenate([np.arange(e-s+1) for s,e in zip(starts,ends)]).astype(np.int16),
        "open": target.open.to_numpy(float), "close": target_close,
        "p0_target": p0_target,
        "r_lead": lead_close / p0_lead - 1.0,
        "r_target": target_close / p0_target - 1.0,
        "daily_lead_close": lead_close[ends], "daily_target_close": target_close[ends],
        "starts": starts, "ends": ends,
        "last": last, "penultimate": penultimate,
    }


def model_arrays(base: dict[str, Any], beta_mode: str, beta_days: int, window: int) -> dict[str, Any]:
    n_days = len(base["unique_days"])
    if beta_mode.startswith("fixed_"):
        beta_day = np.full(n_days, float(beta_mode.removeprefix("fixed_")))
    else:
        lr = pd.Series(base["daily_lead_close"]).pct_change()
        tr = pd.Series(base["daily_target_close"]).pct_change()
        beta_day = (tr.rolling(beta_days, min_periods=beta_days).cov(lr) /
                    lr.rolling(beta_days, min_periods=beta_days).var()).shift(1).clip(0.2,4.0).fillna(1.5).to_numpy()
    spread = base["r_target"] - beta_day[base["day"]] * base["r_lead"]
    z = np.full(len(spread), np.nan)
    for start, end in zip(base["starts"], base["ends"]):
        x = spread[start:end+1]
        count = np.minimum(np.arange(1, len(x)+1), window)
        starts = np.maximum(0, np.arange(len(x)) - window + 1)
        cs = np.r_[0.0, np.cumsum(x)]; cs2 = np.r_[0.0, np.cumsum(x*x)]
        total = cs[np.arange(1,len(x)+1)] - cs[starts]
        total2 = cs2[np.arange(1,len(x)+1)] - cs2[starts]
        mean = total / count
        var = np.divide(total2 - total*total/count, count-1,
                        out=np.full(len(x), np.nan), where=count>1)
        std = np.sqrt(np.maximum(var, 0.0))
        values = np.divide(x-mean, std, out=np.full(len(x), np.nan), where=std>1e-8)
        values[:15] = np.nan
        z[start:end+1] = values
    out = {k:v for k,v in base.items() if k not in {"r_lead","r_target","daily_lead_close","daily_target_close","starts","ends"}}
    out["z"] = z
    fair_price = base["p0_target"] * (1.0 + beta_day[base["day"]] * base["r_lead"])
    out["abs_dev"] = base["close"] - fair_price
    return out


def slice_arrays(a: dict[str, Any], first_day: int, last_day: int) -> dict[str, Any]:
    mask = (a["day"] >= first_day) & (a["day"] < last_day)
    idx = np.flatnonzero(mask)
    out = {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(a["day"]) else v)
           for k, v in a.items()}
    out["day"] = out["day"] - first_day
    out["unique_days"] = a["unique_days"][first_day:last_day]
    return out


def _ratios(daily: np.ndarray, initial: float) -> tuple[float, float]:
    equity_before = initial + np.r_[0.0, np.cumsum(daily[:-1])]
    returns = np.divide(daily, equity_before, out=np.zeros_like(daily), where=equity_before != 0)
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return 0.0, 0.0
    sharpe = math.sqrt(252) * returns.mean() / returns.std(ddof=1)
    downside = np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2))
    sortino = math.sqrt(252) * returns.mean() / downside if downside else 0.0
    return float(sharpe), float(sortino)


def simulate(a: dict[str, Any], p: dict[str, Any], collect: bool = False) -> dict[str, Any]:
    initial, size, commission, slip = 100_000.0, 20_000.0, 0.0035, 0.0002
    n, n_days = len(a["z"]), len(a["unique_days"])
    daily_net, daily_gross = np.zeros(n_days), np.zeros(n_days)
    trade_net, trade_gross, durations, records = [], [], [], []
    long_net = short_net = 0.0
    cash = initial
    peak, max_dd = initial, 0.0
    position = armed = armed_age = 0
    pending = 0  # -2 close, -1 short, +1 long
    pending_z = 0.0
    entry_ref = entry_effective = shares = entry_z = 0.0
    entry_i = -1
    locked = False

    def close_trade(i: int, raw_exit: float, reason: str, exit_z: float):
        nonlocal position, cash, long_net, short_net
        effective_exit = raw_exit * (1.0 - slip if position == 1 else 1.0 + slip)
        gross = position * (raw_exit - entry_ref) * shares
        slip_cost = (abs(entry_effective - entry_ref) + abs(effective_exit - raw_exit)) * shares
        costs = 2.0 * shares * commission + slip_cost
        net = gross - costs
        day = int(a["day"][i])
        daily_net[day] += net
        daily_gross[day] += gross
        trade_net.append(net); trade_gross.append(gross); durations.append(i - entry_i)
        if position == 1: long_net += net
        else: short_net += net
        if collect:
            records.append({
                "entry_time": pd.Timestamp(a["timestamp"][entry_i]),
                "exit_time": pd.Timestamp(a["timestamp"][i]),
                "direction": "LONG" if position == 1 else "SHORT",
                "entry_price": entry_effective, "exit_price": effective_exit,
                "shares": int(shares), "entry_z": entry_z, "exit_z": exit_z,
                "duration_bars": i - entry_i, "gross_pnl": gross,
                "costs": costs, "net_pnl": net, "exit_reason": reason,
            })
        cash += net
        position = 0

    current_day = -1
    for i in range(n):
        day = int(a["day"][i]); z = float(a["z"][i]); raw_open = float(a["open"][i])
        if day != current_day:
            current_day, armed, armed_age, locked = day, 0, 0, False

        if pending:
            if pending == -2 and position:
                close_trade(i, raw_open, "TAKE_PROFIT", z)
            elif pending in (-1, 1) and not position:
                position = pending
                entry_ref = raw_open
                entry_effective = raw_open * (1.0 + slip if position == 1 else 1.0 - slip)
                shares = math.floor(size / entry_effective)
                entry_z, entry_i = pending_z, i
            pending = 0

        if math.isfinite(z):
            lockout = p["z_lockout"]
            if lockout is not None and abs(z) >= lockout:
                locked, armed = True, 0

            if a["time"][i] >= 15 * 60 + 55 or a["penultimate"][i]:
                if position: pending, pending_z = -2, z
                armed = 0
            elif position == 1:
                if z >= p["exit_band"]: pending, pending_z = -2, z
            elif position == -1:
                if z <= -p["exit_band"]: pending, pending_z = -2, z
            elif not locked and a["bar"][i] >= 15:
                allow_long = p["direction"] in ("both", "long")
                allow_short = p["direction"] in ("both", "short")
                hook = p["hook_delta"]
                entry_mode = p.get("entry_mode", "z_only")
                abs_threshold = p.get("abs_threshold_usd")
                anchor_filter = bool(p.get("anchor_filter", False))
                abs_dev = float(a["abs_dev"][i])
                price = float(a["close"][i])
                anchor_price = float(a["p0_target"][i])
                def anchor_ok(direction: int) -> bool:
                    return (not anchor_filter) or (direction == 1 and price < anchor_price) or (direction == -1 and price > anchor_price)
                if armed == 0:
                    z_long, z_short = z <= -p["z_entry"], z >= p["z_entry"]
                    abs_long = abs_threshold is not None and abs_dev <= -abs_threshold
                    abs_short = abs_threshold is not None and abs_dev >= abs_threshold
                    if entry_mode == "z_only":
                        long_hit, short_hit = z_long, z_short
                    elif entry_mode == "abs_only":
                        long_hit, short_hit = abs_long, abs_short
                    elif abs_long or abs_short:
                        long_hit, short_hit = abs_long, abs_short
                    else:
                        long_hit, short_hit = z_long, z_short
                    if long_hit and allow_long and anchor_ok(1):
                        if hook == 0: pending, pending_z = 1, z
                        else: armed, armed_age, extreme_z = 1, 0, z
                    elif short_hit and allow_short and anchor_ok(-1):
                        if hook == 0: pending, pending_z = -1, z
                        else: armed, armed_age, extreme_z = -1, 0, z
                elif armed == 1:
                    armed_age += 1; extreme_z = min(extreme_z, z)
                    if not anchor_ok(1): armed = 0
                    elif z - extreme_z >= hook: pending, pending_z, armed = 1, z, 0
                    elif armed_age >= p["hook_timeout"]: armed = 0
                else:
                    armed_age += 1; extreme_z = max(extreme_z, z)
                    if not anchor_ok(-1): armed = 0
                    elif extreme_z - z >= hook: pending, pending_z, armed = -1, z, 0
                    elif armed_age >= p["hook_timeout"]: armed = 0

        if a["last"][i]:
            if position:
                close_trade(i, float(a["close"][i]), "FORCED_EOD", z)
            pending, armed = 0, 0

        unrealized = position * (float(a["close"][i]) - entry_effective) * shares if position else 0.0
        equity = cash + unrealized
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    nets = np.asarray(trade_net); grosses = np.asarray(trade_gross)
    wins, losses = nets[nets > 0], nets[nets <= 0]
    gross_wins, gross_losses = grosses[grosses > 0], grosses[grosses <= 0]
    net_sharpe, net_sortino = _ratios(daily_net, initial)
    gross_sharpe, _ = _ratios(daily_gross, initial)
    result = {
        **p, "trades": len(nets), "net_pnl": float(nets.sum()),
        "gross_pnl": float(grosses.sum()), "costs": float(grosses.sum() - nets.sum()),
        "net_return_pct": float(nets.sum() / initial * 100),
        "gross_return_pct": float(grosses.sum() / initial * 100),
        "net_sharpe": net_sharpe, "gross_sharpe": gross_sharpe,
        "net_sortino": net_sortino, "max_drawdown_pct": float(max_dd / initial * 100),
        "win_rate_pct": float((nets > 0).mean() * 100) if len(nets) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() else 0.0,
        "gross_profit_factor": float(gross_wins.sum() / abs(gross_losses.sum())) if len(gross_losses) and gross_losses.sum() else 0.0,
        "avg_net_trade": float(nets.mean()) if len(nets) else 0.0,
        "median_net_trade": float(np.median(nets)) if len(nets) else 0.0,
        "avg_duration": float(np.mean(durations)) if durations else 0.0,
        "long_net_pnl": long_net, "short_net_pnl": short_net,
    }
    if collect:
        result["trades_df"] = pd.DataFrame(records)
        result["daily_net"] = daily_net
        result["daily_gross"] = daily_gross
    return result


def parameter_key(p):
    return tuple((k, p[k]) for k in ["beta_mode", "beta_days", "window", "z_entry",
                                               "hook_delta", "hook_timeout", "exit_band",
                                               "z_lockout", "direction"])


def main():
    OUT.mkdir(exist_ok=True)
    cfg = AppConfig.load(str(ROOT / "configs" / "default_config.yaml"))
    lead, target = DataLoader(cfg.cache_dir, "alpaca", "sip").get_synchronized_pair("QQQ", "NVDA")
    cache: dict[tuple, dict[str, Any]] = {}

    raw_base = prepare_market(lead, target)
    def arrays_for(beta_mode: str, beta_days: int, window: int):
        key = (beta_mode, beta_days, window)
        if key not in cache:
            cache[key] = model_arrays(raw_base, beta_mode, beta_days, window)
        return cache[key]

    base = arrays_for("dynamic_rolling", 10, 30)
    n_days = len(base["unique_days"])
    dev_end, val_end = n_days // 2, n_days * 3 // 4
    dev_base = slice_arrays(base, 0, dev_end)
    print(f"Data ready: {n_days} sessions; dev={dev_end}, validation={val_end-dev_end}, holdout={n_days-val_end}", flush=True)

    # Stage A: signal mechanics on the MD default model.
    hooks = [(0.0, 0)] + [(d, t) for d in [0.05, 0.10, 0.15, 0.25, 0.35] for t in [5, 10, 20]]
    stage_a = []
    print("Stage A: Z / hook / timeout / 4-sigma lockout", flush=True)
    for z_entry in [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]:
        for hook_delta, timeout in hooks:
            for lockout in [None, 4.0]:
                p = {"beta_mode": "dynamic_rolling", "beta_days": 10, "window": 30,
                     "z_entry": z_entry, "hook_delta": hook_delta, "hook_timeout": timeout,
                     "exit_band": 0.0, "z_lockout": lockout, "direction": "both"}
                stage_a.append(simulate(dev_base, p))
    a_df = pd.DataFrame(stage_a).sort_values(["net_sharpe", "net_pnl"], ascending=False)
    a_df.to_csv(OUT / "stage_a_signal_grid.csv", index=False)
    print(f"Stage A complete: {len(a_df)} configurations", flush=True)

    # Stage B: beta/window structure for the strongest 10 signal candidates.
    top_signals = a_df[a_df.trades >= 100].head(10)
    beta_specs = [("dynamic_rolling", d) for d in [5, 10, 20, 30]] + [(f"fixed_{b}", 10) for b in [1.0, 1.25, 1.5, 1.75, 2.0]]
    stage_b = []
    print("Stage B: beta model / rolling window", flush=True)
    for _, sig in top_signals.iterrows():
        for beta_mode, beta_days in beta_specs:
            for window in [15, 30, 60, 120]:
                full_a = arrays_for(beta_mode, beta_days, window)
                dev_a = slice_arrays(full_a, 0, dev_end)
                p = {"beta_mode": beta_mode, "beta_days": beta_days, "window": window,
                     "z_entry": float(sig.z_entry), "hook_delta": float(sig.hook_delta),
                     "hook_timeout": int(sig.hook_timeout), "exit_band": 0.0,
                     "z_lockout": sig.z_lockout if pd.notna(sig.z_lockout) else None,
                     "direction": "both"}
                stage_b.append(simulate(dev_a, p))
    b_df = pd.DataFrame(stage_b).drop_duplicates(["beta_mode","beta_days","window","z_entry","hook_delta","hook_timeout","z_lockout"])
    b_df = b_df.sort_values(["net_sharpe", "net_pnl"], ascending=False)
    b_df.to_csv(OUT / "stage_b_model_grid.csv", index=False)
    print(f"Stage B complete: {len(b_df)} configurations", flush=True)

    # Stage C: convergence target and structural-dislocation lockout.
    stage_c = []
    print("Stage C: exit band / dislocation lockout", flush=True)
    for _, candidate in b_df[b_df.trades >= 100].head(20).iterrows():
        full_a = arrays_for(candidate.beta_mode, int(candidate.beta_days), int(candidate.window))
        dev_a = slice_arrays(full_a, 0, dev_end)
        for exit_band in [0.0, 0.25, 0.5]:
            for lockout in [None, 3.5, 4.0, 4.5]:
                p = {"beta_mode": candidate.beta_mode, "beta_days": int(candidate.beta_days),
                     "window": int(candidate.window), "z_entry": float(candidate.z_entry),
                     "hook_delta": float(candidate.hook_delta), "hook_timeout": int(candidate.hook_timeout),
                     "exit_band": exit_band, "z_lockout": lockout, "direction": "both"}
                stage_c.append(simulate(dev_a, p))
    c_df = pd.DataFrame(stage_c).drop_duplicates(["beta_mode","beta_days","window","z_entry","hook_delta","hook_timeout","exit_band","z_lockout"])
    c_df = c_df.sort_values(["net_sharpe", "net_pnl"], ascending=False)
    c_df.to_csv(OUT / "stage_c_exit_lockout_grid.csv", index=False)
    print(f"Stage C complete: {len(c_df)} configurations", flush=True)

    # Validation sees only the top 50 frozen development candidates.
    validation = []
    print("Validation: 50 frozen finalists", flush=True)
    for _, candidate in c_df[c_df.trades >= 100].head(50).iterrows():
        full_a = arrays_for(candidate.beta_mode, int(candidate.beta_days), int(candidate.window))
        val_a = slice_arrays(full_a, dev_end, val_end)
        p = {k: candidate[k] for k in ["beta_mode","beta_days","window","z_entry","hook_delta","hook_timeout","exit_band","z_lockout","direction"]}
        p["beta_days"], p["window"], p["hook_timeout"] = int(p["beta_days"]), int(p["window"]), int(p["hook_timeout"])
        p["z_lockout"] = p["z_lockout"] if pd.notna(p["z_lockout"]) else None
        val = simulate(val_a, p)
        for name in ["net_pnl","gross_pnl","costs","net_return_pct","gross_return_pct","net_sharpe","gross_sharpe","profit_factor","trades"]:
            val[f"dev_{name}"] = candidate[name]
        validation.append(val)
    v_df = pd.DataFrame(validation)
    v_df["robust_score"] = np.minimum(v_df.net_sharpe, v_df.dev_net_sharpe)
    v_df = v_df.sort_values(["robust_score", "net_pnl"], ascending=False)
    v_df.to_csv(OUT / "validation_finalists.csv", index=False)
    print("Validation complete; running the single selected holdout", flush=True)
    selected = v_df.iloc[0]
    p = {k: selected[k] for k in ["beta_mode","beta_days","window","z_entry","hook_delta","hook_timeout","exit_band","z_lockout","direction"]}
    p["beta_days"], p["window"], p["hook_timeout"] = int(p["beta_days"]), int(p["window"]), int(p["hook_timeout"])
    p["z_lockout"] = p["z_lockout"] if pd.notna(p["z_lockout"]) else None

    selected_arrays = arrays_for(p["beta_mode"], p["beta_days"], p["window"])
    dev_result = simulate(slice_arrays(selected_arrays, 0, dev_end), p, collect=True)
    val_result = simulate(slice_arrays(selected_arrays, dev_end, val_end), p, collect=True)
    hold_result = simulate(slice_arrays(selected_arrays, val_end, n_days), p, collect=True)
    full_result = simulate(selected_arrays, p, collect=True)
    hold_result["trades_df"].to_csv(OUT / "selected_holdout_trades.csv", index=False)
    full_result["trades_df"].to_csv(OUT / "selected_full_trades.csv", index=False)

    # Diagnostics: direction, entry time, month, and bootstrap CI of daily net PnL.
    trades = hold_result["trades_df"].copy()
    trades["entry_time"] = pd.to_datetime(trades.entry_time)
    trades["entry_hour"] = trades.entry_time.dt.hour
    trades["month"] = trades.entry_time.dt.to_period("M").astype(str)
    by_direction = trades.groupby("direction").agg(trades=("net_pnl","size"), gross_pnl=("gross_pnl","sum"), net_pnl=("net_pnl","sum"), avg_net=("net_pnl","mean"))
    by_hour = trades.groupby("entry_hour").agg(trades=("net_pnl","size"), net_pnl=("net_pnl","sum"), avg_net=("net_pnl","mean"))
    by_month = trades.groupby("month").agg(trades=("net_pnl","size"), net_pnl=("net_pnl","sum"))
    by_direction.to_csv(OUT / "holdout_by_direction.csv")
    by_hour.to_csv(OUT / "holdout_by_entry_hour.csv")
    by_month.to_csv(OUT / "holdout_by_month.csv")
    rng = np.random.default_rng(20260823)
    daily = hold_result["daily_net"]
    boot = rng.choice(daily, size=(10_000, len(daily)), replace=True).mean(axis=1)
    ci = np.quantile(boot, [0.025, 0.975]).tolist()

    def clean(r):
        return {k: v for k, v in r.items() if k not in {"trades_df","daily_net","daily_gross"}}
    summary = {
        "data": {"feed":"Alpaca SIP", "sessions":n_days, "bars":len(base["z"]),
                 "development_sessions":dev_end, "validation_sessions":val_end-dev_end,
                 "holdout_sessions":n_days-val_end},
        "selected_parameters": p,
        "development": clean(dev_result), "validation": clean(val_result),
        "holdout": clean(hold_result), "full_descriptive": clean(full_result),
        "holdout_daily_net_pnl_mean_95pct_bootstrap_ci": ci,
        "tested_configurations": {"stage_a":len(a_df),"stage_b":len(b_df),"stage_c":len(c_df),"validation":len(v_df)},
    }
    (OUT / "base_strategy_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def row(name, r):
        return f"<tr><td>{name}</td><td>{r['trades']}</td><td>{r['gross_return_pct']:.2f}%</td><td>{r['net_return_pct']:.2f}%</td><td>{r['net_sharpe']:.2f}</td><td>{r['profit_factor']:.2f}</td><td>{r['max_drawdown_pct']:.2f}%</td></tr>"
    edge = hold_result["net_pnl"] > 0 and hold_result["net_sharpe"] > 0 and ci[0] > 0
    report = f'''<!doctype html><meta charset="utf-8"><title>Исследование базового intraday edge</title>
<style>body{{font:16px system-ui;max-width:1150px;margin:auto;padding:30px;background:#0b0e14;color:#e6edf3}}td,th{{padding:8px;border:1px solid #344054}}table{{border-collapse:collapse}}code{{color:#36c5f0}}</style>
<h1>Базовый intraday QQQ→NVDA: поиск edge</h1><p>Без stop-loss и time-stop. Проверено {len(a_df)+len(b_df)+len(c_df):,} development-конфигураций; 50 finalists проверены на validation; одна конфигурация заморожена до holdout.</p>
<p><b>Вердикт: {'EDGE ПОДТВЕРЖДЁН' if edge else 'EDGE НЕ ПОДТВЕРЖДЁН'}.</b> Критерий: положительный holdout net PnL, Sharpe и нижняя граница bootstrap CI среднего дневного PnL.</p>
<p>Параметры: <code>{json.dumps(p, ensure_ascii=False)}</code></p>
<table><tr><th>Период</th><th>Сделки</th><th>Gross return</th><th>Net return</th><th>Net Sharpe</th><th>PF</th><th>Max DD</th></tr>{row('Development',dev_result)}{row('Validation',val_result)}{row('Holdout',hold_result)}{row('Full, описательно',full_result)}</table>
<p>Holdout 95% bootstrap CI среднего дневного net PnL: <code>${ci[0]:.2f} … ${ci[1]:.2f}</code>.</p>
<h2>Что лежит рядом</h2><ul><li>stage_a_signal_grid.csv — Z/hook/timeout</li><li>stage_b_model_grid.csv — beta/window</li><li>stage_c_exit_lockout_grid.csv — exit/lockout</li><li>validation_finalists.csv — финалисты без доступа к holdout</li><li>selected_holdout_trades.csv — все OOS сделки</li><li>holdout_by_*.csv — направления, часы входа, месяцы</li></ul>'''
    (OUT / "BASE_STRATEGY_REPORT.html").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
