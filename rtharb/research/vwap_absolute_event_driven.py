"""Independent event-driven VWAP-Z backtest with absolute NVDA brackets.

Unlike the archived ``old/frozen_vwap_absolute/research.py``, this simulator does not reuse a
convergence-strategy trade cohort.  It walks every synchronized raw SIP minute
and generates a fresh causal VWAP-Z signal whenever it is flat.  Signals are
observed at bar close and filled at the next bar open; while a bracket is open,
all signals are ignored.  After a bracket exit, the same raw stream can create
another signal and therefore another trade.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rtharb.config import AppConfig
from rtharb.data.loader import DataLoader
from rtharb.research.risk_reward import CAPITAL, COMMISSION, SIZE, SLIP
from rtharb.research.vwap_strategy import vwap_arrays


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_output" / "vwap_absolute_event_driven"
START_DATE = pd.Timestamp("2025-08-22").date()
END_DATE = pd.Timestamp("2026-08-21").date()
DISTANCES = tuple(round(i * 0.25, 2) for i in range(1, 13))
TOP_DEVELOPMENT = 10
ENTRY_Z = 2.5
BETA_DAYS = 5
WINDOW = 60
WARMUP = 30


def _json(value: Any):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def market() -> tuple[dict[str, np.ndarray], list[object]]:
    """Load exact synchronized Alpaca SIP bars and causal VWAP arrays."""
    cfg = AppConfig.load(str(ROOT / "configs" / "default_config.yaml"))
    lead, target = DataLoader(cfg.cache_dir, "alpaca", "sip").get_synchronized_pair("QQQ", "NVDA")
    full_common = lead.index.intersection(target.index)
    lead, target = lead.loc[full_common], target.loc[full_common]
    # Critical causality detail: compute the rolling 5-session beta on all
    # available earlier history, then slice the requested year.  Computing it
    # after slicing would incorrectly replace the first five betas with 1.5.
    full = vwap_arrays(lead, target, BETA_DAYS, WINDOW, WARMUP)
    mask = np.fromiter((START_DATE <= ts.date() <= END_DATE for ts in full_common), bool, len(full_common))
    old_day0 = int(full["day"][np.flatnonzero(mask)[0]])
    old_day1 = int(full["day"][np.flatnonzero(mask)[-1]]) + 1
    a = {k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(full_common) else v)
         for k, v in full.items()}
    a["day"] = a["day"] - old_day0
    a["unique_days"] = full["unique_days"][old_day0:old_day1]
    year_target = target.loc[full_common[mask]]
    # Every vector is one-to-one with the exact raw target bars.  No
    # resampling, procedural fill, or interpolation is performed.
    a["high"] = year_target.high.to_numpy(float)
    a["low"] = year_target.low.to_numpy(float)
    a["volume_target"] = year_target.volume.to_numpy(float)
    days = list(a["unique_days"])
    if len(days) != 251 or pd.Timestamp(days[0]).date() != START_DATE or pd.Timestamp(days[-1]).date() != END_DATE:
        raise AssertionError(f"Expected completed 251-session year, got {len(days)}: {days[:1]} .. {days[-1:]}")
    if len(a["timestamp"]) != len(year_target) or len(year_target) < 97_000:
        raise AssertionError(f"Unexpected synchronized raw bar count: {len(year_target)}")
    return a, days


def _ratios(daily: np.ndarray) -> tuple[float, float]:
    prior = CAPITAL + np.r_[0.0, np.cumsum(daily[:-1])]
    returns = np.divide(daily, prior, out=np.zeros_like(daily), where=prior != 0)
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return 0.0, 0.0
    sharpe = math.sqrt(252) * returns.mean() / returns.std(ddof=1)
    downside = np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2))
    sortino = math.sqrt(252) * returns.mean() / downside if downside else 0.0
    return float(sharpe), float(sortino)


def simulate(a: dict[str, np.ndarray], first_day: int, last_day: int,
             stop_usd: float, target_usd: float, collect: bool = False) -> dict[str, Any]:
    """Walk raw bars for [first_day,last_day), regenerating entries while flat."""
    idx = np.flatnonzero((a["day"] >= first_day) & (a["day"] < last_day))
    n_days = last_day - first_day
    daily_net = np.zeros(n_days, dtype=float)
    daily_gross = np.zeros(n_days, dtype=float)
    trades: list[dict[str, Any]] = []
    mtm_rows: list[dict[str, Any]] = []
    cash = CAPITAL
    position = 0
    pending = 0
    pending_z = math.nan
    pending_signal_i = -1
    entry_i = -1
    entry_ref = entry_eff = 0.0
    shares = 0
    stop_price = target_price = 0.0
    entry_commission = 0.0
    signal_count = ignored_open_signals = 0
    stops = targets = forced_eod = 0
    commissions_total = slippage_total = 0.0
    shares_total = 0
    nets: list[float] = []
    grosses: list[float] = []
    durations: list[int] = []
    max_dd = 0.0
    peak = CAPITAL

    def close_trade(i: int, raw_exit: float, reason: str) -> None:
        nonlocal position, cash, stops, targets, forced_eod
        nonlocal commissions_total, slippage_total, shares_total
        exit_eff = raw_exit * (1.0 - SLIP if position == 1 else 1.0 + SLIP)
        gross = position * (raw_exit - entry_ref) * shares
        slippage = (abs(entry_eff - entry_ref) + abs(exit_eff - raw_exit)) * shares
        commissions = 2.0 * shares * COMMISSION
        costs = slippage + commissions
        commissions_total += commissions
        slippage_total += slippage
        shares_total += shares
        if reason == "STOP":
            stops += 1
        elif reason == "TAKE_PROFIT_BRACKET":
            targets += 1
        elif reason == "FORCED_EOD":
            forced_eod += 1
        net = gross - costs
        day_local = int(a["day"][i]) - first_day
        daily_net[day_local] += net
        daily_gross[day_local] += gross
        nets.append(net); grosses.append(gross); durations.append(i - entry_i)
        if collect:
            trades.append({
                "signal_time": pd.Timestamp(a["timestamp"][pending_signal_i]),
                "entry_time": pd.Timestamp(a["timestamp"][entry_i]),
                "exit_time": pd.Timestamp(a["timestamp"][i]),
                "direction": "LONG" if position == 1 else "SHORT",
                "entry_z": float(pending_z),
                "signal_nvda_close": float(a["close"][pending_signal_i]),
                "signal_nvda_vwap": float(a["vwap_target"][pending_signal_i]),
                "signal_qqq_vwap": float(a["vwap_lead"][pending_signal_i]),
                "signal_fair_nvda": float(a["fair_price"][pending_signal_i]),
                "entry_reference": entry_ref, "entry_price": entry_eff,
                "exit_reference": raw_exit, "exit_price": exit_eff,
                "shares": shares, "stop_usd_per_share": stop_usd,
                "target_usd_per_share": target_usd,
                "gross_risk_usd": stop_usd * shares,
                "gross_reward_usd": target_usd * shares,
                "risk_reward_ratio": target_usd / stop_usd,
                "stop_price": stop_price, "target_price": target_price,
                "exit_reason": reason, "duration_bars": i - entry_i,
                "gross_pnl": gross, "slippage": slippage,
                "commissions": commissions, "costs": costs, "net_pnl": net,
            })
        cash += net
        position = 0

    for i in idx:
        day = int(a["day"][i])
        is_first = i == idx[0] or int(a["day"][i - 1]) != day
        is_last = bool(a["last"][i])
        if is_first and (position or pending):
            raise AssertionError("Position/pending signal leaked across RTH sessions")

        # The close-t signal is executed at next bar's raw open.
        if pending:
            position = pending
            pending = 0
            entry_i = i
            entry_ref = float(a["open"][i])
            entry_eff = entry_ref * (1.0 + SLIP if position == 1 else 1.0 - SLIP)
            shares = math.floor(SIZE / entry_eff)
            if shares <= 0:
                raise AssertionError("Invalid position size")
            stop_price = entry_ref - stop_usd if position == 1 else entry_ref + stop_usd
            target_price = entry_ref + target_usd if position == 1 else entry_ref - target_usd
            entry_commission = shares * COMMISSION

        # Brackets are active on the entry bar.  Conservative same-bar rule:
        # stop wins; a gap through a stop fills at the adverse raw open.
        if position:
            op, hi, lo = float(a["open"][i]), float(a["high"][i]), float(a["low"][i])
            stop_hit = (op <= stop_price or lo <= stop_price) if position == 1 else (op >= stop_price or hi >= stop_price)
            target_hit = hi >= target_price if position == 1 else lo <= target_price
            if stop_hit:
                gap = op <= stop_price if position == 1 else op >= stop_price
                close_trade(i, op if gap else stop_price, "STOP")
            elif target_hit:
                close_trade(i, target_price, "TAKE_PROFIT_BRACKET")
            elif is_last:
                close_trade(i, float(a["close"][i]), "FORCED_EOD")

        z = float(a["z"][i])
        # Once the bracket exited above, this close is eligible to generate a
        # fresh event.  No event is emitted on the final session bar because it
        # has no same-session next-open execution.
        if math.isfinite(z) and not is_last:
            hit = 1 if z <= -ENTRY_Z else (-1 if z >= ENTRY_Z else 0)
            if hit:
                if position:
                    ignored_open_signals += 1
                else:
                    pending = hit
                    pending_z = z
                    pending_signal_i = i
                    signal_count += 1

        if is_last and pending:
            raise AssertionError("Final-bar signal should never become pending")

        if collect:
            if position:
                equity = cash - entry_commission + position * (float(a["close"][i]) - entry_eff) * shares
            else:
                equity = cash
            peak = max(peak, equity)
            max_dd = max(max_dd, peak - equity)
            mtm_rows.append({"timestamp": pd.Timestamp(a["timestamp"][i]), "equity": equity,
                             "running_peak": peak, "drawdown_usd": peak - equity,
                             "drawdown_pct": (peak - equity) / peak * 100.0})

    if position or pending:
        raise AssertionError("Simulation ended with live state")
    net_a, gross_a = np.asarray(nets), np.asarray(grosses)
    wins, losses = net_a[net_a > 0], net_a[net_a <= 0]
    sharpe, sortino = _ratios(daily_net)
    # For grid runs max DD is exact closed-daily equity.  The selected collect
    # run additionally publishes exact raw-minute close MTM below.
    daily_eq = CAPITAL + np.cumsum(daily_net)
    daily_curve = np.r_[CAPITAL, daily_eq]
    daily_peak = np.maximum.accumulate(daily_curve)
    daily_dd = daily_peak - daily_curve
    result: dict[str, Any] = {
        "stop_usd": stop_usd, "target_usd": target_usd,
        "sessions": n_days, "raw_bars": len(idx), "trades": len(net_a),
        "generated_flat_signals": signal_count,
        "ignored_signals_while_open": ignored_open_signals,
        "gross_pnl": float(gross_a.sum()), "costs": float(gross_a.sum() - net_a.sum()),
        "commissions": float(commissions_total), "slippage": float(slippage_total),
        "net_pnl": float(net_a.sum()), "net_return_pct": float(net_a.sum() / CAPITAL * 100.0),
        "win_rate_pct": float((net_a > 0).mean() * 100.0) if len(net_a) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() else 0.0,
        "net_sharpe": sharpe, "net_sortino": sortino,
        "max_drawdown_usd_daily": float(daily_dd.max()),
        "max_drawdown_pct_daily": float(np.max(daily_dd / daily_peak) * 100.0),
        "avg_net_trade": float(net_a.mean()) if len(net_a) else 0.0,
        "avg_duration_bars": float(np.mean(durations)) if durations else 0.0,
        "stops": stops, "targets": targets, "forced_eod": forced_eod,
        "avg_gross_risk_usd": float(stop_usd * shares_total / len(net_a)) if len(net_a) else 0.0,
        "avg_gross_reward_usd": float(target_usd * shares_total / len(net_a)) if len(net_a) else 0.0,
    }
    if collect:
        trade_df = pd.DataFrame(trades)
        mtm_df = pd.DataFrame(mtm_rows)
        result["trades_df"] = trade_df
        result["mtm_df"] = mtm_df
        result["daily_net"] = daily_net
        result["daily_gross"] = daily_gross
        result["max_drawdown_usd_mtm"] = float(mtm_df.drawdown_usd.max())
        result["max_drawdown_pct_mtm"] = float(mtm_df.drawdown_pct.max())
        result["final_equity"] = float(mtm_df.equity.iloc[-1])
        if len(trade_df) != signal_count or abs(trade_df.net_pnl.sum() - result["net_pnl"]) > 1e-8:
            raise AssertionError("Generated-event/trade/P&L reconciliation failed")
        if abs(result["final_equity"] - (CAPITAL + result["net_pnl"])) > 1e-8:
            raise AssertionError("MTM final equity reconciliation failed")
    return result


def clean(r: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in r.items() if k not in {"trades_df", "mtm_df", "daily_net", "daily_gross"}}


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    a, days = market()
    dev_end, val_end = 125, 188

    dev_rows = []
    for stop in DISTANCES:
        for target in DISTANCES:
            r = simulate(a, 0, dev_end, stop, target)
            dev_rows.append(clean(r))
    dev = pd.DataFrame(dev_rows).sort_values(["net_sharpe", "net_pnl"], ascending=False)
    dev.to_csv(OUT / "development_grid.csv", index=False)
    top = dev.head(TOP_DEVELOPMENT)[["stop_usd", "target_usd", "net_sharpe", "net_pnl"]].copy()
    top = top.rename(columns={"net_sharpe": "development_net_sharpe", "net_pnl": "development_net_pnl"})

    val_rows = []
    for row in top.itertuples(index=False):
        r = simulate(a, dev_end, val_end, float(row.stop_usd), float(row.target_usd))
        val_rows.append({"stop_usd": row.stop_usd, "target_usd": row.target_usd,
                         "development_net_sharpe": row.development_net_sharpe,
                         "development_net_pnl": row.development_net_pnl,
                         "validation_net_sharpe": r["net_sharpe"],
                         "validation_net_pnl": r["net_pnl"], **{f"validation_{k}": v for k, v in clean(r).items()
                                                                   if k not in {"stop_usd", "target_usd", "net_sharpe", "net_pnl"}}})
    finalists = pd.DataFrame(val_rows)
    finalists["robust_score"] = np.minimum(finalists.development_net_sharpe, finalists.validation_net_sharpe)
    finalists = finalists.sort_values(["robust_score", "validation_net_pnl"], ascending=False).reset_index(drop=True)
    finalists.to_csv(OUT / "validation_finalists.csv", index=False)
    chosen = finalists.iloc[0]
    selected = {"stop_usd": float(chosen.stop_usd), "target_usd": float(chosen.target_usd)}

    periods = {"development": (0, dev_end), "validation": (dev_end, val_end),
               "holdout": (val_end, len(days)), "full": (0, len(days))}
    selected_results: dict[str, dict[str, Any]] = {}
    selected_frames: dict[str, pd.DataFrame] = {}
    for name, (lo, hi) in periods.items():
        r = simulate(a, lo, hi, **selected, collect=True)
        selected_results[name] = clean(r)
        selected_frames[name] = r["trades_df"]
        r["trades_df"].to_csv(OUT / f"selected_{name}_trades.csv", index=False)
        r["mtm_df"].to_csv(OUT / f"selected_{name}_equity.csv", index=False)

    split_trades = sum(selected_results[p]["trades"] for p in ("development", "validation", "holdout"))
    split_net = sum(selected_results[p]["net_pnl"] for p in ("development", "validation", "holdout"))
    full = selected_results["full"]
    frozen_path = ROOT / "old" / "frozen_vwap_absolute" / "output" / "results" / "summary.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    summary = {
        "study": "Independent raw event-driven VWAP-Z with absolute NVDA stop/target",
        "period": {"start": str(START_DATE), "end": str(END_DATE), "sessions": len(days), "raw_bars": len(a["timestamp"])},
        "symbols": {"lead": "QQQ", "traded": "NVDA"},
        "entry_parameters": {"beta_days": BETA_DAYS, "warmup_bars": WARMUP, "window": WINDOW,
                             "z_entry": ENTRY_Z, "hook_delta": 0.0, "z_lockout": None,
                             "entry": "signal at raw bar close t; fill raw next-bar open t+1"},
        "splits": {"development": {"sessions": 125, "start": str(days[0]), "end": str(days[124])},
                   "validation": {"sessions": 63, "start": str(days[125]), "end": str(days[187])},
                   "holdout": {"sessions": 63, "start": str(days[188]), "end": str(days[-1])}},
        "grid": {"stop_usd": list(DISTANCES), "target_usd": list(DISTANCES), "combinations": 144,
                 "top_development_sent_to_validation": TOP_DEVELOPMENT},
        "selection": {"method": "top 10 development by net Sharpe/net P&L, then max min(dev,val Sharpe), validation P&L tie-break; holdout opened once",
                      "development_net_sharpe": float(chosen.development_net_sharpe),
                      "validation_net_sharpe": float(chosen.validation_net_sharpe),
                      "robust_score": float(chosen.robust_score), "holdout_opened_after_selection": True,
                      "no_confirmed_edge": bool(float(chosen.robust_score) <= 0 or selected_results["holdout"]["net_pnl"] <= 0)},
        "selected": selected, "selected_results": selected_results,
        "execution": {"raw_data": "exact synchronized Alpaca SIP 1-minute QQQ/NVDA RTH; no resampling/interpolation",
                      "position_notional_usd": SIZE, "starting_capital_usd": CAPITAL,
                      "commission_usd_per_share_per_side": COMMISSION,
                      "slippage_fraction_per_execution": SLIP,
                      "same_bar_ambiguity": "stop first", "stop_gap": "adverse raw open",
                      "while_position_open": "signals ignored", "after_exit": "fresh raw close signal allowed",
                      "convergence_exit": False},
        "frozen_cohort_comparison": {
            "old_method": "pre-generated convergence cohort; no fresh signal regeneration after early bracket exit",
            "old_selected": frozen["selected"], "old_full": frozen["selected_results"]["full"],
            "old_holdout": frozen["selected_results"]["holdout"],
            "selected_pair_changed": selected != frozen["selected"],
            "full_net_difference_usd": full["net_pnl"] - frozen["selected_results"]["full"]["net_pnl"],
            "holdout_net_difference_usd": selected_results["holdout"]["net_pnl"] - frozen["selected_results"]["holdout"]["net_pnl"],
        },
        "reconciliation": {"split_trades_equal_full": split_trades == full["trades"],
                           "split_net_equal_full": abs(split_net - full["net_pnl"]) <= 1e-8,
                           "full_generated_signals_equal_trades": full["generated_flat_signals"] == full["trades"],
                           "full_final_equity_equal_capital_plus_net": abs(full["final_equity"] - (CAPITAL + full["net_pnl"])) <= 1e-8,
                           "full_exit_reasons_equal_trades": full["stops"] + full["targets"] + full["forced_eod"] == full["trades"],
                           "full_commission_plus_slippage_equal_costs": abs(full["commissions"] + full["slippage"] - full["costs"]) <= 1e-8},
    }
    if not all(summary["reconciliation"].values()):
        raise AssertionError(f"Reconciliation failed: {summary['reconciliation']}")
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json))


if __name__ == "__main__":
    main()
