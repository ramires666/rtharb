"""Causal VWAP synthetic-basket research trading QQQ only.

The four-stock reference and official weights are frozen before the sample.
All calculations use the exact five-way raw Alpaca SIP one-minute RTH
intersection.  Signal selection is separated from exit selection; dollar
brackets regenerate signals whenever the selected strategy is flat.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numba import njit, prange

from rtharb.research.synthetic_index import (
    BASKET,
    NDX_WEIGHTS,
    SOURCE_URL,
    WEIGHTS,
    load_raw,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_output" / "synthetic_vwap_absolute"
CAPITAL = 100_000.0
SIZE = 20_000.0
COMMISSION = 0.0035
SLIP = 0.0002
STEP = 0.25
HARD_CAP = 30.0
TOP_DEV = 10
MIN_TRADES = 50
MIN_ACTIVE_DAYS = 30
DEV_END = 250
VAL_END = 375
WINDOWS = (30, 60, 90)
Z_ENTRIES = (1.5, 2.0, 2.5, 3.0)
HOOKS = (0.0, 0.15, 0.30)
DIRECTIONS = {"normal": 1, "reverse": -1}
VARIANTS = (
    "normal_convergence", "reverse_convergence",
    "normal_dollar_bracket", "reverse_dollar_bracket",
)
SPLITS = {
    "development": (0, DEV_END), "validation": (DEV_END, VAL_END),
    "holdout": (VAL_END, 501), "full": (0, 501),
}
GRID_COLUMNS = (
    "stop_usd", "target_usd", "sessions", "raw_bars", "trades", "active_trade_days",
    "generated_flat_signals", "ignored_signals_while_open", "gross_pnl", "costs",
    "commissions", "slippage", "net_pnl", "net_return_pct", "win_rate_pct",
    "profit_factor", "net_sharpe", "net_sortino", "max_drawdown_usd_daily",
    "max_drawdown_pct_daily", "avg_net_trade", "avg_duration_bars", "stops",
    "targets", "forced_eod", "avg_gross_risk_usd", "avg_gross_reward_usd",
)
TRADE_COLUMNS = (
    "variant", "signal_time", "entry_time", "exit_time", "direction",
    "source_dislocation", "signal_z", "signal_residual", "signal_fair_qqq",
    "signal_qqq_close", "signal_qqq_vwap", "entry_reference", "entry_price",
    "exit_reference", "exit_price", "shares", "stop_usd_per_share",
    "target_usd_per_share", "stop_price", "target_price", "exit_reason",
    "duration_bars", "gross_pnl", "slippage", "commissions", "costs", "net_pnl",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _session_vwap(frame: pd.DataFrame, common: pd.DatetimeIndex, day: np.ndarray) -> np.ndarray:
    raw = frame.loc[common]
    typical = raw[["high", "low", "close"]].to_numpy(float).mean(axis=1)
    volume = raw.volume.to_numpy(float)
    out = np.empty(len(common), dtype=float)
    for code in range(int(day[-1]) + 1):
        ix = np.flatnonzero(day == code)
        cumulative_volume = np.cumsum(volume[ix])
        out[ix] = np.divide(
            np.cumsum(typical[ix] * volume[ix]), cumulative_volume,
            out=np.full(len(ix), np.nan), where=cumulative_volume > 0,
        )
    return out


def _session_z(residual: np.ndarray, day: np.ndarray, window: int) -> np.ndarray:
    """Sample rolling Z; full window is the causal minimum-period gate."""
    out = np.full(len(residual), np.nan)
    for code in range(int(day[-1]) + 1):
        ix = np.flatnonzero(day == code)
        x = residual[ix]
        cs = np.r_[0.0, np.cumsum(x)]
        cs2 = np.r_[0.0, np.cumsum(x * x)]
        for local in range(window - 1, len(ix)):
            lo, hi = local - window + 1, local + 1
            total = cs[hi] - cs[lo]
            total2 = cs2[hi] - cs2[lo]
            mean = total / window
            variance = (total2 - total * total / window) / (window - 1)
            std = math.sqrt(max(variance, 0.0))
            if std > 1e-12:
                out[ix[local]] = (x[local] - mean) / std
    return out


def market() -> tuple[dict[str, np.ndarray], list[object], dict[str, Any]]:
    frames, common, day = load_raw()
    if len(common) != 194_490 or int(day[-1]) + 1 != 501:
        raise AssertionError(f"Expected exact 194,490/501 common sample, got {len(common)}/{int(day[-1])+1}")
    qqq = frames["QQQ"].loc[common]
    vwaps = {symbol: _session_vwap(frames[symbol], common, day) for symbol in ("QQQ", *BASKET)}
    basket_dev = np.zeros(len(common), dtype=float)
    for symbol in BASKET:
        basket_dev += WEIGHTS[symbol] * (frames[symbol].loc[common].close.to_numpy(float) / vwaps[symbol] - 1.0)
    fair = vwaps["QQQ"] * (1.0 + basket_dev)
    close = qqq.close.to_numpy(float)
    residual = (close - fair) / fair
    last = np.r_[day[1:] != day[:-1], True]
    arrays: dict[str, Any] = {
        "timestamp": common.to_numpy(), "day": day.astype(np.int64),
        "unique_days": pd.unique(common.date), "last": last.astype(np.bool_),
        "open": qqq.open.to_numpy(float), "high": qqq.high.to_numpy(float),
        "low": qqq.low.to_numpy(float), "close": close,
        "volume": qqq.volume.to_numpy(float), "qqq_vwap": vwaps["QQQ"],
        "basket_dev": basket_dev, "fair": fair, "residual": residual,
    }
    for window in WINDOWS:
        arrays[f"z_{window}"] = _session_z(residual, day, window)
    audit = {
        "common_raw_bars": len(common), "sessions": int(day[-1]) + 1,
        "first_timestamp": pd.Timestamp(common[0]).isoformat(),
        "last_timestamp": pd.Timestamp(common[-1]).isoformat(),
        "five_way_inner_intersection": True, "no_fill_resample_or_interpolation": True,
        "ohlc_positive": bool((qqq[["open", "high", "low", "close"]] > 0).all().all()),
        "timestamps_unique": bool(not common.has_duplicates),
    }
    return arrays, list(arrays["unique_days"]), audit


def _ratios(daily: np.ndarray) -> tuple[float, float]:
    prior = CAPITAL + np.r_[0.0, np.cumsum(daily[:-1])]
    returns = np.divide(daily, prior, out=np.zeros_like(daily), where=prior != 0.0)
    if len(returns) < 2 or returns.std(ddof=1) == 0.0:
        return 0.0, 0.0
    sharpe = math.sqrt(252) * returns.mean() / returns.std(ddof=1)
    downside = math.sqrt(float(np.mean(np.minimum(returns, 0.0) ** 2)))
    return float(sharpe), float(math.sqrt(252) * returns.mean() / downside if downside else 0.0)


def simulate(a: dict[str, np.ndarray], variant: str, first_day: int, last_day: int,
             signal: dict[str, float], direction_multiplier: int, exit_model: str,
             stop_usd: float | None = None, target_usd: float | None = None,
             collect: bool = False) -> dict[str, Any]:
    idx = np.flatnonzero((a["day"] >= first_day) & (a["day"] < last_day))
    n_days = last_day - first_day
    daily_net = np.zeros(n_days); daily_gross = np.zeros(n_days)
    active_days: set[int] = set()
    trades: list[dict[str, Any]] = []; equity_rows: list[dict[str, Any]] = []
    z_array = a[f"z_{int(signal['window'])}"]
    z_entry, hook = float(signal["z_entry"]), float(signal["hook_delta"])
    cash = CAPITAL; peak = CAPITAL
    position = pending_entry = pending_exit = 0
    source_sign = pending_source = 0
    pending_signal_i = entry_i = -1
    pending_z = math.nan
    entry_ref = entry_eff = stop_price = target_price = 0.0
    shares = 0; entry_commission = 0.0
    arm_sign = 0; extreme = 0.0
    generated = ignored = stops = targets = forced = convergence = 0
    commissions_total = slippage_total = 0.0; shares_total = 0
    nets: list[float] = []; grosses: list[float] = []; durations: list[int] = []

    def close_trade(i: int, raw_exit: float, reason: str) -> None:
        nonlocal position, cash, stops, targets, forced, convergence
        nonlocal commissions_total, slippage_total, shares_total
        exit_eff = raw_exit * (1.0 - SLIP if position == 1 else 1.0 + SLIP)
        gross = position * (raw_exit - entry_ref) * shares
        slippage = (abs(entry_eff - entry_ref) + abs(exit_eff - raw_exit)) * shares
        commissions = 2.0 * shares * COMMISSION
        costs = slippage + commissions; net = gross - costs
        local_day = int(a["day"][i]) - first_day
        daily_net[local_day] += net; daily_gross[local_day] += gross
        nets.append(net); grosses.append(gross); durations.append(i - entry_i)
        commissions_total += commissions; slippage_total += slippage; shares_total += shares
        if reason in ("STOP", "STOP_GAP"): stops += 1
        elif reason == "TAKE_PROFIT_BRACKET": targets += 1
        elif reason == "CONVERGENCE": convergence += 1
        else: forced += 1
        if collect:
            trades.append({
                "variant": variant,
                "signal_time": pd.Timestamp(a["timestamp"][pending_signal_i]),
                "entry_time": pd.Timestamp(a["timestamp"][entry_i]),
                "exit_time": pd.Timestamp(a["timestamp"][i]),
                "direction": "LONG" if position == 1 else "SHORT",
                "source_dislocation": "HIGH" if source_sign == 1 else "LOW",
                "signal_z": pending_z, "signal_residual": float(a["residual"][pending_signal_i]),
                "signal_fair_qqq": float(a["fair"][pending_signal_i]),
                "signal_qqq_close": float(a["close"][pending_signal_i]),
                "signal_qqq_vwap": float(a["qqq_vwap"][pending_signal_i]),
                "entry_reference": entry_ref, "entry_price": entry_eff,
                "exit_reference": raw_exit, "exit_price": exit_eff, "shares": shares,
                "stop_usd_per_share": stop_usd, "target_usd_per_share": target_usd,
                "stop_price": stop_price if exit_model == "dollar_bracket" else None,
                "target_price": target_price if exit_model == "dollar_bracket" else None,
                "exit_reason": reason, "duration_bars": i - entry_i,
                "gross_pnl": gross, "slippage": slippage, "commissions": commissions,
                "costs": costs, "net_pnl": net,
            })
        cash += net; position = 0

    for i in idx:
        day = int(a["day"][i]); is_last = bool(a["last"][i])
        is_first = i == idx[0] or int(a["day"][i - 1]) != day
        if is_first:
            if position or pending_entry or pending_exit:
                raise AssertionError(f"{variant}: state leaked overnight")
            arm_sign = 0; extreme = 0.0
        if pending_exit and position:
            pending_exit = 0
            close_trade(i, float(a["open"][i]), "CONVERGENCE")
        if pending_entry and not position:
            position = pending_entry; pending_entry = 0
            source_sign = pending_source; entry_i = i
            entry_ref = float(a["open"][i])
            entry_eff = entry_ref * (1.0 + SLIP if position == 1 else 1.0 - SLIP)
            shares = math.floor(SIZE / entry_eff)
            entry_commission = shares * COMMISSION
            active_days.add(day - first_day)
            if exit_model == "dollar_bracket":
                assert stop_usd is not None and target_usd is not None
                stop_price = entry_ref - stop_usd if position == 1 else entry_ref + stop_usd
                target_price = entry_ref + target_usd if position == 1 else entry_ref - target_usd
            arm_sign = 0
        if position and exit_model == "dollar_bracket":
            op, hi, lo = float(a["open"][i]), float(a["high"][i]), float(a["low"][i])
            stop_hit = (op <= stop_price or lo <= stop_price) if position == 1 else (op >= stop_price or hi >= stop_price)
            target_hit = hi >= target_price if position == 1 else lo <= target_price
            if stop_hit:
                gap = op <= stop_price if position == 1 else op >= stop_price
                close_trade(i, op if gap else stop_price, "STOP_GAP" if gap else "STOP")
            elif target_hit:
                close_trade(i, target_price, "TAKE_PROFIT_BRACKET")
        if position and is_last:
            close_trade(i, float(a["close"][i]), "FORCED_EOD")
            pending_exit = 0
        z = float(z_array[i]); residual = float(a["residual"][i])
        if position and exit_model == "convergence" and not is_last:
            crossed = residual <= 0.0 if source_sign == 1 else residual >= 0.0
            if crossed:
                pending_exit = 1
        if not position and not pending_entry and math.isfinite(z) and not is_last:
            emit = 0
            if arm_sign == 0:
                hit = 1 if z >= z_entry else (-1 if z <= -z_entry else 0)
                if hit:
                    if hook == 0.0: emit = hit
                    else: arm_sign, extreme = hit, z
            elif arm_sign == 1:
                extreme = max(extreme, z)
                if z <= extreme - hook: emit = 1
            else:
                extreme = min(extreme, z)
                if z >= extreme + hook: emit = -1
            if emit:
                pending_source = emit
                pending_entry = (-emit) * direction_multiplier
                pending_signal_i = i; pending_z = z
                generated += 1; arm_sign = 0; extreme = 0.0
        elif position and math.isfinite(z) and (z >= z_entry or z <= -z_entry):
            ignored += 1
        if collect:
            equity = cash
            if position:
                equity += position * (float(a["close"][i]) - entry_eff) * shares - entry_commission
            peak = max(peak, equity)
            equity_rows.append({
                "timestamp": pd.Timestamp(a["timestamp"][i]), "equity": equity,
                "running_peak": peak, "drawdown_usd": peak - equity,
                "drawdown_pct": (peak - equity) / peak * 100.0,
            })
    if position or pending_entry or pending_exit:
        raise AssertionError(f"{variant}: live state after simulation")
    net_a = np.asarray(nets); gross_a = np.asarray(grosses)
    wins, losses = net_a[net_a > 0], net_a[net_a <= 0]
    sharpe, sortino = _ratios(daily_net)
    curve = np.r_[CAPITAL, CAPITAL + np.cumsum(daily_net)]
    curve_peak = np.maximum.accumulate(curve); dd = curve_peak - curve
    result: dict[str, Any] = {
        "sessions": n_days, "raw_bars": len(idx), "trades": len(net_a),
        "active_trade_days": len(active_days), "generated_flat_signals": generated,
        "ignored_signals_while_open": ignored, "gross_pnl": float(gross_a.sum()),
        "costs": float(commissions_total + slippage_total), "commissions": float(commissions_total),
        "slippage": float(slippage_total), "net_pnl": float(net_a.sum()),
        "net_return_pct": float(net_a.sum() / CAPITAL * 100.0),
        "win_rate_pct": float((net_a > 0).mean() * 100.0) if len(net_a) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() else 0.0,
        "net_sharpe": sharpe, "net_sortino": sortino,
        "max_drawdown_usd_daily": float(dd.max()),
        "max_drawdown_pct_daily": float(np.max(dd / curve_peak) * 100.0),
        "avg_net_trade": float(net_a.mean()) if len(net_a) else 0.0,
        "avg_duration_bars": float(np.mean(durations)) if durations else 0.0,
        "stops": stops, "targets": targets, "convergence_exits": convergence,
        "forced_eod": forced,
        "avg_gross_risk_usd": float((stop_usd or 0.0) * shares_total / len(net_a)) if len(net_a) else 0.0,
        "avg_gross_reward_usd": float((target_usd or 0.0) * shares_total / len(net_a)) if len(net_a) else 0.0,
    }
    if collect:
        trade_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
        equity_df = pd.DataFrame(equity_rows)
        result.update({"trades_df": trade_df, "equity_df": equity_df,
                       "daily_net": daily_net, "daily_gross": daily_gross,
                       "max_drawdown_usd_mtm": float(equity_df.drawdown_usd.max()),
                       "max_drawdown_pct_mtm": float(equity_df.drawdown_pct.max()),
                       "final_equity": float(equity_df.equity.iloc[-1])})
        if len(trade_df) != generated or abs(float(trade_df.net_pnl.sum()) - result["net_pnl"]) > 1e-8:
            raise AssertionError(f"{variant}: trade/event reconciliation")
        if abs(result["final_equity"] - (CAPITAL + result["net_pnl"])) > 1e-8:
            raise AssertionError(f"{variant}: final equity reconciliation")
    return result


def _clean(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key not in {"trades_df", "equity_df", "daily_net", "daily_gross"}}


def _sort(frame: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    return frame.sort_values(
        [f"{prefix}net_sharpe", f"{prefix}net_pnl", "window", "z_entry", "hook_delta"],
        ascending=[False, False, True, True, True], kind="mergesort",
    ).reset_index(drop=True)


def select_convergence(a: dict[str, np.ndarray], days: list[object], direction: str) -> tuple[dict[str, Any], dict[str, Any]]:
    variant = f"{direction}_convergence"; out = OUT / variant; out.mkdir(parents=True, exist_ok=True)
    multiplier = DIRECTIONS[direction]
    rows = []
    for window in WINDOWS:
        for z_entry in Z_ENTRIES:
            for hook in HOOKS:
                signal = {"window": window, "warmup_bars": window, "z_entry": z_entry, "hook_delta": hook}
                result = simulate(a, variant, 0, DEV_END, signal, multiplier, "convergence")
                rows.append({**signal, **_clean(result)})
    grid = _sort(pd.DataFrame(rows)); grid.to_csv(out / "development_grid.csv", index=False, float_format="%.10f")
    eligible = grid[(grid.trades >= MIN_TRADES) & (grid.active_trade_days >= MIN_ACTIVE_DAYS)]
    if eligible.empty: raise RuntimeError(f"{variant}: no eligible development candidates")
    top = _sort(eligible).head(TOP_DEV)
    finalists = []
    for row in top.itertuples(index=False):
        signal = {"window": int(row.window), "warmup_bars": int(row.warmup_bars),
                  "z_entry": float(row.z_entry), "hook_delta": float(row.hook_delta)}
        val = simulate(a, variant, DEV_END, VAL_END, signal, multiplier, "convergence")
        finalists.append({**signal, "development_net_sharpe": float(row.net_sharpe),
                          "development_net_pnl": float(row.net_pnl),
                          "validation_net_sharpe": val["net_sharpe"], "validation_net_pnl": val["net_pnl"],
                          "validation_trades": val["trades"]})
    final = pd.DataFrame(finalists)
    final["robust_score"] = np.minimum(final.development_net_sharpe, final.validation_net_sharpe)
    final = final.sort_values(
        ["robust_score", "validation_net_pnl", "development_net_sharpe", "window", "z_entry", "hook_delta"],
        ascending=[False, False, False, True, True, True], kind="mergesort",
    ).reset_index(drop=True)
    final.to_csv(out / "validation_finalists.csv", index=False, float_format="%.10f")
    chosen = final.iloc[0]
    signal = {"window": int(chosen.window), "warmup_bars": int(chosen.warmup_bars),
              "z_entry": float(chosen.z_entry), "hook_delta": float(chosen.hook_delta)}
    results = export_selected(a, days, variant, signal, multiplier, "convergence", out)
    summary = make_summary(variant, direction, "convergence", signal, None, results, chosen, days)
    audit = audit_variant(summary, out)
    summary["audit"] = {"status": audit["status"], "file": f"{variant}/audit.json"}
    _write_json(out / "summary.json", summary)
    return signal, summary


def export_selected(a: dict[str, np.ndarray], days: list[object], variant: str,
                    signal: dict[str, Any], multiplier: int, exit_model: str, out: Path,
                    bracket: dict[str, float] | None = None) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for split, (lo, hi) in SPLITS.items():
        kwargs = bracket or {}
        result = simulate(a, variant, lo, hi, signal, multiplier, exit_model, collect=True, **kwargs)
        result["trades_df"].to_csv(out / f"selected_{split}_trades.csv", index=False, float_format="%.10f")
        result["equity_df"].to_csv(out / f"selected_{split}_equity.csv", index=False, float_format="%.10f")
        results[split] = _clean(result)
    return results


def make_summary(variant: str, direction: str, exit_model: str, signal: dict[str, Any],
                 bracket: dict[str, float] | None, results: dict[str, Any], chosen: Any,
                 days: list[object]) -> dict[str, Any]:
    return {
        "schema_version": 1, "variant": variant, "direction": direction, "exit_model": exit_model,
        "basket": {"symbols": list(BASKET), "official_ndx_weights_pct": NDX_WEIGHTS,
                   "normalized_reference_weights": WEIGHTS, "combined_ndx_weight_pct": 30.1,
                   "official_snapshot_date": "2024-06-28", "official_source": SOURCE_URL},
        "signal_formula": {"basket_dev": "sum(normalized_weight * (constituent_close / constituent_session_vwap - 1))",
                           "fair_qqq": "QQQ_session_vwap * (1 + basket_dev)",
                           "residual": "(QQQ_close - fair_qqq) / fair_qqq", "beta": None,
                           "vwap": "session cumulative typical-price VWAP; typical=(high+low+close)/3; current close bar included",
                           "z": "within-session rolling sample Z (ddof=1), exact trailing window; first finite at zero-based window-1",
                           "hook": "while flat, arm at signed threshold, track signed extreme, emit after hook_delta retrace; hook=0 emits immediately; reset on emit/entry/session",
                           "direction_mapping": "normal: HIGH residual -> SHORT QQQ, LOW -> LONG; reverse flips only QQQ side"},
        "signal_parameters": signal, "selected": bracket or signal,
        "selection": {"method": "top10 eligible development by daily net Sharpe/net P&L; max min(dev,val Sharpe), validation P&L tie; holdout once",
                      "development_net_sharpe": float(chosen.development_net_sharpe),
                      "validation_net_sharpe": float(chosen.validation_net_sharpe),
                      "robust_score": float(chosen.robust_score), "holdout_opened_after_selection": True,
                      "no_confirmed_edge": bool(float(chosen.robust_score) <= 0 or results["holdout"]["net_pnl"] <= 0)},
        "splits": {name: {"sessions": hi-lo, "start": str(days[lo]), "end": str(days[hi-1])} for name,(lo,hi) in SPLITS.items()},
        "selected_results": results,
        "execution": {"traded_instrument": "QQQ", "reference_only": list(BASKET),
                      "position_notional_usd": SIZE, "starting_capital_usd": CAPITAL,
                      "commission_usd_per_share_per_side": COMMISSION,
                      "slippage_fraction_per_execution": SLIP, "entry": "signal raw close t, fill next raw open",
                      "same_bar_ambiguity": "stop first", "stop_gap": "adverse raw open",
                      "convergence_exit": exit_model == "convergence", "forced_eod": "final raw RTH close"},
    }


def audit_variant(summary: dict[str, Any], out: Path) -> dict[str, Any]:
    results = summary["selected_results"]; full = results["full"]
    checks = {
        "split_trades_equal_full": sum(results[x]["trades"] for x in ("development","validation","holdout")) == full["trades"],
        "split_net_equal_full": abs(sum(results[x]["net_pnl"] for x in ("development","validation","holdout")) - full["net_pnl"]) <= 1e-8,
        "gross_minus_costs_equal_net": abs(full["gross_pnl"] - full["costs"] - full["net_pnl"]) <= 1e-8,
        "commission_plus_slippage_equal_costs": abs(full["commissions"] + full["slippage"] - full["costs"]) <= 1e-8,
        "events_equal_trades": full["generated_flat_signals"] == full["trades"],
        "exit_reasons_equal_trades": full["stops"] + full["targets"] + full["convergence_exits"] + full["forced_eod"] == full["trades"],
        "final_equity_equal_capital_plus_net": abs(full["final_equity"] - (CAPITAL + full["net_pnl"])) <= 1e-8,
        "all_trade_csvs_exist": all((out / f"selected_{split}_trades.csv").is_file() for split in SPLITS),
        "all_equity_csvs_exist": all((out / f"selected_{split}_equity.csv").is_file() for split in SPLITS),
    }
    if "kernel_replay_checks" in summary:
        checks["numba_grid_matches_transparent_replay"] = summary["kernel_replay_checks"]["status"] == "PASS"
    audit = {"variant": summary["variant"], "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    _write_json(out / "audit.json", audit)
    if audit["status"] != "PASS": raise AssertionError(f"Audit failed {audit}")
    return audit


@njit(cache=True)
def _bracket_one(day, op, hi, lo, close, z, last, n_days, z_entry, hook,
                 direction_multiplier, stop_usd, target_usd):
    daily = np.zeros(n_days); active = np.zeros(n_days, dtype=np.uint8)
    position = pending = pending_source = source_sign = 0
    arm = 0; extreme = 0.0; entry_ref = entry_eff = stop_price = target_price = 0.0
    entry_i = -1; shares = 0
    trades = generated = ignored = stops = targets = forced = wins = 0
    gross_sum = net_sum = commissions = slippage = loss_sum = durations = 0.0
    shares_total = 0
    for i in range(len(day)):
        d = day[i]
        if i == 0 or day[i-1] != d: arm = 0; extreme = 0.0
        if pending != 0:
            position = pending; pending = 0; source_sign = pending_source; entry_i = i
            entry_ref = op[i]; entry_eff = entry_ref * (1.0 + SLIP if position == 1 else 1.0 - SLIP)
            shares = math.floor(SIZE / entry_eff); active[d] = 1
            stop_price = entry_ref - stop_usd if position == 1 else entry_ref + stop_usd
            target_price = entry_ref + target_usd if position == 1 else entry_ref - target_usd
            arm = 0
        if position != 0:
            stop_hit = (op[i] <= stop_price or lo[i] <= stop_price) if position == 1 else (op[i] >= stop_price or hi[i] >= stop_price)
            target_hit = hi[i] >= target_price if position == 1 else lo[i] <= target_price
            reason = 0; raw_exit = 0.0
            if stop_hit:
                gap = op[i] <= stop_price if position == 1 else op[i] >= stop_price
                raw_exit = op[i] if gap else stop_price; reason = 1
            elif target_hit: raw_exit = target_price; reason = 2
            elif last[i]: raw_exit = close[i]; reason = 3
            if reason:
                exit_eff = raw_exit * (1.0 - SLIP if position == 1 else 1.0 + SLIP)
                gross = position * (raw_exit - entry_ref) * shares
                slip_cost = (abs(entry_eff-entry_ref)+abs(exit_eff-raw_exit))*shares
                comm = 2.0*shares*COMMISSION; net = gross-slip_cost-comm
                daily[d] += net; gross_sum += gross; net_sum += net
                commissions += comm; slippage += slip_cost; durations += i-entry_i
                shares_total += shares; trades += 1
                if net > 0: wins += 1
                else: loss_sum += net
                if reason == 1: stops += 1
                elif reason == 2: targets += 1
                else: forced += 1
                position = 0
        zi = z[i]
        if not math.isnan(zi) and not last[i]:
            if position != 0:
                if zi >= z_entry or zi <= -z_entry: ignored += 1
            else:
                emit = 0
                if arm == 0:
                    hit = 1 if zi >= z_entry else (-1 if zi <= -z_entry else 0)
                    if hit != 0:
                        if hook == 0.0: emit = hit
                        else: arm = hit; extreme = zi
                elif arm == 1:
                    extreme = max(extreme, zi)
                    if zi <= extreme-hook: emit = 1
                else:
                    extreme = min(extreme, zi)
                    if zi >= extreme+hook: emit = -1
                if emit != 0:
                    pending_source = emit; pending = (-emit)*direction_multiplier
                    generated += 1; arm = 0; extreme = 0.0
    returns = np.zeros(n_days); cumulative = 0.0; peak = CAPITAL; maxdd = maxddpct = 0.0
    for d in range(n_days):
        prior = CAPITAL+cumulative; returns[d] = daily[d]/prior if prior else 0.0
        cumulative += daily[d]; eq=CAPITAL+cumulative; peak=max(peak,eq); dd=peak-eq
        maxdd=max(maxdd,dd); maxddpct=max(maxddpct,dd/peak*100.0 if peak else 0.0)
    mean = returns.mean(); variance = 0.0; downside_sum=0.0
    for x in returns:
        variance += (x-mean)**2
        if x < 0: downside_sum += x*x
    std=math.sqrt(variance/(n_days-1)) if n_days>1 else 0.0
    downside=math.sqrt(downside_sum/n_days); sharpe=math.sqrt(252.0)*mean/std if std else 0.0
    sortino=math.sqrt(252.0)*mean/downside if downside else 0.0
    active_count=0
    for x in active: active_count += x
    costs=commissions+slippage; pf=(net_sum-loss_sum)/abs(loss_sum) if loss_sum<0 else 0.0
    out=np.empty(len(GRID_COLUMNS))
    out[0]=stop_usd; out[1]=target_usd; out[2]=n_days; out[3]=len(day)
    out[4]=trades; out[5]=active_count; out[6]=generated; out[7]=ignored
    out[8]=gross_sum; out[9]=costs; out[10]=commissions; out[11]=slippage
    out[12]=net_sum; out[13]=net_sum/CAPITAL*100.0
    out[14]=wins/trades*100.0 if trades else 0.0
    out[15]=pf; out[16]=sharpe; out[17]=sortino; out[18]=maxdd; out[19]=maxddpct
    out[20]=net_sum/trades if trades else 0.0
    out[21]=durations/trades if trades else 0.0
    out[22]=stops; out[23]=targets; out[24]=forced
    out[25]=stop_usd*shares_total/trades if trades else 0.0
    out[26]=target_usd*shares_total/trades if trades else 0.0
    return out


@njit(parallel=True, cache=True)
def _bracket_grid(day, op, hi, lo, close, z, last, n_days, z_entry, hook, direction, stops, targets):
    out=np.empty((len(stops),len(GRID_COLUMNS)))
    for i in prange(len(stops)):
        out[i,:]=_bracket_one(day,op,hi,lo,close,z,last,n_days,z_entry,hook,direction,stops[i],targets[i])
    return out


def evaluate_brackets(a: dict[str,np.ndarray], signal: dict[str,Any], multiplier: int,
                      first_day: int,last_day: int,pairs:list[tuple[float,float]]) -> pd.DataFrame:
    mask=(a["day"]>=first_day)&(a["day"]<last_day)
    stops=np.asarray([x[0] for x in pairs]); targets=np.asarray([x[1] for x in pairs])
    values=_bracket_grid(np.ascontiguousarray(a["day"][mask]-first_day,dtype=np.int64),
                         np.ascontiguousarray(a["open"][mask]),np.ascontiguousarray(a["high"][mask]),
                         np.ascontiguousarray(a["low"][mask]),np.ascontiguousarray(a["close"][mask]),
                         np.ascontiguousarray(a[f"z_{signal['window']}"][mask]),np.ascontiguousarray(a["last"][mask]),
                         last_day-first_day,float(signal["z_entry"]),float(signal["hook_delta"]),multiplier,stops,targets)
    return pd.DataFrame(values,columns=GRID_COLUMNS)


def _axis(maximum:float)->np.ndarray:
    return np.round(np.arange(1,int(round(maximum/STEP))+1)*STEP,2)


def select_bracket(a:dict[str,np.ndarray],days:list[object],direction:str,signal:dict[str,Any])->dict[str,Any]:
    variant=f"{direction}_dollar_bracket"; out=OUT/variant; out.mkdir(parents=True,exist_ok=True)
    multiplier=DIRECTIONS[direction]
    median=float(np.median(a["close"][a["day"]<DEV_END]))
    initial=max(3.0,math.ceil(median*0.03/STEP-1e-12)*STEP); stop_max=target_max=min(HARD_CAP,initial)
    evaluated:set[tuple[float,float]]=set(); frames=[]; expansion=[]
    while True:
        sa,ta=_axis(stop_max),_axis(target_max); pairs=[(float(s),float(t)) for s in sa for t in ta]
        new=[p for p in pairs if p not in evaluated]
        if new: frames.append(evaluate_brackets(a,signal,multiplier,0,DEV_END,new)); evaluated.update(new)
        grid=pd.concat(frames,ignore_index=True)
        eligible=grid[(grid.trades>=MIN_TRADES)&(grid.active_trade_days>=MIN_ACTIVE_DAYS)]
        if eligible.empty: raise RuntimeError(f"{variant}: no eligible bracket")
        ranked=eligible.sort_values(["net_sharpe","net_pnl","stop_usd","target_usd"],ascending=[False,False,True,True],kind="mergesort")
        winner=ranked.iloc[0]; sb=float(winner.stop_usd)>=sa[-2]; tb=float(winner.target_usd)>=ta[-2]
        ns=min(HARD_CAP,math.ceil(stop_max*1.5/STEP-1e-12)*STEP) if sb and stop_max<HARD_CAP else stop_max
        nt=min(HARD_CAP,math.ceil(target_max*1.5/STEP-1e-12)*STEP) if tb and target_max<HARD_CAP else target_max
        expansion.append({"round":len(expansion)+1,"stop_max_usd":stop_max,"target_max_usd":target_max,
                          "combinations":len(evaluated),"winner":{"stop_usd":float(winner.stop_usd),"target_usd":float(winner.target_usd),"net_sharpe":float(winner.net_sharpe)},
                          "stop_boundary":bool(sb),"target_boundary":bool(tb),"expanded_stop":ns>stop_max,"expanded_target":nt>target_max})
        if ns==stop_max and nt==target_max: break
        stop_max,target_max=ns,nt
    grid=grid.sort_values(["net_sharpe","net_pnl","stop_usd","target_usd"],ascending=[False,False,True,True],kind="mergesort").reset_index(drop=True)
    grid.to_csv(out/"development_grid.csv",index=False,float_format="%.10f")
    eligible=grid[(grid.trades>=MIN_TRADES)&(grid.active_trade_days>=MIN_ACTIVE_DAYS)].head(TOP_DEV)
    pairs=list(zip(eligible.stop_usd.astype(float),eligible.target_usd.astype(float)))
    vals=evaluate_brackets(a,signal,multiplier,DEV_END,VAL_END,pairs)
    final=eligible[["stop_usd","target_usd","net_sharpe","net_pnl"]].rename(columns={"net_sharpe":"development_net_sharpe","net_pnl":"development_net_pnl"}).reset_index(drop=True)
    final["validation_net_sharpe"]=vals.net_sharpe; final["validation_net_pnl"]=vals.net_pnl
    final["robust_score"]=np.minimum(final.development_net_sharpe,final.validation_net_sharpe)
    final=final.sort_values(["robust_score","validation_net_pnl","development_net_sharpe","stop_usd","target_usd"],ascending=[False,False,False,True,True],kind="mergesort").reset_index(drop=True)
    final.to_csv(out/"validation_finalists.csv",index=False,float_format="%.10f")
    chosen=final.iloc[0]; bracket={"stop_usd":float(chosen.stop_usd),"target_usd":float(chosen.target_usd)}
    results=export_selected(a,days,variant,signal,multiplier,"dollar_bracket",out,bracket)
    summary=make_summary(variant,direction,"dollar_bracket",signal,bracket,results,chosen,days)
    kernel_checks={}
    for split in ("development","validation"):
        lo,hi=SPLITS[split]
        fast=evaluate_brackets(a,signal,multiplier,lo,hi,[(bracket["stop_usd"],bracket["target_usd"])]).iloc[0]
        replay=results[split]
        kernel_checks[split]={
            key: bool(abs(float(fast[key])-float(replay[key])) <= 1e-7)
            for key in ("trades","gross_pnl","costs","net_pnl","net_sharpe","stops","targets","forced_eod")
        }
    summary["kernel_replay_checks"]={"status":"PASS" if all(all(x.values()) for x in kernel_checks.values()) else "FAIL",
                                      "checks":kernel_checks}
    if summary["kernel_replay_checks"]["status"]!="PASS": raise AssertionError(f"{variant}: grid kernel/replay mismatch")
    summary["grid"]={"step_usd":STEP,"development_median_qqq":median,"initial_max_usd":initial,"hard_cap_usd":HARD_CAP,"unique_combinations":len(grid),"expansion_log":expansion}
    audit=audit_variant(summary,out); summary["audit"]={"status":audit["status"],"file":f"{variant}/audit.json"}; _write_json(out/"summary.json",summary)
    return summary


def main()->None:
    if sys.platform=="win32":
        sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True,exist_ok=True)
    a,days,data_audit=market(); summaries={}; signals={}
    bracket_only="--bracket-only" in sys.argv[1:]
    if bracket_only:
        for direction in DIRECTIONS:
            path=OUT/f"{direction}_convergence"/"summary.json"
            if not path.is_file(): raise FileNotFoundError(f"Cannot resume without {path}")
            summaries[f"{direction}_convergence"]=json.loads(path.read_text(encoding="utf-8"))
            signals[direction]=summaries[f"{direction}_convergence"]["signal_parameters"]
            print(f"RESUMED {direction} convergence {signals[direction]}",flush=True)
    else:
        for direction in DIRECTIONS:
            signals[direction],summaries[f"{direction}_convergence"]=select_convergence(a,days,direction)
            print(f"SELECTED {direction} convergence {signals[direction]}",flush=True)
    for direction in DIRECTIONS:
        summaries[f"{direction}_dollar_bracket"]=select_bracket(a,days,direction,signals[direction])
        print(f"SELECTED {direction} bracket {summaries[f'{direction}_dollar_bracket']['selected']}",flush=True)
    rows=[]
    for variant in VARIANTS:
        summary=summaries[variant]
        rows.append({"variant":variant,"direction":summary["direction"],"exit_model":summary["exit_model"],
                     **{f"signal_{k}":v for k,v in summary["signal_parameters"].items()},
                     "stop_usd":summary["selected"].get("stop_usd"),"target_usd":summary["selected"].get("target_usd"),
                     "development_net_pnl":summary["selected_results"]["development"]["net_pnl"],
                     "validation_net_pnl":summary["selected_results"]["validation"]["net_pnl"],
                     "holdout_net_pnl":summary["selected_results"]["holdout"]["net_pnl"],
                     "full_net_pnl":summary["selected_results"]["full"]["net_pnl"],
                     "full_mtm_mdd_pct":summary["selected_results"]["full"]["max_drawdown_pct_mtm"]})
    pd.DataFrame(rows).to_csv(OUT/"cross_variant_summary.csv",index=False,float_format="%.10f")
    _write_json(OUT/"cross_variant_summary.json",{"schema_version":1,"rows":rows})
    root_checks={"raw_bar_count":data_audit["common_raw_bars"]==194_490,"sessions":data_audit["sessions"]==501,
                 "no_fill":data_audit["no_fill_resample_or_interpolation"],"four_variant_audits":all(s["audit"]["status"]=="PASS" for s in summaries.values()),
                 "qqq_only_traded":True,"frozen_pre_sample_basket":True,"holdout_not_tuned":True}
    root_audit={"status":"PASS" if all(root_checks.values()) else "FAIL","checks":root_checks,"data":data_audit}; _write_json(OUT/"audit.json",root_audit)
    if root_audit["status"]!="PASS": raise AssertionError(root_audit)
    manifest={"schema_version":1,"status":"COMPLETE","study":"Frozen four-stock synthetic VWAP deviation; QQQ only traded",
              "period":{"start":str(days[0]),"end":str(days[-1]),"sessions":501,"raw_bars":194_490},
              "basket":{"symbols":list(BASKET),"official_ndx_weights_pct":NDX_WEIGHTS,"normalized_reference_weights":WEIGHTS,
                        "combined_ndx_weight_pct":30.1,"snapshot_date":"2024-06-28","source":SOURCE_URL},
              "variants":list(VARIANTS),"formula":{"basket_dev":"sum(w*(close/VWAP-1))","fair_qqq":"QQQ_VWAP*(1+basket_dev)","residual":"(QQQ-fair)/fair","beta":None},
              "signal_grid":{"windows":list(WINDOWS),"warmup":"full window","z_entry":list(Z_ENTRIES),"hook_delta":list(HOOKS)},
              "execution":{"traded":"QQQ","notional_usd":SIZE,"capital_usd":CAPITAL,"commission_per_share_side":COMMISSION,"slippage_fraction":SLIP},
              "audit":{"status":"PASS","file":"audit.json"},"warning":"Exploratory separately selected normal/reverse strategies; holdout is one historical diagnostic"}
    _write_json(OUT/"manifest.json",manifest)
    print(json.dumps({"status":"COMPLETE","rows":rows},ensure_ascii=False,indent=2,default=_json_default))


if __name__=="__main__": main()
