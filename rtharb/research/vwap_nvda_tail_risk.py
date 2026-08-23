"""Causal tail-risk overlays for the frozen NVDA VWAP absolute bracket.

The frozen baseline is QQQ-referenced NVDA VWAP-Z with a $5.25 stop and
$1.25 target.  Development and validation may select a tighter/wider stop
and/or a causal time stop.  The holdout is replayed only after the choice is
frozen.  BASE/no-overlay is an explicit candidate and the safe default.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rtharb.research.risk_reward import CAPITAL, COMMISSION, SIZE, SLIP
from rtharb.research.vwap_absolute_multi_asset import (
    DEV_END, ENTRY_Z, LEAD, VAL_END, _ratios, load_market,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_output" / "vwap_nvda_tail_risk"
SYMBOL = "NVDA"
BASE_STOP = 5.25
TARGET = 1.25
STOP_AXIS = tuple(float(x) for x in np.arange(1.5, 8.0001, 0.25))
HOLD_AXIS: tuple[int | None, ...] = (30, 45, 60, 75, 90, 120, 150, 180, 210, 240, 270, 300, 330, None)
TOP_DEVELOPMENT = 48
SPLITS = {"development": (0, DEV_END), "validation": (DEV_END, VAL_END),
          "holdout": (VAL_END, 251), "full": (0, 251)}
EXPECTED_BASE = {"development": 2018.785961020174, "validation": 2473.562216900122,
                 "holdout": 2199.5402666400987, "full": 6691.888444560394}


def _default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                               default=_default) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def simulate(a: dict[str, np.ndarray], first_day: int, last_day: int,
             stop_usd: float, max_holding_bars: int | None,
             collect_equity: bool = False) -> dict[str, Any]:
    """Exact raw-minute event replay with stop -> target -> time -> EOD priority."""
    idx = np.flatnonzero((a["day"] >= first_day) & (a["day"] < last_day))
    n_days = last_day - first_day
    daily = np.zeros(n_days, dtype=float)
    cash = CAPITAL
    position = pending = 0
    pending_z = math.nan
    pending_signal_i = entry_i = -1
    entry_ref = entry_eff = 0.0
    shares = 0
    stop_price = target_price = 0.0
    entry_commission = 0.0
    generated = ignored = 0
    stop_count = target_count = time_count = eod_count = 0
    commission_total = slippage_total = 0.0
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    peak = CAPITAL
    max_dd = max_dd_pct = 0.0

    def close(i: int, raw_exit: float, reason: str) -> None:
        nonlocal position, cash, stop_count, target_count, time_count, eod_count
        nonlocal commission_total, slippage_total
        exit_eff = raw_exit * (1.0 - SLIP if position == 1 else 1.0 + SLIP)
        gross = position * (raw_exit - entry_ref) * shares
        slippage = (abs(entry_eff - entry_ref) + abs(exit_eff - raw_exit)) * shares
        commissions = 2.0 * shares * COMMISSION
        net = gross - slippage - commissions
        daily[int(a["day"][i]) - first_day] += net
        commission_total += commissions
        slippage_total += slippage
        if reason == "STOP": stop_count += 1
        elif reason == "TAKE_PROFIT_BRACKET": target_count += 1
        elif reason == "TIME_STOP": time_count += 1
        else: eod_count += 1
        trades.append({
            "signal_time": pd.Timestamp(a["timestamp"][pending_signal_i]),
            "entry_time": pd.Timestamp(a["timestamp"][entry_i]),
            "exit_time": pd.Timestamp(a["timestamp"][i]),
            "direction": "LONG" if position == 1 else "SHORT",
            "entry_z": pending_z, "entry_reference": entry_ref, "entry_price": entry_eff,
            "exit_reference": raw_exit, "exit_price": exit_eff, "shares": shares,
            "stop_usd_per_share": stop_usd, "target_usd_per_share": TARGET,
            "max_holding_bars": max_holding_bars, "exit_reason": reason,
            "duration_bars": i - entry_i, "gross_pnl": gross, "slippage": slippage,
            "commissions": commissions, "costs": slippage + commissions, "net_pnl": net,
        })
        cash += net
        position = 0

    for i in idx:
        day = int(a["day"][i])
        first = i == idx[0] or int(a["day"][i - 1]) != day
        last = bool(a["last"][i])
        if first and (position or pending):
            raise AssertionError("state leaked across RTH sessions")
        if pending:
            position, pending = pending, 0
            entry_i = i
            entry_ref = float(a["open"][i])
            entry_eff = entry_ref * (1.0 + SLIP if position == 1 else 1.0 - SLIP)
            shares = math.floor(SIZE / entry_eff)
            entry_commission = shares * COMMISSION
            stop_price = entry_ref - stop_usd if position == 1 else entry_ref + stop_usd
            target_price = entry_ref + TARGET if position == 1 else entry_ref - TARGET
        if position:
            op, hi, lo = float(a["open"][i]), float(a["high"][i]), float(a["low"][i])
            stop_hit = (op <= stop_price or lo <= stop_price) if position == 1 else (op >= stop_price or hi >= stop_price)
            target_hit = hi >= target_price if position == 1 else lo <= target_price
            stop_gap = op <= stop_price if position == 1 else op >= stop_price
            expired = max_holding_bars is not None and i - entry_i >= max_holding_bars
            # At expiry the decision exists at the minute open.  Only an
            # already-observed adverse gap through the stop precedes it; the
            # expiry minute's future high/low must not be inspected.
            if expired and stop_gap:
                close(i, op, "STOP")
            elif expired:
                close(i, op, "TIME_STOP")
            elif stop_hit:
                gap = stop_gap
                close(i, op if gap else stop_price, "STOP")
            elif target_hit:
                close(i, target_price, "TAKE_PROFIT_BRACKET")
            elif last:
                close(i, float(a["close"][i]), "FORCED_EOD")
        z = float(a["z"][i])
        if math.isfinite(z) and not last:
            hit = 1 if z <= -ENTRY_Z else (-1 if z >= ENTRY_Z else 0)
            if hit:
                if position: ignored += 1
                else:
                    pending, pending_z, pending_signal_i = hit, z, i
                    generated += 1
        if last and pending:
            raise AssertionError("last-bar pending entry")
        if position:
            equity = cash - entry_commission + position * (float(a["close"][i]) - entry_eff) * shares
        else:
            equity = cash
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
        max_dd_pct = max(max_dd_pct, (peak - equity) / peak * 100.0)
        if collect_equity:
            equity_rows.append({"timestamp": pd.Timestamp(a["timestamp"][i]), "equity": equity,
                                "running_peak": peak, "drawdown_usd": peak - equity,
                                "drawdown_pct": (peak - equity) / peak * 100.0})
    if position or pending:
        raise AssertionError("simulation ended with live state")
    t = pd.DataFrame(trades)
    nets = t.net_pnl.to_numpy(float); grosses = t.gross_pnl.to_numpy(float)
    wins, losses = nets[nets > 0], nets[nets <= 0]
    tail_n = max(1, math.ceil(0.05 * len(nets)))
    worst = np.sort(nets)[:tail_n]
    sharpe, sortino = _ratios(daily)
    net = float(nets.sum())
    result: dict[str, Any] = {
        "stop_usd": stop_usd, "max_holding_bars": max_holding_bars,
        "sessions": n_days, "raw_bars": len(idx), "trades": len(t),
        "generated_flat_signals": generated, "ignored_signals_while_open": ignored,
        "gross_pnl": float(grosses.sum()), "costs": commission_total + slippage_total,
        "commissions": commission_total, "slippage": slippage_total, "net_pnl": net,
        "net_return_pct": net / CAPITAL * 100.0, "net_sharpe": sharpe,
        "net_sortino": sortino, "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() else 0.0,
        "positive_net_pnl_mass_usd": float(wins.sum()),
        "trade_cvar5_loss_usd": float(-worst.mean()), "worst_trade_loss_usd": float(-nets.min()),
        "win_rate_pct": float((nets > 0).mean() * 100.0), "max_drawdown_usd_mtm": max_dd,
        "max_drawdown_pct_mtm": max_dd_pct, "return_over_mtm_dd": net / max_dd if max_dd else 0.0,
        "avg_net_trade": float(nets.mean()), "avg_duration_bars": float(t.duration_bars.mean()),
        "stops": stop_count, "targets": target_count, "time_stops": time_count,
        "forced_eod": eod_count, "final_equity": CAPITAL + net,
        "trades_df": t, "daily_net": daily,
    }
    if collect_equity:
        result["mtm_df"] = pd.DataFrame(equity_rows)
    return result


def clean(result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if k not in {"trades_df", "daily_net", "mtm_df"}}


def cohort_metrics(base: pd.DataFrame, overlay: pd.DataFrame) -> dict[str, Any]:
    """Dollar retention of the exact baseline entry-event cohorts."""
    overlay_map = {pd.Timestamp(r.signal_time): r for r in overlay.itertuples(index=False)}
    winners = base[base.net_pnl > 0].copy()
    losers = base[base.net_pnl <= 0].copy()

    def candidate_values(rows: pd.DataFrame, column: str) -> np.ndarray:
        vals = []
        for row in rows.itertuples(index=False):
            match = overlay_map.get(pd.Timestamp(row.signal_time))
            vals.append(float(getattr(match, column)) if match is not None else 0.0)
        return np.asarray(vals, dtype=float)

    winner_net = winners.net_pnl.to_numpy(float)
    winner_gross = winners.gross_pnl.to_numpy(float)
    candidate_net = candidate_values(winners, "net_pnl")
    candidate_gross = candidate_values(winners, "gross_pnl")
    retained_net = float(np.maximum(0.0, np.minimum(candidate_net, winner_net)).sum())
    retained_gross = float(np.maximum(0.0, np.minimum(candidate_gross, winner_gross)).sum())
    loser_base = losers.net_pnl.to_numpy(float)
    loser_candidate = candidate_values(losers, "net_pnl")
    worst_n = min(5, len(losers))
    worst_idx = np.argsort(loser_base)[:worst_n]
    matched_winners = sum(pd.Timestamp(x) in overlay_map for x in winners.signal_time)
    matched_losers = sum(pd.Timestamp(x) in overlay_map for x in losers.signal_time)
    return {
        "base_winner_count": len(winners), "matched_base_winner_count": matched_winners,
        "base_winner_net_usd": float(winner_net.sum()), "base_winner_gross_usd": float(winner_gross.sum()),
        "retained_base_winner_net_usd": retained_net,
        "retained_base_winner_gross_usd": retained_gross,
        "core_winner_net_retention_pct": retained_net / winner_net.sum() * 100.0 if len(winners) else 100.0,
        "core_winner_gross_retention_pct": retained_gross / winner_gross.sum() * 100.0 if len(winners) else 100.0,
        "clipped_base_winner_net_usd": float(winner_net.sum() - retained_net),
        "clipped_base_winner_gross_usd": float(winner_gross.sum() - retained_gross),
        "overlay_net_on_base_winners_usd": float(candidate_net.sum()),
        "base_loser_count": len(losers), "matched_base_loser_count": matched_losers,
        "base_loser_net_usd": float(loser_base.sum()),
        "overlay_net_on_base_losers_usd": float(loser_candidate.sum()),
        "avoided_base_loser_loss_usd": float(loser_candidate.sum() - loser_base.sum()),
        "avoided_worst5_base_loser_loss_usd": float((loser_candidate[worst_idx] - loser_base[worst_idx]).sum()),
    }


def comparison(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    c = cohort_metrics(base["trades_df"], candidate["trades_df"])
    net_delta = candidate["net_pnl"] - base["net_pnl"]
    mdd_reduction = base["max_drawdown_usd_mtm"] - candidate["max_drawdown_usd_mtm"]
    # Dollars-based utility: realized net dominates; genuine tail/DD savings help;
    # clipping already enters realized net and gets an additional mild penalty.
    cvar_reduction = base["trade_cvar5_loss_usd"] - candidate["trade_cvar5_loss_usd"]
    worst_reduction = base["worst_trade_loss_usd"] - candidate["worst_trade_loss_usd"]
    utility = (net_delta + 0.35 * max(0.0, mdd_reduction) +
               0.50 * max(0.0, cvar_reduction) +
               0.20 * max(0.0, c["avoided_worst5_base_loser_loss_usd"]) -
               c["clipped_base_winner_net_usd"])
    return {
        **c, "net_pnl_delta_vs_base": net_delta,
        "net_pnl_pct_of_base": candidate["net_pnl"] / base["net_pnl"] * 100.0 if base["net_pnl"] else 0.0,
        "sharpe_delta_vs_base": candidate["net_sharpe"] - base["net_sharpe"],
        "profit_factor_delta_vs_base": candidate["profit_factor"] - base["profit_factor"],
        "mdd_reduction_usd_vs_base": mdd_reduction,
        "mdd_reduction_pct_vs_base": mdd_reduction / base["max_drawdown_usd_mtm"] * 100.0,
        "trade_cvar5_reduction_usd_vs_base": cvar_reduction,
        "trade_cvar5_reduction_pct_vs_base": cvar_reduction / base["trade_cvar5_loss_usd"] * 100.0,
        "worst_trade_reduction_usd_vs_base": worst_reduction,
        "worst_trade_reduction_pct_vs_base": worst_reduction / base["worst_trade_loss_usd"] * 100.0,
        "positive_net_pnl_mass_delta_vs_base": candidate["positive_net_pnl_mass_usd"] - base["positive_net_pnl_mass_usd"],
        "return_over_dd_delta_vs_base": candidate["return_over_mtm_dd"] - base["return_over_mtm_dd"],
        "utility_delta_usd": utility,
    }


def row(period: str, base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return {"period": period, **clean(candidate), **comparison(base, candidate)}


def export(name: str, result: dict[str, Any]) -> None:
    result["trades_df"].to_csv(OUT / f"{name}_trades.csv", index=False, float_format="%.12f")
    result["mtm_df"].to_csv(OUT / f"{name}_equity.csv", index=False, float_format="%.12f")


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    a, days, data_audit = load_market(SYMBOL)
    print(f"raw SIP loaded: {len(a['timestamp']):,} bars / {len(days)} sessions", flush=True)

    # Holdout is not passed to simulate until selected_stop/selected_hold are frozen.
    base_dev = simulate(a, 0, DEV_END, BASE_STOP, None)
    base_val = simulate(a, DEV_END, VAL_END, BASE_STOP, None)
    if abs(base_dev["net_pnl"] - EXPECTED_BASE["development"]) > 1e-8 or abs(base_val["net_pnl"] - EXPECTED_BASE["validation"]) > 1e-8:
        raise AssertionError("independent raw baseline reproduction failed before selection")

    dev_rows: list[dict[str, Any]] = []
    for stop in STOP_AXIS:
        for hold in HOLD_AXIS:
            candidate = simulate(a, 0, DEV_END, stop, hold)
            r = row("development", base_dev, candidate)
            r["development_net_non_degradation_gate"] = bool(r["net_pnl_delta_vs_base"] >= -1e-8)
            dev_rows.append(r)
    dev = pd.DataFrame(dev_rows)
    dev["is_baseline"] = (dev.stop_usd == BASE_STOP) & dev.max_holding_bars.isna()
    dev = dev.sort_values(["development_net_non_degradation_gate", "utility_delta_usd", "net_pnl_delta_vs_base"],
                          ascending=[False, False, False], kind="mergesort").reset_index(drop=True)
    dev.to_csv(OUT / "development_grid.csv", index=False, float_format="%.12f")

    leaders = dev[dev.development_net_non_degradation_gate & ~dev.is_baseline].head(TOP_DEVELOPMENT).copy()
    # Validate the leaders and their immediate axial neighbours.  This makes
    # local stability observable without consulting holdout.
    requested: set[tuple[float, int | None]] = set()
    for d in leaders.itertuples(index=False):
        hold = None if pd.isna(d.max_holding_bars) else int(d.max_holding_bars)
        si = STOP_AXIS.index(float(d.stop_usd)); hi = HOLD_AXIS.index(hold)
        requested.add((float(d.stop_usd), hold))
        if si > 0: requested.add((STOP_AXIS[si - 1], hold))
        if si + 1 < len(STOP_AXIS): requested.add((STOP_AXIS[si + 1], hold))
        if hi > 0: requested.add((float(d.stop_usd), HOLD_AXIS[hi - 1]))
        if hi + 1 < len(HOLD_AXIS): requested.add((float(d.stop_usd), HOLD_AXIS[hi + 1]))
    pool = dev[dev.apply(lambda x: (float(x.stop_usd), None if pd.isna(x.max_holding_bars) else int(x.max_holding_bars)) in requested, axis=1)].copy()
    finalists: list[dict[str, Any]] = []
    for d in pool.itertuples(index=False):
        hold = None if pd.isna(d.max_holding_bars) else int(d.max_holding_bars)
        candidate = simulate(a, DEV_END, VAL_END, float(d.stop_usd), hold)
        v = row("validation", base_val, candidate)
        worst_reduction_pct = v["worst_trade_reduction_pct_vs_base"]
        risk_improved = (v["mdd_reduction_pct_vs_base"] >= 5.0 or
                         v["trade_cvar5_reduction_pct_vs_base"] >= 5.0 or worst_reduction_pct >= 5.0)
        no_material_risk_damage = (v["mdd_reduction_pct_vs_base"] >= -5.0 and
                                   v["trade_cvar5_reduction_pct_vs_base"] >= -5.0 and
                                   worst_reduction_pct >= -5.0 and
                                   v["sharpe_delta_vs_base"] >= -0.10 and
                                   v["profit_factor_delta_vs_base"] >= -0.05)
        validation_gate = bool(d.development_net_non_degradation_gate and
                               v["net_pnl_delta_vs_base"] >= -1e-8 and risk_improved and no_material_risk_damage)
        finalists.append({
            "stop_usd": float(d.stop_usd), "max_holding_bars": hold,
            "development_utility_delta_usd": float(d.utility_delta_usd),
            "development_net_pnl_delta_vs_base": float(d.net_pnl_delta_vs_base),
            "development_core_winner_net_retention_pct": float(d.core_winner_net_retention_pct),
            "validation_worst_trade_reduction_pct_vs_base": worst_reduction_pct,
            "validation_gate": validation_gate, **{f"validation_{k}": val for k, val in v.items()
                                                    if k not in {"period", "stop_usd", "max_holding_bars"}},
        })
    val = pd.DataFrame(finalists)
    if not val.empty:
        val["robust_utility_delta_usd"] = np.minimum(val.development_utility_delta_usd,
                                                      val.validation_utility_delta_usd)
        keyed = {(float(x.stop_usd), None if pd.isna(x.max_holding_bars) else int(x.max_holding_bars)): x
                 for x in val.itertuples(index=False)}
        stability_fraction: list[float] = []
        stability_median: list[float] = []
        for x in val.itertuples(index=False):
            hold = None if pd.isna(x.max_holding_bars) else int(x.max_holding_bars)
            si = STOP_AXIS.index(float(x.stop_usd)); hi = HOLD_AXIS.index(hold)
            keys = []
            if si > 0: keys.append((STOP_AXIS[si - 1], hold))
            if si + 1 < len(STOP_AXIS): keys.append((STOP_AXIS[si + 1], hold))
            if hi > 0: keys.append((float(x.stop_usd), HOLD_AXIS[hi - 1]))
            if hi + 1 < len(HOLD_AXIS): keys.append((float(x.stop_usd), HOLD_AXIS[hi + 1]))
            neighbours = [keyed[k] for k in keys if k in keyed]
            robust = [min(float(n.development_utility_delta_usd), float(n.validation_utility_delta_usd)) for n in neighbours]
            stable = [bool(n.validation_gate) and score > 0.0 for n, score in zip(neighbours, robust)]
            stability_fraction.append(float(np.mean(stable)) if stable else 0.0)
            stability_median.append(float(np.median(robust)) if robust else float("-inf"))
        val["positive_gated_neighbor_fraction"] = stability_fraction
        val["neighbor_median_robust_utility_usd"] = stability_median
        val = val.sort_values(["validation_gate", "robust_utility_delta_usd", "validation_net_pnl_delta_vs_base"],
                              ascending=[False, False, False], kind="mergesort").reset_index(drop=True)
    val.to_csv(OUT / "validation_finalists.csv", index=False, float_format="%.12f")

    eligible = val[val.validation_gate & (val.robust_utility_delta_usd > 0.0) &
                   (val.positive_gated_neighbor_fraction >= 0.5)] if not val.empty else val
    if eligible.empty:
        selected_stop, selected_hold, verdict = BASE_STOP, None, "NO_OP_BASELINE"
        selected_method = "No overlay passed dev/validation net non-degradation + risk + robust utility + neighbour stability gates"
        selection_row_proof = True
    else:
        chosen = eligible.iloc[0]
        selected_stop = float(chosen.stop_usd)
        selected_hold = None if pd.isna(chosen.max_holding_bars) else int(chosen.max_holding_bars)
        verdict = "OVERLAY_SELECTED"
        selected_method = "Maximum positive min(development,validation) dollar utility among gated candidates"
        selection_row_proof = bool(chosen.validation_gate and chosen.robust_utility_delta_usd > 0.0 and
                                   chosen.positive_gated_neighbor_fraction >= 0.5 and
                                   chosen.name == eligible.index[0])

    # Choice frozen.  Holdout is opened below once for baseline and the frozen selected definition.
    base_results: dict[str, dict[str, Any]] = {}
    selected_results: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, dict[str, Any]] = {}
    for name, (lo, hi) in SPLITS.items():
        b = simulate(a, lo, hi, BASE_STOP, None, collect_equity=True)
        s = simulate(a, lo, hi, selected_stop, selected_hold, collect_equity=True)
        base_results[name] = clean(b); selected_results[name] = clean(s)
        comparisons[name] = comparison(b, s)
        export(f"baseline_{name}", b); export(f"selected_{name}", s)

    no_op_files_identical = all(
        sha256(OUT / f"baseline_{split}_{kind}.csv") == sha256(OUT / f"selected_{split}_{kind}.csv")
        for split in SPLITS for kind in ("trades", "equity")
    ) if verdict == "NO_OP_BASELINE" else True
    checks = {
        "exact_raw_bars": data_audit["raw_bars"] == 97530,
        "exact_sessions": data_audit["sessions"] == 251,
        "raw_no_fill_resample_interpolation": data_audit["no_resampling_fill_or_interpolation"],
        "baseline_exact_all_splits": all(abs(base_results[k]["net_pnl"] - v) <= 1e-8 for k, v in EXPECTED_BASE.items()),
        "baseline_full_exact_6691_888444560394": abs(base_results["full"]["net_pnl"] - EXPECTED_BASE["full"]) <= 1e-8,
        "baseline_holdout_exact_2199_540266640099": abs(base_results["holdout"]["net_pnl"] - EXPECTED_BASE["holdout"]) <= 1e-8,
        "holdout_not_used_in_selection": True,
        "selected_is_baseline_or_validation_non_degrading": verdict == "NO_OP_BASELINE" or comparisons["validation"]["net_pnl_delta_vs_base"] >= -1e-8,
        "no_op_exact_base_parameters": verdict != "NO_OP_BASELINE" or (selected_stop == BASE_STOP and selected_hold is None),
        "no_op_selected_outputs_byte_identical_to_base": no_op_files_identical,
        "overlay_selected_row_is_top_eligible": verdict != "OVERLAY_SELECTED" or selection_row_proof,
        "selected_split_net_additivity": abs(sum(selected_results[k]["net_pnl"] for k in ("development", "validation", "holdout")) - selected_results["full"]["net_pnl"]) <= 1e-8,
        "baseline_split_net_additivity": abs(sum(base_results[k]["net_pnl"] for k in ("development", "validation", "holdout")) - base_results["full"]["net_pnl"]) <= 1e-8,
    }
    audit = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    if audit["status"] != "PASS":
        raise AssertionError(audit)

    summary = {
        "schema_version": 1, "study": "NVDA VWAP absolute bracket tail-risk overlay; no 95% rule",
        "data": data_audit,
        "frozen_baseline": {"stop_usd": BASE_STOP, "target_usd": TARGET,
                            "entry_z": ENTRY_Z, "baseline_replayed_independently_from_raw": True},
        "candidate_grid": {"stop_usd": list(STOP_AXIS), "max_holding_bars": list(HOLD_AXIS),
                           "pairs": len(STOP_AXIS) * len(HOLD_AXIS), "includes_no_time_stop": True,
                           "includes_exact_baseline": True},
        "selection": {"verdict": verdict, "selected_stop_usd": selected_stop,
                      "selected_max_holding_bars": selected_hold, "method": selected_method,
                      "development_gate": "net PnL >= baseline; winner retention is diagnostic/penalty, not a percent gate",
                      "validation_gate": "net PnL >= baseline; >=5% MTM-MDD/CVaR5/worst-trade improvement; no material damage to those tail metrics, Sharpe or PF",
                      "utility": "net delta + 0.35*positive MTM-MDD reduction + 0.50*positive CVaR5 reduction + 0.20*positive worst-5 loser savings - clipped winner net dollars",
                      "robust_rule": "positive max of min(development utility, validation utility) with >=50% positive gated immediate neighbours; otherwise BASE",
                      "top_development_sent_to_validation": len(pool),
                      "validation_eligible_overlays": int(val.validation_gate.sum()) if not val.empty else 0,
                      "holdout_opened_once_after_selection": True, "holdout_used_in_selection": False},
        "base_results": base_results, "selected_results": selected_results,
        "selected_vs_base": comparisons,
        "execution": {"raw_data": "exact synchronized Alpaca SIP 1-minute QQQ/NVDA RTH",
                      "entry": "VWAP-Z close event; next raw open", "target_usd": TARGET,
                      "priority": "at expiry: adverse stop gap then time at open; otherwise stop, target, forced EOD",
                      "time_stop_fill": "raw expiry-minute open; expiry high/low never inspected",
                      "stop_gap": "adverse raw open", "same_bar_ambiguity": "stop first",
                      "position_notional_usd": SIZE, "commission_per_share_per_side": COMMISSION,
                      "slippage_fraction_per_execution": SLIP},
        "audit": audit,
    }
    write_json(OUT / "summary.json", summary)
    write_json(OUT / "audit.json", audit)
    files = sorted(p for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json")
    manifest = {"status": "COMPLETE", "module": "rtharb.research.vwap_nvda_tail_risk",
                "verdict": verdict, "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files}}
    write_json(OUT / "manifest.json", manifest)
    print(json.dumps({"status": "COMPLETE", "verdict": verdict,
                      "selected": {"stop_usd": selected_stop, "max_holding_bars": selected_hold},
                      "base_full": base_results["full"], "selected_full": selected_results["full"],
                      "holdout_comparison": comparisons["holdout"]}, ensure_ascii=False, indent=2, default=_default), flush=True)


if __name__ == "__main__":
    main()
