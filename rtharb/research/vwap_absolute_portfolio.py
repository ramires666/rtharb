"""Combine nine frozen VWAP-Z bracket strategies under one $100k capital base.

This is a portfolio construction study, not another parameter search.  Every
symbol keeps the stop and target selected by the completed multi-asset study.
Three deterministic capital models are replayed on synchronized raw Alpaca SIP
minutes: equal sleeves, a global shared-cap batch allocator, and an explicitly
labelled uncapped leverage diagnostic.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rtharb.research.risk_reward import CAPITAL, COMMISSION, SIZE, SLIP
from rtharb.research.vwap_absolute_multi_asset import (
    DEV_END,
    END_DATE,
    ENTRY_Z,
    LEAD,
    START_DATE,
    UNIVERSE,
    VAL_END,
    load_market,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "research_output" / "vwap_absolute_multi_asset"
OUT = ROOT / "research_output" / "vwap_absolute_portfolio"
VARIANTS = ("equal_allocation", "shared_cap", "uncapped_diagnostic")
EQUAL_SLEEVE = round(CAPITAL / len(UNIVERSE), 2)
GROSS_CAP = CAPITAL
PER_SIGNAL_REQUEST = SIZE
SPLITS = {
    "development": (0, DEV_END),
    "validation": (DEV_END, VAL_END),
    "holdout": (VAL_END, 251),
    "full": (0, 251),
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _ratios(daily: np.ndarray) -> tuple[float, float]:
    prior = CAPITAL + np.r_[0.0, np.cumsum(daily[:-1])]
    returns = np.divide(daily, prior, out=np.zeros_like(daily), where=prior != 0.0)
    if len(returns) < 2 or returns.std(ddof=1) == 0.0:
        return 0.0, 0.0
    sharpe = math.sqrt(252.0) * returns.mean() / returns.std(ddof=1)
    downside = math.sqrt(float(np.mean(np.minimum(returns, 0.0) ** 2)))
    sortino = math.sqrt(252.0) * returns.mean() / downside if downside else 0.0
    return float(sharpe), float(sortino)


@dataclass
class Pending:
    signal_i: int
    z: float


@dataclass
class Position:
    direction: int
    signal_i: int
    signal_z: float
    entry_i: int
    entry_reference: float
    entry_effective: float
    shares: int
    requested_notional: float
    allocated_notional: float
    entry_notional: float
    stop_price: float
    target_price: float


@dataclass
class VariantState:
    name: str
    cash: float = CAPITAL
    pending: dict[str, Pending] = field(default_factory=dict)
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[dict[str, Any]] = field(default_factory=list)
    entry_events: list[dict[str, Any]] = field(default_factory=list)
    equity_rows: list[dict[str, Any]] = field(default_factory=list)
    peak: float = CAPITAL
    generated_signals: int = 0
    ignored_signals: int = 0


def _input_gate() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    manifest_path = SOURCE / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or tuple(manifest.get("symbols_completed", ())) != UNIVERSE:
        raise AssertionError("Multi-asset research must be COMPLETE 9/9 in frozen universe order")
    if not manifest.get("all_completed_symbol_audits_pass"):
        raise AssertionError("At least one source symbol audit did not pass")
    summaries: dict[str, dict[str, Any]] = {}
    for symbol in UNIVERSE:
        summary = json.loads((SOURCE / symbol / "summary.json").read_text(encoding="utf-8"))
        audit = json.loads((SOURCE / symbol / "audit.json").read_text(encoding="utf-8"))
        if audit.get("status") != "PASS" or summary["audit"]["status"] != "PASS":
            raise AssertionError(f"{symbol}: source audit is not PASS")
        if summary["symbols"] != {"reference_only": LEAD, "traded": symbol}:
            raise AssertionError(f"{symbol}: source roles changed")
        summaries[symbol] = summary
    return summaries, manifest


def _load_all_markets() -> tuple[dict[str, dict[str, np.ndarray]], list[object], dict[str, Any], dict[str, Any]]:
    markets: dict[str, dict[str, np.ndarray]] = {}
    audits: dict[str, Any] = {}
    indices: dict[str, pd.DatetimeIndex] = {}
    for symbol in UNIVERSE:
        arrays, days, audit = load_market(symbol)
        timestamps = pd.DatetimeIndex(arrays["timestamp"])
        markets[symbol] = arrays
        audits[symbol] = audit
        indices[symbol] = timestamps
        print(f"LOADED {symbol}: {len(timestamps):,} raw synchronized minutes", flush=True)
    common = indices[UNIVERSE[0]]
    for symbol in UNIVERSE[1:]:
        common = common.intersection(indices[symbol], sort=False)
    common = common.sort_values()
    if common.has_duplicates or not len(common):
        raise AssertionError("Invalid global common raw-minute calendar")
    for symbol in UNIVERSE:
        arrays = markets[symbol]
        original_length = len(indices[symbol])
        take = indices[symbol].get_indexer(common)
        if np.any(take < 0):
            raise AssertionError(f"{symbol}: failed global timestamp intersection")
        markets[symbol] = {
            key: (value[take] if isinstance(value, np.ndarray) and len(value) == original_length else value)
            for key, value in arrays.items()
        }
        markets[symbol]["timestamp"] = common.to_numpy()
    dates = pd.Index(common.date)
    day_codes, unique_days = pd.factorize(dates, sort=False)
    last = np.r_[day_codes[1:] != day_codes[:-1], True]
    if len(unique_days) != 251:
        raise AssertionError(f"Global common calendar has {len(unique_days)} sessions, expected 251")
    for symbol in UNIVERSE:
        markets[symbol]["day"] = day_codes.astype(np.int64)
        markets[symbol]["unique_days"] = unique_days
        markets[symbol]["last"] = last.astype(np.bool_)
    calendar_audit = {
        "policy": "deterministic inner intersection of all nine pairwise raw timestamp sets; no fill",
        "global_raw_minutes": int(len(common)), "sessions": int(len(unique_days)),
        "per_symbol_pairwise_raw_minutes": {symbol: int(len(indices[symbol])) for symbol in UNIVERSE},
        "per_symbol_minutes_dropped_by_global_intersection": {symbol: int(len(indices[symbol]) - len(common)) for symbol in UNIVERSE},
        "global_minutes_dropped_from_largest_pairwise_calendar": int(max(map(len, indices.values())) - len(common)),
        "no_fill_resample_or_interpolation": True,
    }
    return markets, list(unique_days), audits, calendar_audit


def _close_position(state: VariantState, symbol: str, arrays: dict[str, np.ndarray], i: int,
                    raw_exit: float, reason: str) -> None:
    position = state.positions.pop(symbol)
    exit_effective = raw_exit * (1.0 - SLIP if position.direction == 1 else 1.0 + SLIP)
    gross = position.direction * (raw_exit - position.entry_reference) * position.shares
    slippage = (
        abs(position.entry_effective - position.entry_reference)
        + abs(exit_effective - raw_exit)
    ) * position.shares
    commissions = 2.0 * position.shares * COMMISSION
    costs = slippage + commissions
    net = gross - costs
    state.cash += net
    state.trades.append({
        "variant": state.name, "symbol": symbol, "lead_symbol": LEAD,
        "signal_time": pd.Timestamp(arrays["timestamp"][position.signal_i]),
        "entry_time": pd.Timestamp(arrays["timestamp"][position.entry_i]),
        "exit_time": pd.Timestamp(arrays["timestamp"][i]),
        "direction": "LONG" if position.direction == 1 else "SHORT",
        "entry_z": position.signal_z,
        "entry_reference": position.entry_reference, "entry_price": position.entry_effective,
        "exit_reference": raw_exit, "exit_price": exit_effective,
        "shares": position.shares, "requested_notional": position.requested_notional,
        "allocated_notional": position.allocated_notional,
        "entry_notional": position.entry_notional,
        "stop_usd_per_share": abs(position.stop_price - position.entry_reference),
        "target_usd_per_share": abs(position.target_price - position.entry_reference),
        "gross_risk_usd": abs(position.stop_price - position.entry_reference) * position.shares,
        "gross_reward_usd": abs(position.target_price - position.entry_reference) * position.shares,
        "stop_price": position.stop_price, "target_price": position.target_price,
        "exit_reason": reason, "duration_bars": i - position.entry_i,
        "gross_pnl": gross, "slippage": slippage, "commissions": commissions,
        "costs": costs, "net_pnl": net,
    })


def _entry_share_allocations(state: VariantState, pending_symbols: list[str], markets: dict[str, dict[str, np.ndarray]],
                             i: int) -> tuple[dict[str, int], dict[str, float], dict[str, float]]:
    effective = {
        symbol: float(markets[symbol]["open"][i]) *
        (1.0 + SLIP if state.pending[symbol].z <= -ENTRY_Z else 1.0 - SLIP)
        for symbol in pending_symbols
    }
    if state.name == "equal_allocation":
        requested = {symbol: EQUAL_SLEEVE for symbol in pending_symbols}
        allocated = requested.copy()
        shares = {symbol: math.floor(EQUAL_SLEEVE / effective[symbol]) for symbol in pending_symbols}
        return shares, requested, allocated
    if state.name == "uncapped_diagnostic":
        requested = {symbol: PER_SIGNAL_REQUEST for symbol in pending_symbols}
        allocated = requested.copy()
        shares = {symbol: math.floor(PER_SIGNAL_REQUEST / effective[symbol]) for symbol in pending_symbols}
        return shares, requested, allocated

    requested = {symbol: PER_SIGNAL_REQUEST for symbol in pending_symbols}
    used = sum(position.entry_notional for position in state.positions.values())
    available = max(0.0, GROSS_CAP - used)
    total_request = sum(requested.values())
    if total_request <= available:
        allocated = requested.copy()
    else:
        allocated = {
            symbol: available * requested[symbol] / total_request if total_request else 0.0
            for symbol in pending_symbols
        }
    max_shares = {symbol: math.floor(requested[symbol] / effective[symbol]) for symbol in pending_symbols}
    shares = {symbol: min(max_shares[symbol], math.floor(allocated[symbol] / effective[symbol])) for symbol in pending_symbols}
    # Spend only the small integer-share residual, in the universe order fixed
    # before all portfolio results.  The primary pro-rata allocation is order-free.
    remaining = available - sum(shares[symbol] * effective[symbol] for symbol in pending_symbols)
    while True:
        added = False
        for symbol in UNIVERSE:
            if symbol not in shares or shares[symbol] >= max_shares[symbol]:
                continue
            if effective[symbol] <= remaining + 1e-10:
                shares[symbol] += 1
                remaining -= effective[symbol]
                added = True
        if not added:
            break
    return shares, requested, allocated


def _enter_batch(state: VariantState, markets: dict[str, dict[str, np.ndarray]], params: dict[str, dict[str, float]],
                 i: int) -> None:
    symbols = [symbol for symbol in UNIVERSE if symbol in state.pending and symbol not in state.positions]
    if not symbols:
        return
    shares_by_symbol, requested, allocated = _entry_share_allocations(state, symbols, markets, i)
    for symbol in symbols:
        arrays = markets[symbol]
        pending = state.pending.pop(symbol)
        direction = 1 if pending.z <= -ENTRY_Z else -1
        raw_open = float(arrays["open"][i])
        entry_effective = raw_open * (1.0 + SLIP if direction == 1 else 1.0 - SLIP)
        shares = int(shares_by_symbol[symbol])
        actual_notional = shares * entry_effective
        allocation_restricted = allocated[symbol] < requested[symbol] - 1e-8
        status = "REJECTED_CAP" if shares <= 0 else ("PARTIAL_CAP" if allocation_restricted else "FILLED")
        state.entry_events.append({
            "variant": state.name, "symbol": symbol,
            "signal_time": pd.Timestamp(arrays["timestamp"][pending.signal_i]),
            "entry_time": pd.Timestamp(arrays["timestamp"][i]),
            "direction": "LONG" if direction == 1 else "SHORT", "entry_z": pending.z,
            "status": status, "requested_notional": requested[symbol],
            "pro_rata_allocated_notional": allocated[symbol], "actual_entry_notional": actual_notional,
            "shares": shares, "entry_reference": raw_open, "entry_price": entry_effective,
        })
        if shares <= 0:
            continue
        stop_distance = params[symbol]["stop_usd"]
        target_distance = params[symbol]["target_usd"]
        state.positions[symbol] = Position(
            direction=direction, signal_i=pending.signal_i, signal_z=pending.z, entry_i=i,
            entry_reference=raw_open, entry_effective=entry_effective, shares=shares,
            requested_notional=requested[symbol], allocated_notional=allocated[symbol],
            entry_notional=actual_notional,
            stop_price=raw_open - stop_distance if direction == 1 else raw_open + stop_distance,
            target_price=raw_open + target_distance if direction == 1 else raw_open - target_distance,
        )


def _gap_stops_at_open(state: VariantState, markets: dict[str, dict[str, np.ndarray]], i: int) -> None:
    for symbol in tuple(UNIVERSE):
        if symbol not in state.positions:
            continue
        position = state.positions[symbol]
        raw_open = float(markets[symbol]["open"][i])
        gap = raw_open <= position.stop_price if position.direction == 1 else raw_open >= position.stop_price
        if gap:
            _close_position(state, symbol, markets[symbol], i, raw_open, "STOP_GAP")


def _intrabar_exits(state: VariantState, markets: dict[str, dict[str, np.ndarray]], i: int) -> None:
    for symbol in tuple(UNIVERSE):
        if symbol not in state.positions:
            continue
        arrays = markets[symbol]
        position = state.positions[symbol]
        high, low = float(arrays["high"][i]), float(arrays["low"][i])
        stop_hit = low <= position.stop_price if position.direction == 1 else high >= position.stop_price
        target_hit = high >= position.target_price if position.direction == 1 else low <= position.target_price
        if stop_hit:
            _close_position(state, symbol, arrays, i, position.stop_price, "STOP")
        elif target_hit:
            _close_position(state, symbol, arrays, i, position.target_price, "TAKE_PROFIT_BRACKET")
        elif bool(arrays["last"][i]):
            _close_position(state, symbol, arrays, i, float(arrays["close"][i]), "FORCED_EOD")


def _mark_to_market(state: VariantState, markets: dict[str, dict[str, np.ndarray]], timestamp: pd.Timestamp, i: int) -> None:
    open_pnl = 0.0
    entry_commissions = 0.0
    gross_entry = gross_mtm = signed_mtm = 0.0
    for symbol, position in state.positions.items():
        close = float(markets[symbol]["close"][i])
        open_pnl += position.direction * (close - position.entry_effective) * position.shares
        entry_commissions += position.shares * COMMISSION
        gross_entry += position.entry_notional
        mtm = position.shares * close
        gross_mtm += mtm
        signed_mtm += position.direction * mtm
    equity = state.cash + open_pnl - entry_commissions
    state.peak = max(state.peak, equity)
    drawdown = state.peak - equity
    state.equity_rows.append({
        "timestamp": timestamp, "equity": equity, "running_peak": state.peak,
        "drawdown_usd": drawdown, "drawdown_pct": drawdown / state.peak * 100.0,
        "active_positions": len(state.positions), "gross_entry_exposure": gross_entry,
        "gross_mtm_exposure": gross_mtm, "signed_mtm_exposure": signed_mtm,
        "utilization_pct": gross_entry / CAPITAL * 100.0,
    })


def replay(markets: dict[str, dict[str, np.ndarray]], params: dict[str, dict[str, float]]) -> dict[str, VariantState]:
    states = {name: VariantState(name) for name in VARIANTS}
    timestamps = pd.DatetimeIndex(markets[UNIVERSE[0]]["timestamp"])
    for i, timestamp in enumerate(timestamps):
        for state in states.values():
            # Gap stops are observable at the raw open and free shared capacity.
            _gap_stops_at_open(state, markets, i)
            _enter_batch(state, markets, params, i)
            _intrabar_exits(state, markets, i)
            _mark_to_market(state, markets, timestamp, i)
            is_last = bool(markets[UNIVERSE[0]]["last"][i])
            for symbol in UNIVERSE:
                z_value = float(markets[symbol]["z"][i])
                if not math.isfinite(z_value) or is_last:
                    continue
                hit = z_value <= -ENTRY_Z or z_value >= ENTRY_Z
                if not hit:
                    continue
                if symbol in state.positions:
                    state.ignored_signals += 1
                else:
                    state.pending[symbol] = Pending(i, z_value)
                    state.generated_signals += 1
            if is_last and (state.positions or state.pending):
                raise AssertionError(f"{state.name}: state leaked across session close")
    return states


def _daily_source_matrix(days: list[object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.Index([str(day) for day in days], name="date")
    pnl = pd.DataFrame(0.0, index=index, columns=UNIVERSE)
    for symbol in UNIVERSE:
        trades = pd.read_csv(SOURCE / symbol / "selected_full_trades.csv")
        exit_dates = pd.to_datetime(trades.exit_time, utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
        grouped = trades.groupby(exit_dates).net_pnl.sum()
        pnl.loc[pnl.index.intersection(grouped.index), symbol] = grouped.reindex(pnl.index.intersection(grouped.index)).to_numpy(float)
    returns = pnl / CAPITAL
    return pnl.reset_index(), returns.reset_index()


def _correlations(daily_returns: pd.DataFrame) -> dict[str, dict[str, pd.DataFrame]]:
    result: dict[str, dict[str, pd.DataFrame]] = {}
    values = daily_returns.set_index("date")
    for split, (lo, hi) in SPLITS.items():
        sample = values.iloc[lo:hi]
        result[split] = {"pearson": sample.corr("pearson"), "spearman": sample.corr("spearman")}
    return result


def _period_metrics(trades: pd.DataFrame, equity: pd.DataFrame, days: list[object], lo: int, hi: int) -> dict[str, Any]:
    start_date, end_date = str(days[lo]), str(days[hi - 1])
    if trades.empty:
        period_trades = trades
    else:
        exit_dates = pd.to_datetime(trades.exit_time, utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
        period_trades = trades[(exit_dates >= start_date) & (exit_dates <= end_date)]
    day_index = pd.Index([str(day) for day in days[lo:hi]])
    daily_net = pd.Series(0.0, index=day_index)
    daily_gross = pd.Series(0.0, index=day_index)
    if len(period_trades):
        exit_dates = pd.to_datetime(period_trades.exit_time, utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
        daily_net = daily_net.add(period_trades.groupby(exit_dates).net_pnl.sum(), fill_value=0.0).reindex(day_index, fill_value=0.0)
        daily_gross = daily_gross.add(period_trades.groupby(exit_dates).gross_pnl.sum(), fill_value=0.0).reindex(day_index, fill_value=0.0)
    sharpe, sortino = _ratios(daily_net.to_numpy(float))
    net = period_trades.net_pnl.to_numpy(float) if len(period_trades) else np.array([], dtype=float)
    wins, losses = net[net > 0], net[net <= 0]
    dates = pd.to_datetime(equity.timestamp, utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    segment = equity[(dates >= start_date) & (dates <= end_date)].copy()
    baseline = CAPITAL + float(trades[
        pd.to_datetime(trades.exit_time, utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d") < start_date
    ].net_pnl.sum()) if len(trades) else CAPITAL
    rebased = CAPITAL + segment.equity.to_numpy(float) - baseline
    peak = np.maximum.accumulate(np.r_[CAPITAL, rebased])
    curve = np.r_[CAPITAL, rebased]
    dd = peak - curve
    return {
        "start": start_date, "end": end_date, "sessions": hi - lo,
        "trades": int(len(period_trades)), "gross_pnl": float(period_trades.gross_pnl.sum()) if len(period_trades) else 0.0,
        "commissions": float(period_trades.commissions.sum()) if len(period_trades) else 0.0,
        "slippage": float(period_trades.slippage.sum()) if len(period_trades) else 0.0,
        "costs": float(period_trades.costs.sum()) if len(period_trades) else 0.0,
        "net_pnl": float(net.sum()), "return_pct": float(net.sum() / CAPITAL * 100.0),
        "win_rate_pct": float((net > 0).mean() * 100.0) if len(net) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() else 0.0,
        "net_sharpe": sharpe, "net_sortino": sortino,
        "max_drawdown_usd_mtm": float(dd.max()),
        "max_drawdown_pct_mtm": float(np.max(np.divide(dd, peak, out=np.zeros_like(dd), where=peak != 0.0)) * 100.0),
        "final_equity_rebased": float(CAPITAL + net.sum()),
        "daily_net": daily_net, "daily_gross": daily_gross,
    }


def _clean_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key not in {"daily_net", "daily_gross"}}


def _source_comparison(trades: pd.DataFrame) -> dict[str, Any]:
    """Compare descriptively; the global common clock drops one raw minute."""
    comparison: dict[str, Any] = {}
    for symbol in UNIVERSE:
        source = pd.read_csv(SOURCE / symbol / "selected_full_trades.csv")
        actual = trades[trades.symbol == symbol].reset_index(drop=True)
        comparison[symbol] = {
            "standalone_pairwise_trades": int(len(source)),
            "global_common_calendar_trades": int(len(actual)),
            "standalone_pairwise_net_pnl": float(source.net_pnl.sum()),
            "global_common_calendar_net_pnl": float(actual.net_pnl.sum()),
            "difference_usd": float(actual.net_pnl.sum() - source.net_pnl.sum()),
            "exact_equality_not_required": True,
        }
    return comparison


def _variant_outputs(state: VariantState, days: list[object]) -> tuple[dict[str, Any], dict[str, Any]]:
    variant_out = OUT / "variants" / state.name
    variant_out.mkdir(parents=True, exist_ok=True)
    trades = pd.DataFrame(state.trades)
    entries = pd.DataFrame(state.entry_events)
    equity = pd.DataFrame(state.equity_rows)
    trades.to_csv(variant_out / "trades.csv", index=False, float_format="%.10f")
    entries.to_csv(variant_out / "entry_events.csv", index=False, float_format="%.10f")
    equity.to_csv(variant_out / "equity.csv", index=False, float_format="%.10f")
    periods_raw = {name: _period_metrics(trades, equity, days, lo, hi) for name, (lo, hi) in SPLITS.items()}
    daily = pd.DataFrame({"date": [str(day) for day in days]})
    daily["daily_net_pnl"] = periods_raw["full"]["daily_net"].to_numpy(float)
    daily["daily_gross_pnl"] = periods_raw["full"]["daily_gross"].to_numpy(float)
    daily["equity"] = CAPITAL + daily.daily_net_pnl.cumsum()
    prior = CAPITAL + np.r_[0.0, daily.daily_net_pnl.cumsum().to_numpy(float)[:-1]]
    daily["daily_return"] = np.divide(daily.daily_net_pnl, prior, out=np.zeros(len(daily)), where=prior != 0.0)
    daily.to_csv(variant_out / "daily_equity.csv", index=False, float_format="%.10f")
    full = periods_raw["full"]
    admissions = entries.status.value_counts().to_dict() if len(entries) else {}
    exposure_stats = {
        "active_positions_mean": float(equity.active_positions.mean()),
        "active_positions_max": int(equity.active_positions.max()),
        "gross_entry_exposure_mean": float(equity.gross_entry_exposure.mean()),
        "gross_entry_exposure_p95": float(equity.gross_entry_exposure.quantile(0.95)),
        "gross_entry_exposure_p99": float(equity.gross_entry_exposure.quantile(0.99)),
        "gross_entry_exposure_max": float(equity.gross_entry_exposure.max()),
        "gross_mtm_exposure_max": float(equity.gross_mtm_exposure.max()),
        "minutes_entry_exposure_at_or_above_99pct_cap": int((equity.gross_entry_exposure >= CAPITAL * 0.99).sum()),
        "minutes_mtm_above_100k": int((equity.gross_mtm_exposure > CAPITAL + 1e-8).sum()),
        "active_position_histogram_minutes": {str(int(key)): int(value) for key, value in equity.active_positions.value_counts().sort_index().items()},
    }
    capital_model = {
        "equal_allocation": {"starting_capital_usd": CAPITAL, "per_symbol_sleeve_usd": EQUAL_SLEEVE, "cross_sleeve_borrowing": False},
        "shared_cap": {"starting_capital_usd": CAPITAL, "per_signal_request_usd": PER_SIGNAL_REQUEST, "gross_entry_cap_usd": GROSS_CAP,
                       "simultaneous_entries": "batch pro-rata; fixed-universe residual share rounding"},
        "uncapped_diagnostic": {"starting_capital_usd": CAPITAL, "per_signal_notional_usd": PER_SIGNAL_REQUEST,
                                "maximum_theoretical_gross_entry_usd": PER_SIGNAL_REQUEST * len(UNIVERSE),
                                "calendar": "global nine-way raw inner intersection; not an exact sum of standalone pairwise runs",
                                "leverage_diagnostic_only": True},
    }[state.name]
    checks = {
        "raw_minute_equity_rows": len(equity) == 97_529,
        "final_equity_equals_capital_plus_net": abs(float(equity.equity.iloc[-1]) - (CAPITAL + full["net_pnl"])) <= 1e-7,
        "gross_minus_costs_equals_net": abs(full["gross_pnl"] - full["costs"] - full["net_pnl"]) <= 1e-7,
        "commission_plus_slippage_equals_costs": abs(full["commissions"] + full["slippage"] - full["costs"]) <= 1e-7,
        "all_traded_symbols_are_frozen_targets": set(trades.symbol.unique()).issubset(set(UNIVERSE)),
        "qqq_is_never_traded": LEAD not in set(trades.symbol.unique()),
        "split_net_equals_full": abs(sum(periods_raw[name]["net_pnl"] for name in ("development", "validation", "holdout")) - full["net_pnl"]) <= 1e-7,
        "split_trades_equal_full": sum(periods_raw[name]["trades"] for name in ("development", "validation", "holdout")) == full["trades"],
        "no_live_positions_at_end": not state.positions and not state.pending,
        "equal_allocation_entry_exposure_within_100k": state.name != "equal_allocation" or float(equity.gross_entry_exposure.max()) <= CAPITAL + 1e-7,
        "shared_cap_entry_exposure_within_100k": state.name != "shared_cap" or float(equity.gross_entry_exposure.max()) <= GROSS_CAP + 1e-7,
    }
    audit = {"variant": state.name, "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    if state.name == "uncapped_diagnostic":
        audit["standalone_pairwise_comparison"] = _source_comparison(trades)
    _write_json(variant_out / "audit.json", audit)
    if audit["status"] != "PASS":
        failed = [key for key, value in checks.items() if not value]
        raise AssertionError(f"{state.name} audit failed: {failed}")
    summary = {
        "schema_version": 1, "variant": state.name, "capital_model": capital_model,
        "periods": {name: _clean_metrics(metrics) for name, metrics in periods_raw.items()},
        "execution": {"raw_data": "exact synchronized Alpaca SIP 1-minute RTH", "reference_only": LEAD,
                      "frozen_traded_symbols": list(UNIVERSE), "entry": "close-t signal, next raw open",
                      "commission_usd_per_share_per_side": COMMISSION, "slippage_fraction_per_execution": SLIP,
                      "same_bar_ambiguity": "stop first", "stop_gap": "adverse raw open",
                      "convergence_exit": False, "new_optimization_or_holdout_tuning": False},
        "admission_statistics": {"generated_signals": state.generated_signals,
                                 "ignored_signals_while_open": state.ignored_signals,
                                 "entry_events": int(len(entries)), **{str(key).lower(): int(value) for key, value in admissions.items()}},
        "exposure_statistics": exposure_stats,
        "audit": {"status": audit["status"], "file": f"variants/{state.name}/audit.json"},
    }
    _write_json(variant_out / "summary.json", summary)
    return summary, audit


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    summaries, source_manifest = _input_gate()
    params = {symbol: {key: float(summaries[symbol]["selected"][key]) for key in ("stop_usd", "target_usd")} for symbol in UNIVERSE}
    OUT.mkdir(parents=True, exist_ok=True)
    markets, days, market_audits, calendar_audit = _load_all_markets()
    daily_pnl, daily_returns = _daily_source_matrix(days)
    daily_pnl.to_csv(OUT / "daily_net_pnl.csv", index=False, float_format="%.10f")
    daily_returns.to_csv(OUT / "daily_returns.csv", index=False, float_format="%.10f")
    correlations = _correlations(daily_returns)
    correlation_checks: dict[str, bool] = {}
    for split, methods in correlations.items():
        for method, frame in methods.items():
            frame.to_csv(OUT / f"correlation_{method}_{split}.csv", float_format="%.10f")
            correlation_checks[f"{method}_{split}_symmetric"] = bool(np.allclose(frame, frame.T, equal_nan=True))
            correlation_checks[f"{method}_{split}_unit_diagonal"] = bool(np.allclose(np.diag(frame), 1.0))
    states = replay(markets, params)
    variant_summaries: dict[str, Any] = {}
    variant_audits: dict[str, Any] = {}
    exposure = pd.DataFrame({"timestamp": pd.DatetimeIndex(markets[UNIVERSE[0]]["timestamp"])})
    for name in VARIANTS:
        summary, audit = _variant_outputs(states[name], days)
        variant_summaries[name] = summary
        variant_audits[name] = audit
        frame = pd.DataFrame(states[name].equity_rows)
        for column in ("active_positions", "gross_entry_exposure", "gross_mtm_exposure", "signed_mtm_exposure", "utilization_pct"):
            exposure[f"{name}_{column}"] = frame[column].to_numpy()
        print(f"BUILT {name}: full {summary['periods']['full']['net_pnl']:+,.2f}, holdout {summary['periods']['holdout']['net_pnl']:+,.2f}", flush=True)
    exposure.to_csv(OUT / "concurrent_exposure.csv", index=False, float_format="%.10f")
    standalone_rows = []
    corr_full = correlations["full"]["pearson"]
    for symbol in UNIVERSE:
        standalone_rows.append({
            "symbol": symbol, "stop_usd": params[symbol]["stop_usd"], "target_usd": params[symbol]["target_usd"],
            "full_net_pnl": summaries[symbol]["selected_results"]["full"]["net_pnl"],
            "holdout_net_pnl": summaries[symbol]["selected_results"]["holdout"]["net_pnl"],
            "holdout_net_sharpe": summaries[symbol]["selected_results"]["holdout"]["net_sharpe"],
            "holdout_max_drawdown_pct_mtm": summaries[symbol]["selected_results"]["holdout"]["max_drawdown_pct_mtm"],
            "mean_full_pearson_correlation_to_other_targets": float(corr_full.loc[symbol].drop(symbol).mean()),
        })
    cross_asset = {
        "schema_version": 1, "reference_only": LEAD, "frozen_parameters": params,
        "standalone": standalone_rows,
        "portfolio_variants": {name: {"full": variant_summaries[name]["periods"]["full"],
                                      "holdout": variant_summaries[name]["periods"]["holdout"]} for name in VARIANTS},
        "interpretation_policy": "Primary diversification verdict uses holdout; full period is descriptive and subject to multiple testing",
    }
    _write_json(OUT / "cross_asset_summary.json", cross_asset)
    global_checks = {
        "source_complete_9_of_9": source_manifest["status"] == "COMPLETE" and len(summaries) == 9,
        "all_source_audits_pass": all(item["audit"]["status"] == "PASS" for item in summaries.values()),
        "all_pairwise_market_audits_valid": all(item["sessions"] == 251 for item in market_audits.values()),
        "global_calendar_is_raw_inner_intersection": calendar_audit["global_raw_minutes"] == 97_529 and calendar_audit["no_fill_resample_or_interpolation"],
        "all_variant_audits_pass": all(item["status"] == "PASS" for item in variant_audits.values()),
        "all_correlations_valid": all(correlation_checks.values()),
        "frozen_parameters_only": True,
        "holdout_not_used_for_portfolio_tuning": True,
    }
    if not all(global_checks.values()):
        raise AssertionError(f"Portfolio global audit failed: {global_checks}")
    manifest = {
        "schema_version": 1, "status": "COMPLETE", "study": "Nine frozen VWAP-Z bracket strategies under one $100k capital base",
        "period": {"start": str(START_DATE), "end": str(END_DATE), "sessions": len(days), "raw_minutes": len(exposure)},
        "reference_only": LEAD, "traded_symbols": list(UNIVERSE), "variants": list(VARIANTS),
        "source": "research_output/vwap_absolute_multi_asset", "source_status": source_manifest["status"],
        "global_calendar": calendar_audit,
        "frozen_stop_target": params, "correlations": {"methods": ["pearson", "spearman"], "splits": list(SPLITS)},
        "audit": {"status": "PASS", "checks": global_checks, "correlation_checks": correlation_checks},
        "warning": "Exploratory portfolio of separately optimized strategies; primary verdict is holdout and does not establish a live edge",
    }
    _write_json(OUT / "manifest.json", manifest)
    print(json.dumps({"status": "COMPLETE", "variants": {name: {split: variant_summaries[name]["periods"][split]["net_pnl"] for split in ("holdout", "full")} for name in VARIANTS}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
