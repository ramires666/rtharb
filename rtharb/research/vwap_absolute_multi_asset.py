"""Event-driven VWAP-Z fixed-bracket research for nine mega-cap stocks.

QQQ is a reference instrument only.  Every target is traded independently at
the next raw one-minute open after a causal close-bar VWAP-Z signal.  The
development grid uses independent absolute dollar stop and target distances;
validation selects a robust finalist and holdout is opened exactly once.

The large $0.25 grid is evaluated by a compiled Numba kernel.  The selected
configuration is then replayed by the deliberately explicit Python simulator
which exports exact trades and raw-minute mark-to-market equity for audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from numba import njit, prange
except ImportError as exc:  # pragma: no cover - guarded with a clear runtime error
    njit = None
    prange = range
    _NUMBA_IMPORT_ERROR = exc
else:
    _NUMBA_IMPORT_ERROR = None

from rtharb.config import AppConfig
from rtharb.data.loader import DataLoader
from rtharb.research.risk_reward import CAPITAL, COMMISSION, SIZE, SLIP
from rtharb.research.vwap_strategy import vwap_arrays


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_output" / "vwap_absolute_multi_asset"
INPUT_MANIFEST = ROOT / "data_cache" / "mega_cap_sip_manifest.json"
UNIVERSE = ("NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "AMD")
LEAD = "QQQ"
START_DATE = pd.Timestamp("2025-08-22").date()
END_DATE = pd.Timestamp("2026-08-21").date()
DEV_END = 125
VAL_END = 188
ENTRY_Z = 2.5
BETA_DAYS = 5
WINDOW = 60
WARMUP = 30
STEP = 0.25
MIN_INITIAL_MAX = 3.0
HARD_CAP = 20.0
TOP_DEVELOPMENT = 10
MIN_DEV_TRADES = 50
MIN_DEV_TRADING_DAYS = 30

GRID_COLUMNS = (
    "stop_usd", "target_usd", "sessions", "raw_bars", "trades",
    "active_trade_days", "generated_flat_signals", "ignored_signals_while_open",
    "gross_pnl", "costs", "commissions", "slippage", "net_pnl",
    "net_return_pct", "win_rate_pct", "profit_factor", "net_sharpe",
    "net_sortino", "max_drawdown_usd_daily", "max_drawdown_pct_daily",
    "avg_net_trade", "avg_duration_bars", "stops", "targets", "forced_eod",
    "avg_gross_risk_usd", "avg_gross_reward_usd",
)
TRADE_COLUMNS = (
    "symbol", "lead_symbol", "signal_time", "entry_time", "exit_time", "direction",
    "entry_z", "signal_target_close", "signal_target_vwap", "signal_qqq_close",
    "signal_qqq_vwap", "signal_fair_target", "entry_reference", "entry_price",
    "exit_reference", "exit_price", "shares", "notional_usd", "stop_usd_per_share",
    "target_usd_per_share", "gross_risk_usd", "gross_reward_usd", "reward_to_risk",
    "stop_price", "target_price", "exit_reason", "duration_bars", "gross_pnl",
    "slippage", "commissions", "costs", "net_pnl",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean(result: dict[str, Any]) -> dict[str, Any]:
    excluded = {"trades_df", "mtm_df", "daily_net", "daily_gross"}
    return {key: value for key, value in result.items() if key not in excluded}


def _axis(maximum: float) -> np.ndarray:
    count = int(round(maximum / STEP))
    return np.round(np.arange(1, count + 1, dtype=float) * STEP, 2)


def _expanded_max(current: float) -> float:
    expanded = math.ceil((current * 1.5) / STEP - 1e-12) * STEP
    return float(min(HARD_CAP, round(expanded, 2)))


def load_market(symbol: str) -> tuple[dict[str, np.ndarray], list[object], dict[str, Any]]:
    """Load exact synchronized raw Alpaca SIP bars and causal VWAP-Z arrays."""
    cfg = AppConfig.load(str(ROOT / "configs" / "default_config.yaml"))
    lead, target = DataLoader(cfg.cache_dir, "alpaca", "sip").get_synchronized_pair(LEAD, symbol)
    common = lead.index.intersection(target.index)
    lead, target = lead.loc[common], target.loc[common]
    # Beta is computed on all available prehistory before the study slice.
    full = vwap_arrays(lead, target, BETA_DAYS, WINDOW, WARMUP)
    mask = np.fromiter((START_DATE <= ts.date() <= END_DATE for ts in common), bool, len(common))
    selected_idx = np.flatnonzero(mask)
    if not len(selected_idx):
        raise AssertionError(f"{symbol}: no bars in study period")
    old_day0 = int(full["day"][selected_idx[0]])
    old_day1 = int(full["day"][selected_idx[-1]]) + 1
    arrays = {
        key: (value[mask] if isinstance(value, np.ndarray) and len(value) == len(common) else value)
        for key, value in full.items()
    }
    arrays["day"] = np.asarray(arrays["day"], dtype=np.int64) - old_day0
    arrays["unique_days"] = full["unique_days"][old_day0:old_day1]
    raw_target = target.loc[common[mask]]
    raw_lead = lead.loc[common[mask]]
    arrays["high"] = raw_target.high.to_numpy(float)
    arrays["low"] = raw_target.low.to_numpy(float)
    arrays["volume_target"] = raw_target.volume.to_numpy(float)
    arrays["close_lead"] = raw_lead.close.to_numpy(float)
    # Normalize the small vectors passed to Numba.
    arrays["last"] = np.asarray(arrays["last"], dtype=np.bool_)
    arrays["open"] = np.asarray(arrays["open"], dtype=np.float64)
    arrays["close"] = np.asarray(arrays["close"], dtype=np.float64)
    arrays["z"] = np.asarray(arrays["z"], dtype=np.float64)
    arrays["high"] = np.asarray(arrays["high"], dtype=np.float64)
    arrays["low"] = np.asarray(arrays["low"], dtype=np.float64)
    days = list(arrays["unique_days"])
    if len(days) != 251 or pd.Timestamp(days[0]).date() != START_DATE or pd.Timestamp(days[-1]).date() != END_DATE:
        raise AssertionError(f"{symbol}: expected 251 sessions, got {len(days)}")
    if len(arrays["timestamp"]) != len(raw_target):
        raise AssertionError(f"{symbol}: raw target/vector row mismatch")
    if not raw_target.index.equals(pd.DatetimeIndex(arrays["timestamp"])):
        raise AssertionError(f"{symbol}: timestamp order mismatch")
    data_audit = {
        "symbol": symbol,
        "lead": LEAD,
        "raw_bars": int(len(raw_target)),
        "sessions": int(len(days)),
        "first_timestamp": pd.Timestamp(raw_target.index[0]).isoformat(),
        "last_timestamp": pd.Timestamp(raw_target.index[-1]).isoformat(),
        "timestamps_unique": bool(not raw_target.index.has_duplicates),
        "all_ohlc_positive": bool((raw_target[["open", "high", "low", "close"]] > 0).all().all()),
        "raw_pairwise_intersection_only": True,
        "no_resampling_fill_or_interpolation": True,
    }
    return arrays, days, data_audit


if njit is not None:
    @njit(cache=True)
    def _single_grid_metric(day, op, hi, lo, close, z, last, n_days, stop_usd, target_usd):
        daily_net = np.zeros(n_days, dtype=np.float64)
        daily_gross = np.zeros(n_days, dtype=np.float64)
        active_days = np.zeros(n_days, dtype=np.uint8)
        position = 0
        pending = 0
        entry_ref = 0.0
        entry_eff = 0.0
        shares = 0
        stop_price = 0.0
        target_price = 0.0
        entry_i = -1
        generated = 0
        ignored = 0
        trades = 0
        wins = 0
        gross_sum = 0.0
        net_sum = 0.0
        loss_sum = 0.0
        commissions = 0.0
        slippage = 0.0
        durations = 0.0
        stops = 0
        targets = 0
        eod = 0
        shares_total = 0

        for i in range(len(day)):
            d = day[i]
            if pending != 0:
                position = pending
                pending = 0
                entry_i = i
                entry_ref = op[i]
                entry_eff = entry_ref * (1.0 + SLIP if position == 1 else 1.0 - SLIP)
                shares = math.floor(SIZE / entry_eff)
                stop_price = entry_ref - stop_usd if position == 1 else entry_ref + stop_usd
                target_price = entry_ref + target_usd if position == 1 else entry_ref - target_usd
                active_days[d] = 1

            if position != 0:
                stop_hit = (op[i] <= stop_price or lo[i] <= stop_price) if position == 1 else (op[i] >= stop_price or hi[i] >= stop_price)
                target_hit = hi[i] >= target_price if position == 1 else lo[i] <= target_price
                reason = 0
                raw_exit = 0.0
                if stop_hit:
                    gap = op[i] <= stop_price if position == 1 else op[i] >= stop_price
                    raw_exit = op[i] if gap else stop_price
                    reason = 1
                elif target_hit:
                    raw_exit = target_price
                    reason = 2
                elif last[i]:
                    raw_exit = close[i]
                    reason = 3
                if reason != 0:
                    exit_eff = raw_exit * (1.0 - SLIP if position == 1 else 1.0 + SLIP)
                    gross = position * (raw_exit - entry_ref) * shares
                    slip_cost = (abs(entry_eff - entry_ref) + abs(exit_eff - raw_exit)) * shares
                    commission = 2.0 * shares * COMMISSION
                    net = gross - slip_cost - commission
                    daily_net[d] += net
                    daily_gross[d] += gross
                    gross_sum += gross
                    net_sum += net
                    commissions += commission
                    slippage += slip_cost
                    durations += i - entry_i
                    shares_total += shares
                    trades += 1
                    if net > 0.0:
                        wins += 1
                    else:
                        loss_sum += net
                    if reason == 1:
                        stops += 1
                    elif reason == 2:
                        targets += 1
                    else:
                        eod += 1
                    position = 0

            zi = z[i]
            if not math.isnan(zi) and not last[i]:
                hit = 1 if zi <= -ENTRY_Z else (-1 if zi >= ENTRY_Z else 0)
                if hit != 0:
                    if position != 0:
                        ignored += 1
                    else:
                        pending = hit
                        generated += 1

        # Daily return ratios and exact closed-daily drawdown.
        returns = np.zeros(n_days, dtype=np.float64)
        cumulative = 0.0
        peak = CAPITAL
        max_dd = 0.0
        max_dd_pct = 0.0
        for d in range(n_days):
            prior = CAPITAL + cumulative
            returns[d] = daily_net[d] / prior if prior != 0.0 else 0.0
            cumulative += daily_net[d]
            equity = CAPITAL + cumulative
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd
            dd_pct = dd / peak * 100.0 if peak != 0.0 else 0.0
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
        mean = 0.0
        for d in range(n_days):
            mean += returns[d]
        mean /= n_days
        variance = 0.0
        downside_sum = 0.0
        for d in range(n_days):
            variance += (returns[d] - mean) ** 2
            if returns[d] < 0.0:
                downside_sum += returns[d] ** 2
        std = math.sqrt(variance / (n_days - 1)) if n_days > 1 else 0.0
        downside = math.sqrt(downside_sum / n_days)
        sharpe = math.sqrt(252.0) * mean / std if std > 0.0 else 0.0
        sortino = math.sqrt(252.0) * mean / downside if downside > 0.0 else 0.0
        active_count = 0
        for d in range(n_days):
            active_count += active_days[d]
        costs = commissions + slippage
        profit_factor = (net_sum - loss_sum) / abs(loss_sum) if loss_sum < 0.0 else 0.0
        out = np.empty(len(GRID_COLUMNS), dtype=np.float64)
        out[0] = stop_usd; out[1] = target_usd; out[2] = n_days; out[3] = len(day)
        out[4] = trades; out[5] = active_count; out[6] = generated; out[7] = ignored
        out[8] = gross_sum; out[9] = costs; out[10] = commissions; out[11] = slippage
        out[12] = net_sum; out[13] = net_sum / CAPITAL * 100.0
        out[14] = wins / trades * 100.0 if trades else 0.0
        out[15] = profit_factor; out[16] = sharpe; out[17] = sortino
        out[18] = max_dd; out[19] = max_dd_pct
        out[20] = net_sum / trades if trades else 0.0
        out[21] = durations / trades if trades else 0.0
        out[22] = stops; out[23] = targets; out[24] = eod
        out[25] = stop_usd * shares_total / trades if trades else 0.0
        out[26] = target_usd * shares_total / trades if trades else 0.0
        return out


    @njit(parallel=True, cache=True)
    def _grid_kernel(day, op, hi, lo, close, z, last, n_days, stops, targets):
        result = np.empty((len(stops), len(GRID_COLUMNS)), dtype=np.float64)
        for i in prange(len(stops)):
            result[i, :] = _single_grid_metric(
                day, op, hi, lo, close, z, last, n_days, stops[i], targets[i]
            )
        return result


def _period_vectors(arrays: dict[str, np.ndarray], first_day: int, last_day: int) -> tuple[np.ndarray, ...]:
    mask = (arrays["day"] >= first_day) & (arrays["day"] < last_day)
    return (
        np.ascontiguousarray(arrays["day"][mask] - first_day, dtype=np.int64),
        np.ascontiguousarray(arrays["open"][mask], dtype=np.float64),
        np.ascontiguousarray(arrays["high"][mask], dtype=np.float64),
        np.ascontiguousarray(arrays["low"][mask], dtype=np.float64),
        np.ascontiguousarray(arrays["close"][mask], dtype=np.float64),
        np.ascontiguousarray(arrays["z"][mask], dtype=np.float64),
        np.ascontiguousarray(arrays["last"][mask], dtype=np.bool_),
    )


def evaluate_grid(arrays: dict[str, np.ndarray], first_day: int, last_day: int,
                  pairs: Iterable[tuple[float, float]]) -> pd.DataFrame:
    if njit is None:
        raise RuntimeError("Numba is required for the exact $0.25 multi-asset grid") from _NUMBA_IMPORT_ERROR
    pair_list = list(pairs)
    if not pair_list:
        return pd.DataFrame(columns=GRID_COLUMNS)
    stops = np.asarray([pair[0] for pair in pair_list], dtype=np.float64)
    targets = np.asarray([pair[1] for pair in pair_list], dtype=np.float64)
    vectors = _period_vectors(arrays, first_day, last_day)
    values = _grid_kernel(*vectors, last_day - first_day, stops, targets)
    return pd.DataFrame(values, columns=GRID_COLUMNS)


def _ratios(daily: np.ndarray) -> tuple[float, float]:
    prior = CAPITAL + np.r_[0.0, np.cumsum(daily[:-1])]
    returns = np.divide(daily, prior, out=np.zeros_like(daily), where=prior != 0)
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return 0.0, 0.0
    sharpe = math.sqrt(252) * returns.mean() / returns.std(ddof=1)
    downside = np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2))
    sortino = math.sqrt(252) * returns.mean() / downside if downside else 0.0
    return float(sharpe), float(sortino)


def simulate(arrays: dict[str, np.ndarray], symbol: str, first_day: int, last_day: int,
             stop_usd: float, target_usd: float, collect: bool = False) -> dict[str, Any]:
    """Transparent exact replay used for selected results and detailed exports."""
    idx = np.flatnonzero((arrays["day"] >= first_day) & (arrays["day"] < last_day))
    n_days = last_day - first_day
    daily_net = np.zeros(n_days, dtype=float)
    daily_gross = np.zeros(n_days, dtype=float)
    active_days: set[int] = set()
    trades: list[dict[str, Any]] = []
    mtm_rows: list[dict[str, Any]] = []
    cash = CAPITAL
    position = pending = 0
    pending_z = math.nan
    pending_signal_i = -1
    entry_i = -1
    entry_ref = entry_eff = 0.0
    shares = 0
    stop_price = target_price = 0.0
    entry_commission = 0.0
    signal_count = ignored_signals = 0
    stops = targets = forced_eod = 0
    commissions_total = slippage_total = 0.0
    shares_total = 0
    nets: list[float] = []
    grosses: list[float] = []
    durations: list[int] = []
    peak = CAPITAL

    def close_trade(i: int, raw_exit: float, reason: str) -> None:
        nonlocal position, cash, stops, targets, forced_eod
        nonlocal commissions_total, slippage_total, shares_total
        exit_eff = raw_exit * (1.0 - SLIP if position == 1 else 1.0 + SLIP)
        gross = position * (raw_exit - entry_ref) * shares
        slippage = (abs(entry_eff - entry_ref) + abs(exit_eff - raw_exit)) * shares
        commissions = 2.0 * shares * COMMISSION
        costs = slippage + commissions
        net = gross - costs
        day_local = int(arrays["day"][i]) - first_day
        daily_net[day_local] += net
        daily_gross[day_local] += gross
        nets.append(net); grosses.append(gross); durations.append(i - entry_i)
        commissions_total += commissions; slippage_total += slippage; shares_total += shares
        if reason == "STOP":
            stops += 1
        elif reason == "TAKE_PROFIT_BRACKET":
            targets += 1
        else:
            forced_eod += 1
        if collect:
            trades.append({
                "symbol": symbol, "lead_symbol": LEAD,
                "signal_time": pd.Timestamp(arrays["timestamp"][pending_signal_i]),
                "entry_time": pd.Timestamp(arrays["timestamp"][entry_i]),
                "exit_time": pd.Timestamp(arrays["timestamp"][i]),
                "direction": "LONG" if position == 1 else "SHORT",
                "entry_z": float(pending_z),
                "signal_target_close": float(arrays["close"][pending_signal_i]),
                "signal_target_vwap": float(arrays["vwap_target"][pending_signal_i]),
                "signal_qqq_close": float(arrays["close_lead"][pending_signal_i]),
                "signal_qqq_vwap": float(arrays["vwap_lead"][pending_signal_i]),
                "signal_fair_target": float(arrays["fair_price"][pending_signal_i]),
                "entry_reference": entry_ref, "entry_price": entry_eff,
                "exit_reference": raw_exit, "exit_price": exit_eff,
                "shares": shares, "notional_usd": shares * entry_eff,
                "stop_usd_per_share": stop_usd, "target_usd_per_share": target_usd,
                "gross_risk_usd": stop_usd * shares,
                "gross_reward_usd": target_usd * shares,
                "reward_to_risk": target_usd / stop_usd,
                "stop_price": stop_price, "target_price": target_price,
                "exit_reason": reason, "duration_bars": i - entry_i,
                "gross_pnl": gross, "slippage": slippage,
                "commissions": commissions, "costs": costs, "net_pnl": net,
            })
        cash += net
        position = 0

    for i in idx:
        day = int(arrays["day"][i])
        is_first = i == idx[0] or int(arrays["day"][i - 1]) != day
        is_last = bool(arrays["last"][i])
        if is_first and (position or pending):
            raise AssertionError(f"{symbol}: state leaked across RTH sessions")
        if pending:
            position = pending
            pending = 0
            entry_i = i
            entry_ref = float(arrays["open"][i])
            entry_eff = entry_ref * (1.0 + SLIP if position == 1 else 1.0 - SLIP)
            shares = math.floor(SIZE / entry_eff)
            if shares <= 0:
                raise AssertionError(f"{symbol}: invalid position size")
            stop_price = entry_ref - stop_usd if position == 1 else entry_ref + stop_usd
            target_price = entry_ref + target_usd if position == 1 else entry_ref - target_usd
            entry_commission = shares * COMMISSION
            active_days.add(day - first_day)
        if position:
            op = float(arrays["open"][i]); hi = float(arrays["high"][i]); lo = float(arrays["low"][i])
            stop_hit = (op <= stop_price or lo <= stop_price) if position == 1 else (op >= stop_price or hi >= stop_price)
            target_hit = hi >= target_price if position == 1 else lo <= target_price
            if stop_hit:
                gap = op <= stop_price if position == 1 else op >= stop_price
                close_trade(i, op if gap else stop_price, "STOP")
            elif target_hit:
                close_trade(i, target_price, "TAKE_PROFIT_BRACKET")
            elif is_last:
                close_trade(i, float(arrays["close"][i]), "FORCED_EOD")
        z_value = float(arrays["z"][i])
        if math.isfinite(z_value) and not is_last:
            hit = 1 if z_value <= -ENTRY_Z else (-1 if z_value >= ENTRY_Z else 0)
            if hit:
                if position:
                    ignored_signals += 1
                else:
                    pending = hit
                    pending_z = z_value
                    pending_signal_i = i
                    signal_count += 1
        if is_last and pending:
            raise AssertionError(f"{symbol}: final-bar signal became pending")
        if collect:
            equity = (cash - entry_commission + position * (float(arrays["close"][i]) - entry_eff) * shares) if position else cash
            peak = max(peak, equity)
            mtm_rows.append({
                "timestamp": pd.Timestamp(arrays["timestamp"][i]), "equity": equity,
                "running_peak": peak, "drawdown_usd": peak - equity,
                "drawdown_pct": (peak - equity) / peak * 100.0,
            })
    if position or pending:
        raise AssertionError(f"{symbol}: simulation ended with live state")
    net_array = np.asarray(nets); gross_array = np.asarray(grosses)
    wins = net_array[net_array > 0]; losses = net_array[net_array <= 0]
    sharpe, sortino = _ratios(daily_net)
    curve = np.r_[CAPITAL, CAPITAL + np.cumsum(daily_net)]
    curve_peak = np.maximum.accumulate(curve)
    drawdown = curve_peak - curve
    result: dict[str, Any] = {
        "stop_usd": stop_usd, "target_usd": target_usd,
        "sessions": n_days, "raw_bars": len(idx), "trades": len(net_array),
        "active_trade_days": len(active_days),
        "generated_flat_signals": signal_count,
        "ignored_signals_while_open": ignored_signals,
        "gross_pnl": float(gross_array.sum()),
        "costs": float(commissions_total + slippage_total),
        "commissions": float(commissions_total), "slippage": float(slippage_total),
        "net_pnl": float(net_array.sum()), "net_return_pct": float(net_array.sum() / CAPITAL * 100.0),
        "win_rate_pct": float((net_array > 0).mean() * 100.0) if len(net_array) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() else 0.0,
        "net_sharpe": sharpe, "net_sortino": sortino,
        "max_drawdown_usd_daily": float(drawdown.max()),
        "max_drawdown_pct_daily": float(np.max(drawdown / curve_peak) * 100.0),
        "avg_net_trade": float(net_array.mean()) if len(net_array) else 0.0,
        "avg_duration_bars": float(np.mean(durations)) if durations else 0.0,
        "stops": stops, "targets": targets, "forced_eod": forced_eod,
        "avg_gross_risk_usd": float(stop_usd * shares_total / len(net_array)) if len(net_array) else 0.0,
        "avg_gross_reward_usd": float(target_usd * shares_total / len(net_array)) if len(net_array) else 0.0,
    }
    if collect:
        trade_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
        mtm_df = pd.DataFrame(mtm_rows)
        result.update({"trades_df": trade_df, "mtm_df": mtm_df, "daily_net": daily_net, "daily_gross": daily_gross})
        result["max_drawdown_usd_mtm"] = float(mtm_df.drawdown_usd.max())
        result["max_drawdown_pct_mtm"] = float(mtm_df.drawdown_pct.max())
        result["final_equity"] = float(mtm_df.equity.iloc[-1])
        if len(trade_df) != signal_count or abs(float(trade_df.net_pnl.sum()) - result["net_pnl"]) > 1e-8:
            raise AssertionError(f"{symbol}: event/trade/P&L reconciliation failed")
        if abs(result["final_equity"] - (CAPITAL + result["net_pnl"])) > 1e-8:
            raise AssertionError(f"{symbol}: final equity reconciliation failed")
    return result


def _sort_development(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["net_sharpe", "net_pnl", "stop_usd", "target_usd"],
        ascending=[False, False, True, True], kind="mergesort",
    ).reset_index(drop=True)


def development_grid(arrays: dict[str, np.ndarray], median_price: float) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    initial = max(MIN_INITIAL_MAX, math.ceil((median_price * 0.03) / STEP - 1e-12) * STEP)
    stop_max = target_max = float(min(HARD_CAP, round(initial, 2)))
    evaluated: set[tuple[float, float]] = set()
    frames: list[pd.DataFrame] = []
    expansion_log: list[dict[str, Any]] = []
    while True:
        stop_axis, target_axis = _axis(stop_max), _axis(target_max)
        all_pairs = [(float(s), float(t)) for s in stop_axis for t in target_axis]
        new_pairs = [pair for pair in all_pairs if pair not in evaluated]
        if new_pairs:
            frame = evaluate_grid(arrays, 0, DEV_END, new_pairs)
            frames.append(frame)
            evaluated.update(new_pairs)
        grid = pd.concat(frames, ignore_index=True)
        eligible = grid[(grid.trades >= MIN_DEV_TRADES) & (grid.active_trade_days >= MIN_DEV_TRADING_DAYS)]
        if eligible.empty:
            raise RuntimeError("No development candidate meets 50 trades across 30 trading days")
        winner = _sort_development(eligible).iloc[0]
        stop_boundary = bool(float(winner.stop_usd) >= stop_axis[max(0, len(stop_axis) - 2)])
        target_boundary = bool(float(winner.target_usd) >= target_axis[max(0, len(target_axis) - 2)])
        new_stop_max = _expanded_max(stop_max) if stop_boundary and stop_max < HARD_CAP else stop_max
        new_target_max = _expanded_max(target_max) if target_boundary and target_max < HARD_CAP else target_max
        expansion_log.append({
            "round": len(expansion_log) + 1,
            "stop_max_usd": stop_max, "target_max_usd": target_max,
            "combinations_evaluated_total": len(evaluated),
            "eligible_combinations": int(len(eligible)),
            "development_winner": {"stop_usd": float(winner.stop_usd), "target_usd": float(winner.target_usd),
                                   "net_sharpe": float(winner.net_sharpe), "net_pnl": float(winner.net_pnl)},
            "winner_in_top_two_stop_values": stop_boundary,
            "winner_in_top_two_target_values": target_boundary,
            "expanded_stop_axis": bool(new_stop_max > stop_max),
            "expanded_target_axis": bool(new_target_max > target_max),
        })
        if new_stop_max == stop_max and new_target_max == target_max:
            break
        stop_max, target_max = new_stop_max, new_target_max
    return _sort_development(grid), expansion_log


def _match_kernel_and_replay(kernel_row: pd.Series, replay: dict[str, Any], tolerance: float = 1e-7) -> bool:
    integer_fields = ("trades", "active_trade_days", "generated_flat_signals", "ignored_signals_while_open", "stops", "targets", "forced_eod")
    numeric_fields = ("gross_pnl", "costs", "commissions", "slippage", "net_pnl", "net_sharpe", "max_drawdown_usd_daily")
    return all(int(round(kernel_row[field])) == int(replay[field]) for field in integer_fields) and all(
        abs(float(kernel_row[field]) - float(replay[field])) <= tolerance for field in numeric_fields
    )


def run_symbol(symbol: str, input_manifest: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    symbol_out = OUT / symbol
    symbol_out.mkdir(parents=True, exist_ok=True)
    arrays, days, data_audit = load_market(symbol)
    dev_mask = arrays["day"] < DEV_END
    median_price = float(np.median(arrays["close"][dev_mask]))
    grid, expansion_log = development_grid(arrays, median_price)
    grid.to_csv(symbol_out / "development_grid.csv", index=False, float_format="%.10f")
    eligible = grid[(grid.trades >= MIN_DEV_TRADES) & (grid.active_trade_days >= MIN_DEV_TRADING_DAYS)]
    top = _sort_development(eligible).head(TOP_DEVELOPMENT)
    validation_pairs = list(zip(top.stop_usd.astype(float), top.target_usd.astype(float)))
    validation_metrics = evaluate_grid(arrays, DEV_END, VAL_END, validation_pairs)
    finalists = top[["stop_usd", "target_usd", "net_sharpe", "net_pnl", "trades", "active_trade_days"]].copy()
    finalists = finalists.rename(columns={
        "net_sharpe": "development_net_sharpe", "net_pnl": "development_net_pnl",
        "trades": "development_trades", "active_trade_days": "development_active_trade_days",
    }).reset_index(drop=True)
    finalists["validation_net_sharpe"] = validation_metrics.net_sharpe
    finalists["validation_net_pnl"] = validation_metrics.net_pnl
    finalists["validation_trades"] = validation_metrics.trades.astype(int)
    finalists["validation_active_trade_days"] = validation_metrics.active_trade_days.astype(int)
    finalists["robust_score"] = np.minimum(finalists.development_net_sharpe, finalists.validation_net_sharpe)
    finalists = finalists.sort_values(
        ["robust_score", "validation_net_pnl", "development_net_sharpe", "stop_usd", "target_usd"],
        ascending=[False, False, False, True, True], kind="mergesort",
    ).reset_index(drop=True)
    finalists.to_csv(symbol_out / "validation_finalists.csv", index=False, float_format="%.10f")
    chosen = finalists.iloc[0]
    selected = {"stop_usd": float(chosen.stop_usd), "target_usd": float(chosen.target_usd)}
    periods = {
        "development": (0, DEV_END), "validation": (DEV_END, VAL_END),
        "holdout": (VAL_END, len(days)), "full": (0, len(days)),
    }
    selected_results: dict[str, dict[str, Any]] = {}
    replay_results: dict[str, dict[str, Any]] = {}
    for name, (lo, hi) in periods.items():
        replay = simulate(arrays, symbol, lo, hi, **selected, collect=True)
        selected_results[name] = _clean(replay)
        replay_results[name] = replay
        replay["trades_df"].to_csv(symbol_out / f"selected_{name}_trades.csv", index=False, float_format="%.10f")
        replay["mtm_df"].to_csv(symbol_out / f"selected_{name}_equity.csv", index=False, float_format="%.10f")
    full = selected_results["full"]
    selected_dev_row = grid[(grid.stop_usd == selected["stop_usd"]) & (grid.target_usd == selected["target_usd"])].iloc[0]
    selected_val_row = validation_metrics[
        (validation_metrics.stop_usd == selected["stop_usd"]) & (validation_metrics.target_usd == selected["target_usd"])
    ].iloc[0]
    split_names = ("development", "validation", "holdout")
    audit_checks = {
        "data_has_251_sessions": len(days) == 251,
        "data_timestamps_unique": data_audit["timestamps_unique"],
        "data_all_ohlc_positive": data_audit["all_ohlc_positive"],
        "grid_pairs_unique": not grid[["stop_usd", "target_usd"]].duplicated().any(),
        "grid_uses_exact_quarter_dollar_step": bool(np.allclose(grid.stop_usd / STEP, np.round(grid.stop_usd / STEP)) and np.allclose(grid.target_usd / STEP, np.round(grid.target_usd / STEP))),
        "selected_meets_minimum_development_sample": bool(selected_dev_row.trades >= MIN_DEV_TRADES and selected_dev_row.active_trade_days >= MIN_DEV_TRADING_DAYS),
        "selected_development_kernel_equals_replay": _match_kernel_and_replay(selected_dev_row, replay_results["development"]),
        "selected_validation_kernel_equals_replay": _match_kernel_and_replay(selected_val_row, replay_results["validation"]),
        "split_trades_equal_full": sum(selected_results[name]["trades"] for name in split_names) == full["trades"],
        "split_net_equal_full": abs(sum(selected_results[name]["net_pnl"] for name in split_names) - full["net_pnl"]) <= 1e-8,
        "full_generated_signals_equal_trades": full["generated_flat_signals"] == full["trades"],
        "full_exit_reasons_equal_trades": full["stops"] + full["targets"] + full["forced_eod"] == full["trades"],
        "full_commission_plus_slippage_equal_costs": abs(full["commissions"] + full["slippage"] - full["costs"]) <= 1e-8,
        "full_final_equity_equal_capital_plus_net": abs(full["final_equity"] - (CAPITAL + full["net_pnl"])) <= 1e-8,
        "holdout_evaluated_after_selection": True,
    }
    audit = {
        "symbol": symbol, "status": "PASS" if all(audit_checks.values()) else "FAIL",
        "checks": audit_checks, "data": data_audit,
        "raw_input": input_manifest.get("symbols", {}).get(symbol, {}),
    }
    _write_json(symbol_out / "audit.json", audit)
    if audit["status"] != "PASS":
        raise AssertionError(f"{symbol}: audit failed: {audit_checks}")
    summary = {
        "schema_version": 1,
        "study": "Independent raw event-driven QQQ-referenced VWAP-Z fixed dollar brackets",
        "symbols": {"reference_only": LEAD, "traded": symbol},
        "period": {"start": str(START_DATE), "end": str(END_DATE), "sessions": len(days), "raw_bars": len(arrays["timestamp"])},
        "entry_parameters": {"beta_days": BETA_DAYS, "window": WINDOW, "warmup_bars": WARMUP, "z_entry": ENTRY_Z, "hook_delta": 0.0,
                             "entry": "signal at raw close t; traded target fills raw next-minute open t+1"},
        "splits": {
            "development": {"session_indices": [0, DEV_END], "start": str(days[0]), "end": str(days[DEV_END - 1])},
            "validation": {"session_indices": [DEV_END, VAL_END], "start": str(days[DEV_END]), "end": str(days[VAL_END - 1])},
            "holdout": {"session_indices": [VAL_END, len(days)], "start": str(days[VAL_END]), "end": str(days[-1])},
        },
        "grid": {"step_usd": STEP, "development_median_price": median_price,
                 "initial_max_usd": expansion_log[0]["stop_max_usd"], "hard_cap_usd": HARD_CAP,
                 "boundary_rule": "expand an axis by 50% when the development winner is in its top two values; development only",
                 "minimum_sample": {"trades": MIN_DEV_TRADES, "active_trade_days": MIN_DEV_TRADING_DAYS},
                 "unique_combinations": int(len(grid)), "expansion_log": expansion_log},
        "selection": {"method": "top 10 eligible development by daily net Sharpe/net P&L; max min(dev,val Sharpe), validation net P&L tie-break; holdout once",
                      "development_net_sharpe": float(chosen.development_net_sharpe),
                      "validation_net_sharpe": float(chosen.validation_net_sharpe),
                      "robust_score": float(chosen.robust_score), "holdout_opened_after_selection": True,
                      "no_confirmed_edge": bool(float(chosen.robust_score) <= 0 or selected_results["holdout"]["net_pnl"] <= 0)},
        "selected": selected, "selected_results": selected_results,
        "execution": {"raw_data": f"exact synchronized Alpaca SIP 1-minute {LEAD}/{symbol} RTH; no resampling/interpolation",
                      "traded_instrument": symbol, "reference_not_traded": LEAD,
                      "position_notional_usd": SIZE, "starting_capital_usd": CAPITAL,
                      "commission_usd_per_share_per_side": COMMISSION, "slippage_fraction_per_execution": SLIP,
                      "same_bar_ambiguity": "stop first", "stop_gap": "adverse raw open",
                      "while_position_open": "signals ignored", "after_exit": "fresh raw close signal allowed",
                      "convergence_exit": False},
        "audit": {"status": audit["status"], "file": f"{symbol}/audit.json"},
    }
    _write_json(symbol_out / "summary.json", summary)
    cross_row = {
        "symbol": symbol, "reference": LEAD,
        "stop_usd": selected["stop_usd"], "target_usd": selected["target_usd"],
        "reward_to_risk": selected["target_usd"] / selected["stop_usd"],
        "robust_score": float(chosen.robust_score),
        "development_net_pnl": selected_results["development"]["net_pnl"],
        "validation_net_pnl": selected_results["validation"]["net_pnl"],
        "holdout_net_pnl": selected_results["holdout"]["net_pnl"],
        "holdout_trades": selected_results["holdout"]["trades"],
        "holdout_net_sharpe": selected_results["holdout"]["net_sharpe"],
        "full_net_pnl": full["net_pnl"], "full_trades": full["trades"],
        "full_win_rate_pct": full["win_rate_pct"], "full_profit_factor": full["profit_factor"],
        "full_net_sharpe": full["net_sharpe"],
        "full_max_drawdown_usd_mtm": full["max_drawdown_usd_mtm"],
        "full_max_drawdown_pct_mtm": full["max_drawdown_pct_mtm"],
        "full_commissions": full["commissions"], "full_slippage": full["slippage"],
        "full_costs": full["costs"], "audit_status": audit["status"],
    }
    print(f"{symbol}: stop ${selected['stop_usd']:.2f}, target ${selected['target_usd']:.2f}, holdout {selected_results['holdout']['net_pnl']:+,.2f}, full {full['net_pnl']:+,.2f}", flush=True)
    return summary, cross_row


def _selected_symbols(value: str | None) -> tuple[str, ...]:
    if not value:
        return UNIVERSE
    requested = tuple(item.strip().upper() for item in value.split(",") if item.strip())
    invalid = [symbol for symbol in requested if symbol not in UNIVERSE]
    if invalid:
        raise ValueError(f"Unknown symbols {invalid}; allowed: {list(UNIVERSE)}")
    return tuple(symbol for symbol in UNIVERSE if symbol in requested)


def smoke(symbol: str) -> None:
    arrays, days, data_audit = load_market(symbol)
    frame = evaluate_grid(arrays, 0, DEV_END, [(1.0, 1.0)])
    replay = simulate(arrays, symbol, 0, DEV_END, 1.0, 1.0)
    passed = _match_kernel_and_replay(frame.iloc[0], replay)
    print(json.dumps({"smoke": "PASS" if passed else "FAIL", "symbol": symbol,
                      "sessions": len(days), "raw_bars": data_audit["raw_bars"],
                      "kernel_replay_match": passed, "metrics": frame.iloc[0].to_dict()},
                     ensure_ascii=False, indent=2, default=_json_default))
    if not passed:
        raise AssertionError("Compiled kernel does not match Python replay")


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", help="Comma-separated subset for resumable staged execution")
    parser.add_argument("--smoke", action="store_true", help="Compile and reconcile one $1/$1 development run only")
    args = parser.parse_args()
    symbols = _selected_symbols(args.symbols)
    if args.smoke:
        smoke(symbols[0])
        return
    if not INPUT_MANIFEST.is_file():
        raise FileNotFoundError(f"Run rtharb.data.download_mega_cap first: {INPUT_MANIFEST}")
    input_manifest = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))
    if tuple(input_manifest.get("frozen_universe", ())) != UNIVERSE:
        raise AssertionError("Frozen input manifest universe does not match research universe")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1, "status": "RUNNING", "reference_only": LEAD,
        "traded_separately": list(UNIVERSE), "symbols_requested_this_run": list(symbols),
        "study_period": {"start": str(START_DATE), "end": str(END_DATE)},
        "raw_input_manifest": str(INPUT_MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "raw_input_manifest_sha256": _sha256(INPUT_MANIFEST),
        "data_policy": "raw synchronized Alpaca SIP 1-minute RTH only; no synthetic, mock, resampled, filled or interpolated quotes",
        "selection_policy": "development grid and boundary expansion; top 10 to validation; holdout once",
    }
    _write_json(OUT / "manifest.json", manifest)
    rows: list[dict[str, Any]] = []
    existing_path = OUT / "cross_asset_summary.csv"
    if existing_path.is_file():
        existing = pd.read_csv(existing_path)
        rows.extend(existing[~existing.symbol.isin(symbols)].to_dict(orient="records"))
    for symbol in symbols:
        _, row = run_symbol(symbol, input_manifest)
        rows.append(row)
        partial = pd.DataFrame(rows)
        partial["_order"] = partial.symbol.map({symbol_: i for i, symbol_ in enumerate(UNIVERSE)})
        partial = partial.sort_values("_order").drop(columns="_order").reset_index(drop=True)
        partial.to_csv(OUT / "cross_asset_summary.csv", index=False, float_format="%.10f")
        _write_json(OUT / "cross_asset_summary.json", {"schema_version": 1, "rows": partial.to_dict(orient="records")})
    manifest["status"] = "COMPLETE" if set(pd.DataFrame(rows).symbol) == set(UNIVERSE) else "PARTIAL"
    manifest["symbols_completed"] = [symbol for symbol in UNIVERSE if symbol in set(pd.DataFrame(rows).symbol)]
    manifest["all_completed_symbol_audits_pass"] = bool(all(row.get("audit_status") == "PASS" for row in rows))
    _write_json(OUT / "manifest.json", manifest)
    print(f"{manifest['status']}: {len(manifest['symbols_completed'])}/{len(UNIVERSE)} symbols; {OUT}")


if __name__ == "__main__":
    main()
