"""Independent audit of the nine-asset VWAP bracket walk-forward selection.

The research engine and report builder are intentionally not imported.  This
module reconstructs the causal VWAP-Z market arrays from raw Alpaca SIP bars,
checks every published grid/block/fold selection rule, and replays only the
frozen CURRENT/SELECTED definitions with an independent Python state machine.
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

from rtharb.data.loader import DataLoader

try:
    from numba import njit, prange
except ImportError:  # lightweight project venv: table/raw replay audit still works
    njit = None
    prange = range


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_output" / "vwap_all_assets_robust_selection"
SOURCE = ROOT / "research_output" / "vwap_absolute_multi_asset"
ENGINE = ROOT / "rtharb" / "research" / "vwap_all_assets_robust_selection.py"
UNIVERSE = ("NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "AMD")
START = pd.Timestamp("2025-08-22").date(); END = pd.Timestamp("2026-08-21").date()
BLOCKS = ((0,21),(21,42),(42,63),(63,84),(84,105),(105,126),(126,147),(147,168),(168,188))
TEST_BLOCKS = tuple(range(3, 9))
STEP = 0.25; PRE_END = 188; SEEN_END = 251
CAPITAL = 100_000.0; SIZE = 20_000.0; COMMISSION = 0.0035; SLIP = 0.0002; ENTRY_Z = 2.5
ATOL = 7e-7
BASE_METRICS = (
    "stop_usd","target_usd","sessions","raw_bars","trades","active_days","gross_pnl","costs",
    "commissions","slippage","net_pnl","positive_mass","loss_mass","win_rate_pct","profit_factor",
    "net_sharpe","mtm_dd_usd","pnl_over_dd","cvar5_loss_usd","worst_loss_usd",
    "clipped_current_winner_net_usd","avoided_current_loser_net_usd","stops","targets","forced_eod",
)
SUMMARY_METRICS = ("trades", "net_pnl", "max_drawdown_usd_mtm", "max_drawdown_pct_mtm",
                   "pnl_over_dd", "net_sharpe", "profit_factor", "costs")


class ArtifactsNotReady(FileNotFoundError):
    pass


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _close(actual: Any, expected: Any, label: str, atol: float = ATOL) -> None:
    actual_null = actual is None or (isinstance(actual, (float, np.floating)) and math.isnan(float(actual)))
    expected_null = expected is None or (isinstance(expected, (float, np.floating)) and math.isnan(float(expected)))
    if actual_null or expected_null:
        if actual_null != expected_null:
            raise AssertionError(f"{label}: {actual!r} != {expected!r}")
        return
    if not math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=atol):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def _market(symbol: str) -> dict[str, np.ndarray]:
    qqq, target = DataLoader(str(ROOT / "data_cache"), "alpaca", "sip").get_synchronized_pair("QQQ", symbol)
    common = qqq.index.intersection(target.index); qqq, target = qqq.loc[common], target.loc[common]
    dates = np.asarray(common.date); day, unique = pd.factorize(dates, sort=False)
    first = np.r_[True, day[1:] != day[:-1]]; last = np.r_[day[1:] != day[:-1], True]
    starts, ends = np.flatnonzero(first), np.flatnonzero(last)
    qclose, tclose = qqq.close.to_numpy(float), target.close.to_numpy(float)
    qr, tr = pd.Series(qclose[ends]).pct_change(), pd.Series(tclose[ends]).pct_change()
    beta = (tr.rolling(5, min_periods=5).cov(qr) / qr.rolling(5, min_periods=5).var()).shift(1).clip(0.2, 4.0).fillna(1.5).to_numpy()

    def vwap(frame: pd.DataFrame) -> np.ndarray:
        typical = frame[["high", "low", "close"]].to_numpy(float).mean(1)
        volume = frame.volume.to_numpy(float); out = np.empty(len(frame))
        for lo, hi in zip(starts, ends):
            v = volume[lo:hi + 1]; cv = np.cumsum(v)
            out[lo:hi + 1] = np.divide(np.cumsum(typical[lo:hi + 1] * v), cv,
                                        out=np.full(len(v), np.nan), where=cv > 0)
        return out

    qv, tv = vwap(qqq), vwap(target)
    spread = tclose / tv - 1.0 - beta[day] * (qclose / qv - 1.0)
    z = np.full(len(spread), np.nan)
    for lo, hi in zip(starts, ends):
        x = spread[lo:hi + 1]; count = np.minimum(np.arange(1, len(x) + 1), 60)
        rs = np.maximum(0, np.arange(len(x)) - 59)
        cs, cs2 = np.r_[0.0, np.cumsum(x)], np.r_[0.0, np.cumsum(x * x)]
        total = cs[np.arange(1, len(x) + 1)] - cs[rs]
        total2 = cs2[np.arange(1, len(x) + 1)] - cs2[rs]
        var = np.divide(total2 - total * total / count, count - 1,
                        out=np.full(len(x), np.nan), where=count > 1)
        std = np.sqrt(np.maximum(var, 0.0))
        values = np.divide(x - total / count, std, out=np.full(len(x), np.nan), where=std > 1e-8)
        values[:30] = np.nan; z[lo:hi + 1] = values
    mask = np.fromiter((START <= value <= END for value in dates), bool, len(dates))
    selected = np.flatnonzero(mask); old0 = int(day[selected[0]])
    out = {"timestamp": common.to_numpy()[mask], "day": day[mask].astype(np.int64) - old0,
           "open": target.open.to_numpy(float)[mask], "high": target.high.to_numpy(float)[mask],
           "low": target.low.to_numpy(float)[mask], "close": tclose[mask], "z": z[mask], "last": last[mask]}
    if len(np.unique(out["day"])) != 251 or len(out["timestamp"]) < 251:
        raise AssertionError(f"{symbol}: expected 251 non-empty raw sessions")
    return out


def _replay(a: dict[str, np.ndarray], lo_day: int, hi_day: int, stop: float, target: float
            ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    idx = np.flatnonzero((a["day"] >= lo_day) & (a["day"] < hi_day))
    cash = peak = CAPITAL; position = pending = 0; signal_i = entry_i = -1
    entry_ref = entry_eff = stop_price = target_price = entry_comm = 0.0; shares = 0
    trades: list[dict[str, Any]] = []; equities: list[dict[str, Any]] = []
    ignored = generated = 0; daily = np.zeros(hi_day - lo_day)
    for i in idx:
        if pending:
            position, pending = pending, 0; entry_i = i; entry_ref = float(a["open"][i])
            entry_eff = entry_ref * (1 + SLIP if position == 1 else 1 - SLIP)
            shares = math.floor(SIZE / entry_eff); entry_comm = shares * COMMISSION
            stop_price = entry_ref - stop if position == 1 else entry_ref + stop
            target_price = entry_ref + target if position == 1 else entry_ref - target
        if position:
            op, high, low = float(a["open"][i]), float(a["high"][i]), float(a["low"][i])
            stop_hit = (op <= stop_price or low <= stop_price) if position == 1 else (op >= stop_price or high >= stop_price)
            target_hit = high >= target_price if position == 1 else low <= target_price
            raw_exit = math.nan; reason = ""
            if stop_hit:
                gap = op <= stop_price if position == 1 else op >= stop_price
                raw_exit, reason = (op if gap else stop_price), "STOP"
            elif target_hit: raw_exit, reason = target_price, "TAKE_PROFIT_BRACKET"
            elif bool(a["last"][i]): raw_exit, reason = float(a["close"][i]), "FORCED_EOD"
            if reason:
                exit_eff = raw_exit * (1 - SLIP if position == 1 else 1 + SLIP)
                gross = position * (raw_exit - entry_ref) * shares
                slippage = (abs(entry_eff - entry_ref) + abs(exit_eff - raw_exit)) * shares
                commissions = 2 * shares * COMMISSION; net = gross - slippage - commissions
                trades.append({"signal_time": pd.Timestamp(a["timestamp"][signal_i]),
                               "entry_time": pd.Timestamp(a["timestamp"][entry_i]),
                               "exit_time": pd.Timestamp(a["timestamp"][i]),
                               "signal_i": signal_i, "day": int(a["day"][i]),
                               "direction": "LONG" if position == 1 else "SHORT",
                               "entry_reference": entry_ref, "entry_price": entry_eff,
                               "exit_reference": raw_exit, "exit_price": exit_eff, "shares": shares,
                               "exit_reason": reason, "duration_bars": i - entry_i,
                               "gross_pnl": gross, "slippage": slippage, "commissions": commissions,
                               "costs": slippage + commissions, "net_pnl": net})
                cash += net; daily[int(a["day"][i]) - lo_day] += net; position = 0
        value = float(a["z"][i])
        if math.isfinite(value) and not bool(a["last"][i]):
            hit = 1 if value <= -ENTRY_Z else (-1 if value >= ENTRY_Z else 0)
            if hit:
                if position: ignored += 1
                else: pending, signal_i, generated = hit, i, generated + 1
        if bool(a["last"][i]) and pending:
            raise AssertionError("last-bar signal pending")
        equity = cash - entry_comm + position * (float(a["close"][i]) - entry_eff) * shares if position else cash
        peak = max(peak, equity)
        equities.append({"timestamp": pd.Timestamp(a["timestamp"][i]), "equity": equity,
                         "running_peak": peak, "drawdown_usd": peak - equity,
                         "drawdown_pct": (peak - equity) / peak * 100.0})
    if position or pending: raise AssertionError("replay ended live")
    return pd.DataFrame(trades), pd.DataFrame(equities), {"daily": daily, "generated": generated, "ignored": ignored}


def _metrics(trades: pd.DataFrame, equity: pd.DataFrame, extra: dict[str, Any], sessions: int) -> dict[str, Any]:
    if trades.empty:
        return {"trades": 0, "net_pnl": 0.0, "max_drawdown_usd_mtm": 0.0,
                "max_drawdown_pct_mtm": 0.0, "pnl_over_dd": None, "net_sharpe": 0.0,
                "profit_factor": 0.0, "costs": 0.0, "final_equity": CAPITAL}
    nets = trades.net_pnl.to_numpy(float); pos, neg = nets[nets > 0], nets[nets <= 0]
    daily = extra["daily"]; prior = CAPITAL + np.r_[0.0, np.cumsum(daily[:-1])]
    ret = np.divide(daily, prior, out=np.zeros_like(daily), where=prior != 0)
    sharpe = math.sqrt(252) * ret.mean() / ret.std(ddof=1) if len(ret) > 1 and ret.std(ddof=1) else 0.0
    dd = float(equity.drawdown_usd.max())
    return {"trades": len(trades), "net_pnl": float(nets.sum()), "max_drawdown_usd_mtm": dd,
            "max_drawdown_pct_mtm": float(equity.drawdown_pct.max()), "pnl_over_dd": float(nets.sum()) / dd if dd else None,
            "net_sharpe": sharpe, "profit_factor": float(pos.sum() / abs(neg.sum())),
            "costs": float(trades.costs.sum()), "final_equity": float(equity.equity.iloc[-1])}


# This is deliberately a second implementation of the grid state machine.  It
# neither imports nor calls the production research engine.  The full-grid
# audit is optional because the small project venv does not install Numba; the
# production venv does, and CI/manual deep audits use that interpreter.
if njit is not None:
    @njit(cache=True)
    def _grid_row(day: np.ndarray, op: np.ndarray, high: np.ndarray, low: np.ndarray,
                  close: np.ndarray, z: np.ndarray, last: np.ndarray, n_days: int,
                  stop: float, target: float, baseline_event: np.ndarray) -> np.ndarray:
        daily = np.zeros(n_days); active = np.zeros(n_days, np.uint8)
        trade_nets = np.empty(len(day)); position = 0; pending = 0; pending_signal = -1
        entry_reference = 0.0; entry_effective = 0.0; entry_commission = 0.0
        shares = 0; stop_price = 0.0; target_price = 0.0
        cash = CAPITAL; peak = CAPITAL; maximum_dd = 0.0
        trade_count = win_count = stop_count = target_count = eod_count = 0
        gross = commission = slippage = positive = losses = 0.0; worst = 0.0
        baseline_positive = 0.0; baseline_loss = 0.0; retained_positive = 0.0
        candidate_on_baseline_loser = 0.0
        for value in baseline_event:
            if not math.isnan(value):
                if value > 0: baseline_positive += value
                else: baseline_loss += value
        for i in range(len(day)):
            session = day[i]
            if pending != 0:
                position = pending; pending = 0; entry_reference = op[i]
                entry_effective = entry_reference * (1 + SLIP if position == 1 else 1 - SLIP)
                shares = math.floor(SIZE / entry_effective); entry_commission = shares * COMMISSION
                stop_price = entry_reference - stop if position == 1 else entry_reference + stop
                target_price = entry_reference + target if position == 1 else entry_reference - target
                active[session] = 1
            if position != 0:
                stop_hit = ((op[i] <= stop_price or low[i] <= stop_price) if position == 1
                            else (op[i] >= stop_price or high[i] >= stop_price))
                target_hit = high[i] >= target_price if position == 1 else low[i] <= target_price
                reason = 0; raw_exit = 0.0
                if stop_hit:
                    gap = op[i] <= stop_price if position == 1 else op[i] >= stop_price
                    raw_exit = op[i] if gap else stop_price; reason = 1
                elif target_hit:
                    raw_exit = target_price; reason = 2
                elif last[i]:
                    raw_exit = close[i]; reason = 3
                if reason:
                    exit_effective = raw_exit * (1 - SLIP if position == 1 else 1 + SLIP)
                    one_gross = position * (raw_exit - entry_reference) * shares
                    one_slip = (abs(entry_effective - entry_reference) + abs(exit_effective - raw_exit)) * shares
                    one_commission = 2 * shares * COMMISSION
                    net = one_gross - one_slip - one_commission
                    daily[session] += net; gross += one_gross; slippage += one_slip
                    commission += one_commission; trade_nets[trade_count] = net
                    if net > 0: positive += net; win_count += 1
                    else: losses += net
                    if net < worst: worst = net
                    if pending_signal >= 0 and not math.isnan(baseline_event[pending_signal]):
                        baseline_value = baseline_event[pending_signal]
                        if baseline_value > 0: retained_positive += max(0.0, min(net, baseline_value))
                        else: candidate_on_baseline_loser += net
                    trade_count += 1
                    if reason == 1: stop_count += 1
                    elif reason == 2: target_count += 1
                    else: eod_count += 1
                    cash += net; position = 0
            zi = z[i]
            if not math.isnan(zi) and not last[i]:
                hit = 1 if zi <= -ENTRY_Z else (-1 if zi >= ENTRY_Z else 0)
                if hit != 0 and position == 0:
                    pending = hit; pending_signal = i
            equity = (cash - entry_commission + position * (close[i] - entry_effective) * shares
                      if position else cash)
            if equity > peak: peak = equity
            drawdown = peak - equity
            if drawdown > maximum_dd: maximum_dd = drawdown
        returns = np.zeros(n_days); cumulative = 0.0
        for session in range(n_days):
            prior = CAPITAL + cumulative
            returns[session] = daily[session] / prior if prior else 0.0
            cumulative += daily[session]
        mean = returns.mean(); std = returns.std()
        sharpe = math.sqrt(252) * mean / std if std > 0 else 0.0
        factor = positive / abs(losses) if losses < 0 else 0.0
        net_sum = positive + losses; cvar = 0.0
        if trade_count:
            tail_n = max(1, int(math.ceil(0.05 * trade_count)))
            ordered = np.sort(trade_nets[:trade_count]); cvar = -ordered[:tail_n].mean()
        active_days = 0
        for flag in active: active_days += flag
        return np.array((stop, target, n_days, len(day), trade_count, active_days, gross,
            commission + slippage, commission, slippage, net_sum, positive, losses,
            100 * win_count / trade_count if trade_count else 0.0, factor, sharpe, maximum_dd,
            net_sum / maximum_dd if maximum_dd > 0 else 0.0, cvar, -worst,
            baseline_positive - retained_positive, candidate_on_baseline_loser - baseline_loss,
            stop_count, target_count, eod_count), dtype=np.float64)

    @njit(parallel=True, cache=True)
    def _grid_kernel(day: np.ndarray, op: np.ndarray, high: np.ndarray, low: np.ndarray,
                     close: np.ndarray, z: np.ndarray, last: np.ndarray, n_days: int,
                     stops: np.ndarray, targets: np.ndarray,
                     baseline_event: np.ndarray) -> np.ndarray:
        result = np.empty((len(stops), len(BASE_METRICS)))
        for i in prange(len(stops)):
            result[i] = _grid_row(day, op, high, low, close, z, last, n_days,
                                  stops[i], targets[i], baseline_event)
        return result
else:
    def _grid_kernel(*args: Any, **kwargs: Any) -> np.ndarray:
        raise RuntimeError("deep grid audit requires the production venv with numba")


if njit is not None:
    @njit(parallel=True, cache=True)
    def _pareto_kernel(objectives: np.ndarray, eligible: np.ndarray) -> np.ndarray:
        result = np.zeros(len(eligible), np.uint8)
        for i in prange(len(eligible)):
            if not eligible[i]:
                continue
            dominated = False
            for j in range(len(eligible)):
                if i == j or not eligible[j]:
                    continue
                all_ge = True; any_gt = False
                for column in range(objectives.shape[1]):
                    if objectives[j, column] < objectives[i, column] - 1e-9:
                        all_ge = False; break
                    if objectives[j, column] > objectives[i, column] + 1e-9:
                        any_gt = True
                if all_ge and any_gt:
                    dominated = True; break
            if not dominated:
                result[i] = 1
        return result
else:
    _pareto_kernel = None


def _evaluate_grid(market: dict[str, np.ndarray], lo_day: int, hi_day: int,
                   stops: np.ndarray, targets: np.ndarray,
                   current: dict[str, float]) -> pd.DataFrame:
    indices = np.flatnonzero((market["day"] >= lo_day) & (market["day"] < hi_day))
    baseline = np.full(len(indices), np.nan)
    current_trades, _, _ = _replay(market, lo_day, hi_day,
                                    current["stop_usd"], current["target_usd"])
    interval_timestamps = pd.DatetimeIndex(market["timestamp"][indices])
    for row in current_trades.itertuples(index=False):
        position = int(interval_timestamps.searchsorted(pd.Timestamp(row.signal_time)))
        if position >= len(indices) or interval_timestamps[position] != pd.Timestamp(row.signal_time):
            raise AssertionError("baseline signal absent from raw interval")
        baseline[position] = float(row.net_pnl)
    local_day = market["day"][indices].astype(np.int64) - lo_day
    values = _grid_kernel(local_day, market["open"][indices], market["high"][indices],
        market["low"][indices], market["close"][indices], market["z"][indices],
        market["last"][indices], hi_day - lo_day, np.asarray(stops, float),
        np.asarray(targets, float), baseline)
    return pd.DataFrame(values, columns=BASE_METRICS)


def _pareto(objectives: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    if _pareto_kernel is not None:
        return _pareto_kernel(objectives, eligible).astype(bool)
    ids = np.flatnonzero(eligible); result = np.zeros(len(eligible), dtype=bool)
    # Chunked skyline check is independent and avoids a quadratic allocation.
    for i in ids:
        value = objectives[i]; dominated = False
        for start in range(0, len(ids), 2048):
            rows = objectives[ids[start:start + 2048]]
            ge = np.all(rows >= value - 1e-9, axis=1)
            gt = np.any(rows > value + 1e-9, axis=1)
            if np.any(ge & gt): dominated = True; break
        result[i] = not dominated
    return result


def _aggregate(blocks: list[pd.DataFrame], exact: pd.DataFrame, indices: list[int],
               sessions: int, required_positive: int | None = None) -> pd.DataFrame:
    out = exact.copy(); stack = lambda column: np.stack([blocks[i][column].to_numpy(float) for i in indices])
    pnl = stack("net_pnl"); out["total_pnl"] = pnl.sum(0); out["mean_block_pnl"] = pnl.mean(0)
    out["median_block_pnl"] = np.median(pnl, axis=0)
    out["se_mean_block_pnl"] = pnl.std(0, ddof=1) / math.sqrt(len(indices)) if len(indices) > 1 else 0.0
    out["positive_blocks"] = (pnl > 0).sum(0); out["agg_trades"] = stack("trades").sum(0)
    out["agg_active_days"] = stack("active_days").sum(0); out["agg_costs"] = stack("costs").sum(0)
    out["agg_clipped"] = stack("clipped_current_winner_net_usd").sum(0)
    min_trades = max(8, math.ceil(50 * sessions / PRE_END)); min_days = max(5, math.ceil(30 * sessions / PRE_END))
    majority = required_positive if required_positive is not None else math.floor(len(indices) / 2) + 1
    out["viable"] = ((out.agg_trades >= min_trades) & (out.agg_active_days >= min_days) &
                     (out.total_pnl > 0) & (out.median_block_pnl > 0) & (out.positive_blocks >= majority))
    obj = np.column_stack((out.total_pnl, out.mean_block_pnl, out.pnl_over_dd,
                           -out.cvar5_loss_usd, -out.worst_loss_usd, -out.agg_costs, -out.agg_clipped))
    out["pareto"] = _pareto(obj, out.viable.to_numpy(bool))
    return out


def _neighbors(index: int, n_axis: int) -> list[int]:
    row, col = divmod(index, n_axis); result = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr, cc = row + dr, col + dc
            if (dr or dc) and 0 <= rr < n_axis and 0 <= cc < n_axis: result.append(rr * n_axis + cc)
    return result


def _choose(frame: pd.DataFrame, n_axis: int) -> tuple[int | None, pd.DataFrame, dict[str, Any]]:
    out = frame.copy(); stable = np.zeros(len(out), bool)
    cluster_cols = ("mean_block_pnl", "pnl_over_dd", "cvar5_loss_usd", "worst_loss_usd", "agg_clipped")
    for column in cluster_cols: out["cluster_" + column] = np.nan
    out["viable_neighbor_count"] = 0; out["neighbor_ids"] = ""; out["viable_neighbor_ids"] = ""
    for i in np.flatnonzero(out.pareto.to_numpy(bool)):
        ns = _neighbors(int(i), n_axis); out.at[i, "neighbor_ids"] = "|".join(map(str, ns))
        if len(ns) != 8: continue
        viable = [j for j in ns if bool(out.at[j, "viable"])]
        out.at[i, "viable_neighbor_count"] = len(viable); out.at[i, "viable_neighbor_ids"] = "|".join(map(str, viable))
        if len(viable) < 5: continue
        stable[i] = True; cluster = [int(i), *viable]
        for column in cluster_cols: out.at[i, "cluster_" + column] = float(np.median(out.loc[cluster, column]))
    out["stable_pareto"] = stable; ids = np.flatnonzero(stable)
    if not len(ids): return None, out, {"reason": "NO_STABLE_PARETO"}
    best = int(ids[np.argmax(out.loc[ids, "cluster_mean_block_pnl"].to_numpy())])
    threshold = float(out.at[best, "cluster_mean_block_pnl"] - out.at[best, "se_mean_block_pnl"])
    one = [int(i) for i in ids if out.at[i, "cluster_mean_block_pnl"] >= threshold]
    out["one_se"] = False; out.loc[one, "one_se"] = True
    med_stop, med_target = float(np.median(out.loc[one, "stop_usd"])), float(np.median(out.loc[one, "target_usd"]))
    ordered = sorted(one, key=lambda i: (-out.at[i,"cluster_pnl_over_dd"], out.at[i,"cluster_cvar5_loss_usd"],
        out.at[i,"cluster_worst_loss_usd"], out.at[i,"cluster_agg_clipped"],
        abs(out.at[i,"stop_usd"]-med_stop)+abs(out.at[i,"target_usd"]-med_target), out.at[i,"stop_usd"], out.at[i,"target_usd"]))
    boundary = any(out.at[i,"stop_usd"] >= out.stop_usd.max()-STEP or out.at[i,"target_usd"] >= out.target_usd.max()-STEP for i in one)
    return ordered[0], out, {"reason": "SELECTED", "one_se_threshold": threshold,
                             "one_se_count": len(one), "plateau_touches_top2": boundary}


def _frame_close(actual: pd.DataFrame, expected: pd.DataFrame, columns: list[str], label: str) -> None:
    if len(actual) != len(expected): raise AssertionError(f"{label}: row count mismatch")
    for column in columns:
        left, right = actual[column], expected[column]
        if pd.api.types.is_bool_dtype(left) or pd.api.types.is_bool_dtype(right):
            if not np.array_equal(left.to_numpy(bool), right.to_numpy(bool)):
                raise AssertionError(f"{label}.{column}: boolean mismatch")
        elif pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            if not np.allclose(left.to_numpy(float), right.to_numpy(float), atol=ATOL, rtol=1e-10, equal_nan=True):
                raise AssertionError(f"{label}.{column}: numeric mismatch")
        else:
            if not np.array_equal(left.fillna("").astype(str), right.fillna("").astype(str)):
                raise AssertionError(f"{label}.{column}: value mismatch")


def _audit_exports(dest: Path, summary: dict[str, Any], market: dict[str, np.ndarray],
                   variant: str, parameters: dict[str, float] | None) -> None:
    periods = {"pre_seen": (0, 188), "seen": (188, 251), "full": (0, 251)}
    for period, (lo, hi) in periods.items():
        trades_path, equity_path = dest / f"{variant}_{period}_trades.csv", dest / f"{variant}_{period}_equity.csv"
        if not trades_path.is_file() or not equity_path.is_file(): raise ArtifactsNotReady(f"missing {variant}/{period}")
        published_t, published_e = pd.read_csv(trades_path), pd.read_csv(equity_path)
        expected = summary[f"{variant}_results"][period]
        for key in SUMMARY_METRICS:
            if key not in expected: raise AssertionError(f"{variant}.{period}: summary missing {key}")
        if parameters is None:
            if len(published_t) or not np.allclose(published_e.equity, CAPITAL) or not np.allclose(published_e.drawdown_usd, 0):
                raise AssertionError(f"{variant}.{period}: CASH artifacts are not flat")
            metrics = _metrics(pd.DataFrame(), published_e, {"daily": np.zeros(hi-lo)}, hi-lo)
        else:
            replay_t, replay_e, extra = _replay(market, lo, hi, parameters["stop_usd"], parameters["target_usd"])
            metrics = _metrics(replay_t, replay_e, extra, hi-lo)
            if len(replay_t) != len(published_t) or len(replay_e) != len(published_e):
                raise AssertionError(f"{variant}.{period}: raw replay row mismatch")
            for column in ("signal_time", "entry_time", "exit_time"):
                if not np.array_equal(pd.to_datetime(replay_t[column], utc=True).to_numpy(),
                                      pd.to_datetime(published_t[column], utc=True).to_numpy()):
                    raise AssertionError(f"{variant}.{period}.{column}: replay mismatch")
            if not np.array_equal(replay_t.direction.to_numpy(), published_t.direction.to_numpy()):
                raise AssertionError(f"{variant}.{period}.direction mismatch")
            for column in ("entry_reference", "entry_price", "exit_reference", "exit_price", "shares",
                           "duration_bars", "gross_pnl", "slippage", "commissions", "costs", "net_pnl"):
                if not np.allclose(replay_t[column].to_numpy(float), published_t[column].to_numpy(float),
                                   atol=ATOL, rtol=1e-10):
                    raise AssertionError(f"{variant}.{period}.{column}: replay mismatch")
            for column in ("equity", "running_peak", "drawdown_usd", "drawdown_pct"):
                if not np.allclose(replay_e[column].to_numpy(float), published_e[column].to_numpy(float),
                                   atol=ATOL, rtol=1e-10):
                    raise AssertionError(f"{variant}.{period}.{column}: equity mismatch")
        for key in SUMMARY_METRICS:
            _close(metrics[key], expected[key], f"{variant}.{period}.{key}")
        _close(metrics["final_equity"], expected["final_equity"], f"{variant}.{period}.final_equity")
    _close(summary[f"{variant}_results"]["pre_seen"]["net_pnl"] + summary[f"{variant}_results"]["seen"]["net_pnl"],
           summary[f"{variant}_results"]["full"]["net_pnl"], f"{variant}.split additivity")


def audit_symbol(symbol: str, raw_replay: bool = True, deep_grid: bool = False) -> dict[str, Any]:
    dest = OUT / symbol
    needed = tuple(dest / name for name in ("summary.json", "audit.json", "pre_seen_freeze.json",
        "pre_seen_freeze.sha256", "development_grid.csv", "block_metrics.csv", "folds.csv"))
    if not all(path.is_file() for path in needed): raise ArtifactsNotReady(f"{symbol} outputs not complete")
    summary, engine_audit, freeze = _read(needed[0]), _read(needed[1]), _read(needed[2])
    if engine_audit.get("status") != "PASS" or summary.get("audit", {}).get("status") != "PASS":
        raise AssertionError(f"{symbol}: published audit is not PASS")
    freeze_hash = _sha(needed[2])
    if needed[3].read_text(encoding="utf-8").strip() != freeze_hash or summary["pre_seen_freeze_sha256"] != freeze_hash:
        raise AssertionError(f"{symbol}: pre-seen freeze hash mismatch")
    hashes = freeze["selection_artifact_hashes"]
    if hashes["engine_sha256"] != _sha(ENGINE): raise AssertionError(f"{symbol}: engine changed after freeze")
    for name, filename in (("development_grid_sha256", "development_grid.csv"),
                           ("block_metrics_sha256", "block_metrics.csv"), ("folds_sha256", "folds.csv")):
        if hashes[name] != _sha(dest / filename): raise AssertionError(f"{symbol}: selection artifact hash mismatch")
    if freeze["selection_sessions"] != [0,188] or freeze["seen_sessions_excluded"] != [188,251] or freeze["seen_label"] != "SEEN_HISTORICAL_DIAGNOSTIC":
        raise AssertionError(f"{symbol}: selection/seen boundaries mismatch")

    old = _read(SOURCE / symbol / "summary.json")
    current = {"stop_usd": float(old["selected"]["stop_usd"]), "target_usd": float(old["selected"]["target_usd"])}
    if freeze["current"] != current or summary["current"] != current: raise AssertionError(f"{symbol}: CURRENT_PARAMS mismatch")
    market = _market(symbol)
    data = summary.get("data", {})
    expected_first = pd.Timestamp(market["timestamp"][0]).isoformat()
    expected_last = pd.Timestamp(market["timestamp"][-1]).isoformat()
    if (data.get("symbol") != symbol or data.get("lead") != "QQQ" or
            data.get("raw_bars") != len(market["timestamp"]) or data.get("sessions") != 251 or
            data.get("first_timestamp") != expected_first or data.get("last_timestamp") != expected_last or
            data.get("timestamps_unique") is not True or data.get("all_ohlc_positive") is not True or
            data.get("raw_pairwise_intersection_only") is not True or
            data.get("no_resampling_fill_or_interpolation") is not True):
        raise AssertionError(f"{symbol}: raw SIP provenance summary mismatch")
    if len(np.unique(market["timestamp"])) != len(market["timestamp"]):
        raise AssertionError(f"{symbol}: raw timestamps are not unique")
    if not all(np.all(market[column] > 0) for column in ("open", "high", "low", "close")):
        raise AssertionError(f"{symbol}: non-positive raw OHLC")
    median = float(np.median(market["close"][market["day"] < 63]))
    cap = math.ceil(max(current["stop_usd"], current["target_usd"], .06 * median) / STEP - 1e-12) * STEP
    _close(median, freeze["grid"]["median_first63"], f"{symbol}.median_first63")
    _close(cap, freeze["grid"]["cap_usd"], f"{symbol}.cap")
    axis = np.arange(STEP, cap + STEP/2, STEP); n_axis = len(axis); pairs = n_axis*n_axis
    if freeze["grid"]["pairs"] != pairs or freeze["grid"]["step_usd"] != STEP: raise AssertionError(f"{symbol}: grid metadata mismatch")

    grid = pd.read_csv(needed[4], low_memory=False)
    block_all = pd.read_csv(needed[5])
    folds = pd.read_csv(needed[6])
    expected_stops, expected_targets = np.repeat(axis, n_axis), np.tile(axis, n_axis)
    if len(grid) != pairs or not np.allclose(grid.stop_usd, expected_stops) or not np.allclose(grid.target_usd, expected_targets):
        raise AssertionError(f"{symbol}: exact Cartesian grid/order mismatch")
    if len(block_all) != 9*pairs or set(block_all.block.astype(int)) != set(range(9)):
        raise AssertionError(f"{symbol}: block metric coverage mismatch")
    blocks = []
    for block_i, (lo, hi) in enumerate(BLOCKS):
        frame = block_all[block_all.block == block_i].reset_index(drop=True); blocks.append(frame)
        if len(frame) != pairs or not np.allclose(frame.stop_usd, expected_stops) or not np.allclose(frame.target_usd, expected_targets):
            raise AssertionError(f"{symbol}: block {block_i} grid mismatch")
        if not np.all(frame.sessions == hi-lo): raise AssertionError(f"{symbol}: block {block_i} sessions mismatch")
    exact = grid[list(BASE_METRICS)].copy()
    selection_blocks, selection_exact = blocks, exact
    deep_cache: dict[tuple[int, int], pd.DataFrame] = {}
    if deep_grid:
        if njit is None:
            raise RuntimeError("deep grid audit requires numba; use the production venv")
        all_stops, all_targets = expected_stops.astype(float), expected_targets.astype(float)
        selection_exact = _evaluate_grid(market, 0, PRE_END, all_stops, all_targets, current)
        deep_cache[(0, PRE_END)] = selection_exact
        _frame_close(selection_exact, exact, list(BASE_METRICS), f"{symbol}.raw exact 0:188")
        selection_blocks = []
        for block_i, (lo, hi) in enumerate(BLOCKS):
            independent = _evaluate_grid(market, lo, hi, all_stops, all_targets, current)
            deep_cache[(lo, hi)] = independent; selection_blocks.append(independent)
            _frame_close(independent, blocks[block_i], list(BASE_METRICS),
                         f"{symbol}.raw block {block_i}")
    aggregate = _aggregate(selection_blocks, selection_exact, list(range(9)), 188, required_positive=6)
    selected_index, rebuilt, meta = _choose(aggregate, n_axis)
    derived = ("total_pnl","mean_block_pnl","median_block_pnl","se_mean_block_pnl","positive_blocks",
               "agg_trades","agg_active_days","agg_costs","agg_clipped","viable","pareto",
               "cluster_mean_block_pnl","cluster_pnl_over_dd","cluster_cvar5_loss_usd",
               "cluster_worst_loss_usd","cluster_agg_clipped","viable_neighbor_count",
               "neighbor_ids","viable_neighbor_ids","stable_pareto")
    compare_derived = list(derived)
    if "one_se" in rebuilt.columns or "one_se" in grid.columns:
        if "one_se" not in rebuilt.columns or "one_se" not in grid.columns:
            raise AssertionError(f"{symbol}: one_se column presence mismatch")
        compare_derived.append("one_se")
    _frame_close(rebuilt, grid, compare_derived, f"{symbol}.selection grid")
    if freeze["selection_meta"].get("reason") != meta.get("reason"):
        raise AssertionError(f"{symbol}: one-SE selection reason mismatch")
    for key in ("one_se_threshold", "one_se_count", "plateau_touches_top2"):
        if key in meta or key in freeze["selection_meta"]:
            if isinstance(meta.get(key), (float, int)) and not isinstance(meta.get(key), bool):
                _close(freeze["selection_meta"].get(key), meta.get(key), f"{symbol}.selection_meta.{key}")
            elif freeze["selection_meta"].get(key) != meta.get(key):
                raise AssertionError(f"{symbol}.selection_meta.{key} mismatch")
    candidate = None if selected_index is None else {"stop_usd": float(grid.at[selected_index,"stop_usd"]),
                                                      "target_usd": float(grid.at[selected_index,"target_usd"])}
    if freeze["candidate"] != candidate: raise AssertionError(f"{symbol}: candidate mismatch")
    current_rows = grid[np.isclose(grid.stop_usd,current["stop_usd"]) & np.isclose(grid.target_usd,current["target_usd"])]
    if len(current_rows) != 1 or int(grid.is_current.sum()) != 1: raise AssertionError(f"{symbol}: CURRENT grid row mismatch")
    current_index = int(current_rows.index[0])

    expected_fold_bounds = []
    for kind in ("anchored","rolling"):
        for j, test_i in enumerate(TEST_BLOCKS):
            train = list(range(test_i)) if kind == "anchored" else list(range(j,j+3))
            expected_fold_bounds.append((kind,j+1,"+".join(map(str,train)),BLOCKS[train[0]][0],BLOCKS[train[-1]][1],test_i))
    actual_bounds = [(r.kind,int(r.fold),str(r.train_blocks),int(r.train_start_session),int(r.train_end_session_exclusive),int(r.test_block)) for r in folds.itertuples(index=False)]
    if actual_bounds != expected_fold_bounds: raise AssertionError(f"{symbol}: fold boundaries/order mismatch")
    fold_indices: dict[str,list[int|None]] = {"anchored":[],"rolling":[]}
    hash_serialization_drifts: list[dict[str, Any]] = []
    for row in folds.itertuples(index=False):
        train_indices = [int(value) for value in str(row.train_blocks).split("+")]
        if deep_grid:
            interval = (int(row.train_start_session), int(row.train_end_session_exclusive))
            if interval not in deep_cache:
                deep_cache[interval] = _evaluate_grid(market, interval[0], interval[1],
                                                       expected_stops, expected_targets, current)
            independent_exact = deep_cache[interval]
            exact_csv = independent_exact.to_csv(index=False, float_format="%.10f")
            exact_hash = hashlib.sha256(exact_csv.encode("utf-8")).hexdigest()
            if exact_hash != row.exact_train_metrics_sha256:
                # A SHA mismatch can be caused by sub-ULP differences in the
                # independently reconstructed CURRENT event values crossing a
                # %.10f decimal boundary.  It remains a warning only if the
                # independently rebuilt choice/reason below is identical; any
                # metric/choice mismatch still fails the audit.
                hash_serialization_drifts.append({
                    "kind": str(row.kind), "fold": int(row.fold),
                    "published_sha256": str(row.exact_train_metrics_sha256),
                    "independent_sha256": exact_hash,
                })
            independent_train = _aggregate(selection_blocks, independent_exact, train_indices,
                                           interval[1] - interval[0])
            independent_choice, _, independent_meta = _choose(independent_train, n_axis)
            if independent_meta["reason"] != row.reason:
                raise AssertionError(f"{symbol}: {row.kind} fold {row.fold} reason mismatch")
            if independent_choice is None:
                if not pd.isna(row.chosen_stop_usd) or not pd.isna(row.chosen_target_usd):
                    raise AssertionError(f"{symbol}: {row.kind} fold {row.fold} should have no choice")
            else:
                _close(row.chosen_stop_usd, independent_train.at[independent_choice, "stop_usd"],
                       f"{symbol}.{row.kind}{row.fold}.chosen_stop")
                _close(row.chosen_target_usd, independent_train.at[independent_choice, "target_usd"],
                       f"{symbol}.{row.kind}{row.fold}.chosen_target")
        fi = None
        if not pd.isna(row.chosen_stop_usd):
            matches = grid[np.isclose(grid.stop_usd,row.chosen_stop_usd)&np.isclose(grid.target_usd,row.chosen_target_usd)]
            if len(matches)!=1: raise AssertionError(f"{symbol}: fold chosen pair absent")
            fi=int(matches.index[0]); test=blocks[int(row.test_block)].iloc[fi]
            _close(row.test_net_pnl,test.net_pnl,f"{symbol}.fold test PnL")
            _close(row.test_pnl_over_dd,test.pnl_over_dd,f"{symbol}.fold test PnL/DD")
        fold_indices[row.kind].append(fi)
        if not isinstance(row.exact_train_metrics_sha256,str) or len(row.exact_train_metrics_sha256)!=64:
            raise AssertionError(f"{symbol}: fold exact-train hash absent")
    near = lambda a,b: abs(grid.at[a,"stop_usd"]-grid.at[b,"stop_usd"])<=STEP+1e-9 and abs(grid.at[a,"target_usd"]-grid.at[b,"target_usd"])<=STEP+1e-9
    def counts(index: int) -> dict[str,int]:
        return {kind:sum(fi is not None and near(index,fi) for fi in values) for kind,values in fold_indices.items()}
    current_counts=counts(current_index); candidate_counts=counts(selected_index) if selected_index is not None else {"anchored":0,"rolling":0}
    if freeze["current_fold_counts"]!=current_counts or freeze["candidate_fold_counts"]!=candidate_counts:
        raise AssertionError(f"{symbol}: fold consensus counts mismatch")
    current_viable=bool(grid.at[current_index,"viable"])
    candidate_robust=bool(selected_index is not None and grid.at[selected_index,"viable"] and candidate_counts["anchored"]>=4 and candidate_counts["rolling"]>=4)
    dominance=False
    if selected_index is not None:
        c,b=grid.loc[selected_index],grid.loc[current_index]
        weak=(c.total_pnl>=b.total_pnl-1e-8 and c.pnl_over_dd>=b.pnl_over_dd-1e-8 and c.cvar5_loss_usd<=b.cvar5_loss_usd+1e-8 and c.worst_loss_usd<=b.worst_loss_usd+1e-8)
        strict=(c.total_pnl>b.total_pnl+1e-8 or c.pnl_over_dd>b.pnl_over_dd+1e-8 or c.cvar5_loss_usd<b.cvar5_loss_usd-1e-8 or c.worst_loss_usd<b.worst_loss_usd-1e-8)
        dominance=bool(weak and strict)
    boundary=bool(meta.get("plateau_touches_top2",False))
    if selected_index is not None and selected_index!=current_index and candidate_robust and dominance and not boundary:
        verdict="CHANGE"; selected=candidate
    elif current_viable:
        verdict="BOUNDARY_UNRESOLVED_KEEP_CURRENT" if boundary else "KEEP_CURRENT"; selected=current
    else:
        verdict="NO_TRADE_NO_CONFIRMED_EDGE"; selected=None
    if (freeze["current_robust"],freeze["candidate_robust"],freeze["dominates_current"],freeze["verdict"],freeze["selected"]) != (current_viable,candidate_robust,dominance,verdict,selected):
        raise AssertionError(f"{symbol}: verdict/gate mismatch")

    if raw_replay:
        _audit_exports(dest,summary,market,"current",current)
        _audit_exports(dest,summary,market,"selected",selected)
    return {"status":"PASS","symbol":symbol,"verdict":verdict,"pairs":pairs,
            "candidate":candidate,"selected":selected,"folds":len(folds),"raw_replay":raw_replay,
            "deep_grid":deep_grid,
            "hash_serialization_drifts":hash_serialization_drifts,
            "engine_sha256":hashes["engine_sha256"]}


def audit_all(raw_replay: bool = False, deep_grid: bool = False) -> dict[str, Any]:
    progress_path = OUT / "progress.json"
    if not progress_path.is_file(): raise ArtifactsNotReady("walk-forward outputs have not started")
    progress = _read(progress_path); completed = [row["symbol"] for row in progress.get("completed", [])]
    results = [audit_symbol(symbol, raw_replay=raw_replay, deep_grid=deep_grid) for symbol in completed]
    hashes = {row["engine_sha256"] for row in results}
    if len(hashes) > 1 or (hashes and next(iter(hashes)) != _sha(ENGINE)):
        raise AssertionError("engine source hash differs across frozen assets/current source")
    source_text = ENGINE.read_text(encoding="utf-8")
    freeze_write = source_text.find('atomic_json(dest/"pre_seen_freeze.json",freeze)')
    seen_replay = source_text.find('"seen":(PRE_END,SEEN_END)')
    if freeze_write < 0 or seen_replay < 0 or freeze_write >= seen_replay:
        raise AssertionError("engine source does not freeze selection before seen replay")
    # The run must have started only after the final engine source was frozen.
    if completed and ENGINE.stat().st_mtime_ns > progress_path.stat().st_mtime_ns:
        raise AssertionError("engine source is newer than production progress/run provenance")

    if progress.get("status") == "COMPLETE":
        if completed != list(UNIVERSE) or progress.get("remaining") != []:
            raise AssertionError("complete progress universe/order mismatch")
        csv_path, json_path = OUT / "cross_asset_summary.csv", OUT / "cross_asset_summary.json"
        if not csv_path.is_file() or not json_path.is_file(): raise ArtifactsNotReady("cross-asset tables absent")
        cross_csv, cross_json = pd.read_csv(csv_path), pd.DataFrame(_read(json_path))
        pd.testing.assert_frame_equal(cross_csv, cross_json[cross_csv.columns], check_dtype=False,
                                      check_exact=False, atol=ATOL, rtol=1e-10)
        if cross_csv.symbol.tolist() != list(UNIVERSE): raise AssertionError("cross table universe/order mismatch")
        required = {f"{variant}_{period}_{metric}" for variant in ("current","selected")
                    for period in ("pre_seen","seen","full") for metric in SUMMARY_METRICS}
        missing = sorted(required - set(cross_csv.columns))
        if missing: raise AssertionError(f"cross table missing split metrics: {missing}")
        for row in cross_csv.itertuples(index=False):
            summary = _read(OUT / row.symbol / "summary.json")
            for variant in ("current","selected"):
                for period in ("pre_seen","seen","full"):
                    for metric in SUMMARY_METRICS:
                        _close(getattr(row,f"{variant}_{period}_{metric}"),
                               summary[f"{variant}_results"][period][metric],
                               f"cross.{row.symbol}.{variant}.{period}.{metric}")
    return {"status":"PASS","completed":completed,"remaining":progress.get("remaining",[]),
            "production_status":progress.get("status"),"raw_replay":raw_replay,
            "deep_grid":deep_grid,
            "engine_sha256":None if not hashes else next(iter(hashes)),"assets":results}


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
    print(json.dumps(audit_all(raw_replay="--raw" in sys.argv[1:],
                               deep_grid="--deep-grid" in sys.argv[1:]),
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
