"""Independent audit for the NVDA VWAP tail-risk overlay artifacts.

The audited research module is deliberately not imported.  Baseline signals
and fills are reconstructed from raw synchronized Alpaca SIP minute bars; the
published grid, validation gates, metrics, hashes and no-op selection are then
reconciled independently.
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


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_output" / "vwap_nvda_tail_risk"
REPORT = ROOT / "tradingview_vwap_nvda_tail_risk"
START = pd.Timestamp("2025-08-22").date()
END = pd.Timestamp("2026-08-21").date()
SPLITS = {"development": (0, 125), "validation": (125, 188),
          "holdout": (188, 251), "full": (0, 251)}
BASE_STOP = 5.25
TARGET = 1.25
ENTRY_Z = 2.5
CAPITAL = 100_000.0
SIZE = 20_000.0
COMMISSION = 0.0035
SLIP = 0.0002
STOP_AXIS = tuple(float(x) for x in np.arange(1.5, 8.0001, 0.25))
HOLD_AXIS: tuple[int | None, ...] = (30, 45, 60, 75, 90, 120, 150, 180,
                                     210, 240, 270, 300, 330, None)
EXPECTED = {"development": 2018.785961020174, "validation": 2473.562216900122,
            "holdout": 2199.5402666400987, "full": 6691.888444560394}
ATOL = 5e-7


class ArtifactsNotReady(FileNotFoundError):
    pass


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _close(actual: Any, expected: Any, label: str, atol: float = ATOL) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=atol):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def _key(stop: Any, hold: Any) -> tuple[float, int | None]:
    return float(stop), None if pd.isna(hold) else int(hold)


def _independent_market() -> dict[str, np.ndarray]:
    """Rebuild causal VWAP-Z arrays without importing the researched engine."""
    qqq, nvda = DataLoader(str(ROOT / "data_cache"), "alpaca", "sip").get_synchronized_pair("QQQ", "NVDA")
    common = qqq.index.intersection(nvda.index)
    qqq, nvda = qqq.loc[common], nvda.loc[common]
    if len(common) != 194_490 or not qqq.index.equals(nvda.index):
        raise AssertionError("raw QQQ/NVDA SIP clock changed")
    dates = np.asarray(common.date)
    day, unique = pd.factorize(dates, sort=False)
    first = np.r_[True, day[1:] != day[:-1]]
    last = np.r_[day[1:] != day[:-1], True]
    starts, ends = np.flatnonzero(first), np.flatnonzero(last)

    qclose = qqq.close.to_numpy(float)
    nclose = nvda.close.to_numpy(float)
    qret = pd.Series(qclose[ends]).pct_change()
    nret = pd.Series(nclose[ends]).pct_change()
    beta_day = (nret.rolling(5, min_periods=5).cov(qret) /
                qret.rolling(5, min_periods=5).var()).shift(1).clip(0.2, 4.0).fillna(1.5).to_numpy()

    def cumulative_vwap(frame: pd.DataFrame) -> np.ndarray:
        typical = frame[["high", "low", "close"]].to_numpy(float).mean(axis=1)
        volume = frame.volume.to_numpy(float)
        result = np.empty(len(frame), dtype=float)
        for lo, hi in zip(starts, ends):
            v = volume[lo:hi + 1]
            cv = np.cumsum(v)
            result[lo:hi + 1] = np.divide(np.cumsum(typical[lo:hi + 1] * v), cv,
                                           out=np.full(len(v), np.nan), where=cv > 0)
        return result

    qvwap, nvwap = cumulative_vwap(qqq), cumulative_vwap(nvda)
    spread = nclose / nvwap - 1.0 - beta_day[day] * (qclose / qvwap - 1.0)
    z = np.full(len(spread), np.nan)
    for lo, hi in zip(starts, ends):
        x = spread[lo:hi + 1]
        count = np.minimum(np.arange(1, len(x) + 1), 60)
        rolling_start = np.maximum(0, np.arange(len(x)) - 59)
        cs, cs2 = np.r_[0.0, np.cumsum(x)], np.r_[0.0, np.cumsum(x * x)]
        total = cs[np.arange(1, len(x) + 1)] - cs[rolling_start]
        total2 = cs2[np.arange(1, len(x) + 1)] - cs2[rolling_start]
        variance = np.divide(total2 - total * total / count, count - 1,
                             out=np.full(len(x), np.nan), where=count > 1)
        std = np.sqrt(np.maximum(variance, 0.0))
        values = np.divide(x - total / count, std, out=np.full(len(x), np.nan), where=std > 1e-8)
        values[:30] = np.nan
        z[lo:hi + 1] = values

    study = np.fromiter((START <= value <= END for value in dates), bool, len(dates))
    idx = np.flatnonzero(study)
    old_first = int(day[idx[0]])
    fair = nvwap * (1.0 + beta_day[day] * (qclose / qvwap - 1.0))
    arrays = {
        "timestamp": common.to_numpy()[study], "day": day[study].astype(np.int64) - old_first,
        "open": nvda.open.to_numpy(float)[study], "high": nvda.high.to_numpy(float)[study],
        "low": nvda.low.to_numpy(float)[study], "close": nclose[study], "z": z[study],
        "last": last[study], "nvwap": nvwap[study], "fair": fair[study],
        "qo": qqq.open.to_numpy(float)[study], "qh": qqq.high.to_numpy(float)[study],
        "ql": qqq.low.to_numpy(float)[study], "qc": qclose[study], "qvwap": qvwap[study],
    }
    if len(arrays["timestamp"]) != 97_530 or len(np.unique(arrays["day"])) != 251:
        raise AssertionError("expected exact 97,530 bars / 251 sessions")
    return arrays


def _raw_baseline_replay(a: dict[str, np.ndarray], stop_usd: float = BASE_STOP,
                         first_day: int = 0, last_day: int = 251
                         ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Independent event state machine for an absolute-stop diagnostic."""
    position = pending = 0
    pending_signal_i = entry_i = -1
    entry_ref = entry_eff = stop_price = target_price = 0.0
    shares = 0; entry_commission = 0.0
    cash = peak = CAPITAL
    generated = ignored = 0
    rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    idx = np.flatnonzero((a["day"] >= first_day) & (a["day"] < last_day))
    for i in idx:
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
            raw_exit = math.nan
            reason = ""
            if stop_hit:
                gap = op <= stop_price if position == 1 else op >= stop_price
                raw_exit, reason = (op if gap else stop_price), "STOP"
            elif target_hit:
                raw_exit, reason = target_price, "TAKE_PROFIT_BRACKET"
            elif bool(a["last"][i]):
                raw_exit, reason = float(a["close"][i]), "FORCED_EOD"
            if reason:
                exit_eff = raw_exit * (1.0 - SLIP if position == 1 else 1.0 + SLIP)
                gross = position * (raw_exit - entry_ref) * shares
                slippage = (abs(entry_eff - entry_ref) + abs(exit_eff - raw_exit)) * shares
                commissions = 2.0 * shares * COMMISSION
                rows.append({
                    "signal_time": pd.Timestamp(a["timestamp"][pending_signal_i]),
                    "entry_time": pd.Timestamp(a["timestamp"][entry_i]),
                    "exit_time": pd.Timestamp(a["timestamp"][i]), "direction": position,
                    "entry_reference": entry_ref, "exit_reference": raw_exit, "shares": shares,
                    "exit_reason": reason, "duration_bars": i - entry_i, "gross_pnl": gross,
                    "slippage": slippage, "commissions": commissions,
                    "costs": slippage + commissions, "net_pnl": gross - slippage - commissions,
                    "day": int(a["day"][i]),
                })
                cash += gross - slippage - commissions
                position = 0
        value = float(a["z"][i])
        if math.isfinite(value) and not bool(a["last"][i]):
            hit = 1 if value <= -ENTRY_Z else (-1 if value >= ENTRY_Z else 0)
            if hit:
                if position:
                    ignored += 1
                else:
                    pending, pending_signal_i = hit, i
                    generated += 1
        if bool(a["last"][i]) and pending:
            raise AssertionError("raw replay left a final-bar entry pending")
        equity = (cash - entry_commission + position * (float(a["close"][i]) - entry_eff) * shares
                  if position else cash)
        peak = max(peak, equity)
        equity_rows.append({"timestamp": pd.Timestamp(a["timestamp"][i]), "equity": equity,
                            "running_peak": peak, "drawdown_usd": peak - equity,
                            "drawdown_pct": (peak - equity) / peak * 100.0,
                            "day": int(a["day"][i])})
    if position or pending:
        raise AssertionError("raw replay ended with live state")
    return pd.DataFrame(rows), pd.DataFrame(equity_rows), {"generated": generated, "ignored": ignored}


def _published_trade_metrics(path: Path, equity_path: Path, metrics: dict[str, Any], label: str) -> pd.DataFrame:
    trades = pd.read_csv(path)
    equity = pd.read_csv(equity_path)
    nets = trades.net_pnl.to_numpy(float)
    positives, losses = nets[nets > 0], nets[nets <= 0]
    tail_n = max(1, math.ceil(0.05 * len(nets)))
    checks = {
        "trades": len(trades), "gross_pnl": trades.gross_pnl.sum(),
        "slippage": trades.slippage.sum(), "commissions": trades.commissions.sum(),
        "costs": trades.costs.sum(), "net_pnl": nets.sum(),
        "positive_net_pnl_mass_usd": positives.sum(),
        "trade_cvar5_loss_usd": -np.sort(nets)[:tail_n].mean(),
        "worst_trade_loss_usd": -nets.min(), "win_rate_pct": (nets > 0).mean() * 100.0,
        "profit_factor": positives.sum() / abs(losses.sum()),
        "avg_net_trade": nets.mean(), "avg_duration_bars": trades.duration_bars.mean(),
        "max_drawdown_usd_mtm": equity.drawdown_usd.max(),
        "max_drawdown_pct_mtm": equity.drawdown_pct.max(),
        "final_equity": equity.equity.iloc[-1],
    }
    for key, value in checks.items():
        _close(value, metrics[key], f"{label}.{key}")
    if not np.allclose(trades.gross_pnl - trades.costs, trades.net_pnl, atol=ATOL, rtol=1e-10):
        raise AssertionError(f"{label}: gross-cost-net reconciliation failed")
    if not np.allclose(trades.slippage + trades.commissions, trades.costs, atol=ATOL, rtol=1e-10):
        raise AssertionError(f"{label}: cost components do not reconcile")
    _close(metrics["final_equity"], CAPITAL + metrics["net_pnl"], f"{label}.capital_plus_net")
    return trades


def _audit_grid(summary: dict[str, Any]) -> dict[str, Any]:
    dev = pd.read_csv(OUT / "development_grid.csv")
    val = pd.read_csv(OUT / "validation_finalists.csv")
    expected_keys = {(stop, hold) for stop in STOP_AXIS for hold in HOLD_AXIS}
    keys = [_key(r.stop_usd, r.max_holding_bars) for r in dev.itertuples(index=False)]
    if len(dev) != 378 or len(set(keys)) != 378 or set(keys) != expected_keys:
        raise AssertionError("development grid is not the exact 27x14 Cartesian product")
    calculated_gate = dev.net_pnl_delta_vs_base >= -1e-8
    if not np.array_equal(calculated_gate.to_numpy(bool), dev.development_net_non_degradation_gate.to_numpy(bool)):
        raise AssertionError("development non-degradation gate mismatch")
    baseline = dev[(dev.stop_usd == BASE_STOP) & dev.max_holding_bars.isna()]
    if len(baseline) != 1 or not bool(baseline.iloc[0].is_baseline):
        raise AssertionError("grid does not contain exactly one explicit baseline")
    _close(baseline.iloc[0].net_pnl_delta_vs_base, 0.0, "grid baseline delta")

    leaders = dev[dev.development_net_non_degradation_gate & ~dev.is_baseline].head(48)
    requested: set[tuple[float, int | None]] = set()
    for r in leaders.itertuples(index=False):
        stop, hold = _key(r.stop_usd, r.max_holding_bars)
        si, hi = STOP_AXIS.index(stop), HOLD_AXIS.index(hold)
        requested.add((stop, hold))
        if si: requested.add((STOP_AXIS[si - 1], hold))
        if si + 1 < len(STOP_AXIS): requested.add((STOP_AXIS[si + 1], hold))
        if hi: requested.add((stop, HOLD_AXIS[hi - 1]))
        if hi + 1 < len(HOLD_AXIS): requested.add((stop, HOLD_AXIS[hi + 1]))
    if {_key(r.stop_usd, r.max_holding_bars) for r in val.itertuples(index=False)} != requested:
        raise AssertionError("validation pool is not the dev leaders plus axial neighbours")
    if len(val) != int(summary["selection"]["top_development_sent_to_validation"]):
        raise AssertionError("published validation pool count mismatch")

    risk_improved = ((val.validation_mdd_reduction_pct_vs_base >= 5.0) |
                     (val.validation_trade_cvar5_reduction_pct_vs_base >= 5.0) |
                     (val.validation_worst_trade_reduction_pct_vs_base >= 5.0))
    no_damage = ((val.validation_mdd_reduction_pct_vs_base >= -5.0) &
                 (val.validation_trade_cvar5_reduction_pct_vs_base >= -5.0) &
                 (val.validation_worst_trade_reduction_pct_vs_base >= -5.0) &
                 (val.validation_sharpe_delta_vs_base >= -0.10) &
                 (val.validation_profit_factor_delta_vs_base >= -0.05))
    exact_gate = ((val.development_net_pnl_delta_vs_base >= -1e-8) &
                  (val.validation_net_pnl_delta_vs_base >= -1e-8) & risk_improved & no_damage)
    if not np.array_equal(exact_gate.to_numpy(bool), val.validation_gate.to_numpy(bool)):
        raise AssertionError("validation gate mismatch")
    robust = np.minimum(val.development_utility_delta_usd, val.validation_utility_delta_usd)
    if not np.allclose(robust, val.robust_utility_delta_usd, atol=ATOL, rtol=1e-10):
        raise AssertionError("robust utility mismatch")
    eligible = val[val.validation_gate & (val.robust_utility_delta_usd > 0.0) &
                       (val.positive_gated_neighbor_fraction >= 0.5)]
    verdict = summary["selection"]["verdict"]
    if eligible.empty:
        if verdict != "NO_OP_BASELINE" or summary["selection"]["selected_stop_usd"] != BASE_STOP or summary["selection"]["selected_max_holding_bars"] is not None:
            raise AssertionError("empty eligible set did not resolve to exact baseline/no-op")
    elif verdict != "OVERLAY_SELECTED":
        raise AssertionError("eligible overlay exists but summary selected no-op")
    if summary["selection"]["holdout_used_in_selection"] is not False:
        raise AssertionError("holdout leakage flag is not false")
    if any("holdout" in column.lower() for column in [*dev.columns, *val.columns]):
        raise AssertionError("selection tables contain holdout-derived fields")
    return {"pairs": len(dev), "validation_rows": len(val), "eligible_overlays": len(eligible)}


def _replay_metrics(trades: pd.DataFrame, equity: pd.DataFrame, counts: dict[str, int],
                    first_day: int, last_day: int, stop_usd: float) -> dict[str, Any]:
    nets = trades.net_pnl.to_numpy(float)
    grosses = trades.gross_pnl.to_numpy(float)
    positives, losses = nets[nets > 0], nets[nets <= 0]
    tail_n = max(1, math.ceil(0.05 * len(nets)))
    daily = np.zeros(last_day - first_day, dtype=float)
    for row in trades.itertuples(index=False):
        daily[int(row.day) - first_day] += float(row.net_pnl)
    prior = CAPITAL + np.r_[0.0, np.cumsum(daily[:-1])]
    returns = np.divide(daily, prior, out=np.zeros_like(daily), where=prior != 0)
    sharpe = (math.sqrt(252.0) * returns.mean() / returns.std(ddof=1)
              if len(returns) > 1 and returns.std(ddof=1) else 0.0)
    downside = math.sqrt(float(np.mean(np.minimum(returns, 0.0) ** 2)))
    sortino = math.sqrt(252.0) * returns.mean() / downside if downside else 0.0
    net = float(nets.sum())
    mdd = float(equity.drawdown_usd.max())
    reasons = trades.exit_reason.value_counts()
    return {
        "stop_usd": stop_usd, "max_holding_bars": None,
        "sessions": last_day - first_day, "raw_bars": len(equity), "trades": len(trades),
        "generated_flat_signals": counts["generated"], "ignored_signals_while_open": counts["ignored"],
        "gross_pnl": float(grosses.sum()), "costs": float(trades.costs.sum()),
        "commissions": float(trades.commissions.sum()), "slippage": float(trades.slippage.sum()),
        "net_pnl": net, "net_return_pct": net / CAPITAL * 100.0,
        "net_sharpe": sharpe, "net_sortino": sortino,
        "profit_factor": float(positives.sum() / abs(losses.sum())),
        "positive_net_pnl_mass_usd": float(positives.sum()),
        "trade_cvar5_loss_usd": float(-np.sort(nets)[:tail_n].mean()),
        "worst_trade_loss_usd": float(-nets.min()),
        "win_rate_pct": float((nets > 0).mean() * 100.0),
        "max_drawdown_usd_mtm": mdd, "max_drawdown_pct_mtm": float(equity.drawdown_pct.max()),
        "return_over_mtm_dd": net / mdd if mdd else 0.0,
        "avg_net_trade": float(nets.mean()), "avg_duration_bars": float(trades.duration_bars.mean()),
        "stops": int(reasons.get("STOP", 0)), "targets": int(reasons.get("TAKE_PROFIT_BRACKET", 0)),
        "time_stops": 0, "forced_eod": int(reasons.get("FORCED_EOD", 0)),
        "final_equity": float(equity.equity.iloc[-1]),
    }


def _assert_payload_metrics(actual: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    if set(actual) != set(expected):
        raise AssertionError(f"{label}: metric schema mismatch")
    for key, value in actual.items():
        expected_value = expected[key]
        if value is None or expected_value is None:
            if value is not expected_value:
                raise AssertionError(f"{label}.{key}: null mismatch")
        elif isinstance(value, (int, float, np.integer, np.floating)):
            _close(value, expected_value, f"{label}.{key}")
        elif value != expected_value:
            raise AssertionError(f"{label}.{key}: {value!r} != {expected_value!r}")


def _float_vector(values: list[Any]) -> np.ndarray:
    return np.asarray([np.nan if value is None else float(value) for value in values], dtype=float)


def _audit_report(summary: dict[str, Any], market: dict[str, np.ndarray]) -> dict[str, Any]:
    manifest_path = REPORT / "manifest.js"
    overview_path = REPORT / "data" / "overview.json"
    stop3_path = REPORT / "data" / "stop3_diagnostic.json"
    index_path = REPORT / "index.html"
    if not all(path.is_file() for path in (manifest_path, overview_path, stop3_path, index_path)):
        raise ArtifactsNotReady("NVDA tail-risk report is not complete")
    manifest_text = manifest_path.read_text(encoding="utf-8").strip()
    prefix = "window.VWAP_NVDA_TAIL_RISK_MANIFEST="
    if not manifest_text.startswith(prefix) or not manifest_text.endswith(";"):
        raise AssertionError("browser manifest assignment/schema mismatch")
    manifest = json.loads(manifest_text[len(prefix):-1])
    if (manifest.get("schema_version"), manifest.get("status"), manifest.get("sessions"), manifest.get("verdict")) != (1, "COMPLETE", 251, "NO_OP_BASELINE"):
        raise AssertionError("browser manifest metadata mismatch")
    if (overview_path.stat().st_size != manifest["overview"]["bytes"] or
            _sha(overview_path) != manifest["overview"]["sha256"]):
        raise AssertionError("overview bytes/hash mismatch")
    for key, source_path in (("source", OUT / "summary.json"), ("audit", OUT / "audit.json")):
        if (source_path.stat().st_size != manifest[key]["bytes"] or
                _sha(source_path) != manifest[key]["sha256"]):
            raise AssertionError(f"manifest {key} source mismatch")

    overview = _read(overview_path)
    meta = overview["meta"]
    if (meta["verdict"], meta["period"]["sessions"], meta["period"]["raw_bars"],
            meta["grid_pairs"], meta["eligible_overlays"]) != ("NO_OP_BASELINE", 251, 97_530, 378, 0):
        raise AssertionError("overview headline metadata mismatch")
    if meta["selected"] != {"stop_usd": 5.25, "target_usd": 1.25, "max_holding_bars": None}:
        raise AssertionError("overview selected baseline mismatch")
    for name, spec in meta["source_files"].items():
        source_path = ROOT / spec["path"]
        if not source_path.is_file() or source_path.stat().st_size != spec["bytes"] or _sha(source_path) != spec["sha256"]:
            raise AssertionError(f"overview source hash mismatch: {name}")

    dev = pd.read_csv(OUT / "development_grid.csv")
    finalists = pd.read_csv(OUT / "validation_finalists.csv")
    report_grid = pd.DataFrame(overview["grid"])
    report_finalists = pd.DataFrame(overview["finalists"])
    if not set(report_grid.columns).issubset(dev.columns) or not set(report_finalists.columns).issubset(finalists.columns):
        raise AssertionError("overview grid/finalist browser schema has unknown fields")
    pd.testing.assert_frame_equal(report_grid, dev[report_grid.columns],
                                  check_dtype=False, check_exact=False, atol=ATOL, rtol=1e-10)
    pd.testing.assert_frame_equal(report_finalists, finalists[report_finalists.columns],
                                  check_dtype=False, check_exact=False, atol=ATOL, rtol=1e-10)
    for split in SPLITS:
        _assert_payload_metrics(overview["results"][split], summary["base_results"][split],
                                f"overview.results.{split}")

    stop3 = _read(stop3_path)
    if stop3["definition"] != {"stop_usd": 3.0, "target_usd": 1.25,
                               "max_holding_bars": None, "selection_role": "diagnostic_only"}:
        raise AssertionError("$3 diagnostic definition mismatch")
    if stop3["results"] != overview["stop3_comparison"]["$3_stop"]:
        raise AssertionError("overview/$3 payload mismatch")
    stop3_results: dict[str, dict[str, Any]] = {}
    for split, (lo, hi) in SPLITS.items():
        trades, equity, counts = _raw_baseline_replay(market, 3.0, lo, hi)
        metrics = _replay_metrics(trades, equity, counts, lo, hi, 3.0)
        _assert_payload_metrics(metrics, stop3["results"][split], f"stop3.{split}")
        stop3_results[split] = metrics
    _close(sum(stop3_results[x]["net_pnl"] for x in ("development", "validation", "holdout")),
           stop3_results["full"]["net_pnl"], "stop3 split additivity")

    sessions = overview["sessions"]
    if len(sessions) != 251 or len({item["date"] for item in sessions}) != 251:
        raise AssertionError("overview does not list 251 unique sessions")
    if sum(int(item["bytes"]) for item in sessions) != int(manifest["session_bytes"]):
        raise AssertionError("session byte total mismatch")
    baseline_equity = pd.read_csv(OUT / "baseline_full_equity.csv")
    baseline_equity["timestamp"] = pd.to_datetime(baseline_equity.timestamp, utc=True)
    published_trades = pd.read_csv(OUT / "baseline_full_trades.csv")
    timestamps = pd.DatetimeIndex(market["timestamp"])
    epoch = np.asarray([int(pd.Timestamp(value).timestamp()) for value in timestamps], dtype=np.int64)
    all_session_trades: list[dict[str, Any]] = []
    daily_t: list[int] = []; daily_equity: list[float] = []; daily_dd: list[float] = []; daily_dd_pct: list[float] = []
    total_bars = total_trades = 0; total_net = 0.0
    array_keys = ("no", "nh", "nl", "nc", "nvwap", "fair", "z",
                  "qo", "qh", "ql", "qc", "qvwap", "equity", "drawdown", "drawdown_pct")
    market_map = {"no": "open", "nh": "high", "nl": "low", "nc": "close",
                  "nvwap": "nvwap", "fair": "fair", "z": "z", "qo": "qo", "qh": "qh",
                  "ql": "ql", "qc": "qc", "qvwap": "qvwap"}
    for day_i, item in enumerate(sessions):
        path = ROOT / "tradingview_vwap_nvda_tail_risk" / item["data"]
        if path.stat().st_size != item["bytes"] or _sha(path) != item["sha256"]:
            raise AssertionError(f"session bytes/hash mismatch: {item['date']}")
        payload = _read(path)
        loc = np.flatnonzero(market["day"] == day_i)
        bars = payload["bars"]
        if payload["date"] != item["date"] or payload["split"] != item["split"] or len(loc) != item["bars"]:
            raise AssertionError(f"session metadata mismatch: {item['date']}")
        if not np.array_equal(np.asarray(bars["t"], dtype=np.int64), epoch[loc]):
            raise AssertionError(f"session timestamps mismatch: {item['date']}")
        if set(bars) != {"t", *array_keys}:
            raise AssertionError(f"session bar schema mismatch: {item['date']}")
        for browser_key, market_key in market_map.items():
            if not np.allclose(_float_vector(bars[browser_key]), market[market_key][loc],
                               atol=ATOL, rtol=1e-10, equal_nan=True):
                raise AssertionError(f"session raw/model mismatch: {item['date']}/{browser_key}")
        eq = baseline_equity.iloc[loc]
        for browser_key, source_key in (("equity", "equity"), ("drawdown", "drawdown_usd"),
                                        ("drawdown_pct", "drawdown_pct")):
            if not np.allclose(_float_vector(bars[browser_key]), eq[source_key].to_numpy(float),
                               atol=ATOL, rtol=1e-10, equal_nan=True):
                raise AssertionError(f"session equity mismatch: {item['date']}/{browser_key}")
        if len(payload["trades"]) != item["trades"]:
            raise AssertionError(f"session trade count mismatch: {item['date']}")
        session_net = sum(float(trade["net_pnl"]) for trade in payload["trades"])
        _close(session_net, item["net_pnl"], f"session net.{item['date']}")
        all_session_trades.extend(payload["trades"])
        total_bars += len(loc); total_trades += len(payload["trades"]); total_net += session_net
        daily_t.append(int(bars["t"][-1])); daily_equity.append(float(bars["equity"][-1]))
        daily_dd.append(float(bars["drawdown"][-1])); daily_dd_pct.append(float(bars["drawdown_pct"][-1]))
    if (total_bars, total_trades) != (97_530, 456):
        raise AssertionError("lazy session bar/trade additivity mismatch")
    _close(total_net, EXPECTED["full"], "lazy session net additivity")
    if [daily_t, daily_equity, daily_dd, daily_dd_pct] != [overview["daily"][key]
            for key in ("t", "equity", "drawdown", "drawdown_pct")]:
        raise AssertionError("overview daily arrays do not equal session closes")
    if len(all_session_trades) != len(published_trades):
        raise AssertionError("lazy trades do not equal published full trade count")
    for i, (browser_trade, source_trade) in enumerate(zip(all_session_trades, published_trades.itertuples(index=False)), 1):
        if browser_trade["id"] != i or browser_trade["side"] != source_trade.direction or browser_trade["exit_reason"] != source_trade.exit_reason:
            raise AssertionError(f"lazy trade identity mismatch: {i}")
        for key in ("entry_reference", "entry_price", "exit_reference", "exit_price", "gross_pnl",
                    "slippage", "commissions", "costs", "net_pnl"):
            _close(browser_trade[key], getattr(source_trade, key), f"lazy trade {i}.{key}")
        _close(browser_trade["gross_pnl"] - browser_trade["costs"], browser_trade["net_pnl"],
               f"lazy trade {i}.gross-cost-net")

    recon = overview["reconciliation"]
    if not all(recon[key] for key in ("audit_pass", "base_selected_all_split_files_byte_identical",
                                      "stop3_exact_raw_replay", "session_trade_additivity",
                                      "session_net_additivity")):
        raise AssertionError("overview reconciliation flag is false")
    _close(recon["full_trade_net"], EXPECTED["full"], "overview reconciliation net")
    _close(recon["full_final_equity"], CAPITAL + EXPECTED["full"], "overview reconciliation equity")
    _close(recon["full_mtm_mdd"], summary["base_results"]["full"]["max_drawdown_usd_mtm"],
           "overview reconciliation DD")

    html = index_path.read_text(encoding="utf-8")
    required_tokens = (
        "window.VWAP_NVDA_TAIL_RISK_MANIFEST", "fetch(M.overview.data", "fetch(a.data",
        "O.stop3_comparison", "O.grid", "O.finalists", "O.sessions", "S.bars", "S.trades",
        "B.nvwap", "B.qvwap", "B.fair", "B.z", "B.equity", "B.drawdown",
        "lightweight-charts.standalone.production.js", "cache:'no-store'",
    )
    missing = [token for token in required_tokens if token not in html]
    if missing:
        raise AssertionError(f"browser schema tokens missing: {missing}")
    return {"status": "PASS", "sessions": 251, "bars": total_bars, "trades": total_trades,
            "overview_bytes": overview_path.stat().st_size,
            "session_bytes": int(manifest["session_bytes"]), "stop3_full_net": stop3_results["full"]["net_pnl"]}


def audit(raw_replay: bool = True, report: bool = True) -> dict[str, Any]:
    required = (OUT / "summary.json", OUT / "audit.json", OUT / "manifest.json",
                OUT / "development_grid.csv", OUT / "validation_finalists.csv")
    if not all(path.is_file() for path in required):
        raise ArtifactsNotReady("NVDA tail-risk outputs are not complete")
    summary, engine_audit, manifest = (_read(required[0]), _read(required[1]), _read(required[2]))
    if summary["selection"]["verdict"] != "NO_OP_BASELINE":
        raise AssertionError("expected strict selection to resolve to BASE/no-overlay")
    if engine_audit.get("status") != "PASS" or manifest.get("status") != "COMPLETE":
        raise AssertionError("published audit/manifest is not complete PASS")

    actual_files = {p.name for p in OUT.iterdir() if p.is_file() and p.name != "manifest.json"}
    if set(manifest["files"]) != actual_files:
        raise AssertionError("manifest file inventory mismatch")
    for name, meta in manifest["files"].items():
        path = OUT / name
        if path.stat().st_size != int(meta["bytes"]) or _sha(path) != meta["sha256"]:
            raise AssertionError(f"manifest mismatch: {name}")

    grid_result = _audit_grid(summary)
    published: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        base_trade = OUT / f"baseline_{split}_trades.csv"
        base_equity = OUT / f"baseline_{split}_equity.csv"
        selected_trade = OUT / f"selected_{split}_trades.csv"
        selected_equity = OUT / f"selected_{split}_equity.csv"
        if _sha(base_trade) != _sha(selected_trade) or _sha(base_equity) != _sha(selected_equity):
            raise AssertionError(f"{split}: selected no-op artifacts are not byte-identical to baseline")
        published[split] = _published_trade_metrics(base_trade, base_equity,
                                                     summary["base_results"][split], f"base.{split}")
        _published_trade_metrics(selected_trade, selected_equity,
                                 summary["selected_results"][split], f"selected.{split}")
        _close(summary["base_results"][split]["net_pnl"], EXPECTED[split], f"expected.{split}")
        for key, value in summary["selected_vs_base"][split].items():
            if key.endswith("_vs_base") or key.startswith("clipped_") or key.startswith("avoided_"):
                _close(value, 0.0, f"no_op_comparison.{split}.{key}")

    _close(sum(summary["base_results"][x]["net_pnl"] for x in ("development", "validation", "holdout")),
           summary["base_results"]["full"]["net_pnl"], "split additivity")

    raw = {"performed": False}
    report_result = {"status": "SKIPPED"}
    market = _independent_market() if raw_replay or report else None
    if raw_replay:
        assert market is not None
        replay, replay_equity, _ = _raw_baseline_replay(market)
        full = published["full"].copy()
        for column in ("signal_time", "entry_time", "exit_time"):
            full[column] = pd.to_datetime(full[column], utc=True)
            replay[column] = pd.to_datetime(replay[column], utc=True)
        if len(replay) != len(full):
            raise AssertionError("raw replay trade count mismatch")
        if not np.array_equal(replay.direction.to_numpy(int),
                              np.where(full.direction.eq("LONG"), 1, -1)):
            raise AssertionError("raw replay direction mismatch")
        for column in ("signal_time", "entry_time", "exit_time", "exit_reason"):
            if not np.array_equal(replay[column].to_numpy(), full[column].to_numpy()):
                raise AssertionError(f"raw replay {column} mismatch")
        for column in ("entry_reference", "exit_reference", "shares", "duration_bars",
                       "gross_pnl", "slippage", "commissions", "costs", "net_pnl"):
            if not np.allclose(replay[column].to_numpy(float), full[column].to_numpy(float),
                               atol=ATOL, rtol=1e-10):
                raise AssertionError(f"raw replay {column} mismatch")
        for split, (lo, hi) in SPLITS.items():
            subset = replay[(replay.day >= lo) & (replay.day < hi)]
            _close(subset.net_pnl.sum(), EXPECTED[split], f"raw replay pnl.{split}")
        raw = {"performed": True, "bars": 97_530, "sessions": 251,
               "trades": len(replay), "net_pnl": float(replay.net_pnl.sum())}

    if report:
        assert market is not None
        report_result = _audit_report(summary, market)

    return {"status": "PASS", "verdict": "NO_OP_BASELINE", "grid": grid_result,
            "raw_replay": raw, "report": report_result,
            "manifest_files": len(manifest["files"])}


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    print(json.dumps(audit(raw_replay=True, report=True), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
