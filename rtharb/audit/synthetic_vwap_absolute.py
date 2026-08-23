"""Independent raw-event audit for synthetic VWAP absolute strategies.

The implementation deliberately imports neither the research engine nor the
report builder.  It reconstructs exact raw Alpaca SIP bars, session VWAP,
four-stock fair value, residual/Z, hook state, every selected execution and
minute MTM equity directly from frozen artifacts.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rtharb.data.loader import DataLoader


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_output" / "synthetic_vwap_absolute"
REPORT = ROOT / "tradingview_synthetic_vwap_absolute"
SYMBOLS = ("QQQ", "MSFT", "AAPL", "NVDA", "AMZN")
BASKET = ("MSFT", "AAPL", "NVDA", "AMZN")
RAW_WEIGHTS = {"MSFT": 8.6, "AAPL": 8.4, "NVDA": 7.9, "AMZN": 5.2}
WEIGHTS = {key: value / sum(RAW_WEIGHTS.values()) for key, value in RAW_WEIGHTS.items()}
VARIANTS = ("normal_convergence", "reverse_convergence", "normal_dollar_bracket", "reverse_dollar_bracket")
SPLITS = {"development": (0, 250), "validation": (250, 375), "holdout": (375, 501), "full": (0, 501)}
START = pd.Timestamp("2024-08-22").date()
END = pd.Timestamp("2026-08-21").date()
CAPITAL = 100_000.0
NOTIONAL = 20_000.0
COMMISSION = 0.0035
SLIP = 0.0002
ATOL = 1e-7
NY = "America/New_York"


class ArtifactsNotReady(FileNotFoundError):
    pass


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _close(actual: Any, expected: Any, label: str, atol: float = ATOL) -> None:
    a, e = float(actual), float(expected)
    if not (math.isfinite(a) and math.isfinite(e)) or not math.isclose(a, e, abs_tol=atol, rel_tol=1e-10):
        raise AssertionError(f"{label}: {a!r} != {e!r}")


def _epoch(value: Any) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise AssertionError(f"Naive timestamp: {value!r}")
    return int(ts.tz_convert("UTC").timestamp())


def _time_frame(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in columns:
        frame[column] = pd.to_datetime(frame[column], format="mixed", utc=True).dt.tz_convert(NY)
    return frame


@dataclass
class RawMarket:
    common: pd.DatetimeIndex
    frames: dict[str, pd.DataFrame]
    day: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    last: np.ndarray
    vwap: dict[str, np.ndarray]
    basket_dev: np.ndarray
    fair: np.ndarray
    residual: np.ndarray
    by_epoch: dict[int, int]


def _load_raw() -> RawMarket:
    loader = DataLoader(str(ROOT / "data_cache"), "alpaca", "sip")
    frames: dict[str, pd.DataFrame] = {}
    common: pd.DatetimeIndex | None = None
    for symbol in SYMBOLS:
        frame = loader.storage.load_bars(symbol, "1m")
        if frame is None or frame.empty or frame.index.tz is None:
            raise AssertionError(f"{symbol}: raw timezone-aware parquet missing")
        frame = loader._filter_official_rth(frame, "09:30", "16:00")
        frame = frame[(frame.index.date >= START) & (frame.index.date <= END)].sort_index()
        if frame.index.has_duplicates:
            raise AssertionError(f"{symbol}: duplicate raw timestamps")
        frames[symbol] = frame
        common = frame.index if common is None else common.intersection(frame.index, sort=False)
    assert common is not None
    common = common.sort_values()
    frames = {symbol: frame.loc[common] for symbol, frame in frames.items()}
    if len(common) != 194_490 or common.has_duplicates or len(pd.unique(common.date)) != 501:
        raise AssertionError("Raw five-way calendar is not exact 194,490-minute/501-session inner intersection")
    dates = np.asarray(common.date)
    day = pd.factorize(dates, sort=False)[0].astype(np.int64)
    starts = np.r_[0, np.flatnonzero(day[1:] != day[:-1]) + 1]
    ends = np.r_[starts[1:] - 1, len(common) - 1]
    last = np.zeros(len(common), dtype=bool); last[ends] = True
    vwap: dict[str, np.ndarray] = {}
    for symbol, frame in frames.items():
        typical = (frame.high.to_numpy(float) + frame.low.to_numpy(float) + frame.close.to_numpy(float)) / 3.0
        volume = frame.volume.to_numpy(float)
        values = np.full(len(common), np.nan)
        for lo, hi in zip(starts, ends):
            v = volume[lo:hi + 1]; cv = np.cumsum(v)
            values[lo:hi + 1] = np.divide(
                np.cumsum(typical[lo:hi + 1] * v), cv,
                out=np.full(len(v), np.nan), where=cv > 0,
            )
            _close(values[lo], typical[lo], f"{symbol} VWAP reset {common[lo]}", 1e-9)
        vwap[symbol] = values
    basket_dev = sum(
        WEIGHTS[symbol] * (frames[symbol].close.to_numpy(float) / vwap[symbol] - 1.0)
        for symbol in BASKET
    )
    fair = vwap["QQQ"] * (1.0 + basket_dev)
    residual = (frames["QQQ"].close.to_numpy(float) - fair) / fair
    return RawMarket(
        common, frames, day, starts, ends, last, vwap, basket_dev, fair, residual,
        dict(zip((common.as_unit("ns").asi8 // 1_000_000_000).astype(int), range(len(common)))),
    )


def _rolling_z(market: RawMarket, window: int) -> np.ndarray:
    out = np.full(len(market.common), np.nan)
    for lo, hi in zip(market.starts, market.ends):
        x = market.residual[lo:hi + 1]
        right = np.arange(1, len(x) + 1)
        left = np.maximum(0, right - window)
        count = right - left
        cs, cs2 = np.r_[0.0, np.cumsum(x)], np.r_[0.0, np.cumsum(x * x)]
        total, total2 = cs[right] - cs[left], cs2[right] - cs2[left]
        variance = np.divide(total2 - total * total / count, count - 1,
                             out=np.full(len(x), np.nan), where=count > 1)
        std = np.sqrt(np.maximum(variance, 0.0))
        values = np.divide(x - total / count, std, out=np.full(len(x), np.nan), where=std > 1e-12)
        values[count < window] = np.nan
        out[lo:hi + 1] = values
    return out


def _gate() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = OUT / "manifest.json"
    if not path.is_file():
        raise ArtifactsNotReady("Run python -m rtharb.research.synthetic_vwap_absolute first")
    manifest = _read(path)
    if manifest.get("status") != "COMPLETE" or tuple(manifest.get("variants", ())) != VARIANTS:
        raise ArtifactsNotReady(f"Synthetic VWAP absolute stage is not COMPLETE: {manifest.get('status')}")
    period = manifest.get("period", {})
    if (int(period.get("raw_bars", -1)), int(period.get("sessions", -1))) != (194_490, 501):
        raise AssertionError("Manifest raw five-way calendar differs from 194,490/501")
    if manifest.get("formula", {}).get("beta", "unexpected") is not None:
        raise AssertionError("Manifest violates the frozen no-beta formula")
    root_audit = _read(OUT / "audit.json")
    required_root_checks = (
        "raw_bar_count", "sessions", "no_fill", "four_variant_audits",
        "qqq_only_traded", "frozen_pre_sample_basket", "holdout_not_tuned",
    )
    if root_audit.get("status") != "PASS" or not all(
        root_audit.get("checks", {}).get(key) is True for key in required_root_checks
    ):
        raise AssertionError("Root synthetic VWAP audit/check set is not PASS")
    basket = manifest.get("basket", {})
    if tuple(basket.get("symbols", ())) != BASKET:
        raise AssertionError("Frozen official four-stock basket differs")
    published_raw = basket.get("official_ndx_weights_pct", basket.get("raw_weights", {}))
    published_normalized = basket.get("normalized_reference_weights", basket.get("normalized_weights", {}))
    for symbol in BASKET:
        _close(published_raw[symbol], RAW_WEIGHTS[symbol], f"{symbol} official weight", 1e-12)
        _close(published_normalized[symbol], WEIGHTS[symbol], f"{symbol} normalized weight", 1e-12)
    if basket.get("snapshot_date") != "2024-06-28" or not math.isclose(float(basket.get("combined_ndx_weight_pct", 0)), 30.1, abs_tol=1e-12):
        raise AssertionError("Official pre-sample basket snapshot/combined weight differs")
    execution = manifest.get("execution", {})
    if execution.get("traded") != "QQQ":
        raise AssertionError("Roles must say basket reference-only and QQQ-only traded")
    summaries = {name: _read(OUT / name / "summary.json") for name in VARIANTS}
    for name, summary in summaries.items():
        if summary.get("basket", {}).get("official_snapshot_date") != "2024-06-28":
            raise AssertionError(f"{name}: pre-sample basket snapshot differs")
        if summary.get("signal_formula", {}).get("beta", "unexpected") is not None:
            raise AssertionError(f"{name}: forbidden beta appeared in frozen no-beta formula")
        execution = summary.get("execution", {})
        if execution.get("traded_instrument") != "QQQ" or tuple(execution.get("reference_only", ())) != BASKET:
            raise AssertionError(f"{name}: QQQ-only/reference roles differ")
        _close(execution["position_notional_usd"], NOTIONAL, f"{name} notional")
        _close(execution["starting_capital_usd"], CAPITAL, f"{name} capital")
        _close(execution["commission_usd_per_share_per_side"], COMMISSION, f"{name} commission")
        _close(execution["slippage_fraction_per_execution"], SLIP, f"{name} slippage")
    return manifest, summaries


def _selected(summary: dict[str, Any]) -> dict[str, Any]:
    value = summary.get("selected", {})
    signal = summary.get("signal_parameters", summary.get("parameters", value))
    result = {
        "window": int(signal.get("window", signal.get("z_window"))),
        "threshold": float(signal.get("z_entry", signal.get("threshold"))),
        "hook": float(signal.get("hook_delta", signal.get("hook", 0.0))),
    }
    if "dollar_bracket" in str(summary.get("variant", "")):
        result["stop_usd"] = float(value.get("stop_usd", value.get("stop_usd_per_share")))
        result["target_usd"] = float(value.get("target_usd", value.get("target_usd_per_share")))
    return result


def _artifact(name: str, stem: str) -> Path:
    candidates = (OUT / name / f"{stem}.csv", OUT / f"{name}_{stem}.csv")
    return next((path for path in candidates if path.is_file()), candidates[0])


def _audit_selection(name: str, summary: dict[str, Any]) -> None:
    dev_path, val_path = _artifact(name, "development_grid"), _artifact(name, "validation_finalists")
    if not dev_path.is_file() or not val_path.is_file():
        raise AssertionError(f"{name}: development/validation selection artifacts missing")
    dev, finalists = pd.read_csv(dev_path), pd.read_csv(val_path)
    if name.endswith("convergence"):
        params = ["window", "z_entry", "hook_delta"]
        order_params = params
    else:
        params = ["stop_usd", "target_usd"]
        order_params = params
    if not {"net_sharpe", "net_pnl", "trades", "active_trade_days"}.issubset(dev):
        raise AssertionError(f"{name}: development grid metric schema differs")
    minimum = summary.get("grid", {}).get("minimum_sample", {"trades": 50, "active_trade_days": 30})
    eligible = dev[(dev.trades >= int(minimum.get("trades", 50))) &
                   (dev.active_trade_days >= int(minimum.get("active_trade_days", 30)))]
    top = eligible.sort_values(["net_sharpe", "net_pnl", *order_params],
                               ascending=[False, False, *([True] * len(order_params))], kind="mergesort").head(10)
    if set(map(tuple, top[params].to_numpy())) != set(map(tuple, finalists[params].to_numpy())):
        raise AssertionError(f"{name}: validation finalists are not exact development top 10")
    if any("holdout" in column.casefold() for column in finalists):
        raise AssertionError(f"{name}: holdout leaked into selection table")
    robust = np.minimum(finalists.development_net_sharpe, finalists.validation_net_sharpe)
    if not np.allclose(finalists.robust_score, robust, atol=1e-10):
        raise AssertionError(f"{name}: robust score differs")
    winner = finalists.assign(_robust=robust).sort_values(
        ["_robust", "validation_net_pnl", "development_net_sharpe", *params],
        ascending=[False, False, False, *([True] * len(params))], kind="mergesort",
    ).iloc[0]
    selected = _selected(summary)
    for parameter in params:
        key = "threshold" if parameter == "z_entry" else "hook" if parameter == "hook_delta" else parameter
        _close(selected[key], winner[parameter], f"{name} selected {parameter}")
    if summary.get("selection", {}).get("holdout_opened_after_selection") is not True:
        raise AssertionError(f"{name}: holdout isolation attestation missing")
    if name.endswith("dollar_bracket"):
        if not np.allclose(dev.stop_usd / 0.25, np.round(dev.stop_usd / 0.25)) or not np.allclose(dev.target_usd / 0.25, np.round(dev.target_usd / 0.25)):
            raise AssertionError(f"{name}: dollar grid is not exact $0.25 lattice")
        grid = summary.get("grid", {})
        if float(grid.get("step_usd", 0)) != 0.25 or int(grid.get("unique_combinations", -1)) != len(dev):
            raise AssertionError(f"{name}: adaptive grid metadata differs")


def _hook_signal(z: float, threshold: float, hook: float, armed: int, extreme: float) -> tuple[int, float, int]:
    """Return updated arm/extreme and emitted source sign (last item)."""
    if not math.isfinite(z):
        return armed, extreme, 0
    if armed == 0:
        if abs(z) < threshold:
            return 0, math.nan, 0
        sign = 1 if z > 0 else -1
        if hook == 0:
            return 0, math.nan, sign
        return sign, z, 0
    if armed > 0:
        extreme = max(extreme, z); retrace = extreme - z
    else:
        extreme = min(extreme, z); retrace = z - extreme
    if retrace >= hook:
        return 0, math.nan, armed
    return armed, extreme, 0


def _effective(raw: float, direction: int, entry: bool) -> float:
    return raw * (1.0 + (direction if entry else -direction) * SLIP)


def _simulate(name: str, market: RawMarket, params: dict[str, Any],
              first_day: int = 0, last_day: int = 501,
              z: np.ndarray | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    if z is None:
        z = _rolling_z(market, params["window"])
    q_open = market.frames["QQQ"].open.to_numpy(float)
    q_high = market.frames["QQQ"].high.to_numpy(float)
    q_low = market.frames["QQQ"].low.to_numpy(float)
    q_close = market.frames["QQQ"].close.to_numpy(float)
    reverse = name.startswith("reverse")
    bracket = name.endswith("dollar_bracket")
    rows: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    cash = CAPITAL; peak = CAPITAL
    position: dict[str, Any] | None = None
    pending_entry: dict[str, Any] | None = None
    pending_exit = False
    armed = 0; extreme = math.nan
    generated = ignored = 0

    def close_position(i: int, raw_exit: float, reason: str) -> None:
        nonlocal cash, position
        assert position is not None
        direction, shares = position["direction_value"], position["shares"]
        exit_fill = _effective(raw_exit, direction, False)
        gross = direction * (raw_exit - position["entry_reference"]) * shares
        slippage = (abs(position["entry_price"] - position["entry_reference"]) + abs(exit_fill - raw_exit)) * shares
        commissions = 2 * shares * COMMISSION; costs = slippage + commissions; net = gross - costs
        row = {**position, "exit_time": market.common[i], "exit_reference": raw_exit,
               "exit_price": exit_fill, "exit_reason": reason, "duration_bars": i - position["entry_i"],
               "gross_pnl": gross, "commissions": commissions, "slippage": slippage,
               "costs": costs, "net_pnl": net}
        row.pop("entry_i"); row.pop("source_sign")
        rows.append(row); cash += net; position = None

    indices = np.flatnonzero((market.day >= first_day) & (market.day < last_day))
    for i in indices:
        timestamp = market.common[i]
        if i == market.starts[market.day[i]]:
            armed = 0; extreme = math.nan
            if position or pending_entry or pending_exit:
                raise AssertionError(f"{name}: state leaked across session")
        if pending_exit and position is not None:
            close_position(i, float(q_open[i]), "CONVERGENCE")
            pending_exit = False; armed = 0; extreme = math.nan
        if pending_entry is not None:
            source_sign = pending_entry["source_sign"]
            direction = source_sign if reverse else -source_sign
            raw = float(q_open[i]); fill = _effective(raw, direction, True)
            shares = math.floor(NOTIONAL / fill)
            position = {"variant": name,
                        "signal_time": pending_entry["signal_time"], "entry_time": timestamp,
                        "direction": "LONG" if direction == 1 else "SHORT", "direction_value": direction,
                        "source_sign": source_sign, "entry_z": pending_entry["z"],
                        "signal_residual": pending_entry["residual"], "entry_i": i,
                        "entry_reference": raw, "entry_price": fill, "shares": shares}
            position.update({"source_dislocation": "HIGH" if source_sign == 1 else "LOW",
                             "signal_z": pending_entry["z"],
                             "signal_fair_qqq": float(market.fair[pending_entry["signal_i"]]),
                             "signal_qqq_close": float(q_close[pending_entry["signal_i"]]),
                             "signal_qqq_vwap": float(market.vwap["QQQ"][pending_entry["signal_i"]])})
            if bracket:
                stop = raw - params["stop_usd"] if direction == 1 else raw + params["stop_usd"]
                target = raw + params["target_usd"] if direction == 1 else raw - params["target_usd"]
                position.update({"stop_usd_per_share": params["stop_usd"],
                                 "target_usd_per_share": params["target_usd"],
                                 "stop_price": stop, "target_price": target})
            pending_entry = None
        if position is not None and bracket:
            direction = position["direction_value"]
            op = float(q_open[i]); hi = float(q_high[i]); lo = float(q_low[i])
            stop, target = position["stop_price"], position["target_price"]
            gap = op <= stop if direction == 1 else op >= stop
            stop_hit = lo <= stop if direction == 1 else hi >= stop
            target_hit = hi >= target if direction == 1 else lo <= target
            if gap: close_position(i, op, "STOP_GAP")
            elif stop_hit: close_position(i, stop, "STOP")
            elif target_hit: close_position(i, target, "TAKE_PROFIT_BRACKET")
        if position is not None:
            if market.last[i]:
                close_position(i, float(q_close[i]), "FORCED_EOD")
                pending_exit = False
            elif not bracket:
                source_sign = position["source_sign"]
                converged = market.residual[i] <= 0 if source_sign > 0 else market.residual[i] >= 0
                if converged:
                    pending_exit = True
        if not market.last[i]:
            if position is None and pending_entry is None:
                armed, extreme, emitted = _hook_signal(float(z[i]), params["threshold"], params["hook"], armed, extreme)
                if emitted:
                    pending_entry = {"source_sign": emitted, "signal_time": timestamp,
                                     "signal_i": i, "z": float(z[i]), "residual": float(market.residual[i])}
                    generated += 1; armed = 0; extreme = math.nan
            elif position is not None and math.isfinite(float(z[i])) and abs(float(z[i])) >= params["threshold"]:
                ignored += 1
        qclose = float(q_close[i])
        if position is None:
            equity = cash; active = 0
        else:
            direction, shares = position["direction_value"], position["shares"]
            equity = cash - shares * COMMISSION + direction * (qclose - position["entry_price"]) * shares
            active = 1
        peak = max(peak, equity)
        equity_rows.append({"timestamp": timestamp, "equity": equity, "running_peak": peak,
                            "drawdown_usd": peak - equity, "drawdown_pct": (peak - equity) / peak * 100,
                            "active_position": active})
    if position or pending_entry or pending_exit:
        raise AssertionError(f"{name}: live state after replay")
    return pd.DataFrame(rows), pd.DataFrame(equity_rows), {
        "generated_flat_signals": generated,
        "ignored_signals_while_open": ignored,
    }


def _audit_frame(name: str, expected: pd.DataFrame, saved: pd.DataFrame) -> None:
    if len(expected) != len(saved):
        raise AssertionError(f"{name}: trade count differs ({len(expected)} != {len(saved)})")
    time_cols = ("signal_time", "entry_time", "exit_time")
    for column in time_cols:
        if [_epoch(value) for value in expected[column]] != [_epoch(value) for value in saved[column]]:
            raise AssertionError(f"{name}: {column} differs")
    for column in ("variant", "direction", "source_dislocation", "exit_reason", "shares", "duration_bars"):
        if expected[column].astype(str).tolist() != saved[column].astype(str).tolist():
            raise AssertionError(f"{name}: trade {column} differs")
    numeric = [column for column in (
        "signal_z", "signal_residual", "signal_fair_qqq", "signal_qqq_close", "signal_qqq_vwap",
        "entry_reference", "entry_price", "exit_reference", "exit_price",
        "stop_usd_per_share", "target_usd_per_share", "stop_price", "target_price",
        "gross_pnl", "commissions", "slippage", "costs", "net_pnl",
    ) if column in expected and column in saved]
    for column in numeric:
        if not np.allclose(expected[column], saved[column], atol=1e-7, rtol=1e-10):
            raise AssertionError(f"{name}: trade {column} differs")


def _audit_equity(name: str, expected: pd.DataFrame, saved: pd.DataFrame,
                  metrics: dict[str, Any]) -> None:
    raw_epoch = pd.DatetimeIndex(expected.timestamp).as_unit("ns").asi8 // 1_000_000_000
    saved_epoch = pd.DatetimeIndex(saved.timestamp).as_unit("ns").asi8 // 1_000_000_000
    if not np.array_equal(raw_epoch, saved_epoch):
        raise AssertionError(f"{name}: equity timestamps differ from raw clock")
    for column in ("equity", "running_peak", "drawdown_usd", "drawdown_pct"):
        if not np.allclose(expected[column], saved[column], atol=2e-7, rtol=1e-10):
            raise AssertionError(f"{name}: MTM {column} differs")
    _close(saved.equity.iloc[-1], CAPITAL + metrics["net_pnl"], f"{name} final equity", 2e-7)
    _close(saved.drawdown_usd.max(), metrics["max_drawdown_usd_mtm"], f"{name} MTM DD", 2e-7)


def _audit_splits(name: str, summary: dict[str, Any]) -> None:
    results = summary["selected_results"]
    split_net = 0.0; split_trades = 0
    for split in SPLITS:
        trades = _time_frame(_artifact(name, f"selected_{split}_trades"), ("signal_time", "entry_time", "exit_time"))
        metrics = results[split]
        if len(trades) != int(metrics["trades"]):
            raise AssertionError(f"{name} {split}: trade count differs")
        for column, key in (("gross_pnl", "gross_pnl"), ("commissions", "commissions"),
                            ("slippage", "slippage"), ("costs", "costs"), ("net_pnl", "net_pnl")):
            _close(trades[column].sum(), metrics[key], f"{name} {split} {key}")
        if split != "full":
            split_net += float(metrics["net_pnl"]); split_trades += int(metrics["trades"])
    _close(split_net, results["full"]["net_pnl"], f"{name} split/full net")
    if split_trades != int(results["full"]["trades"]):
        raise AssertionError(f"{name}: split/full trades differ")


def _audit_report(manifest: dict[str, Any], market: RawMarket | None = None,
                  summaries: dict[str, dict[str, Any]] | None = None) -> None:
    js_path = REPORT / "manifest.js"
    if not js_path.is_file():
        return
    html = REPORT / "index.html"
    if not html.is_file() or html.stat().st_size < 1000:
        raise AssertionError("Synthetic VWAP absolute report index missing")
    js = js_path.read_text(encoding="utf-8").strip()
    prefix = "window.SYNTHETIC_VWAP_ABSOLUTE_MANIFEST="
    if not js.startswith(prefix) or not js.endswith(";"):
        raise AssertionError("Synthetic VWAP report manifest wrapper malformed")
    report = json.loads(js[len(prefix):-1])
    if tuple(report.get("variants", ())) != VARIANTS or report.get("traded") != "QQQ":
        raise AssertionError("Synthetic VWAP report variants/roles differ")
    report_source = report.get("source", {})
    source_manifest = OUT / "manifest.json"
    if (report_source.get("sha256") != _sha(source_manifest) or
            int(report_source.get("bytes", -1)) != source_manifest.stat().st_size):
        raise AssertionError("Synthetic VWAP report source-manifest provenance differs")
    assets = report.get("assets", [])
    if [item.get("variant") for item in assets] != list(VARIANTS):
        raise AssertionError("Synthetic VWAP report assets differ")
    for asset in assets:
        path = REPORT / asset["data"]
        if not path.is_file() or path.stat().st_size != int(asset["bytes"]) or _sha(path) != asset["sha256"]:
            raise AssertionError(f"{asset['variant']}: lazy report path/size/hash differs")
        payload = _read(path)
        if set(payload) != {"meta", "bars", "trades", "results"}:
            raise AssertionError(f"{asset['variant']}: lazy payload schema differs")
        if {len(value) for value in payload["bars"].values()} != {194_490}:
            raise AssertionError(f"{asset['variant']}: lazy bars not aligned")
        source = _read(OUT / asset["variant"] / "summary.json")
        _close(payload["results"]["full"]["net_pnl"], source["selected_results"]["full"]["net_pnl"],
               f"{asset['variant']} report full net")
        if len(payload["trades"]) != int(source["selected_results"]["full"]["trades"]):
            raise AssertionError(f"{asset['variant']}: lazy report trade count differs")
        meta = payload["meta"]
        if meta.get("traded") != "QQQ" or tuple(meta.get("reference_constituents", ())) != BASKET:
            raise AssertionError(f"{asset['variant']}: lazy payload roles differ")
        for source_item in meta.get("sources", {}).values():
            provenance_path = ROOT / source_item["path"]
            if (not provenance_path.is_file() or source_item.get("sha256") != _sha(provenance_path)
                    or int(source_item.get("bytes", -1)) != provenance_path.stat().st_size):
                raise AssertionError(f"{asset['variant']}: lazy payload source provenance differs")
        if market is not None and summaries is not None:
            bars = payload["bars"]
            raw_epoch = (market.common.as_unit("ns").asi8 // 1_000_000_000).tolist()
            if bars["t"] != raw_epoch:
                raise AssertionError(f"{asset['variant']}: lazy timestamps differ from raw five-way clock")
            exact = {
                "qo": market.frames["QQQ"].open.to_numpy(float),
                "qh": market.frames["QQQ"].high.to_numpy(float),
                "ql": market.frames["QQQ"].low.to_numpy(float),
                "qc": market.frames["QQQ"].close.to_numpy(float),
                "qvwap": market.vwap["QQQ"], "fair": market.fair,
                "residual": market.residual, "basket_dev": market.basket_dev,
                "z": _rolling_z(market, _selected(summaries[asset["variant"]])["window"]),
            }
            for key, values in exact.items():
                published = np.asarray([np.nan if value is None else value for value in bars[key]], float)
                if not np.allclose(published, values, atol=1e-9, rtol=1e-10, equal_nan=True):
                    raise AssertionError(f"{asset['variant']}: lazy {key} differs from independent raw reconstruction")
            equity = pd.read_csv(_artifact(asset["variant"], "selected_full_equity"))
            if not np.allclose(bars["equity"], equity.equity, atol=1e-7):
                raise AssertionError(f"{asset['variant']}: lazy equity differs")


def audit(*, raw_replay: bool = True) -> dict[str, Any]:
    manifest, summaries = _gate()
    market = _load_raw() if raw_replay else None
    z_cache = ({window: _rolling_z(market, window) for window in
                {int(summary["signal_parameters"]["window"]) for summary in summaries.values()}}
               if market is not None else {})
    for direction in ("normal", "reverse"):
        convergence = summaries[f"{direction}_convergence"]["signal_parameters"]
        bracket = summaries[f"{direction}_dollar_bracket"]["signal_parameters"]
        if convergence != bracket:
            raise AssertionError(f"{direction}: dollar bracket did not freeze convergence signal parameters")
    cross_csv, cross_json = OUT / "cross_variant_summary.csv", OUT / "cross_variant_summary.json"
    if not cross_csv.is_file() or not cross_json.is_file():
        raise AssertionError("Cross-variant summaries missing")
    cross = pd.read_csv(cross_csv); cross_payload = _read(cross_json).get("rows", [])
    if cross.variant.tolist() != list(VARIANTS) or [row.get("variant") for row in cross_payload] != list(VARIANTS):
        raise AssertionError("Cross-variant summary order/schema differs")
    for row in cross.itertuples(index=False):
        summary = summaries[row.variant]
        _close(row.full_net_pnl, summary["selected_results"]["full"]["net_pnl"],
               f"{row.variant} cross full net")
        _close(row.holdout_net_pnl, summary["selected_results"]["holdout"]["net_pnl"],
               f"{row.variant} cross holdout net")
        _close(row.full_mtm_mdd_pct, summary["selected_results"]["full"]["max_drawdown_pct_mtm"],
               f"{row.variant} cross MTM DD", 2e-7)
    _audit_report(manifest, market, summaries)
    results: dict[str, Any] = {}
    for name in VARIANTS:
        summary = summaries[name]
        if summary.get("execution", {}).get("traded_instrument") != "QQQ" or "QQQ" in summary.get("execution", {}).get("reference_only", []):
            raise AssertionError(f"{name}: QQQ/reference roles differ")
        _audit_selection(name, summary)
        _audit_splits(name, summary)
        if raw_replay:
            assert market is not None
            params = _selected(summary)
            for split, (first_day, last_day) in SPLITS.items():
                expected_trades, expected_equity, stats = _simulate(
                    name, market, params, first_day, last_day,
                    z=z_cache[params["window"]],
                )
                saved_trades = _time_frame(
                    _artifact(name, f"selected_{split}_trades"),
                    ("signal_time", "entry_time", "exit_time"),
                )
                saved_equity = _time_frame(
                    _artifact(name, f"selected_{split}_equity"), ("timestamp",),
                )
                label = f"{name} {split}"
                _audit_frame(label, expected_trades, saved_trades)
                _audit_equity(label, expected_equity, saved_equity,
                              summary["selected_results"][split])
                split_metrics = summary["selected_results"][split]
                for key, value in stats.items():
                    if key in split_metrics and int(split_metrics[key]) != value:
                        raise AssertionError(f"{label}: {key} differs")
        results[name] = summary["selected_results"]
    return results


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    try:
        results = audit()
    except ArtifactsNotReady as exc:
        print(f"NOT READY: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print("PASS synthetic VWAP absolute: 194,490 exact five-way raw SIP minutes")
    for name in VARIANTS:
        full, holdout = results[name]["full"], results[name]["holdout"]
        print(f"{name}: {full['trades']:,} trades, full ${full['net_pnl']:,.2f}, "
              f"MTM DD ${full['max_drawdown_usd_mtm']:,.2f}, holdout ${holdout['net_pnl']:,.2f}")


if __name__ == "__main__":
    main()
