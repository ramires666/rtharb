"""Independent audit of the nine-stock event-driven VWAP bracket study.

The audit intentionally does not import the research simulator or report
builder.  It reloads exact Alpaca SIP parquet bars, constructs each QQQ/stock
intersection once, reconstructs causal session VWAP/fair-value/Z arrays, and
then proves every published fill, bracket exit, cost, P&L and MTM equity row.
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
MANIFEST = ROOT / "data_cache" / "mega_cap_sip_manifest.json"
SOURCE = ROOT / "research_output" / "vwap_absolute_multi_asset"
REPORT_CANDIDATES = (
    ROOT / "tradingview_vwap_absolute_multi_asset",
    ROOT / "tradingview_multi_asset",
)
NY = "America/New_York"
ATOL = 1e-8


class ArtifactsNotReady(FileNotFoundError):
    """Raised when research/report artifacts have not been built yet."""


def _close(actual: Any, expected: Any, label: str, atol: float = ATOL) -> None:
    a, e = float(actual), float(expected)
    if not (math.isfinite(a) and math.isfinite(e)) or not math.isclose(a, e, abs_tol=atol, rel_tol=1e-10):
        raise AssertionError(f"{label}: {a!r} != {e!r}")


def _epoch(value: Any) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise AssertionError(f"Naive timestamp: {value!r}")
    return int(ts.tz_convert("UTC").timestamp())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_source() -> dict[str, Any]:
    if not MANIFEST.is_file():
        raise ArtifactsNotReady(
            "Missing data_cache/mega_cap_sip_manifest.json; run "
            "python -m rtharb.data.download_mega_cap --download-missing first"
        )
    path = SOURCE / "manifest.json"
    if not path.is_file():
        raise ArtifactsNotReady(
            "Multi-asset research is not ready; run "
            "python -m rtharb.research.vwap_absolute_multi_asset first, then run "
            "python -m rtharb.reporting.vwap_absolute_multi_asset to build the HTML report"
        )
    return _read_json(path)


def _symbol_summary(summary: dict[str, Any], symbol: str) -> dict[str, Any]:
    for key in ("symbols", "per_symbol", "results"):
        value = summary.get(key)
        if isinstance(value, dict) and isinstance(value.get(symbol), dict):
            return value[symbol]
    path = SOURCE / symbol / "summary.json"
    if path.is_file():
        return _read_json(path)
    path = SOURCE / f"{symbol}_summary.json"
    if path.is_file():
        return _read_json(path)
    raise AssertionError(f"{symbol}: per-symbol summary not found")


def _artifact(symbol: str, stem: str) -> Path:
    candidates = (
        SOURCE / symbol / f"{stem}.csv",
        SOURCE / symbol.lower() / f"{stem}.csv",
        SOURCE / f"{symbol}_{stem}.csv",
        SOURCE / f"{stem}_{symbol}.csv",
        SOURCE / f"{stem}_{symbol.lower()}.csv",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise AssertionError(f"{symbol}: missing {stem}.csv; checked {', '.join(map(str, candidates))}")


def _get(mapping: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    raise AssertionError(f"Missing aliases {names}")


def _frame_value(row: Any, *names: str) -> Any:
    for name in names:
        if hasattr(row, name):
            return getattr(row, name)
    raise AssertionError(f"Trade row is missing aliases {names}")


def _selected(item: dict[str, Any]) -> dict[str, float]:
    selected = item.get("selected", item.get("parameters", {}))
    return {
        "stop_usd": float(_get(selected, "stop_usd", "stop_usd_per_share")),
        "target_usd": float(_get(selected, "target_usd", "target_usd_per_share")),
    }


def _entry_parameters(root: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("entry_parameters", root.get("entry_parameters", {}))
    return {
        "beta_days": int(value.get("beta_days", 5)),
        "window": int(value.get("window", value.get("z_window", 60))),
        "warmup": int(value.get("warmup_bars", value.get("warmup", 30))),
        "z_entry": float(value.get("z_entry", value.get("threshold", 2.5))),
    }


def _execution(root: dict[str, Any], item: dict[str, Any]) -> dict[str, float]:
    value = item.get("execution", root.get("execution", {}))
    return {
        "capital": float(_get(value, "starting_capital_usd", "capital")),
        "notional": float(_get(value, "position_notional_usd", "notional_usd")),
        "commission": float(_get(value, "commission_usd_per_share_per_side", "commission_per_share")),
        "slip": float(_get(value, "slippage_fraction_per_execution", "slippage_fraction")),
    }


def _period(root: dict[str, Any], item: dict[str, Any]) -> tuple[object, object]:
    value = item.get("period", root.get("period", root.get("study_period", {})))
    return pd.Timestamp(value["start"]).date(), pd.Timestamp(value["end"]).date()


@dataclass
class Market:
    common: pd.DatetimeIndex
    lead: pd.DataFrame
    target: pd.DataFrame
    qvwap: np.ndarray
    tvwap: np.ndarray
    fair: np.ndarray
    z: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    by_epoch: dict[int, int]
    day_end: dict[object, int]


def _market(loader: DataLoader, lead_all: pd.DataFrame, symbol: str, start: object, end: object,
            beta_days: int, window: int, warmup: int) -> Market:
    target_all = loader.storage.load_bars(symbol, "1m")
    if target_all is None or target_all.empty:
        raise AssertionError(f"{symbol}: raw parquet is absent/empty")
    for frame in (lead_all, target_all):
        if frame.index.tz is None:
            raise AssertionError(f"{symbol}: raw timestamps must be timezone-aware")
    lead_rth = loader._filter_official_rth(lead_all, "09:30", "16:00")
    target_rth = loader._filter_official_rth(target_all, "09:30", "16:00")
    full_common = lead_rth.index.intersection(target_rth.index)
    lead_full, target_full = lead_rth.loc[full_common], target_rth.loc[full_common]
    dates = np.asarray(full_common.date)
    day = pd.factorize(dates, sort=False)[0]
    starts = np.r_[0, np.flatnonzero(day[1:] != day[:-1]) + 1]
    ends = np.r_[starts[1:] - 1, len(full_common) - 1]

    def session_vwap(frame: pd.DataFrame) -> np.ndarray:
        typical = (frame.high.to_numpy(float) + frame.low.to_numpy(float) + frame.close.to_numpy(float)) / 3.0
        volume = frame.volume.to_numpy(float)
        out = np.full(len(frame), np.nan)
        for lo, hi in zip(starts, ends):
            v = volume[lo:hi + 1]
            cv = np.cumsum(v)
            out[lo:hi + 1] = np.divide(
                np.cumsum(typical[lo:hi + 1] * v), cv,
                out=np.full(len(v), np.nan), where=cv > 0,
            )
        return out

    qvwap = session_vwap(lead_full)
    tvwap = session_vwap(target_full)
    qdaily = lead_full.close.to_numpy(float)[ends]
    tdaily = target_full.close.to_numpy(float)[ends]
    qr, tr = pd.Series(qdaily).pct_change(), pd.Series(tdaily).pct_change()
    beta = (tr.rolling(beta_days, min_periods=beta_days).cov(qr)
            / qr.rolling(beta_days, min_periods=beta_days).var()).shift(1)
    beta = beta.clip(0.2, 4.0).fillna(1.5).to_numpy(float)
    lead_move = lead_full.close.to_numpy(float) / qvwap - 1.0
    spread = target_full.close.to_numpy(float) / tvwap - 1.0 - beta[day] * lead_move
    fair = tvwap * (1.0 + beta[day] * lead_move)
    z = np.full(len(spread), np.nan)
    for lo, hi in zip(starts, ends):
        x = spread[lo:hi + 1]
        count = np.minimum(np.arange(1, len(x) + 1), window)
        rolling_start = np.maximum(0, np.arange(len(x)) - window + 1)
        cs, cs2 = np.r_[0.0, np.cumsum(x)], np.r_[0.0, np.cumsum(x * x)]
        total = cs[np.arange(1, len(x) + 1)] - cs[rolling_start]
        total2 = cs2[np.arange(1, len(x) + 1)] - cs2[rolling_start]
        variance = np.divide(
            total2 - total * total / count, count - 1,
            out=np.full(len(x), np.nan), where=count > 1,
        )
        std = np.sqrt(np.maximum(variance, 0.0))
        part = np.divide(
            x - total / count, std,
            out=np.full(len(x), np.nan), where=std > 1e-8,
        )
        part[:warmup] = np.nan
        z[lo:hi + 1] = part

    mask = np.fromiter((start <= ts.date() <= end for ts in full_common), bool, len(full_common))
    common = full_common[mask]
    lead, target = lead_full.loc[common], target_full.loc[common]
    period_day = pd.factorize(np.asarray(common.date), sort=False)[0]
    pstarts = np.r_[0, np.flatnonzero(period_day[1:] != period_day[:-1]) + 1]
    pends = np.r_[pstarts[1:] - 1, len(common) - 1]
    return Market(
        common=common, lead=lead, target=target, qvwap=qvwap[mask], tvwap=tvwap[mask],
        fair=fair[mask], z=z[mask], starts=pstarts, ends=pends,
        by_epoch=dict(zip((common.as_unit("ns").asi8 // 1_000_000_000).astype(int), range(len(common)))),
        day_end={common[int(i)].date(): int(i) for i in pends},
    )


def _audit_manifest(manifest: dict[str, Any], summary: dict[str, Any], completed: list[str]) -> list[str]:
    universe = list(manifest.get("frozen_universe", []))
    if len(universe) != 9 or len(set(universe)) != 9 or "QQQ" in universe:
        raise AssertionError(f"Manifest frozen universe must contain nine distinct traded stocks: {universe}")
    if manifest.get("lead") != "QQQ" or manifest.get("provider") != "Alpaca" or manifest.get("feed") != "SIP":
        raise AssertionError("Manifest must identify QQQ lead and Alpaca SIP source")
    if manifest.get("timeframe") not in ("1 minute", "1m"):
        raise AssertionError("Manifest does not identify raw one-minute bars")
    if manifest.get("pairwise_intersection") is not True or manifest.get("no_resampling_or_fill") is not True:
        raise AssertionError("Manifest does not freeze exact inner joins without fill/resampling")
    published = summary.get(
        "frozen_universe",
        summary.get("traded_separately", summary.get("universe")),
    )
    if isinstance(published, dict):
        published = published.get("symbols", published.get("frozen_universe"))
    if published is not None and list(published) != universe:
        raise AssertionError("Research universe differs from the predeclared manifest universe")
    for symbol in ("QQQ", *universe):
        entry = manifest.get("symbols", {}).get(symbol)
        if not isinstance(entry, dict):
            raise AssertionError(f"Manifest is missing {symbol}")
        path = ROOT / str(entry["file"])
        if not path.is_file() or path.stat().st_size != int(entry["bytes"]):
            raise AssertionError(f"{symbol}: parquet path/size differs from manifest")
        # Hash raw inputs participating in this completed stage.  Merely
        # predownloaded future targets are size-checked now and hash-checked
        # once their own research stage is audited.
        if symbol in {"QQQ", *completed} and _sha256(path) != entry["sha256"]:
            raise AssertionError(f"{symbol}: parquet SHA-256 differs from frozen manifest")
    return universe


def _audit_selection(symbol: str, item: dict[str, Any]) -> None:
    dev = pd.read_csv(_artifact(symbol, "development_grid"))
    finalists = pd.read_csv(_artifact(symbol, "validation_finalists"))
    grid = item.get("grid", {})
    if len(dev) != int(grid.get("unique_combinations", len(dev))) or len(dev) < 144:
        raise AssertionError(f"{symbol}: adaptive development grid size is invalid")
    needed = {"stop_usd", "target_usd", "net_sharpe", "net_pnl"}
    if not needed.issubset(dev.columns):
        raise AssertionError(f"{symbol}: development grid is missing {sorted(needed - set(dev.columns))}")
    selection = item.get("selection", {})
    minimum = grid.get("minimum_sample", {})
    eligible = dev[
        (dev.trades >= int(minimum.get("trades", 0)))
        & (dev.active_trade_days >= int(minimum.get("active_trade_days", 0)))
    ]
    count = len(finalists)
    if count != 10:
        raise AssertionError(f"{symbol}: exactly ten development finalists expected, got {count}")
    top = eligible.sort_values(
        ["net_sharpe", "net_pnl", "stop_usd", "target_usd"],
        ascending=[False, False, True, True], kind="mergesort",
    ).head(count)
    expected_pairs = set(zip(top.stop_usd.astype(float), top.target_usd.astype(float)))
    actual_pairs = set(zip(finalists.stop_usd.astype(float), finalists.target_usd.astype(float)))
    if actual_pairs != expected_pairs:
        raise AssertionError(f"{symbol}: validation finalists are not exactly the development top {count}")
    forbidden = [c for c in finalists.columns if "holdout" in c.casefold()]
    if forbidden:
        raise AssertionError(f"{symbol}: holdout leaked into parameter-selection table: {forbidden}")
    robust = np.minimum(finalists.development_net_sharpe, finalists.validation_net_sharpe)
    if "robust_score" not in finalists or not np.allclose(finalists.robust_score, robust, atol=1e-12):
        raise AssertionError(f"{symbol}: robust score is not min(development, validation Sharpe)")
    winner = finalists.assign(_robust=robust).sort_values(
        ["_robust", "validation_net_pnl", "development_net_sharpe", "stop_usd", "target_usd"],
        ascending=[False, False, False, True, True], kind="mergesort",
    ).iloc[0]
    selected = _selected(item)
    _close(selected["stop_usd"], winner.stop_usd, f"{symbol} selected stop")
    _close(selected["target_usd"], winner.target_usd, f"{symbol} selected target")
    if selection.get("holdout_opened_after_selection") is not True:
        raise AssertionError(f"{symbol}: summary does not attest that holdout was opened once after selection")


def _load_trades(symbol: str, split: str = "full") -> pd.DataFrame:
    trades = pd.read_csv(_artifact(symbol, f"selected_{split}_trades"))
    for col in ("signal_time", "entry_time", "exit_time"):
        trades[col] = pd.to_datetime(trades[col], format="mixed", utc=True).dt.tz_convert(NY)
    return trades


def _audit_trades(symbol: str, trades: pd.DataFrame, market: Market,
                  selected: dict[str, float], entry: dict[str, Any], execution: dict[str, float]) -> None:
    for number, row in enumerate(trades.itertuples(index=False), 1):
        si = market.by_epoch.get(_epoch(row.signal_time))
        ei = market.by_epoch.get(_epoch(row.entry_time))
        xi = market.by_epoch.get(_epoch(row.exit_time))
        if si is None or ei is None or xi is None:
            raise AssertionError(f"{symbol} trade {number}: timestamp is absent from exact QQQ/{symbol} intersection")
        if ei != si + 1 or row.signal_time.date() != row.entry_time.date():
            raise AssertionError(f"{symbol} trade {number}: entry is not next synchronized raw open")
        z = float(market.z[si])
        _close(_frame_value(row, "entry_z", "signal_z"), z, f"{symbol} trade {number} Z", 1e-9)
        direction = 1 if str(row.direction).upper() == "LONG" else -1
        if (direction == 1 and z > -entry["z_entry"]) or (direction == -1 and z < entry["z_entry"]):
            raise AssertionError(f"{symbol} trade {number}: direction contradicts causal Z")
        raw_entry = float(market.target.open.iloc[ei])
        _close(_frame_value(row, "entry_reference", "entry_raw"), raw_entry, f"{symbol} trade {number} raw entry")
        fill_entry = raw_entry * (1 + execution["slip"] if direction == 1 else 1 - execution["slip"])
        _close(row.entry_price, fill_entry, f"{symbol} trade {number} entry fill")
        shares = math.floor(execution["notional"] / fill_entry)
        if int(row.shares) != shares or shares <= 0:
            raise AssertionError(f"{symbol} trade {number}: share sizing mismatch")
        _close(_frame_value(row, "stop_usd_per_share", "stop_usd"), selected["stop_usd"], f"{symbol} trade {number} stop distance")
        _close(_frame_value(row, "target_usd_per_share", "target_usd"), selected["target_usd"], f"{symbol} trade {number} target distance")
        stop = raw_entry - selected["stop_usd"] if direction == 1 else raw_entry + selected["stop_usd"]
        target = raw_entry + selected["target_usd"] if direction == 1 else raw_entry - selected["target_usd"]
        _close(row.stop_price, stop, f"{symbol} trade {number} stop price")
        _close(row.target_price, target, f"{symbol} trade {number} target price")

        expected_i = None
        expected_raw = None
        expected_reason = None
        for i in range(ei, market.day_end[row.entry_time.date()] + 1):
            op = float(market.target.open.iloc[i])
            hi = float(market.target.high.iloc[i])
            lo = float(market.target.low.iloc[i])
            stop_hit = (op <= stop or lo <= stop) if direction == 1 else (op >= stop or hi >= stop)
            target_hit = hi >= target if direction == 1 else lo <= target
            if stop_hit:  # Conservative same-bar ambiguity: stop always wins.
                gap = op <= stop if direction == 1 else op >= stop
                expected_i, expected_raw, expected_reason = i, (op if gap else stop), "STOP"
                break
            if target_hit:
                expected_i, expected_raw, expected_reason = i, target, "TAKE_PROFIT_BRACKET"
                break
        if expected_i is None:
            expected_i = market.day_end[row.entry_time.date()]
            expected_raw = float(market.target.close.iloc[expected_i])
            expected_reason = "FORCED_EOD"
        if xi != expected_i or str(row.exit_reason) != expected_reason:
            raise AssertionError(f"{symbol} trade {number}: first permissible stop-first bracket exit differs")
        _close(_frame_value(row, "exit_reference", "exit_raw"), expected_raw, f"{symbol} trade {number} raw exit")
        fill_exit = expected_raw * (1 - execution["slip"] if direction == 1 else 1 + execution["slip"])
        _close(row.exit_price, fill_exit, f"{symbol} trade {number} exit fill")
        gross = direction * (expected_raw - raw_entry) * shares
        slippage = (abs(fill_entry - raw_entry) + abs(fill_exit - expected_raw)) * shares
        commission = 2 * shares * execution["commission"]
        _close(row.gross_pnl, gross, f"{symbol} trade {number} gross P&L")
        _close(_frame_value(row, "slippage", "slippage_usd"), slippage, f"{symbol} trade {number} slippage")
        _close(_frame_value(row, "commissions", "commission"), commission, f"{symbol} trade {number} commission")
        _close(row.costs, slippage + commission, f"{symbol} trade {number} total costs")
        _close(row.net_pnl, gross - slippage - commission, f"{symbol} trade {number} net P&L")

    # Independently walk close events after proving every exit.  A qualifying
    # close while flat must create exactly one next-open trade; closes while a
    # bracket remains open are ignored.  Exit bars themselves may immediately
    # create a fresh close signal, exactly as the event-driven design states.
    signal_indices = [market.by_epoch[_epoch(value)] for value in trades.signal_time]
    entry_indices = [market.by_epoch[_epoch(value)] for value in trades.entry_time]
    exit_indices = [market.by_epoch[_epoch(value)] for value in trades.exit_time]
    expected: list[int] = []
    cursor = 0
    session_ends = set(map(int, market.ends))
    for i, value in enumerate(market.z):
        while cursor < len(trades) and i >= exit_indices[cursor]:
            cursor += 1
        open_after_bracket = cursor < len(trades) and entry_indices[cursor] <= i < exit_indices[cursor]
        is_last = i in session_ends
        if not open_after_bracket and not is_last and math.isfinite(float(value)) and abs(float(value)) >= entry["z_entry"]:
            expected.append(i)
    if expected != signal_indices:
        mismatch = next((i for i, pair in enumerate(zip(expected, signal_indices)) if pair[0] != pair[1]), None)
        raise AssertionError(
            f"{symbol}: flat causal signal stream differs from trades "
            f"(expected={len(expected)}, saved={len(signal_indices)}, first mismatch={mismatch})"
        )


def _audit_equity(symbol: str, trades: pd.DataFrame, market: Market,
                  execution: dict[str, float], full_metrics: dict[str, Any]) -> None:
    equity = pd.read_csv(_artifact(symbol, "selected_full_equity"))
    equity["timestamp"] = pd.to_datetime(equity.timestamp, format="mixed", utc=True).dt.tz_convert(NY)
    saved_epoch = pd.DatetimeIndex(equity.timestamp).as_unit("ns").asi8 // 1_000_000_000
    raw_epoch = market.common.as_unit("ns").asi8 // 1_000_000_000
    if len(equity) != len(market.common) or not np.array_equal(saved_epoch, raw_epoch):
        raise AssertionError(f"{symbol}: equity timestamps are not the exact raw QQQ/{symbol} intersection")
    expected = np.full(len(market.common), execution["capital"], dtype=float)
    cash = execution["capital"]
    cursor = 0
    for row in trades.itertuples(index=False):
        ei = market.by_epoch[_epoch(row.entry_time)]
        xi = market.by_epoch[_epoch(row.exit_time)]
        expected[cursor:ei] = cash
        direction = 1 if str(row.direction).upper() == "LONG" else -1
        shares = int(row.shares)
        expected[ei:xi] = (
            cash - shares * execution["commission"]
            + direction * (market.target.close.to_numpy(float)[ei:xi] - float(row.entry_price)) * shares
        )
        cash += float(row.net_pnl)
        expected[xi] = cash
        cursor = xi + 1
    expected[cursor:] = cash
    if not np.allclose(equity.equity.to_numpy(float), expected, atol=1e-7, rtol=1e-11):
        i = int(np.nanargmax(np.abs(equity.equity.to_numpy(float) - expected)))
        raise AssertionError(f"{symbol}: MTM equity differs at {market.common[i]}")
    peak = np.maximum.accumulate(expected)
    dd = peak - expected
    dd_pct = np.divide(dd, peak, out=np.zeros_like(dd), where=peak != 0) * 100
    for column, values in (("running_peak", peak), ("drawdown_usd", dd), ("drawdown_pct", dd_pct)):
        if column in equity and not np.allclose(equity[column].to_numpy(float), values, atol=1e-7, rtol=1e-11):
            raise AssertionError(f"{symbol}: {column} differs from independent MTM reconstruction")
    _close(expected[-1], execution["capital"] + float(full_metrics["net_pnl"]), f"{symbol} final equity", 1e-7)
    _close(dd.max(), full_metrics["max_drawdown_usd_mtm"], f"{symbol} MTM drawdown USD", 1e-7)
    _close(dd_pct.max(), full_metrics["max_drawdown_pct_mtm"], f"{symbol} MTM drawdown percent", 1e-7)


def _audit_splits(symbol: str, item: dict[str, Any], full_trades: pd.DataFrame) -> dict[str, Any]:
    results = item.get("selected_results", item.get("results"))
    if not isinstance(results, dict) or not all(name in results for name in ("development", "validation", "holdout", "full")):
        raise AssertionError(f"{symbol}: four selected split result blocks are required")
    split_frames = {name: _load_trades(symbol, name) for name in ("development", "validation", "holdout")}
    if sum(len(frame) for frame in split_frames.values()) != len(full_trades):
        raise AssertionError(f"{symbol}: split trade counts do not equal full")
    if abs(sum(frame.net_pnl.sum() for frame in split_frames.values()) - full_trades.net_pnl.sum()) > ATOL:
        raise AssertionError(f"{symbol}: split net P&L does not equal full")
    for name, frame in (*split_frames.items(), ("full", full_trades)):
        metrics = results[name]
        if int(metrics["trades"]) != len(frame):
            raise AssertionError(f"{symbol} {name}: metric/CSV trade counts differ")
        _close(frame.net_pnl.sum(), metrics["net_pnl"], f"{symbol} {name} net P&L")
        _close(frame.gross_pnl.sum(), metrics["gross_pnl"], f"{symbol} {name} gross P&L")
        _close(frame.costs.sum(), metrics["costs"], f"{symbol} {name} costs")
        if int(metrics["stops"]) + int(metrics["targets"]) + int(metrics["forced_eod"]) != len(frame):
            raise AssertionError(f"{symbol} {name}: exit reasons do not reconcile")
    return results["full"]


def _audit_report(summary: dict[str, Any], universe: list[str]) -> None:
    report = next((path for path in REPORT_CANDIDATES if path.exists()), None)
    # A report is optional while research is being generated.  Once any report
    # directory exists it must be complete and carry all nine symbols/equities.
    if report is None or not (report / "manifest.js").is_file():
        return
    html_path, js_path = report / "index.html", report / "manifest.js"
    if not html_path.is_file() or html_path.stat().st_size < 1000 or js_path.stat().st_size < 1000:
        raise AssertionError("Built lazy report index/manifest is absent or implausibly small")
    js = js_path.read_text(encoding="utf-8").strip()
    prefix = "window.VWAP_MULTI_ASSET_MANIFEST="
    if not js.startswith(prefix) or not js.endswith(";"):
        raise AssertionError("Lazy multi-asset manifest.js wrapper is malformed")
    manifest = json.loads(js[len(prefix):-1])
    if manifest.get("lead") != "QQQ" or list(manifest.get("targets", [])) != universe:
        raise AssertionError("Lazy report manifest roles/universe differ from research")
    assets = manifest.get("assets", [])
    if [item.get("symbol") for item in assets] != universe:
        raise AssertionError("Lazy report asset ordering differs from frozen universe")
    for asset in assets:
        symbol = asset["symbol"]
        path = report / asset["data"]
        if not path.is_file() or path.stat().st_size != int(asset["bytes"]):
            raise AssertionError(f"{symbol}: lazy payload path/size mismatch")
        if _sha256(path) != asset["sha256"]:
            raise AssertionError(f"{symbol}: lazy payload hash mismatch")
        payload = _read_json(path)
        if set(payload) != {"meta", "bars", "trades", "results"}:
            raise AssertionError(f"{symbol}: unexpected lazy payload schema")
        if payload["meta"].get("lead") != "QQQ" or payload["meta"].get("target") != symbol:
            raise AssertionError(f"{symbol}: lazy payload role mismatch")
        bars = payload["bars"]
        lengths = {len(value) for value in bars.values()}
        if lengths != {int(asset["raw_bars"])}:
            raise AssertionError(f"{symbol}: lazy payload arrays are not aligned")
        source_summary = _read_json(SOURCE / symbol / "summary.json")
        full = source_summary["selected_results"]["full"]
        if len(payload["trades"]) != int(full["trades"]):
            raise AssertionError(f"{symbol}: lazy payload/source trade counts differ")
        _close(payload["results"]["full"]["net_pnl"], full["net_pnl"], f"{symbol} payload net P&L")
        if bars["t"] != sorted(bars["t"]) or len(set(bars["t"])) != len(bars["t"]):
            raise AssertionError(f"{symbol}: lazy payload timestamps are not unique/ordered")
        equity = pd.read_csv(SOURCE / symbol / "selected_full_equity.csv")
        raw_epochs = (
            pd.DatetimeIndex(pd.to_datetime(equity.timestamp, format="mixed", utc=True))
            .as_unit("ns").asi8 // 1_000_000_000
        ).tolist()
        if bars["t"] != raw_epochs:
            raise AssertionError(f"{symbol}: lazy payload timestamps differ from audited raw equity timeline")
    html = (report / "index.html").read_text(encoding="utf-8").casefold()
    for token in ("qqq", "vwap", "stop", "target", "equity", "drawdown"):
        if token not in html:
            raise AssertionError(f"Multi-asset HTML does not explain/show {token}")
    generated = str(manifest.get("generated_from", ""))
    if generated and generated not in {
        "research_output/vwap_absolute_multi_asset/manifest.json",
        "research_output/vwap_absolute_multi_asset/summary.json",
    }:
        raise AssertionError("Lazy report provenance does not identify multi-asset research")


def audit() -> dict[str, Any]:
    summary = _require_source()
    manifest = _read_json(MANIFEST)
    completed = list(summary.get("symbols_completed", []))
    universe = _audit_manifest(manifest, summary, completed)
    if not completed or completed != [symbol for symbol in universe if symbol in completed]:
        raise ArtifactsNotReady(
            "No ordered completed symbol stage is available; run multi-asset research first"
        )
    loader = DataLoader(str(ROOT / "data_cache"), "alpaca", "sip")
    lead_all = loader.storage.load_bars("QQQ", "1m")
    if lead_all is None or lead_all.empty:
        raise AssertionError("QQQ raw parquet is absent/empty")
    audited: dict[str, Any] = {}
    for symbol in completed:
        item = _symbol_summary(summary, symbol)
        start, end = _period(summary, item)
        entry = _entry_parameters(summary, item)
        execution = _execution(summary, item)
        selected = _selected(item)
        _audit_selection(symbol, item)
        market = _market(
            loader, lead_all, symbol, start, end,
            beta_days=entry["beta_days"], window=entry["window"], warmup=entry["warmup"],
        )
        manifest_rows = int(manifest["symbols"][symbol]["pairwise_rows_with_qqq"])
        if len(market.common) != manifest_rows:
            raise AssertionError(f"{symbol}: pairwise raw rows differ from frozen manifest")
        expected_bars = item.get("period", item).get("raw_bars") if isinstance(item.get("period", item), dict) else None
        if expected_bars is not None and int(expected_bars) != len(market.common):
            raise AssertionError(f"{symbol}: summary raw bars differ from exact pairwise intersection")
        # Prove VWAP reset and Z warmup on every session once per symbol.
        for lo in market.starts:
            qtyp = (market.lead.high.iloc[lo] + market.lead.low.iloc[lo] + market.lead.close.iloc[lo]) / 3
            ttyp = (market.target.high.iloc[lo] + market.target.low.iloc[lo] + market.target.close.iloc[lo]) / 3
            _close(market.qvwap[lo], qtyp, f"{symbol} QQQ VWAP reset {market.common[lo]}", 1e-9)
            _close(market.tvwap[lo], ttyp, f"{symbol} target VWAP reset {market.common[lo]}", 1e-9)
            if np.any(np.isfinite(market.z[lo:lo + entry["warmup"]])):
                raise AssertionError(f"{symbol}: finite Z inside causal warmup at {market.common[lo]}")
        trades = _load_trades(symbol)
        _audit_trades(symbol, trades, market, selected, entry, execution)
        full = _audit_splits(symbol, item, trades)
        _audit_equity(symbol, trades, market, execution, full)
        audited[symbol] = {"bars": len(market.common), "trades": len(trades), "net_pnl": float(full["net_pnl"])}
    cross_csv = SOURCE / "cross_asset_summary.csv"
    cross_json = SOURCE / "cross_asset_summary.json"
    if not cross_csv.is_file() or not cross_json.is_file():
        raise AssertionError("Cross-asset summary CSV/JSON is absent")
    cross = pd.read_csv(cross_csv)
    cross_payload = _read_json(cross_json).get("rows", [])
    if cross.symbol.tolist() != completed or [row.get("symbol") for row in cross_payload] != completed:
        raise AssertionError("Cross-asset summaries do not preserve the completed frozen-universe order")
    for row in cross.itertuples(index=False):
        value = audited[row.symbol]
        if int(row.full_trades) != value["trades"]:
            raise AssertionError(f"{row.symbol}: cross/full trade counts differ")
        _close(row.full_net_pnl, value["net_pnl"], f"{row.symbol} cross/full net P&L")
    _audit_report(summary, completed)
    return audited


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
    print(f"PASS multi-asset VWAP audit: {len(results)} frozen symbols")
    for symbol, item in results.items():
        print(f"{symbol}: {item['bars']:,} exact pairwise SIP bars, {item['trades']:,} trades, net ${item['net_pnl']:,.2f}")


if __name__ == "__main__":
    main()
