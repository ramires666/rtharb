"""Independent raw-event audit of the nine-strategy VWAP portfolio.

No portfolio research or reporting implementation is imported.  The auditor
reconstructs the common nine-way raw Alpaca SIP clock and causal VWAP-Z arrays,
then verifies all admissions, fills, brackets, costs, trades, combined MTM,
capital constraints, split metrics, correlations and lazy report payloads.
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

from rtharb.audit.vwap_absolute_multi_asset import Market, _market
from rtharb.data.loader import DataLoader


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "research_output" / "vwap_absolute_multi_asset"
OUT = ROOT / "research_output" / "vwap_absolute_portfolio"
REPORT = ROOT / "tradingview_vwap_absolute_portfolio"
SYMBOLS = ("NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "AMD")
VARIANTS = ("equal_allocation", "shared_cap", "uncapped_diagnostic")
SPLITS = {"development": (0, 125), "validation": (125, 188), "holdout": (188, 251), "full": (0, 251)}
CAPITAL = 100_000.0
COMMISSION = 0.0035
SLIP = 0.0002
ENTRY_Z = 2.5
ATOL = 1e-7
NY = "America/New_York"


class ArtifactsNotReady(FileNotFoundError):
    pass


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _close(actual: Any, expected: Any, label: str, atol: float = ATOL) -> None:
    a, e = float(actual), float(expected)
    if not (math.isfinite(a) and math.isfinite(e)) or not math.isclose(a, e, abs_tol=atol, rel_tol=1e-10):
        raise AssertionError(f"{label}: {a!r} != {e!r}")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _epoch(value: Any) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise AssertionError(f"Naive timestamp: {value!r}")
    return int(ts.tz_convert("UTC").timestamp())


def _time_columns(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.DataFrame:
    for name in names:
        frame[name] = pd.to_datetime(frame[name], format="mixed", utc=True).dt.tz_convert(NY)
    return frame


def _gate() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = OUT / "manifest.json"
    if not manifest_path.is_file():
        raise ArtifactsNotReady("Run python -m rtharb.research.vwap_absolute_portfolio first")
    manifest = _read(manifest_path)
    if manifest.get("status") != "COMPLETE":
        raise ArtifactsNotReady(f"Portfolio research is not COMPLETE: {manifest.get('status')}")
    if tuple(manifest.get("traded_symbols", ())) != SYMBOLS or manifest.get("reference_only") != "QQQ":
        raise AssertionError("Portfolio roles/frozen universe differ")
    if tuple(manifest.get("variants", ())) != VARIANTS:
        raise AssertionError("Portfolio variants differ from the frozen three-model design")
    source_manifest = _read(SOURCE / "manifest.json")
    if source_manifest.get("status") != "COMPLETE" or tuple(source_manifest.get("symbols_completed", ())) != SYMBOLS:
        raise AssertionError("Source multi-asset study is not COMPLETE 9/9")
    summaries: dict[str, dict[str, Any]] = {}
    for symbol in SYMBOLS:
        summaries[symbol] = _read(SOURCE / symbol / "summary.json")
        selected = summaries[symbol]["selected"]
        frozen = manifest["frozen_stop_target"][symbol]
        _close(selected["stop_usd"], frozen["stop_usd"], f"{symbol} frozen stop")
        _close(selected["target_usd"], frozen["target_usd"], f"{symbol} frozen target")
        if summaries[symbol]["symbols"] != {"reference_only": "QQQ", "traded": symbol}:
            raise AssertionError(f"{symbol}: source roles changed")
    return manifest, summaries


def _global_markets(summaries: dict[str, dict[str, Any]]) -> tuple[pd.DatetimeIndex, dict[str, Market], dict[str, np.ndarray]]:
    loader = DataLoader(str(ROOT / "data_cache"), "alpaca", "sip")
    lead = loader.storage.load_bars("QQQ", "1m")
    if lead is None or lead.empty:
        raise AssertionError("QQQ raw parquet missing")
    markets: dict[str, Market] = {}
    common: pd.DatetimeIndex | None = None
    for symbol in SYMBOLS:
        summary = summaries[symbol]
        entry = summary["entry_parameters"]
        market = _market(
            loader, lead, symbol,
            pd.Timestamp(summary["period"]["start"]).date(), pd.Timestamp(summary["period"]["end"]).date(),
            int(entry["beta_days"]), int(entry["window"]), int(entry["warmup_bars"]),
        )
        markets[symbol] = market
        common = market.common if common is None else common.intersection(market.common, sort=False)
    assert common is not None
    common = common.sort_values()
    if len(common) != 97_529 or common.has_duplicates or len(pd.unique(common.date)) != 251:
        raise AssertionError("Global raw calendar is not the exact 97,529-minute/251-session inner intersection")
    takes: dict[str, np.ndarray] = {}
    for symbol, market in markets.items():
        take = market.common.get_indexer(common)
        if np.any(take < 0):
            raise AssertionError(f"{symbol}: global timestamp is absent from pairwise raw market")
        takes[symbol] = take
    return common, markets, takes


def _variant_files(name: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folder = OUT / "variants" / name
    required = tuple(folder / filename for filename in ("summary.json", "entry_events.csv", "trades.csv", "equity.csv"))
    if not all(path.is_file() for path in required):
        raise ArtifactsNotReady(f"Missing portfolio variant artifacts: {folder}")
    summary = _read(required[0])
    entries = _time_columns(pd.read_csv(required[1]), ("signal_time", "entry_time"))
    trades = _time_columns(pd.read_csv(required[2]), ("signal_time", "entry_time", "exit_time"))
    equity = _time_columns(pd.read_csv(required[3]), ("timestamp",))
    return summary, entries, trades, equity


def _map_indices(frame: pd.DataFrame, columns: tuple[str, ...], by_epoch: dict[int, int], label: str) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for column in columns:
        values = np.asarray([by_epoch.get(_epoch(value), -1) for value in frame[column]], dtype=int)
        if np.any(values < 0):
            raise AssertionError(f"{label}: {column} contains a non-global timestamp")
        result[column] = values
    return result


def _audit_admissions(name: str, entries: pd.DataFrame, trades: pd.DataFrame, common: pd.DatetimeIndex,
                      markets: dict[str, Market], takes: dict[str, np.ndarray], summary: dict[str, Any]) -> dict[str, np.ndarray]:
    by_epoch = dict(zip((common.as_unit("ns").asi8 // 1_000_000_000).astype(int), range(len(common))))
    entry_idx = _map_indices(entries, ("signal_time", "entry_time"), by_epoch, f"{name} entries")
    trade_idx = _map_indices(trades, ("signal_time", "entry_time", "exit_time"), by_epoch, f"{name} trades")
    if np.any(entry_idx["entry_time"] != entry_idx["signal_time"] + 1):
        raise AssertionError(f"{name}: admission is not next raw global open")
    if set(entries.symbol) - set(SYMBOLS) or set(trades.symbol) - set(SYMBOLS) or "QQQ" in set(trades.symbol):
        raise AssertionError(f"{name}: non-frozen/QQQ traded leg found")

    # Every saved admission must be a causal threshold event in the same direction.
    for row_no, row in enumerate(entries.itertuples(index=False)):
        symbol = row.symbol
        si, ei = entry_idx["signal_time"][row_no], entry_idx["entry_time"][row_no]
        z = float(markets[symbol].z[takes[symbol][si]])
        _close(row.entry_z, z, f"{name} admission {row_no + 1} causal Z", 1e-9)
        expected_direction = "LONG" if z <= -ENTRY_Z else "SHORT" if z >= ENTRY_Z else ""
        if row.direction != expected_direction:
            raise AssertionError(f"{name} admission {row_no + 1}: direction/threshold mismatch")
        raw_open = float(markets[symbol].target.open.iloc[takes[symbol][ei]])
        fill = raw_open * (1 + SLIP if expected_direction == "LONG" else 1 - SLIP)
        _close(row.entry_reference, raw_open, f"{name} admission {row_no + 1} raw open")
        _close(row.entry_price, fill, f"{name} admission {row_no + 1} fill")
        _close(row.actual_entry_notional, int(row.shares) * fill, f"{name} admission {row_no + 1} notional")

    # Filled/partial admissions map one-to-one to trades; rejected events do not.
    admitted = entries[entries.status != "REJECTED_CAP"]
    key_cols = ["symbol", "signal_time", "entry_time", "direction", "shares"]
    left = admitted[key_cols].astype({"shares": int}).sort_values(key_cols).reset_index(drop=True)
    right = trades[key_cols].astype({"shares": int}).sort_values(key_cols).reset_index(drop=True)
    if not left.equals(right):
        raise AssertionError(f"{name}: admitted entries do not map one-to-one to trades")
    admission_by_key = {
        (row.symbol, _epoch(row.signal_time), _epoch(row.entry_time)): row
        for row in admitted.itertuples(index=False)
    }
    for number, row in enumerate(trades.itertuples(index=False), 1):
        event = admission_by_key[(row.symbol, _epoch(row.signal_time), _epoch(row.entry_time))]
        for trade_name, event_name in (
            ("entry_reference", "entry_reference"), ("entry_price", "entry_price"),
            ("requested_notional", "requested_notional"),
            ("allocated_notional", "pro_rata_allocated_notional"),
            ("entry_notional", "actual_entry_notional"),
        ):
            _close(getattr(row, trade_name), getattr(event, event_name), f"{name} trade {number} admission {trade_name}")

    # Reconstruct all causal close signals using saved position lifetimes. This
    # catches omitted/cherry-picked admissions as well as extra admissions.
    expected_signals: list[tuple[str, int]] = []
    ignored = 0
    session_ends = set(np.flatnonzero(np.r_[common.date[1:] != common.date[:-1], True]))
    for symbol in SYMBOLS:
        rows = trades[trades.symbol == symbol]
        intervals = sorted(
            (by_epoch[_epoch(row.entry_time)], by_epoch[_epoch(row.exit_time)])
            for row in rows.itertuples(index=False)
        )
        cursor = 0
        z_values = markets[symbol].z[takes[symbol]]
        for i, z in enumerate(z_values):
            while cursor < len(intervals) and i >= intervals[cursor][1]:
                cursor += 1
            open_at_close = cursor < len(intervals) and intervals[cursor][0] <= i < intervals[cursor][1]
            if i in session_ends or not math.isfinite(float(z)) or abs(float(z)) < ENTRY_Z:
                continue
            if open_at_close:
                ignored += 1
            else:
                expected_signals.append((symbol, i))
    actual_signals = sorted(
        ((row.symbol, by_epoch[_epoch(row.signal_time)]) for row in entries.itertuples(index=False)),
        key=lambda item: (item[1], SYMBOLS.index(item[0])),
    )
    expected_signals.sort(key=lambda item: (item[1], SYMBOLS.index(item[0])))
    if actual_signals != expected_signals:
        raise AssertionError(f"{name}: reconstructed flat causal signal stream differs ({len(actual_signals)} != {len(expected_signals)})")
    stats = summary["admission_statistics"]
    if int(stats["entry_events"]) != len(entries) or int(stats["generated_signals"]) != len(expected_signals):
        raise AssertionError(f"{name}: admission statistics differ")
    if int(stats["ignored_signals_while_open"]) != ignored:
        raise AssertionError(f"{name}: ignored signal count differs")
    return trade_idx


def _audit_capital(name: str, entries: pd.DataFrame, trades: pd.DataFrame,
                   trade_idx: dict[str, np.ndarray], summary: dict[str, Any]) -> None:
    if name in {"equal_allocation", "uncapped_diagnostic"}:
        request = 11_111.11 if name == "equal_allocation" else 20_000.0
        if set(entries.status) != {"FILLED"}:
            raise AssertionError(f"{name}: all events must be filled")
        if not np.allclose(entries.requested_notional, request, atol=1e-9) or not np.allclose(entries.pro_rata_allocated_notional, request, atol=1e-9):
            raise AssertionError(f"{name}: fixed sleeve/request changed")
    else:
        # Independently reconstruct each shared-cap simultaneous pro-rata batch.
        trade_entry_epoch = trades.entry_time.map(_epoch).to_numpy(int)
        trade_exit_epoch = trades.exit_time.map(_epoch).to_numpy(int)
        for entry_i, batch in entries.groupby(pd.to_datetime(entries.entry_time, utc=True)):
            epoch_i = _epoch(batch.entry_time.iloc[0])
            active = trades[
                (trade_entry_epoch < epoch_i)
                & ((trade_exit_epoch > epoch_i)
                   | ((trade_exit_epoch == epoch_i) & (trades.exit_reason.to_numpy() != "STOP_GAP")))
            ]
            used = float(active.entry_notional.sum())
            available = max(0.0, CAPITAL - used)
            total_request = float(batch.requested_notional.sum())
            allocated = np.full(len(batch), 20_000.0) if total_request <= available else batch.requested_notional.to_numpy(float) * available / total_request
            if not np.allclose(batch.pro_rata_allocated_notional.to_numpy(float), allocated, atol=1e-7):
                raise AssertionError(f"shared_cap: pro-rata allocation differs at {batch.entry_time.iloc[0]}")
            effective = batch.entry_price.to_numpy(float)
            max_shares = np.floor(batch.requested_notional.to_numpy(float) / effective).astype(int)
            shares = np.minimum(max_shares, np.floor(allocated / effective).astype(int))
            remaining = available - float(np.sum(shares * effective))
            symbols = batch.symbol.tolist()
            while True:
                added = False
                for symbol in SYMBOLS:
                    if symbol not in symbols:
                        continue
                    j = symbols.index(symbol)
                    if shares[j] < max_shares[j] and effective[j] <= remaining + 1e-10:
                        shares[j] += 1; remaining -= effective[j]; added = True
                if not added:
                    break
            if not np.array_equal(shares, batch.shares.to_numpy(int)):
                raise AssertionError(f"shared_cap: integer residual allocation differs at {batch.entry_time.iloc[0]}")
            restricted = allocated < batch.requested_notional.to_numpy(float) - 1e-8
            expected_status = np.where(shares <= 0, "REJECTED_CAP", np.where(restricted, "PARTIAL_CAP", "FILLED"))
            if not np.array_equal(expected_status, batch.status.to_numpy(str)):
                raise AssertionError(f"shared_cap: admission status differs at {batch.entry_time.iloc[0]}")
            if used + float(batch.actual_entry_notional.sum()) > CAPITAL + ATOL:
                raise AssertionError("shared_cap: batch exceeds $100k gross entry cap")
    model = summary["capital_model"]
    if name == "equal_allocation" and (model.get("cross_sleeve_borrowing") is not False or float(model["per_symbol_sleeve_usd"]) != 11_111.11):
        raise AssertionError("equal allocation is not nine fixed non-borrowing sleeves")
    if name == "shared_cap" and float(model["gross_entry_cap_usd"]) != CAPITAL:
        raise AssertionError("shared cap is not $100k")
    if name == "uncapped_diagnostic":
        if model.get("leverage_diagnostic_only") is not True or float(model["maximum_theoretical_gross_entry_usd"]) != 180_000:
            raise AssertionError("uncapped variant is not explicitly labelled leverage diagnostic")


def _audit_trades(name: str, trades: pd.DataFrame, trade_idx: dict[str, np.ndarray], common: pd.DatetimeIndex,
                  markets: dict[str, Market], takes: dict[str, np.ndarray], frozen: dict[str, Any]) -> None:
    dates = np.asarray(common.date)
    starts = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1]
    ends = np.r_[starts[1:] - 1, len(common) - 1]
    day_end = np.empty(len(common), dtype=int)
    for lo_i, hi_i in zip(starts, ends):
        day_end[lo_i:hi_i + 1] = hi_i
    for number, row in enumerate(trades.itertuples(index=False), 1):
        symbol = row.symbol
        si, ei, xi = (trade_idx[key][number - 1] for key in ("signal_time", "entry_time", "exit_time"))
        direction = 1 if row.direction == "LONG" else -1
        raw_entry = float(markets[symbol].target.open.iloc[takes[symbol][ei]])
        stop_distance = float(frozen[symbol]["stop_usd"])
        target_distance = float(frozen[symbol]["target_usd"])
        _close(row.stop_usd_per_share, stop_distance, f"{name} trade {number} frozen stop")
        _close(row.target_usd_per_share, target_distance, f"{name} trade {number} frozen target")
        stop = raw_entry - stop_distance if direction == 1 else raw_entry + stop_distance
        target = raw_entry + target_distance if direction == 1 else raw_entry - target_distance
        entry_fill = raw_entry * (1 + SLIP if direction == 1 else 1 - SLIP)
        _close(row.entry_reference, raw_entry, f"{name} trade {number} raw entry")
        _close(row.entry_price, entry_fill, f"{name} trade {number} entry fill")
        _close(row.entry_notional, int(row.shares) * entry_fill, f"{name} trade {number} entry notional")
        _close(row.stop_price, stop, f"{name} trade {number} stop price")
        _close(row.target_price, target, f"{name} trade {number} target price")
        expected_i = None; raw_exit = None; reason = None
        for i in range(ei, int(day_end[ei]) + 1):
            op = float(markets[symbol].target.open.iloc[takes[symbol][i]])
            hi = float(markets[symbol].target.high.iloc[takes[symbol][i]])
            lo = float(markets[symbol].target.low.iloc[takes[symbol][i]])
            gap = op <= stop if direction == 1 else op >= stop
            stop_hit = lo <= stop if direction == 1 else hi >= stop
            target_hit = hi >= target if direction == 1 else lo <= target
            if gap:
                expected_i, raw_exit, reason = i, op, "STOP_GAP"; break
            if stop_hit:
                expected_i, raw_exit, reason = i, stop, "STOP"; break
            if target_hit:
                expected_i, raw_exit, reason = i, target, "TAKE_PROFIT_BRACKET"; break
            if i == day_end[ei]:
                expected_i, raw_exit, reason = i, float(markets[symbol].target.close.iloc[takes[symbol][i]]), "FORCED_EOD"; break
        if xi != expected_i or row.exit_reason != reason:
            raise AssertionError(f"{name} trade {number}: first raw stop-first exit differs")
        _close(row.exit_reference, raw_exit, f"{name} trade {number} raw exit")
        fill_exit = raw_exit * (1 - SLIP if direction == 1 else 1 + SLIP)
        _close(row.exit_price, fill_exit, f"{name} trade {number} exit fill")
        shares = int(row.shares)
        gross = direction * (raw_exit - raw_entry) * shares
        slippage = (abs(entry_fill - raw_entry) + abs(fill_exit - raw_exit)) * shares
        commissions = 2 * shares * COMMISSION
        _close(row.gross_pnl, gross, f"{name} trade {number} gross")
        _close(row.slippage, slippage, f"{name} trade {number} slippage")
        _close(row.commissions, commissions, f"{name} trade {number} commissions")
        _close(row.costs, slippage + commissions, f"{name} trade {number} costs")
        _close(row.net_pnl, gross - slippage - commissions, f"{name} trade {number} net")


def _audit_equity(name: str, trades: pd.DataFrame, trade_idx: dict[str, np.ndarray], equity: pd.DataFrame,
                  common: pd.DatetimeIndex, markets: dict[str, Market], takes: dict[str, np.ndarray], summary: dict[str, Any]) -> None:
    saved_epoch = pd.DatetimeIndex(equity.timestamp).as_unit("ns").asi8 // 1_000_000_000
    raw_epoch = common.as_unit("ns").asi8 // 1_000_000_000
    if len(equity) != len(common) or not np.array_equal(saved_epoch, raw_epoch):
        raise AssertionError(f"{name}: equity is not exact global raw calendar")
    n = len(common)
    cash_change = np.zeros(n)
    open_pnl = np.zeros(n)
    active = np.zeros(n, dtype=int)
    gross_entry = np.zeros(n)
    gross_mtm = np.zeros(n)
    signed_mtm = np.zeros(n)
    for number, row in enumerate(trades.itertuples(index=False)):
        ei, xi = trade_idx["entry_time"][number], trade_idx["exit_time"][number]
        cash_change[xi] += float(row.net_pnl)
        if xi <= ei:
            continue
        direction = 1 if row.direction == "LONG" else -1
        close = markets[row.symbol].target.close.to_numpy(float)[takes[row.symbol][ei:xi]]
        open_pnl[ei:xi] += direction * (close - float(row.entry_price)) * int(row.shares) - int(row.shares) * COMMISSION
        active[ei:xi] += 1
        gross_entry[ei:xi] += float(row.entry_notional)
        mtm = int(row.shares) * close
        gross_mtm[ei:xi] += mtm
        signed_mtm[ei:xi] += direction * mtm
    expected = CAPITAL + np.cumsum(cash_change) + open_pnl
    peak = np.maximum.accumulate(expected); dd = peak - expected
    for column, values in (("equity", expected), ("running_peak", peak), ("drawdown_usd", dd),
                           ("active_positions", active), ("gross_entry_exposure", gross_entry),
                           ("gross_mtm_exposure", gross_mtm), ("signed_mtm_exposure", signed_mtm)):
        if not np.allclose(equity[column].to_numpy(float), values, atol=2e-7, rtol=1e-11):
            raise AssertionError(f"{name}: independent combined MTM {column} differs")
    _close(equity.equity.iloc[-1], CAPITAL + trades.net_pnl.sum(), f"{name} final equity", 2e-7)
    full = summary["periods"]["full"]
    _close(dd.max(), full["max_drawdown_usd_mtm"], f"{name} full MTM DD", 2e-7)
    if name in {"equal_allocation", "shared_cap"} and gross_entry.max() > CAPITAL + ATOL:
        raise AssertionError(f"{name}: gross entry exposure exceeds $100k")
    if name == "uncapped_diagnostic" and gross_entry.max() <= CAPITAL:
        raise AssertionError("uncapped diagnostic never demonstrates leverage")


def _audit_periods(name: str, trades: pd.DataFrame, equity: pd.DataFrame, summary: dict[str, Any], common: pd.DatetimeIndex) -> None:
    days = list(pd.unique(common.date))
    exit_dates = trades.exit_time.dt.date
    split_net = 0.0; split_trades = 0
    for split, (lo, hi) in SPLITS.items():
        start, end = days[lo], days[hi - 1]
        selected = trades[(exit_dates >= start) & (exit_dates <= end)]
        metrics = summary["periods"][split]
        if int(metrics["trades"]) != len(selected):
            raise AssertionError(f"{name} {split}: trade count differs")
        for column, key in (("gross_pnl", "gross_pnl"), ("commissions", "commissions"),
                            ("slippage", "slippage"), ("costs", "costs"), ("net_pnl", "net_pnl")):
            _close(selected[column].sum(), metrics[key], f"{name} {split} {key}")
        _close(metrics["commissions"] + metrics["slippage"], metrics["costs"], f"{name} {split} cost reconciliation")
        _close(metrics["gross_pnl"] - metrics["costs"], metrics["net_pnl"], f"{name} {split} net reconciliation")
        _close(CAPITAL + metrics["net_pnl"], metrics["final_equity_rebased"], f"{name} {split} rebased final")
        equity_dates = equity.timestamp.dt.date
        segment = equity[(equity_dates >= start) & (equity_dates <= end)]
        prior_net = float(trades[exit_dates < start].net_pnl.sum())
        rebased = CAPITAL + segment.equity.to_numpy(float) - (CAPITAL + prior_net)
        curve = np.r_[CAPITAL, rebased]
        peak = np.maximum.accumulate(curve); dd = peak - curve
        dd_pct = np.divide(dd, peak, out=np.zeros_like(dd), where=peak != 0) * 100
        _close(dd.max(), metrics["max_drawdown_usd_mtm"], f"{name} {split} MTM DD USD", 2e-7)
        _close(dd_pct.max(), metrics["max_drawdown_pct_mtm"], f"{name} {split} MTM DD pct", 2e-7)
        if split != "full":
            split_net += float(metrics["net_pnl"]); split_trades += int(metrics["trades"])
    _close(split_net, summary["periods"]["full"]["net_pnl"], f"{name} split/full net")
    if split_trades != int(summary["periods"]["full"]["trades"]):
        raise AssertionError(f"{name}: split/full trades differ")


def _audit_correlations() -> None:
    source_daily = pd.DataFrame({"date": pd.read_csv(OUT / "daily_net_pnl.csv").date})
    for symbol in SYMBOLS:
        trades = _time_columns(pd.read_csv(SOURCE / symbol / "selected_full_trades.csv"), ("exit_time",))
        grouped = trades.groupby(trades.exit_time.dt.strftime("%Y-%m-%d")).net_pnl.sum()
        source_daily[symbol] = source_daily.date.map(grouped).fillna(0.0)
    published_pnl = pd.read_csv(OUT / "daily_net_pnl.csv")
    published_returns = pd.read_csv(OUT / "daily_returns.csv")
    if published_pnl.columns.tolist() != ["date", *SYMBOLS] or len(published_pnl) != 251:
        raise AssertionError("Daily constituent P&L schema/calendar differs")
    if not np.allclose(published_pnl[list(SYMBOLS)], source_daily[list(SYMBOLS)], atol=1e-7):
        raise AssertionError("Daily constituent P&L differs from frozen source trades")
    if not np.allclose(published_returns[list(SYMBOLS)], source_daily[list(SYMBOLS)] / CAPITAL, atol=1e-9):
        raise AssertionError("Daily constituent returns differ")
    for split, (lo, hi) in SPLITS.items():
        sample = (source_daily[list(SYMBOLS)] / CAPITAL).iloc[lo:hi]
        for method in ("pearson", "spearman"):
            expected = sample.corr(method)
            actual = pd.read_csv(OUT / f"correlation_{method}_{split}.csv", index_col=0).loc[list(SYMBOLS), list(SYMBOLS)]
            if not np.allclose(actual, expected, atol=1e-9, equal_nan=True):
                raise AssertionError(f"{method} {split}: correlation matrix differs")
            if not np.allclose(actual, actual.T, atol=1e-12, equal_nan=True) or not np.allclose(np.diag(actual), 1.0):
                raise AssertionError(f"{method} {split}: matrix is not symmetric/unit diagonal")


def _audit_report(manifest: dict[str, Any]) -> None:
    js_path = REPORT / "manifest.js"
    if not js_path.is_file():
        return
    html = REPORT / "index.html"
    if not html.is_file() or html.stat().st_size < 1000:
        raise AssertionError("Portfolio report index missing/too small")
    js = js_path.read_text(encoding="utf-8").strip()
    prefix = "window.VWAP_PORTFOLIO_MANIFEST="
    if not js.startswith(prefix) or not js.endswith(";"):
        raise AssertionError("Portfolio manifest.js wrapper malformed")
    report = json.loads(js[len(prefix):-1])
    if report.get("lead") != "QQQ" or tuple(report.get("targets", ())) != SYMBOLS or tuple(report.get("variants", ())) != VARIANTS:
        raise AssertionError("Portfolio report roles/universe/variants differ")
    if report.get("global_calendar") != manifest.get("global_calendar"):
        raise AssertionError("Portfolio report global calendar differs")
    assets = report.get("assets", [])
    if [item.get("variant") for item in assets] != list(VARIANTS):
        raise AssertionError("Portfolio report assets differ")
    for asset in assets:
        name = asset["variant"]; path = REPORT / asset["data"]
        if not path.is_file() or path.stat().st_size != int(asset["bytes"]) or _sha(path) != asset["sha256"]:
            raise AssertionError(f"{name}: lazy payload path/size/hash differs")
        payload = _read(path)
        if set(payload) != {"meta", "bars", "constituents", "correlations", "results"}:
            raise AssertionError(f"{name}: lazy payload schema differs")
        equity = pd.read_csv(OUT / "variants" / name / "equity.csv")
        bars = payload["bars"]
        if {len(value) for value in bars.values()} != {97_529}:
            raise AssertionError(f"{name}: lazy combined arrays not aligned")
        if not np.allclose(bars["equity"], equity.equity, atol=1e-7):
            raise AssertionError(f"{name}: lazy equity differs")
        source = _read(OUT / "variants" / name / "summary.json")
        _close(payload["results"]["full"]["net_pnl"], source["periods"]["full"]["net_pnl"], f"{name} report net")
        if payload["meta"].get("lead") != "QQQ" or "QQQ" in payload["meta"].get("targets", []):
            raise AssertionError(f"{name}: report trades/labels QQQ")
    for split, methods in report.get("correlation_sources", {}).items():
        for method, source in methods.items():
            path = ROOT / source["path"]
            if not path.is_file() or path.stat().st_size != int(source["bytes"]) or _sha(path) != source["sha256"]:
                raise AssertionError(f"report correlation provenance differs: {method}/{split}")


def audit(*, raw_replay: bool = True) -> dict[str, Any]:
    manifest, summaries = _gate()
    calendar = manifest["global_calendar"]
    if int(calendar["global_raw_minutes"]) != 97_529 or calendar.get("no_fill_resample_or_interpolation") is not True:
        raise AssertionError("Published portfolio calendar is not exact 97,529-minute no-fill intersection")
    _audit_correlations()
    common: pd.DatetimeIndex | None = None
    markets: dict[str, Market] = {}; takes: dict[str, np.ndarray] = {}
    if raw_replay:
        common, markets, takes = _global_markets(summaries)
    results: dict[str, Any] = {}
    for name in VARIANTS:
        summary, entries, trades, equity = _variant_files(name)
        if summary.get("variant") != name or summary.get("execution", {}).get("reference_only") != "QQQ":
            raise AssertionError(f"{name}: summary roles differ")
        if raw_replay:
            assert common is not None
            idx = _audit_admissions(name, entries, trades, common, markets, takes, summary)
            _audit_capital(name, entries, trades, idx, summary)
            _audit_trades(name, trades, idx, common, markets, takes, manifest["frozen_stop_target"])
            _audit_equity(name, trades, idx, equity, common, markets, takes, summary)
            _audit_periods(name, trades, equity, summary, common)
        else:
            full = summary["periods"]["full"]
            _close(trades.net_pnl.sum(), full["net_pnl"], f"{name} lightweight net")
            _close(trades.costs.sum(), full["costs"], f"{name} lightweight costs")
            _close(equity.equity.iloc[-1], CAPITAL + full["net_pnl"], f"{name} lightweight final equity")
        results[name] = summary["periods"]
    _audit_report(manifest)
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
    print("PASS independent VWAP portfolio audit: 9 targets, 97,529 raw no-fill minutes")
    for name in VARIANTS:
        full, holdout = results[name]["full"], results[name]["holdout"]
        print(f"{name}: full {full['trades']:,} trades, net ${full['net_pnl']:,.2f}, "
              f"MTM DD ${full['max_drawdown_usd_mtm']:,.2f}; holdout ${holdout['net_pnl']:,.2f}")


if __name__ == "__main__":
    main()
