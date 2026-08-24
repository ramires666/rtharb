"""Independent reconciliation of robust sleeve portfolio aggregation.

No research or reporting implementation is imported.  The auditor rebuilds
both frozen portfolios and the post-SEEN veto diagnostic from immutable source
equity/trade artifacts.
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


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "research_output" / "vwap_all_assets_robust_selection"
OUT = ROOT / "research_output" / "vwap_robust_portfolio"
REPORT = ROOT / "tradingview_vwap_robust_portfolio"
UNIVERSE = ("NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "AMD")
VARIANTS = ("factual_sum", "equal_normalized", "seen_veto_diagnostic")
CAPITAL = 100_000.0


class ArtifactsNotReady(FileNotFoundError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual: Any, expected: Any, label: str, atol: float = 3e-7) -> None:
    if not math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=1e-10):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def times(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ("timestamp", "entry_time", "exit_time", "signal_time"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], format="mixed", utc=True).dt.tz_convert("America/New_York")
    return frame


def source_gate() -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    progress = read_json(SOURCE / "progress.json")
    completed = [str(item.get("symbol")) for item in progress.get("completed", [])]
    if progress.get("status") != "COMPLETE" or completed != list(UNIVERSE) or progress.get("remaining"):
        raise AssertionError("Portfolio source is not exact COMPLETE 9/9")
    summaries = {}
    for symbol in UNIVERSE:
        folder = SOURCE / symbol
        summary, audit = read_json(folder / "summary.json"), read_json(folder / "audit.json")
        if audit.get("status") != "PASS" or summary.get("audit", {}).get("status") != "PASS":
            raise AssertionError(f"{symbol}: source audit differs")
        if sha(folder / "pre_seen_freeze.json") != summary["pre_seen_freeze_sha256"]:
            raise AssertionError(f"{symbol}: frozen pre-seen hash differs")
        summaries[symbol] = summary
    confirmed = [symbol for symbol in UNIVERSE if not summaries[symbol]["verdict"].startswith("NO_TRADE")]
    survivors = [symbol for symbol in confirmed
                 if summaries[symbol]["selected_results"]["seen"]["net_pnl"] > 0 and
                 (summaries[symbol]["selected_results"]["seen"].get("pnl_over_dd") or 0) > 0]
    return summaries, confirmed, survivors


def source_curves(symbols: list[str]) -> tuple[pd.DatetimeIndex, dict[str, pd.Series], pd.DataFrame]:
    frames = {symbol: times(SOURCE / symbol / "selected_full_equity.csv") for symbol in symbols}
    union: pd.DatetimeIndex | None = None
    for frame in frames.values():
        clock = pd.DatetimeIndex(frame.timestamp)
        union = clock if union is None else union.union(clock, sort=False)
    if union is None:
        raise AssertionError("No confirmed source sleeves")
    union = union.sort_values()
    aligned = {symbol: pd.Series(frame.equity.to_numpy(float), index=pd.DatetimeIndex(frame.timestamp))
               .reindex(union).ffill().fillna(CAPITAL) for symbol, frame in frames.items()}
    trade_frames = []
    for symbol in symbols:
        frame = times(SOURCE / symbol / "selected_full_trades.csv").copy()
        frame.insert(0, "portfolio_symbol", symbol)
        if "notional_usd" not in frame:
            frame["notional_usd"] = frame.shares.astype(float) * frame.entry_price.astype(float)
        trade_frames.append(frame)
    return union, aligned, pd.concat(trade_frames, ignore_index=True)


def exposure(union: pd.DatetimeIndex, trades: pd.DataFrame, symbols: list[str]) -> tuple[np.ndarray, float]:
    diff = np.zeros(len(union) + 1, float); events: dict[pd.Timestamp, dict[str, float]] = {}
    for row in trades[trades.portfolio_symbol.isin(symbols)].itertuples(index=False):
        start, end = int(union.searchsorted(row.entry_time)), int(union.searchsorted(row.exit_time))
        notional = float(row.notional_usd)
        diff[start] += notional; diff[end] -= notional
        a = events.setdefault(row.entry_time, {"entry": 0.0, "prior_exit": 0.0, "same_exit": 0.0}); a["entry"] += notional
        b = events.setdefault(row.exit_time, {"entry": 0.0, "prior_exit": 0.0, "same_exit": 0.0})
        b["same_exit" if row.exit_time == row.entry_time else "prior_exit"] += notional
    active = maximum = 0.0
    for timestamp in sorted(events):
        event = events[timestamp]; active -= event["prior_exit"]; active += event["entry"]
        maximum = max(maximum, active); active -= event["same_exit"]
    close(active, 0.0, "event exposure final flat")
    return np.cumsum(diff[:-1]), maximum


def expected(union: pd.DatetimeIndex, aligned: dict[str, pd.Series], trades: pd.DataFrame,
             symbols: list[str], scale: float) -> tuple[pd.DataFrame, dict[str, float]]:
    pnl = sum(((aligned[symbol].to_numpy(float) - CAPITAL) * scale for symbol in symbols),
              start=np.zeros(len(union)))
    equity = CAPITAL + pnl; peak = np.maximum.accumulate(np.r_[CAPITAL, equity])[1:]
    gross, event_max = exposure(union, trades, symbols); gross *= scale
    frame = pd.DataFrame({"timestamp": union, "equity": equity, "running_peak": peak,
                          "drawdown_usd": peak - equity,
                          "drawdown_pct": np.divide(peak - equity, peak, out=np.zeros(len(union)), where=peak != 0) * 100,
                          "gross_exposure_usd": gross, "gross_leverage": gross / CAPITAL})
    selected_trades = trades[trades.portfolio_symbol.isin(symbols)]
    metrics = {"net_pnl": float(equity[-1] - CAPITAL), "trades": len(selected_trades),
               "costs": float(selected_trades.costs.sum() * scale),
               "max_drawdown_usd_mtm": float(frame.drawdown_usd.max()),
               "max_drawdown_pct_mtm": float(frame.drawdown_pct.max()),
               "max_gross_entry_event_exposure_usd": event_max * scale,
               "max_gross_entry_event_leverage": event_max * scale / CAPITAL}
    return frame, metrics


def audit_report(summary: dict[str, Any], expected_frames: dict[str, pd.DataFrame]) -> None:
    path = REPORT / "manifest.js"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8").strip(); prefix = "window.VWAP_ROBUST_PORTFOLIO_MANIFEST="
    if not text.startswith(prefix) or not text.endswith(";"):
        raise AssertionError("Portfolio report manifest wrapper differs")
    report = json.loads(text[len(prefix):-1])
    data_path = REPORT / report["data"]
    if data_path.stat().st_size != report["bytes"] or sha(data_path) != report["sha256"]:
        raise AssertionError("Portfolio lazy payload hash/size differs")
    payload = read_json(data_path); bars = payload["bars"]
    if tuple(report["variants"]) != VARIANTS or {len(value) for value in bars.values()} != {len(expected_frames["factual_sum"])}:
        raise AssertionError("Portfolio report variants/aligned arrays differ")
    for variant, frame in expected_frames.items():
        for key, column in (("equity", "equity"), ("drawdown", "drawdown_usd"),
                            ("exposure", "gross_exposure_usd"), ("leverage", "gross_leverage")):
            if not np.allclose(bars[f"{variant}_{key}"], frame[column], atol=3e-7, rtol=1e-10):
                raise AssertionError(f"Portfolio report {variant} {key} differs")
    if summary["confirmed_count"] < 2 and any(payload["correlations"].values()):
        raise AssertionError("False 1x1 diversification correlation published")
    for item in payload["meta"].get("sources", {}).values():
        source_path = ROOT / item["path"]
        if source_path.stat().st_size != item["bytes"] or sha(source_path) != item["sha256"]:
            raise AssertionError("Portfolio report provenance differs")


def audit() -> dict[str, Any]:
    manifest_path = OUT / "manifest.json"
    if not manifest_path.is_file():
        raise ArtifactsNotReady("Run python -m rtharb.research.vwap_robust_portfolio first")
    manifest, summary, internal = read_json(manifest_path), read_json(OUT / "summary.json"), read_json(OUT / "audit.json")
    if manifest.get("status") != "COMPLETE" or internal.get("status") != "PASS" or tuple(manifest.get("variants", ())) != VARIANTS:
        raise AssertionError("Portfolio manifest/internal audit differs")
    summaries, confirmed, survivors = source_gate()
    if summary["confirmed"] != confirmed or summary["seen_veto_survivors"] != survivors:
        raise AssertionError("Portfolio confirmed/SEEN-veto universe differs")
    union, aligned, trades = source_curves(confirmed)
    specs = {"factual_sum": (confirmed, 1.0), "equal_normalized": (confirmed, 1 / len(confirmed)),
             "seen_veto_diagnostic": (survivors, 1.0)}
    expected_frames = {}
    for variant, (symbols, scale) in specs.items():
        frame, metrics = expected(union, aligned, trades, symbols, scale)
        saved = times(OUT / variant / "full_equity.csv")
        if len(saved) != len(frame) or not pd.DatetimeIndex(saved.timestamp).equals(union):
            raise AssertionError(f"{variant}: exact union clock differs")
        for column in ("equity", "running_peak", "drawdown_usd", "drawdown_pct",
                       "gross_exposure_usd", "gross_leverage"):
            if not np.allclose(saved[column], frame[column], atol=3e-7, rtol=1e-10):
                raise AssertionError(f"{variant}: {column} differs")
        published = summary["variants"][variant]
        for key, value in metrics.items():
            close(value, published[key], f"{variant} {key}")
        expected_frames[variant] = frame
    if len(confirmed) < 2:
        try:
            published_single_corr = pd.read_csv(OUT / "daily_pnl_correlation.csv")
        except pd.errors.EmptyDataError:
            published_single_corr = pd.DataFrame()
        if not published_single_corr.empty:
            raise AssertionError("N<2 daily correlation must be null/empty")
        if not summary["single_asset_warning"] or not np.allclose(
                expected_frames["factual_sum"].equity, expected_frames["equal_normalized"].equity):
            raise AssertionError("Single-sleeve diversification warning/identity differs")
    else:
        aligned_frame = pd.DataFrame(aligned)
        expected_daily = aligned_frame.groupby(aligned_frame.index.date).last().diff().fillna(0).corr()
        expected_minute = aligned_frame.diff().fillna(0).corr()
        for path, expected_corr in ((OUT / "daily_pnl_correlation.csv", expected_daily),
                                    (OUT / "concurrent_mtm_correlation.csv", expected_minute)):
            saved = pd.read_csv(path, index_col=0)
            if not np.allclose(saved, expected_corr, atol=1e-9, rtol=1e-9, equal_nan=True):
                raise AssertionError(f"{path.name}: correlation differs")
    contributions = pd.read_csv(OUT / "marginal_dd_contributions.csv")
    for variant in VARIANTS:
        close(contributions[contributions.variant == variant].marginal_drawdown_usd.sum(),
              summary["variants"][variant]["max_drawdown_usd_mtm"], f"{variant} marginal DD sum")
    if any(symbol in survivors for symbol in summary["seen_vetoed"]):
        raise AssertionError("SEEN-veto universe overlaps survivors")
    audit_report(summary, expected_frames)
    return summary


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        summary = audit()
    except ArtifactsNotReady as exc:
        print(f"NOT READY: {exc}", file=sys.stderr); raise SystemExit(2) from exc
    print(f"PASS robust portfolio: {summary['confirmed_count']} confirmed sleeves; "
          f"diversification_tested={summary['diversification_tested']}")


if __name__ == "__main__":
    main()
