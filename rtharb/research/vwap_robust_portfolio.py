"""Combine confirmed robust-selection sleeves into two explicit portfolios.

This stage is a deterministic aggregation of already executed per-symbol
selected sleeves.  It never replays prices or changes shares:

* ``factual_sum`` uses one shared $100k capital base plus the sum of original
  $20k-notional sleeve P&L.  Concurrent gross exposure can exceed 20%.
* ``equal_normalized`` linearly averages the already executed sleeve P&L.
  This is a research normalization, not a new exact-share backtest.
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
UNIVERSE = ("NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "AMD")
VARIANTS = ("factual_sum", "equal_normalized", "seen_veto_diagnostic")
CAPITAL = 100_000.0


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source(path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha(path)}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                               default=lambda value: value.item() if isinstance(value, np.generic) else str(value)) + "\n",
                    encoding="utf-8")


def read_times(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ("timestamp", "signal_time", "entry_time", "exit_time"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], format="mixed", utc=True).dt.tz_convert("America/New_York")
    return frame


def gate() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    progress_path = SOURCE / "progress.json"
    if not progress_path.is_file():
        raise FileNotFoundError("Robust-selection progress.json отсутствует")
    progress = read_json(progress_path)
    completed = [str(item.get("symbol")) for item in progress.get("completed", [])]
    if progress.get("status") != "COMPLETE" or completed != list(UNIVERSE) or progress.get("remaining"):
        raise RuntimeError(f"Source robust-selection не COMPLETE 9/9: {len(completed)}/9")
    summaries: dict[str, dict[str, Any]] = {}
    for symbol in UNIVERSE:
        folder = SOURCE / symbol
        summary, audit = read_json(folder / "summary.json"), read_json(folder / "audit.json")
        if summary.get("audit", {}).get("status") != "PASS" or audit.get("status") != "PASS":
            raise AssertionError(f"{symbol}: source audit is not PASS")
        if sha(folder / "pre_seen_freeze.json") != summary["pre_seen_freeze_sha256"]:
            raise AssertionError(f"{symbol}: immutable pre-seen freeze hash differs")
        summaries[symbol] = summary
    return progress, summaries


def align_equity(symbols: list[str]) -> tuple[pd.DatetimeIndex, dict[str, pd.Series]]:
    source_frames: dict[str, pd.DataFrame] = {}
    union: pd.DatetimeIndex | None = None
    for symbol in symbols:
        frame = read_times(SOURCE / symbol / "selected_full_equity.csv")
        if frame.timestamp.duplicated().any():
            raise AssertionError(f"{symbol}: duplicate equity timestamps")
        source_frames[symbol] = frame
        clock = pd.DatetimeIndex(frame.timestamp)
        union = clock if union is None else union.union(clock, sort=False)
    if union is None:
        raise AssertionError("No confirmed sleeves")
    union = union.sort_values()
    aligned: dict[str, pd.Series] = {}
    for symbol, frame in source_frames.items():
        series = pd.Series(frame.equity.to_numpy(float), index=pd.DatetimeIndex(frame.timestamp))
        aligned[symbol] = series.reindex(union).ffill().fillna(CAPITAL)
    return union, aligned


def entry_event_max(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    events: dict[pd.Timestamp, dict[str, float]] = {}
    for row in frame.itertuples(index=False):
        notional = float(row.notional_usd)
        entry = events.setdefault(row.entry_time, {"entry": 0.0, "prior_exit": 0.0, "same_exit": 0.0})
        entry["entry"] += notional
        exit_event = events.setdefault(row.exit_time, {"entry": 0.0, "prior_exit": 0.0, "same_exit": 0.0})
        exit_event["same_exit" if row.exit_time == row.entry_time else "prior_exit"] += notional
    active = maximum = 0.0
    for timestamp in sorted(events):
        event = events[timestamp]
        active -= event["prior_exit"]
        active += event["entry"]
        maximum = max(maximum, active)
        active -= event["same_exit"]
    if not math.isclose(active, 0.0, abs_tol=1e-7):
        raise AssertionError("Gross exposure event sweep did not finish flat")
    return maximum


def trades_and_exposure(symbols: list[str], union: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    frames = []; exposure: dict[str, np.ndarray] = {}
    for symbol in symbols:
        frame = read_times(SOURCE / symbol / "selected_full_trades.csv").copy()
        frame.insert(0, "portfolio_symbol", symbol)
        if "notional_usd" not in frame:
            frame["notional_usd"] = frame.shares.astype(float) * frame.entry_price.astype(float)
        frames.append(frame)
        diff = np.zeros(len(union) + 1, float)
        for row in frame.itertuples(index=False):
            start = int(union.searchsorted(row.entry_time)); end = int(union.searchsorted(row.exit_time))
            notional = float(row.notional_usd)
            diff[start] += notional; diff[end] -= notional
        exposure[symbol] = np.cumsum(diff[:-1])
    all_trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return all_trades, exposure


def correlations(aligned: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(aligned) < 2:
        return pd.DataFrame(), pd.DataFrame()
    minute = pd.DataFrame(aligned).diff().fillna(0.0)
    daily_equity = pd.DataFrame(aligned).groupby(pd.DataFrame(aligned).index.date).last()
    daily = daily_equity.diff().fillna(0.0)
    return daily.corr(), minute.corr()


def portfolio_metrics(equity: np.ndarray, union: pd.DatetimeIndex, trades: pd.DataFrame,
                      scale: float, exposure: np.ndarray, event_max: float) -> dict[str, Any]:
    peak = np.maximum.accumulate(np.r_[CAPITAL, equity])[1:]
    dd = peak - equity; dd_pct = np.divide(dd, peak, out=np.zeros_like(dd), where=peak != 0) * 100
    daily = pd.Series(equity, index=union).groupby(union.date).last()
    returns = daily / daily.shift(1).fillna(CAPITAL) - 1.0
    sharpe = math.sqrt(252) * returns.mean() / returns.std(ddof=1) if returns.std(ddof=1) > 0 else 0.0
    net = trades.net_pnl.to_numpy(float) * scale
    wins, losses = net[net > 0], net[net <= 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() < 0 else None
    net_pnl = float(equity[-1] - CAPITAL); max_dd = float(dd.max())
    return {"starting_capital_usd": CAPITAL, "final_equity": float(equity[-1]),
            "net_pnl": net_pnl, "net_return_pct": net_pnl / CAPITAL * 100,
            "trades": len(trades), "costs": float(trades.costs.sum() * scale),
            "commissions": float(trades.commissions.sum() * scale),
            "slippage": float(trades.slippage.sum() * scale), "net_sharpe": float(sharpe),
            "profit_factor": pf, "max_drawdown_usd_mtm": max_dd,
            "max_drawdown_pct_mtm": float(dd_pct.max()),
            "pnl_over_dd": net_pnl / max_dd if max_dd else None,
            "max_gross_minute_close_exposure_usd": float(exposure.max()),
            "max_gross_entry_event_exposure_usd": float(event_max * scale),
            "max_gross_entry_event_leverage": float(event_max * scale / CAPITAL)}


def build_variant(name: str, union: pd.DatetimeIndex, aligned: dict[str, pd.Series],
                  all_trades: pd.DataFrame, exposures: dict[str, np.ndarray],
                  scale: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    pnl = sum(((series.to_numpy(float) - CAPITAL) * scale for series in aligned.values()),
              start=np.zeros(len(union), float))
    equity = CAPITAL + pnl
    gross = sum((values * scale for values in exposures.values()), start=np.zeros(len(union), float))
    peak = np.maximum.accumulate(np.r_[CAPITAL, equity])[1:]
    frame = pd.DataFrame({"timestamp": union, "equity": equity, "running_peak": peak,
                          "drawdown_usd": peak - equity,
                          "drawdown_pct": np.divide(peak - equity, peak, out=np.zeros(len(union)), where=peak != 0) * 100,
                          "gross_exposure_usd": gross, "gross_leverage": gross / CAPITAL})
    trades = all_trades.copy()
    for column in ("gross_pnl", "slippage", "commissions", "costs", "net_pnl"):
        trades[f"portfolio_{column}"] = trades[column].astype(float) * scale
    metrics = portfolio_metrics(equity, union, all_trades, scale, gross, entry_event_max(all_trades))
    trough = int(np.argmax(frame.drawdown_usd.to_numpy(float)))
    peak_i = int(np.argmax(equity[:trough + 1])) if trough >= 0 else 0
    contributions = []
    for symbol, series in aligned.items():
        contribution = scale * (float(series.iloc[peak_i]) - float(series.iloc[trough]))
        contributions.append({"variant": name, "symbol": symbol,
                              "peak_timestamp": union[peak_i], "trough_timestamp": union[trough],
                              "marginal_drawdown_usd": contribution,
                              "share_of_portfolio_dd_pct": contribution / metrics["max_drawdown_usd_mtm"] * 100
                              if metrics["max_drawdown_usd_mtm"] else None})
    contributions_df = pd.DataFrame(contributions, columns=("variant", "symbol", "peak_timestamp",
                                                             "trough_timestamp", "marginal_drawdown_usd",
                                                             "share_of_portfolio_dd_pct"))
    if not math.isclose(contributions_df.marginal_drawdown_usd.sum(), metrics["max_drawdown_usd_mtm"], abs_tol=1e-7):
        raise AssertionError(f"{name}: marginal DD contributions do not add to portfolio DD")
    return frame, trades, metrics, contributions_df


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    progress, summaries = gate()
    confirmed = [symbol for symbol in UNIVERSE if not summaries[symbol]["verdict"].startswith("NO_TRADE")]
    excluded = [symbol for symbol in UNIVERSE if symbol not in confirmed]
    seen_veto_survivors = [symbol for symbol in confirmed
                           if float(summaries[symbol]["selected_results"]["seen"]["net_pnl"]) > 0
                           and float(summaries[symbol]["selected_results"]["seen"].get("pnl_over_dd") or 0) > 0]
    seen_vetoed = [symbol for symbol in confirmed if symbol not in seen_veto_survivors]
    if not confirmed:
        raise RuntimeError("No confirmed sleeves: portfolio is all CASH; publish source universe first")
    union, aligned = align_equity(confirmed)
    all_trades, exposures = trades_and_exposure(confirmed, union)
    daily_corr, minute_corr = correlations(aligned)
    OUT.mkdir(parents=True, exist_ok=True)
    daily_corr.to_csv(OUT / "daily_pnl_correlation.csv", float_format="%.10f")
    minute_corr.to_csv(OUT / "concurrent_mtm_correlation.csv", float_format="%.10f")
    metrics: dict[str, dict[str, Any]] = {}; contribution_frames = []
    variant_specs = (("factual_sum", confirmed, 1.0),
                     ("equal_normalized", confirmed, 1.0 / len(confirmed)),
                     ("seen_veto_diagnostic", seen_veto_survivors, 1.0))
    for name, sleeve_symbols, scale in variant_specs:
        folder = OUT / name; folder.mkdir(parents=True, exist_ok=True)
        sleeve_aligned = {symbol: aligned[symbol] for symbol in sleeve_symbols}
        sleeve_exposures = {symbol: exposures[symbol] for symbol in sleeve_symbols}
        sleeve_trades = all_trades[all_trades.portfolio_symbol.isin(sleeve_symbols)].copy()
        equity, trades, result, contributions = build_variant(
            name, union, sleeve_aligned, sleeve_trades, sleeve_exposures, scale)
        result["sleeves"] = sleeve_symbols
        result["post_seen_diagnostic"] = name == "seen_veto_diagnostic"
        equity.to_csv(folder / "full_equity.csv", index=False, float_format="%.10f")
        trades.to_csv(folder / "trades.csv", index=False, float_format="%.10f")
        write_json(folder / "summary.json", result)
        metrics[name] = result; contribution_frames.append(contributions)
    contributions = pd.concat(contribution_frames, ignore_index=True)
    contributions.to_csv(OUT / "marginal_dd_contributions.csv", index=False, float_format="%.10f")
    universe_rows = []
    for symbol in UNIVERSE:
        summary = summaries[symbol]; result = summary["selected_results"]["full"]
        universe_rows.append({"symbol": symbol, "verdict": summary["verdict"],
                              "portfolio_role": "CONFIRMED_SLEEVE" if symbol in confirmed else "CASH_ZERO_EXPOSURE",
                              "seen_net_pnl": summary["selected_results"]["seen"]["net_pnl"],
                              "seen_pnl_over_dd": summary["selected_results"]["seen"].get("pnl_over_dd"),
                              "seen_veto_diagnostic": symbol in seen_vetoed,
                              "net_pnl": result["net_pnl"], "max_drawdown_usd_mtm": result["max_drawdown_usd_mtm"],
                              "pnl_over_dd": result["pnl_over_dd"], "trades": result["trades"], "costs": result["costs"]})
    universe = pd.DataFrame(universe_rows)
    universe.to_csv(OUT / "universe.csv", index=False, float_format="%.10f")
    best = universe[universe.portfolio_role == "CONFIRMED_SLEEVE"].sort_values(
        ["pnl_over_dd", "net_pnl"], ascending=False, kind="mergesort").iloc[0]
    provenance = {"progress": source(SOURCE / "progress.json"),
                  "cross_asset_summary": source(SOURCE / "cross_asset_summary.json"), "symbols": {}}
    for symbol in UNIVERSE:
        folder = SOURCE / symbol
        provenance["symbols"][symbol] = {name: source(folder / name) for name in
                                         ("summary.json", "audit.json", "pre_seen_freeze.json",
                                          "selected_full_trades.csv", "selected_full_equity.csv")}
    write_json(OUT / "source_provenance.json", provenance)
    diversification_tested = len(confirmed) >= 2
    summary = {"schema_version": 1, "study": "Robust selected sleeve portfolio combination",
               "status": "COMPLETE", "universe": list(UNIVERSE), "confirmed": confirmed,
               "excluded_no_trade": excluded, "confirmed_count": len(confirmed),
               "seen_veto_survivors": seen_veto_survivors, "seen_vetoed": seen_vetoed,
               "seen_veto_warning": "Post-SEEN conservative deployment diagnostic using already-seen sessions 188:251; not untouched/OOS proof and never parameter retuning.",
               "diversification_tested": diversification_tested,
               "single_asset_warning": None if diversification_tested else
               "Only one confirmed sleeve survived; diversification is not tested and both portfolios are identical.",
               "method": {"timestamp_alignment": "exact union; last observed sleeve equity held as cash between its RTH timestamps",
                          "factual_sum": "$100k shared starting capital + sum of original executed $20k-notional sleeve PnL; gross leverage may exceed 20%",
                          "equal_normalized": "$100k + arithmetic mean of already executed sleeve PnL; linear research normalization, not a new rounded-share replay",
                          "costs": "already embedded in each source trade/equity and summed or scaled exactly",
                          "correlations": "daily session-last PnL changes and concurrent union-clock minute MTM changes",
                          "marginal_dd": "sleeve change from portfolio running-peak timestamp to max-DD trough"},
               "variants": metrics, "best_single_confirmed": best.to_dict(),
               "correlations_available": diversification_tested,
               "period": {"first_timestamp": union[0], "last_timestamp": union[-1],
                          "union_minutes": len(union), "sessions": len(pd.unique(union.date))}}
    write_json(OUT / "summary.json", summary)
    checks = {"source_complete_9of9": True, "source_audits_pass": True,
              "confirmed_nonempty": bool(confirmed), "no_trade_zero_exposure": set(excluded).isdisjoint(confirmed),
              "factual_final_additivity": math.isclose(metrics["factual_sum"]["net_pnl"],
                                                        universe[universe.portfolio_role == "CONFIRMED_SLEEVE"].net_pnl.sum(), abs_tol=1e-7),
              "normalized_final_average": math.isclose(metrics["equal_normalized"]["net_pnl"],
                                                        universe[universe.portfolio_role == "CONFIRMED_SLEEVE"].net_pnl.mean(), abs_tol=1e-7),
              "correlations_null_if_single": diversification_tested or (daily_corr.empty and minute_corr.empty),
              "single_portfolios_identical": diversification_tested or math.isclose(
                  metrics["factual_sum"]["net_pnl"], metrics["equal_normalized"]["net_pnl"], abs_tol=1e-9),
              "seen_veto_is_frozen_subset": set(seen_veto_survivors).issubset(confirmed),
              "seen_veto_rule_exact": all(
                  (symbol in seen_veto_survivors) ==
                  (float(summaries[symbol]["selected_results"]["seen"]["net_pnl"]) > 0 and
                   float(summaries[symbol]["selected_results"]["seen"].get("pnl_over_dd") or 0) > 0)
                  for symbol in confirmed)}
    audit = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}
    write_json(OUT / "audit.json", audit)
    if audit["status"] != "PASS":
        raise AssertionError(audit)
    manifest = {"schema_version": 1, "status": "COMPLETE", "audit": {"status": "PASS", "file": "audit.json"},
                "variants": list(VARIANTS), "confirmed": confirmed, "excluded_no_trade": excluded,
                "outputs": {"summary": "summary.json", "universe": "universe.csv",
                            "daily_correlation": "daily_pnl_correlation.csv",
                            "minute_correlation": "concurrent_mtm_correlation.csv",
                            "marginal_dd": "marginal_dd_contributions.csv",
                            "provenance": "source_provenance.json"}}
    write_json(OUT / "manifest.json", manifest)
    print(json.dumps({"status": "COMPLETE", "confirmed": confirmed,
                      "factual": metrics["factual_sum"], "normalized": metrics["equal_normalized"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
