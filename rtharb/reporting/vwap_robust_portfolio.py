"""Publish the robust sleeve portfolio comparison as one lazy payload."""
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
SRC = ROOT / "research_output" / "vwap_robust_portfolio"
SOURCE_SLEEVES = ROOT / "research_output" / "vwap_all_assets_robust_selection"
OUT = ROOT / "tradingview_vwap_robust_portfolio"
DATA = OUT / "data"
VARIANTS = ("factual_sum", "equal_normalized", "seen_veto_diagnostic")


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


def read_times(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ("timestamp", "entry_time", "exit_time", "signal_time"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], format="mixed", utc=True).dt.tz_convert("America/New_York")
    return frame


def number(value: Any) -> int | float | None:
    if value is None or pd.isna(value):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def matrix(path: Path) -> dict[str, Any] | None:
    try:
        frame = pd.read_csv(path, index_col=0)
    except pd.errors.EmptyDataError:
        return None
    if frame.empty or len(frame) < 2:
        return None
    return {"symbols": frame.columns.tolist(),
            "values": [[number(value) for value in row] for row in frame.to_numpy(float)]}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    manifest_path = SRC / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Сначала запустите python -m rtharb.research.vwap_robust_portfolio")
    manifest, summary, audit = read_json(manifest_path), read_json(SRC / "summary.json"), read_json(SRC / "audit.json")
    if manifest.get("status") != "COMPLETE" or audit.get("status") != "PASS":
        raise RuntimeError("Portfolio source is not COMPLETE/PASS")
    frames = {name: read_times(SRC / name / "full_equity.csv") for name in VARIANTS}
    clock = pd.DatetimeIndex(frames["factual_sum"].timestamp)
    if any(len(frame) != len(clock) or not pd.DatetimeIndex(frame.timestamp).equals(clock)
           for frame in frames.values()):
        raise AssertionError("Portfolio variant clocks differ")
    best_symbol = str(summary["best_single_confirmed"]["symbol"])
    best = read_times(SOURCE_SLEEVES / best_symbol / "selected_full_equity.csv")
    best_series = pd.Series(best.equity.to_numpy(float), index=pd.DatetimeIndex(best.timestamp)).reindex(clock).ffill().fillna(100_000.0)
    best_peak = np.maximum.accumulate(np.r_[100_000.0, best_series.to_numpy(float)])[1:]
    universe = pd.read_csv(SRC / "universe.csv")
    contributions = pd.read_csv(SRC / "marginal_dd_contributions.csv")
    bars: dict[str, Any] = {"t": (clock.as_unit("ns").asi8 // 1_000_000_000).astype(int).tolist(),
                            "best_single_equity": best_series.tolist(),
                            "best_single_drawdown": (best_peak - best_series.to_numpy(float)).tolist()}
    for name, frame in frames.items():
        bars[f"{name}_equity"] = frame.equity.astype(float).tolist()
        bars[f"{name}_drawdown"] = frame.drawdown_usd.astype(float).tolist()
        bars[f"{name}_exposure"] = frame.gross_exposure_usd.astype(float).tolist()
        bars[f"{name}_leverage"] = frame.gross_leverage.astype(float).tolist()
    provenance_names = ("manifest.json", "summary.json", "audit.json", "universe.csv",
                        "daily_pnl_correlation.csv", "concurrent_mtm_correlation.csv",
                        "marginal_dd_contributions.csv", "source_provenance.json")
    provenance = {name: source(SRC / name) for name in provenance_names}
    for variant in VARIANTS:
        for name in ("full_equity.csv", "trades.csv", "summary.json"):
            provenance[f"{variant}/{name}"] = source(SRC / variant / name)
    payload = {"meta": {"schema_version": 1, "confirmed": summary["confirmed"],
                         "excluded_no_trade": summary["excluded_no_trade"],
                         "seen_veto_survivors": summary["seen_veto_survivors"],
                         "seen_vetoed": summary["seen_vetoed"],
                         "diversification_tested": summary["diversification_tested"],
                         "single_asset_warning": summary["single_asset_warning"],
                         "seen_veto_warning": summary["seen_veto_warning"],
                         "method": summary["method"], "best_single": summary["best_single_confirmed"],
                         "sources": provenance},
               "bars": bars, "metrics": summary["variants"],
               "correlations": {"daily_pnl": matrix(SRC / "daily_pnl_correlation.csv"),
                                "concurrent_mtm": matrix(SRC / "concurrent_mtm_correlation.csv")},
               "universe": [{key: (value if isinstance(value, str) else number(value))
                              for key, value in row.items()} for row in universe.to_dict(orient="records")],
               "marginal_dd": [{key: (value if isinstance(value, str) else number(value))
                                for key, value in row.items()} for row in contributions.to_dict(orient="records")]}
    OUT.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    data_path = DATA / "portfolio.json"
    data_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    report_manifest = {"schema_version": 1, "data": "data/portfolio.json", "bytes": data_path.stat().st_size,
                       "sha256": sha(data_path), "source": source(manifest_path),
                       "variants": list(VARIANTS), "confirmed_count": len(summary["confirmed"]),
                       "diversification_tested": summary["diversification_tested"]}
    (OUT / "manifest.js").write_text("window.VWAP_ROBUST_PORTFOLIO_MANIFEST=" +
                                      json.dumps(report_manifest, ensure_ascii=False, separators=(",", ":")) + ";\n",
                                      encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "payload_bytes": data_path.stat().st_size}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
