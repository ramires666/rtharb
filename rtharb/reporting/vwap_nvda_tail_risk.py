"""Build the lazy interactive decision report for NVDA VWAP tail risk.

The builder never reruns selection.  It reconciles the completed research,
reconstructs exact raw SIP QQQ/NVDA VWAP/fair/Z vectors, and writes one compact
overview plus one lazy JSON payload per RTH session.
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

from rtharb.config import AppConfig
from rtharb.data.loader import DataLoader
from rtharb.research.vwap_absolute_multi_asset import END_DATE, LEAD, START_DATE, load_market
from rtharb.research.vwap_nvda_tail_risk import clean as clean_tail, simulate as simulate_tail


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "research_output" / "vwap_nvda_tail_risk"
OUT = ROOT / "tradingview_vwap_nvda_tail_risk"
DATA = OUT / "data"
SESSIONS = DATA / "sessions"
NY = "America/New_York"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source(path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}


def epoch(value: Any) -> int:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(NY)
    return int(stamp.tz_convert("UTC").timestamp())


def number(value: Any) -> bool | int | float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    out = float(value)
    return out if math.isfinite(out) else None


def vector(values: Any) -> list[Any]:
    return [number(x) for x in np.asarray(values)]


def read_csv(path: Path, *time_columns: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in time_columns:
        frame[column] = pd.to_datetime(frame[column], format="mixed", utc=True).dt.tz_convert(NY)
    return frame


def split_for(day_index: int) -> str:
    if day_index < 125:
        return "development"
    if day_index < 188:
        return "validation"
    return "holdout"


def compact_grid(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = (
        "stop_usd", "max_holding_bars", "net_pnl", "net_pnl_delta_vs_base",
        "utility_delta_usd", "development_net_non_degradation_gate", "is_baseline",
        "core_winner_net_retention_pct", "core_winner_gross_retention_pct",
        "clipped_base_winner_net_usd", "clipped_base_winner_gross_usd",
        "avoided_base_loser_loss_usd", "avoided_worst5_base_loser_loss_usd",
        "mdd_reduction_usd_vs_base", "mdd_reduction_pct_vs_base",
        "trade_cvar5_reduction_usd_vs_base", "trade_cvar5_reduction_pct_vs_base",
        "worst_trade_reduction_usd_vs_base", "worst_trade_reduction_pct_vs_base",
        "positive_net_pnl_mass_usd", "positive_net_pnl_mass_delta_vs_base",
        "net_sharpe", "profit_factor", "max_drawdown_usd_mtm",
        "trade_cvar5_loss_usd", "worst_trade_loss_usd", "trades", "time_stops",
    )
    return [{key: number(row[key]) for key in columns} for _, row in frame.iterrows()]


def compact_finalists(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = (
        "stop_usd", "max_holding_bars", "development_net_pnl_delta_vs_base",
        "development_utility_delta_usd", "development_core_winner_net_retention_pct",
        "validation_gate", "validation_net_pnl", "validation_net_pnl_delta_vs_base",
        "validation_utility_delta_usd", "robust_utility_delta_usd",
        "validation_core_winner_net_retention_pct", "validation_clipped_base_winner_net_usd",
        "validation_avoided_base_loser_loss_usd", "validation_avoided_worst5_base_loser_loss_usd",
        "validation_mdd_reduction_pct_vs_base", "validation_trade_cvar5_reduction_pct_vs_base",
        "validation_worst_trade_reduction_pct_vs_base", "validation_positive_net_pnl_mass_usd",
        "validation_net_sharpe", "validation_profit_factor", "validation_max_drawdown_usd_mtm",
        "validation_trade_cvar5_loss_usd", "validation_worst_trade_loss_usd",
        "positive_gated_neighbor_fraction", "neighbor_median_robust_utility_usd",
    )
    return [{key: number(row[key]) for key in columns} for _, row in frame.iterrows()]


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
    summary_path, audit_path, research_manifest_path = SRC / "summary.json", SRC / "audit.json", SRC / "manifest.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or summary["selection"]["verdict"] != "NO_OP_BASELINE":
        raise AssertionError("Expected completed PASS/NO_OP research")

    # BASE and selected must be exact byte-identical before reporting one path.
    source_files: dict[str, Any] = {
        "summary": source(summary_path), "audit": source(audit_path),
        "research_manifest": source(research_manifest_path),
        "development_grid": source(SRC / "development_grid.csv"),
        "validation_finalists": source(SRC / "validation_finalists.csv"),
    }
    for split in ("development", "validation", "holdout", "full"):
        for noun in ("trades", "equity"):
            base = SRC / f"baseline_{split}_{noun}.csv"
            selected = SRC / f"selected_{split}_{noun}.csv"
            if sha256(base) != sha256(selected):
                raise AssertionError(f"NO_OP mismatch: {split}/{noun}")
            source_files[f"baseline_{split}_{noun}"] = source(base)
            source_files[f"selected_{split}_{noun}"] = source(selected)

    print("PHASE exact raw SIP QQQ/NVDA + VWAP/fair/Z", flush=True)
    arrays, days, data_audit = load_market("NVDA")
    DATA.mkdir(parents=True, exist_ok=True); SESSIONS.mkdir(parents=True, exist_ok=True)
    stop3_bounds = {"development": (0, 125), "validation": (125, 188),
                    "holdout": (188, 251), "full": (0, 251)}
    stop3_results = {name: clean_tail(simulate_tail(arrays, lo, hi, 3.0, None))
                     for name, (lo, hi) in stop3_bounds.items()}
    stop3_expected = {"development": (864.18, 2481.73, 0.348),
                      "validation": (2699.89, 1576.34, 1.713),
                      "holdout": (3033.45, 1671.06, 1.815),
                      "full": (6597.51, 2481.73, 2.658)}
    for name, (net, dd, ratio) in stop3_expected.items():
        actual = stop3_results[name]
        if (abs(actual["net_pnl"] - net) > 0.01 or
                abs(actual["max_drawdown_usd_mtm"] - dd) > 0.01 or
                abs(actual["return_over_mtm_dd"] - ratio) > 0.001):
            raise AssertionError(f"$3 diagnostic mismatch for {name}: {actual}")
    stop3_path = DATA / "stop3_diagnostic.json"
    stop3_payload = {"definition": {"stop_usd": 3.0, "target_usd": 1.25,
                                     "max_holding_bars": None, "selection_role": "diagnostic_only"},
                     "results": stop3_results,
                     "warning": "Holdout is post-selection diagnostic and cannot be used to select $3 retrospectively"}
    stop3_path.write_text(json.dumps(stop3_payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
                          encoding="utf-8")
    source_files["stop3_diagnostic"] = source(stop3_path)
    timestamps = pd.DatetimeIndex(arrays["timestamp"])
    cfg = AppConfig.load(str(ROOT / "configs" / "default_config.yaml"))
    qqq, nvda = DataLoader(cfg.cache_dir, "alpaca", "sip").get_synchronized_pair(LEAD, "NVDA")
    qqq, nvda = qqq.reindex(timestamps), nvda.reindex(timestamps)
    if qqq.isna().any().any() or nvda.isna().any().any() or len(timestamps) != 97_530:
        raise AssertionError("Raw pair reconstruction mismatch")

    trades_path = SRC / "baseline_full_trades.csv"
    equity_path = SRC / "baseline_full_equity.csv"
    trades = read_csv(trades_path, "signal_time", "entry_time", "exit_time")
    equity = read_csv(equity_path, "timestamp").set_index("timestamp").reindex(timestamps)
    if equity.isna().any().any() or len(trades) != summary["base_results"]["full"]["trades"]:
        raise AssertionError("Full trade/equity source mismatch")
    if not math.isclose(float(trades.net_pnl.sum()), summary["base_results"]["full"]["net_pnl"], abs_tol=1e-8):
        raise AssertionError("Full trade PnL mismatch")
    if not math.isclose(float(equity.equity.iloc[-1]), summary["base_results"]["full"]["final_equity"], abs_tol=1e-8):
        raise AssertionError("Full final equity mismatch")

    day_lookup = {pd.Timestamp(day).date(): i for i, day in enumerate(days)}
    trade_items: list[dict[str, Any]] = []
    for trade_id, row in enumerate(trades.itertuples(index=False), 1):
        direction = 1 if row.direction == "LONG" else -1
        stop = float(row.entry_reference) - direction * float(row.stop_usd_per_share)
        target = float(row.entry_reference) + direction * float(row.target_usd_per_share)
        trade_items.append({
            "id": trade_id, "day": str(pd.Timestamp(row.entry_time).date()),
            "split": split_for(day_lookup[pd.Timestamp(row.entry_time).date()]),
            "direction": direction, "side": str(row.direction),
            "signal_time": epoch(row.signal_time), "entry_time": epoch(row.entry_time), "exit_time": epoch(row.exit_time),
            "entry_z": float(row.entry_z), "entry_reference": float(row.entry_reference),
            "entry_price": float(row.entry_price), "exit_reference": float(row.exit_reference),
            "exit_price": float(row.exit_price), "shares": int(row.shares),
            "stop_price": stop, "target_price": target, "exit_reason": str(row.exit_reason),
            "duration_bars": int(row.duration_bars), "gross_pnl": float(row.gross_pnl),
            "slippage": float(row.slippage), "commissions": float(row.commissions),
            "costs": float(row.costs), "net_pnl": float(row.net_pnl),
        })

    dev = pd.read_csv(SRC / "development_grid.csv")
    finalists = pd.read_csv(SRC / "validation_finalists.csv")
    if len(dev) != 378 or int(finalists.validation_gate.sum()) != 0:
        raise AssertionError("Grid/finalist count changed")

    session_assets: list[dict[str, Any]] = []
    date_array = np.asarray(timestamps.date)
    by_day_trades: dict[str, list[dict[str, Any]]] = {}
    for item in trade_items:
        by_day_trades.setdefault(item["day"], []).append(item)
    for day_i, day in enumerate(days):
        day_text = str(pd.Timestamp(day).date())
        mask = date_array == pd.Timestamp(day).date()
        loc = np.flatnonzero(mask)
        eq = equity.iloc[loc]
        payload = {
            "date": day_text, "split": split_for(day_i),
            "bars": {
                "t": [epoch(x) for x in timestamps[loc]],
                "no": vector(nvda.open.iloc[loc]), "nh": vector(nvda.high.iloc[loc]),
                "nl": vector(nvda.low.iloc[loc]), "nc": vector(nvda.close.iloc[loc]),
                "nvwap": vector(arrays["vwap_target"][loc]), "fair": vector(arrays["fair_price"][loc]),
                "z": vector(arrays["z"][loc]),
                "qo": vector(qqq.open.iloc[loc]), "qh": vector(qqq.high.iloc[loc]),
                "ql": vector(qqq.low.iloc[loc]), "qc": vector(qqq.close.iloc[loc]),
                "qvwap": vector(arrays["vwap_lead"][loc]),
                "equity": vector(eq.equity), "drawdown": vector(eq.drawdown_usd),
                "drawdown_pct": vector(eq.drawdown_pct),
            },
            "trades": by_day_trades.get(day_text, []),
        }
        path = SESSIONS / f"{day_text}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
        check = json.loads(path.read_text(encoding="utf-8"))
        if len(check["bars"]["t"]) != len(loc):
            raise AssertionError(f"{day_text}: session JSON mismatch")
        session_assets.append({"date": day_text, "split": split_for(day_i), "bars": len(loc),
                               "trades": len(payload["trades"]), "net_pnl": sum(x["net_pnl"] for x in payload["trades"]),
                               "data": f"data/sessions/{day_text}.json", "bytes": path.stat().st_size,
                               "sha256": sha256(path)})

    grouped = equity.groupby(equity.index.date)
    daily_last = grouped.tail(1)
    overview = {
        "meta": {
            "schema_version": 1, "study": summary["study"], "verdict": summary["selection"]["verdict"],
            "selected": {"stop_usd": summary["selection"]["selected_stop_usd"],
                         "target_usd": summary["frozen_baseline"]["target_usd"],
                         "max_holding_bars": summary["selection"]["selected_max_holding_bars"]},
            "period": {"start": str(START_DATE), "end": str(END_DATE), "sessions": len(days), "raw_bars": len(timestamps)},
            "grid_pairs": len(dev), "eligible_overlays": int(finalists.validation_gate.sum()),
            "selection": summary["selection"], "execution": summary["execution"], "data_audit": data_audit,
            "source_files": source_files,
        },
        "results": summary["base_results"], "selected_vs_base": summary["selected_vs_base"],
        "stop3_comparison": {"baseline": summary["base_results"], "$3_stop": stop3_results,
                             "selection_role": "diagnostic only; holdout never used to select"},
        "grid": compact_grid(dev), "finalists": compact_finalists(finalists),
        "daily": {"t": [epoch(x) for x in daily_last.index], "equity": vector(daily_last.equity),
                  "drawdown": vector(daily_last.drawdown_usd), "drawdown_pct": vector(daily_last.drawdown_pct)},
        "sessions": session_assets,
        "reconciliation": {
            "audit_pass": audit["status"] == "PASS", "raw_bars": len(timestamps), "sessions": len(days),
            "full_trade_rows": len(trades), "full_trade_net": float(trades.net_pnl.sum()),
            "full_final_equity": float(equity.equity.iloc[-1]),
            "full_mtm_mdd": float(equity.drawdown_usd.max()),
            "base_selected_all_split_files_byte_identical": True,
            "stop3_exact_raw_replay": True,
            "session_trade_additivity": sum(x["trades"] for x in session_assets) == len(trades),
            "session_net_additivity": abs(sum(x["net_pnl"] for x in session_assets) - float(trades.net_pnl.sum())) <= 1e-8,
        },
    }
    if not all(value for key, value in overview["reconciliation"].items() if isinstance(value, bool)):
        raise AssertionError(overview["reconciliation"])
    overview_path = DATA / "overview.json"
    overview_path.write_text(json.dumps(overview, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    readback = json.loads(overview_path.read_text(encoding="utf-8"))
    if len(readback["grid"]) != 378 or len(readback["sessions"]) != 251:
        raise AssertionError("Overview readback mismatch")
    manifest = {
        "schema_version": 1, "status": "COMPLETE", "default_session": max(session_assets, key=lambda x: x["trades"])["date"],
        "overview": {"data": "data/overview.json", "bytes": overview_path.stat().st_size, "sha256": sha256(overview_path)},
        "sessions": len(session_assets), "session_bytes": sum(x["bytes"] for x in session_assets),
        "verdict": summary["selection"]["verdict"], "source": source(summary_path), "audit": source(audit_path),
    }
    (OUT / "manifest.js").write_text("window.VWAP_NVDA_TAIL_RISK_MANIFEST=" +
                                      json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "report": str(OUT / "index.html"),
                      "overview_bytes": overview_path.stat().st_size,
                      "lazy_sessions": len(session_assets), "lazy_session_bytes": manifest["session_bytes"],
                      "reconciliation": overview["reconciliation"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
