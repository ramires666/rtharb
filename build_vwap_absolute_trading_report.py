"""Build and strictly reconcile the event-driven absolute VWAP-Z data payload.

No HTML is generated. Output schema:
``window.VWAP_ABSOLUTE_DATA={meta,bars,trades,results}``.
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

from research_risk_reward import CAPITAL, COMMISSION, SLIP
from research_vwap_absolute_event_driven import clean, market, simulate
from rtharb.config import AppConfig
from rtharb.data.loader import DataLoader


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "tradingview_vwap_absolute"
SRC = ROOT / "research_output" / "vwap_absolute_event_driven"
SUMMARY_PATH = SRC / "summary.json"
TRADES_PATH = SRC / "selected_full_trades.csv"
MTM_PATH = SRC / "selected_full_equity.csv"
NY = "America/New_York"
ATOL = 1e-8


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def epoch(value: Any) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise AssertionError(f"Naive timestamp: {ts}")
    return int(ts.tz_convert("UTC").timestamp())


def number(value: Any) -> int | float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    result = float(value)
    if not math.isfinite(result):
        raise AssertionError(f"Non-finite JSON value: {value!r}")
    return result


def parse_times(frame: pd.DataFrame, *columns: str) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        out[column] = pd.to_datetime(out[column], format="mixed", utc=True).dt.tz_convert(NY)
    return out


def assert_close(actual: Any, expected: Any, label: str, atol: float = ATOL) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=atol):
        raise AssertionError(f"{label}: {actual!r} != {expected!r}")


def assert_metrics(actual: dict, expected: dict, label: str) -> None:
    if set(actual) != set(expected):
        raise AssertionError(f"Metric keys changed for {label}: {set(actual) ^ set(expected)}")
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool):
            assert_close(actual_value, expected_value, f"{label}.{key}")
        elif actual_value != expected_value:
            raise AssertionError(f"{label}.{key}: {actual_value!r} != {expected_value!r}")


def assert_frames(actual: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    """Compare serialized research records with an independent exact replay."""
    if list(actual.columns) != list(expected.columns):
        raise AssertionError(f"{label} columns changed: {list(actual.columns)} != {list(expected.columns)}")
    if len(actual) != len(expected):
        raise AssertionError(f"{label} row count changed: {len(actual)} != {len(expected)}")
    times = {"signal_time", "entry_time", "exit_time", "timestamp"}
    texts = {"direction", "exit_reason"}
    for column in actual.columns:
        left, right = actual[column], expected[column]
        if column in times:
            if not pd.DatetimeIndex(left).equals(pd.DatetimeIndex(right)):
                raise AssertionError(f"{label}.{column} timestamps differ")
        elif column in texts:
            if not np.array_equal(left.astype(str).to_numpy(), right.astype(str).to_numpy()):
                raise AssertionError(f"{label}.{column} text differs")
        elif not np.allclose(left.to_numpy(float), right.to_numpy(float), rtol=0.0,
                            atol=ATOL, equal_nan=True):
            delta = float(np.nanmax(np.abs(left.to_numpy(float) - right.to_numpy(float))))
            raise AssertionError(f"{label}.{column} numeric max delta={delta}")


def split_for_day(day, summary: dict) -> str:
    validation_start = pd.Timestamp(summary["splits"]["validation"]["start"]).date()
    holdout_start = pd.Timestamp(summary["splits"]["holdout"]["start"]).date()
    return "development" if day < validation_start else "validation" if day < holdout_start else "holdout"


def ui_result(result: dict, frame: pd.DataFrame) -> dict:
    ignored = int(result["ignored_signals_while_open"])
    generated = int(result["generated_flat_signals"])
    out = dict(result)
    out.update({
        "candidate_entries": generated + ignored,
        "skipped_overlaps": ignored,
        "max_drawdown_usd": float(result["max_drawdown_usd_daily"]),
        "max_drawdown_pct": float(result["max_drawdown_pct_daily"]),
        "stops": int((frame.exit_reason == "STOP").sum()),
        "targets": int((frame.exit_reason == "TAKE_PROFIT_BRACKET").sum()),
        "forced_eod": int((frame.exit_reason == "FORCED_EOD").sum()),
    })
    if out["candidate_entries"] != out["trades"] + out["skipped_overlaps"]:
        raise AssertionError("Candidate accounting failed")
    return out


def trade_items(trades: pd.DataFrame, common: pd.DatetimeIndex, arrays: dict,
                nvda: pd.DataFrame, summary: dict) -> list[dict]:
    by_time = {ts: i for i, ts in enumerate(common)}
    items: list[dict] = []
    for trade_id, row in enumerate(trades.itertuples(index=False), 1):
        signal_ts, entry_ts, exit_ts = map(pd.Timestamp, (row.signal_time, row.entry_time, row.exit_time))
        signal_i, entry_i, exit_i = by_time.get(signal_ts), by_time.get(entry_ts), by_time.get(exit_ts)
        if signal_i is None or entry_i is None or exit_i is None:
            raise AssertionError(f"Trade {trade_id} timestamp absent from synchronized raw bars")
        if entry_i != signal_i + 1 or signal_ts.date() != entry_ts.date():
            raise AssertionError(f"Trade {trade_id}: fill is not next synchronized open")
        assert_close(arrays["z"][signal_i], row.entry_z, f"trade {trade_id} Z", 1e-10)
        assert_close(arrays["close"][signal_i], row.signal_nvda_close, f"trade {trade_id} signal close")
        assert_close(arrays["vwap_target"][signal_i], row.signal_nvda_vwap, f"trade {trade_id} NVDA VWAP")
        assert_close(arrays["vwap_lead"][signal_i], row.signal_qqq_vwap, f"trade {trade_id} QQQ VWAP")
        assert_close(arrays["fair_price"][signal_i], row.signal_fair_nvda, f"trade {trade_id} fair")
        assert_close(nvda.iloc[entry_i].open, row.entry_reference, f"trade {trade_id} entry open")
        direction = 1 if str(row.direction).upper() == "LONG" else -1
        assert_close(row.entry_price, float(row.entry_reference) *
                     (1 + SLIP if direction == 1 else 1 - SLIP), f"trade {trade_id} entry fill")
        assert_close(row.exit_price, float(row.exit_reference) *
                     (1 - SLIP if direction == 1 else 1 + SLIP), f"trade {trade_id} exit fill")
        assert_close(row.gross_pnl, direction * (float(row.exit_reference) -
                     float(row.entry_reference)) * int(row.shares), f"trade {trade_id} gross")
        assert_close(row.commissions, 2 * int(row.shares) * COMMISSION, f"trade {trade_id} commission")
        assert_close(row.costs, float(row.commissions) + float(row.slippage), f"trade {trade_id} costs")
        assert_close(row.net_pnl, float(row.gross_pnl) - float(row.costs), f"trade {trade_id} net")
        items.append({
            "id": trade_id, "split": split_for_day(entry_ts.date(), summary),
            "side": "LONG" if direction == 1 else "SHORT", "direction": direction,
            "entry_signal_time": epoch(signal_ts), "entry_time": epoch(entry_ts),
            "exit_time": epoch(exit_ts), "entry_z": number(row.entry_z),
            "signal_nvda_close": number(row.signal_nvda_close),
            "signal_nvda_vwap": number(row.signal_nvda_vwap),
            "signal_qqq_vwap": number(row.signal_qqq_vwap),
            "signal_fair_nvda": number(row.signal_fair_nvda),
            "entry_reference": number(row.entry_reference), "entry_price": number(row.entry_price),
            "exit_reference": number(row.exit_reference), "exit_price": number(row.exit_price),
            "shares": int(row.shares), "stop_usd_per_share": number(row.stop_usd_per_share),
            "target_usd_per_share": number(row.target_usd_per_share),
            "gross_risk_usd": number(row.gross_risk_usd),
            "gross_reward_usd": number(row.gross_reward_usd),
            "risk_reward_ratio": number(row.risk_reward_ratio),
            "stop_price": number(row.stop_price), "target_price": number(row.target_price),
            "gross_pnl": number(row.gross_pnl), "slippage": number(row.slippage),
            "commissions": number(row.commissions), "costs": number(row.costs),
            "net_pnl": number(row.net_pnl), "exit_reason": str(row.exit_reason),
            "duration_minutes": int(row.duration_bars),
        })
    return items


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    start, end = (pd.Timestamp(summary["period"][k]).date() for k in ("start", "end"))
    selected, strategy = summary["selected"], summary["entry_parameters"]
    if summary["study"] != "Independent raw event-driven VWAP-Z with absolute NVDA stop/target":
        raise AssertionError("Builder requires independent event-driven research")
    if summary["execution"]["convergence_exit"] is not False:
        raise AssertionError("Convergence exit must be disabled")

    cfg = AppConfig.load(str(ROOT / "configs" / "default_config.yaml"))
    qqq_all, nvda_all = DataLoader(cfg.cache_dir, "alpaca", "sip").get_synchronized_pair("QQQ", "NVDA")
    common_all = qqq_all.index.intersection(nvda_all.index)
    mask = np.fromiter((start <= ts.date() <= end for ts in common_all), bool, len(common_all))
    common = common_all[mask]
    qqq, nvda = qqq_all.loc[common], nvda_all.loc[common]
    if not qqq.index.equals(nvda.index) or not qqq.index.equals(common):
        raise AssertionError("QQQ/NVDA bars are not exactly synchronized")
    if not common.is_monotonic_increasing or common.has_duplicates:
        raise AssertionError("Raw timestamps are not ordered and unique")

    # Reuse the research market constructor exactly.  It computes rolling beta
    # on the full raw history first and only then slices the reporting year, so
    # the first report sessions retain their strictly prior-day beta history.
    arrays, days = market()
    if not pd.DatetimeIndex(arrays["timestamp"]).equals(common):
        raise AssertionError("Event-driven market arrays do not align with emitted raw bars")
    if len(common) != int(summary["period"]["raw_bars"]) or len(days) != int(summary["period"]["sessions"]):
        raise AssertionError("Raw bar/session count changed")

    dev = int(summary["splits"]["development"]["sessions"])
    val = dev + int(summary["splits"]["validation"]["sessions"])
    ranges = {"development": (0, dev), "validation": (dev, val),
              "holdout": (val, len(days)), "full": (0, len(days))}
    replay: dict[str, dict] = {}
    frames: dict[str, pd.DataFrame] = {}
    equities: dict[str, pd.DataFrame] = {}
    for split, (lo, hi) in ranges.items():
        result = simulate(arrays, lo, hi, float(selected["stop_usd"]),
                          float(selected["target_usd"]), collect=True)
        saved = parse_times(pd.read_csv(SRC / f"selected_{split}_trades.csv"),
                            "signal_time", "entry_time", "exit_time")
        saved_eq = parse_times(pd.read_csv(SRC / f"selected_{split}_equity.csv"), "timestamp")
        assert_frames(saved.reset_index(drop=True), result["trades_df"].reset_index(drop=True), f"{split} trades")
        assert_frames(saved_eq.reset_index(drop=True), result["mtm_df"].reset_index(drop=True), f"{split} MTM")
        clean_result = clean(result)
        assert_metrics(clean_result, summary["selected_results"][split], split)
        replay[split], frames[split], equities[split] = clean_result, saved, saved_eq

    full_trades, full_equity = frames["full"], equities["full"]
    if len(full_equity) != len(common) or not pd.DatetimeIndex(full_equity.timestamp).equals(common):
        raise AssertionError("Full MTM is not one-to-one with raw bars")
    assert_close(full_equity.equity.iloc[-1], CAPITAL + full_trades.net_pnl.sum(), "final equity")
    if sum(len(frames[k]) for k in ("development", "validation", "holdout")) != len(full_trades):
        raise AssertionError("Split/full trade count mismatch")
    assert_close(sum(frames[k].net_pnl.sum() for k in ("development", "validation", "holdout")),
                 full_trades.net_pnl.sum(), "split/full net")

    items = trade_items(full_trades, common, arrays, nvda, summary)
    for split in ("development", "validation", "holdout"):
        if sum(x["split"] == split for x in items) != len(frames[split]):
            raise AssertionError(f"Payload split assignment failed: {split}")
    results = {k: ui_result(replay[k], frames[k]) for k in ranges}
    cost_breakdown = {k: {"commissions": float(frames[k].commissions.sum()),
                          "slippage": float(frames[k].slippage.sum()),
                          "total": float(frames[k].costs.sum())} for k in ranges}
    for split, costs in cost_breakdown.items():
        assert_close(costs["commissions"] + costs["slippage"], costs["total"], f"{split} costs")
        assert_close(costs["total"], results[split]["costs"], f"{split} result costs")

    meta_splits = {}
    for split in ("development", "validation", "holdout"):
        first = pd.Timestamp(summary["splits"][split]["start"]).date()
        last = pd.Timestamp(summary["splits"][split]["end"]).date()
        times = common[(common.date >= first) & (common.date <= last)]
        if len(times) != int(results[split]["raw_bars"]):
            raise AssertionError(f"Split bar range mismatch: {split}")
        meta_splits[split] = {**summary["splits"][split],
                              "start_epoch": epoch(times[0]), "end_epoch": epoch(times[-1])}

    bars = {
        "t": [epoch(x) for x in common],
        "qo": [number(x) for x in qqq.open], "qh": [number(x) for x in qqq.high],
        "ql": [number(x) for x in qqq.low], "qc": [number(x) for x in qqq.close],
        "qv": [number(x) for x in qqq.volume], "no": [number(x) for x in nvda.open],
        "nh": [number(x) for x in nvda.high], "nl": [number(x) for x in nvda.low],
        "nc": [number(x) for x in nvda.close], "nv": [number(x) for x in nvda.volume],
        "qvwap": [number(x) for x in arrays["vwap_lead"]],
        "nvwap": [number(x) for x in arrays["vwap_target"]],
        "fair": [number(x) for x in arrays["fair_price"]], "z": [number(x) for x in arrays["z"]],
        "equity": [number(x) for x in full_equity.equity],
        "drawdown": [number(x) for x in full_equity.drawdown_usd],
    }
    if set(map(len, bars.values())) != {len(common)}:
        raise AssertionError("Payload arrays are not aligned")

    strategy_public = {k: strategy[k] for k in
                       ("beta_days", "warmup_bars", "window", "z_entry", "hook_delta", "z_lockout")}
    execution = {**summary["execution"],
                 "slippage_bps_per_execution": float(summary["execution"]["slippage_fraction_per_execution"]) * 10_000,
                 "stop_usd_per_share": float(selected["stop_usd"]),
                 "target_usd_per_share": float(selected["target_usd"]),
                 "reward_risk_ratio": float(selected["target_usd"]) / float(selected["stop_usd"])}
    reconciliation = {
        "raw_bars_and_sessions_equal_summary": True,
        "event_driven_replay_equal_all_trade_csvs": True,
        "event_driven_replay_equal_all_split_metrics": True,
        "signal_close_is_immediately_before_entry_open": True,
        "signal_z_vwap_and_fair_equal_raw_causal_arrays": True,
        "trade_fills_costs_and_pnl_recomputed": True,
        "split_rows_and_net_equal_full": True,
        "mtm_equal_all_saved_equity_series": True,
        "full_mtm_final_equal_capital_plus_net": True,
        "payload_arrays_aligned": True,
    }
    payload = {
        "meta": {
            "source": "Exact synchronized raw Alpaca SIP 1-minute QQQ/NVDA RTH bars; no resampling, interpolation, synthetic quotes, or frozen convergence-cohort reuse",
            "period": summary["period"], "timezone": NY, "symbols": summary["symbols"],
            "strategy": strategy_public, "selected": selected, "selection": summary["selection"],
            "execution": execution, "splits": meta_splits, "cost_breakdown": cost_breakdown,
            "candidate_accounting": {k: {
                "threshold_bars_while_flat": int(results[k]["generated_flat_signals"]),
                "threshold_bars_ignored_while_open": int(results[k]["ignored_signals_while_open"]),
                "executed_trades": int(results[k]["trades"])} for k in ranges},
            "caveat": "The superseded frozen convergence-cohort bracket result is not a trade source: this payload regenerates fresh causal entry events from every raw minute whenever flat.",
            "frozen_comparison": {
                "old_selected": summary["frozen_cohort_comparison"]["old_selected"],
                "old_full_net_pnl": summary["frozen_cohort_comparison"]["old_full"]["net_pnl"],
                "old_holdout_net_pnl": summary["frozen_cohort_comparison"]["old_holdout"]["net_pnl"],
            },
            "sources": {
                "summary": {"path": str(SUMMARY_PATH.relative_to(ROOT)), "sha256": sha256(SUMMARY_PATH)},
                "selected_full_trades": {"path": str(TRADES_PATH.relative_to(ROOT)), "sha256": sha256(TRADES_PATH)},
                "selected_full_equity": {"path": str(MTM_PATH.relative_to(ROOT)), "sha256": sha256(MTM_PATH)}},
            "reconciliation": reconciliation},
        "bars": bars, "trades": items, "results": results}

    compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data.js").write_text("window.VWAP_ABSOLUTE_DATA=" + compact + ";\n", encoding="utf-8")
    (OUT / "report_data.json").write_text(compact + "\n", encoding="utf-8")
    written = json.loads((OUT / "report_data.json").read_text(encoding="utf-8"))
    if len(written["bars"]["t"]) != len(common) or len(written["trades"]) != len(full_trades):
        raise AssertionError("Written payload readback failed")
    print(json.dumps({"bars": len(common), "sessions": len(days), "trades": len(full_trades),
                      "selected": selected, "full_net_pnl": results["full"]["net_pnl"],
                      "holdout_net_pnl": results["holdout"]["net_pnl"],
                      "mtm_max_drawdown_usd": results["full"]["max_drawdown_usd_mtm"],
                      "files": {x: (OUT / x).stat().st_size for x in ("data.js", "report_data.json")},
                      "reconciliation": reconciliation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
