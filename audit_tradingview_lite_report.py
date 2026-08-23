"""Integrity checks for the two-month TradingView Lite report."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "tradingview_lite"


def main() -> None:
    payload = json.loads((OUT / "report_data.json").read_text(encoding="utf-8"))
    bars = payload["bars"]
    n = len(bars["t"])
    assert n > 10_000, f"Expected raw 1-minute history, got only {n} bars"
    for name, values in bars.items():
        assert len(values) == n, f"Array length mismatch: {name}"
    times = np.asarray(bars["t"], dtype=np.int64)
    assert np.all(np.diff(times) > 0), "Timestamps are not strictly increasing"
    ny = pd.to_datetime(times, unit="s", utc=True).tz_convert("America/New_York")
    minute = ny.hour * 60 + ny.minute
    assert np.all((minute >= 570) & (minute < 960)), "Non-RTH bar found"
    assert all(x is not None for key in ("qo", "qh", "ql", "qc", "no", "nh", "nl", "nc") for x in bars[key])
    assert np.all(np.asarray(bars["qh"]) >= np.maximum(bars["qo"], bars["qc"]))
    assert np.all(np.asarray(bars["ql"]) <= np.minimum(bars["qo"], bars["qc"]))
    assert np.all(np.asarray(bars["nh"]) >= np.maximum(bars["no"], bars["nc"]))
    assert np.all(np.asarray(bars["nl"]) <= np.minimum(bars["no"], bars["nc"]))
    assert all(x is not None for key in ("qvwap", "nvwap") for x in bars[key])
    session_first = np.r_[True, ny.date[1:] != ny.date[:-1]]
    q_typical = (np.asarray(bars["qh"]) + np.asarray(bars["ql"]) + np.asarray(bars["qc"])) / 3
    n_typical = (np.asarray(bars["nh"]) + np.asarray(bars["nl"]) + np.asarray(bars["nc"])) / 3
    assert np.allclose(np.asarray(bars["qvwap"])[session_first], q_typical[session_first], atol=1e-4)
    assert np.allclose(np.asarray(bars["nvwap"])[session_first], n_typical[session_first], atol=1e-4)

    timestamp_set = set(bars["t"])
    assert set(payload["variants"]) == {"base", "base_anchor", "hybrid_abs", "hybrid_abs_anchor", "base_inverse", "rr_classic", "vwap_z", "rr_vwap"}
    for variant_name, variant in payload["variants"].items():
        strategy = variant.get("strategy", payload["meta"]["strategy"])
        threshold = strategy["z_entry"]
        trades = variant["trades"]
        assert len(trades) == variant["trades_count"]
        for trade in trades:
            assert trade["entry_time"] in timestamp_set, f"Entry missing: {variant_name} #{trade['id']}"
            assert trade["exit_time"] in timestamp_set, f"Exit missing: {variant_name} #{trade['id']}"
            assert trade["entry_signal_time"] in timestamp_set, f"Entry signal missing: {variant_name} #{trade['id']}"
            assert trade["entry_time"] > trade["entry_signal_time"], f"No next-bar execution: {variant_name} #{trade['id']}"
            assert trade["entry_trigger"] in {"Z", "ABS_USD", "Z+ABS"}
            source_direction = trade["source_signal_direction"]
            z_armed = trade["armed_z"] <= -threshold if source_direction == 1 else trade["armed_z"] >= threshold
            abs_armed = (
                variant["abs_threshold_usd"] is not None
                and abs(trade["armed_abs_deviation_usd"]) >= variant["abs_threshold_usd"] - 1e-4
            )
            assert z_armed or abs_armed, f"No valid arming trigger: {variant_name} #{trade['id']}"
            if variant["entry_mode"] == "z_only":
                assert z_armed
            if variant["inverse"]:
                assert trade["direction"] == -source_direction
            else:
                assert trade["direction"] == source_direction
            if variant["anchor_filter"]:
                if trade["direction"] == 1:
                    assert trade["signal_target_price"] < trade["signal_anchor_price"]
                else:
                    assert trade["signal_target_price"] > trade["signal_anchor_price"]
            assert trade["hook_delta"] >= strategy["hook_delta"] - 1e-4
            assert trade["hook_bars"] <= strategy["hook_timeout"]
            reconciled = trade["gross_pnl"] - trade["commission"] - trade["slippage"]
            assert abs(reconciled - trade["net_pnl"]) < 0.02, f"P&L mismatch: {variant_name} #{trade['id']}"
        assert abs(sum(t["net_pnl"] for t in trades) - variant["net_pnl"]) < 0.02
    trades = payload["trades"]
    assert trades == payload["variants"]["base"]["trades"]
    assert len(trades) == payload["meta"]["trades"]
    assert abs(sum(t["net_pnl"] for t in trades) - payload["meta"]["net_pnl"]) < 0.02

    html = (OUT / "index.html").read_text(encoding="utf-8")
    for token in ("CandlestickSeries", "createSeriesMarkers", "trade-box", "Z-score", "Equity", "updateEquity", "statMdd", "data-ema", "data-hma", "nvdaVwapToggle", "qqqVwapToggle", "toggleQqq", "toggleAbsFilter", "toggleAnchorFilter", "toggleInverse", "toggleRr", "toggleVwapZ", "toggleVwapRr", "researchNote", "QQQ", "NVDA"):
        assert token in html, f"Missing report feature: {token}"
    assert (OUT / "data.js").stat().st_size > 500_000
    assert (OUT / "lightweight-charts.standalone.production.js").stat().st_size > 100_000
    counts = {name: item["trades_count"] for name, item in payload["variants"].items()}
    print(f"PASS: {n:,} raw synchronized 1-minute bars, variants {counts}, timestamps/P&L/hooks/filters reconciled")


if __name__ == "__main__":
    main()
