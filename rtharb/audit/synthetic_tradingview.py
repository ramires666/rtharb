"""Integrity audit for the standalone synthetic-index interactive report."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tradingview_synthetic"


def main() -> None:
    for name in ("index.html", "data.js", "report_data.json"):
        path = OUT / name
        assert path.exists() and path.stat().st_size > 100, f"missing/empty {path}"
    payload = json.loads((OUT / "report_data.json").read_text(encoding="utf-8"))
    bars, meta = payload["bars"], payload["meta"]
    n = len(bars["t"])
    assert n == meta["data"]["bars"] == 194_490
    assert meta["data"]["sessions"] == 501
    assert meta["execution"]["traded_symbol"] == "QQQ"
    assert meta["execution"]["rr_convergence_exit"] is False
    assert bars["t"] == sorted(set(bars["t"]))
    for name in ("qo", "qh", "ql", "qc", "basket", "points", "spread", "conv_metric", "rr_metric"):
        assert len(bars[name]) == n, name
    for o, h, low, c, basket in zip(bars["qo"], bars["qh"], bars["ql"], bars["qc"], bars["basket"]):
        assert h >= max(o, c) and low <= min(o, c)
        assert math.isfinite(basket)
    dates = pd.to_datetime(bars["t"], unit="s", utc=True).tz_convert("America/New_York").date
    starts = [0] + [i for i in range(1, n) if dates[i] != dates[i - 1]]
    assert len(starts) == 501
    assert max(abs(bars["basket"][i] - bars["qc"][i]) for i in starts) < 1e-3
    assert set(payload["variants"]) == {"convergence", "risk_reward"}
    timestamp_set = set(bars["t"])
    for key, variant in payload["variants"].items():
        trades = variant["trades"]
        assert len(trades) == variant["trades_count"]
        assert abs(sum(t["net_pnl"] for t in trades) - variant["net_pnl"]) < 0.02
        assert abs(variant["net_pnl"] - variant["results"]["full"]["net_pnl"]) < 0.02
        previous_exit = 0
        for trade in trades:
            assert trade["entry_signal_time"] in timestamp_set
            assert trade["entry_time"] in timestamp_set and trade["exit_time"] in timestamp_set
            assert trade["entry_signal_time"] < trade["entry_time"] <= trade["exit_time"]
            assert trade["entry_time"] >= previous_exit
            previous_exit = trade["exit_time"]
            reconciled = trade["gross_pnl"] - trade["commission"] - trade["slippage"]
            assert abs(reconciled - trade["net_pnl"]) < 0.02
            if key == "risk_reward":
                assert trade["exit_reason"] in {"STOP", "TAKE_PROFIT", "FORCED_EOD"}
                assert trade["rr"] > 0 and trade["stop_pct"] > 0
    html = (OUT / "index.html").read_text(encoding="utf-8")
    for token in ("CandlestickSeries", "LineSeries", "createSeriesMarkers", "trade-box",
                  "Synthetic basket", "RR без схождения", "updateEquity", "stMdd",
                  "entry_signal_time", "signal_metric", "metricsBody", "showBasket", "data.js",
                  "T=V.trades", "номинально gross риск"):
        assert token in html, token
    assert "addCandlestickSeries" not in html and "addLineSeries" not in html
    print(f"PASS synthetic TradingView: {n:,} raw SIP bars, "
          f"variants { {k: v['trades_count'] for k, v in payload['variants'].items()} }, exact P&L/timestamps")


if __name__ == "__main__":
    main()
