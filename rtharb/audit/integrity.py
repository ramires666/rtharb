"""Fail-fast audit for the recalculated SIP report."""

import json
import re
import sys
from pathlib import Path
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "audit_output"


def audit():
    required = ["REPORT.html", "summary.json", "session_bar_audit.csv",
                "training_parameter_grid.csv", "training_filter_grid.csv",
                "trades_full.csv", "equity_holdout.svg", "session_2026-08-21.svg"]
    for name in required:
        path = OUT / name
        assert path.exists() and path.stat().st_size > 0, f"Missing/empty: {path}"

    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["data"]["feed"] == "Alpaca SIP"
    assert summary["data"]["sessions"] == 501
    assert summary["data"]["bars"] == 194490
    assert summary["data"]["all_session_counts_match_calendar"] is True

    sessions = pd.read_csv(OUT / "session_bar_audit.csv")
    assert len(sessions) == 501 and sessions["ok"].all()
    assert (sessions.actual_bars == sessions.expected_bars).all()
    assert set(sessions.expected_bars) == {210, 390}

    trades = pd.read_csv(OUT / "trades_full.csv")
    expected_net = trades.gross_pnl - trades.commission - trades.slippage
    assert (expected_net - trades.net_pnl).abs().max() < 1e-7
    entry = pd.to_datetime(trades.entry_time, utc=True)
    exit_ = pd.to_datetime(trades.exit_time, utc=True)
    # Convert both to New York before comparing session dates.
    assert (entry.dt.tz_convert("America/New_York").dt.date ==
            exit_.dt.tz_convert("America/New_York").dt.date).all()

    svg = (OUT / "session_2026-08-21.svg").read_text(encoding="utf-8")
    declared = int(re.search(r'data-count="(\d+)"', svg).group(1))
    assert declared == 390
    assert len(re.findall(r'class="wick"', svg)) == declared
    assert len(re.findall(r'class="body"', svg)) == declared
    assert "$127" not in svg

    print("PASS: 501 calendar sessions / 194,490 SIP RTH bars")
    print(f"PASS: {len(trades):,} trades reconcile gross - commission - slippage = net")
    print("PASS: latest chart contains exactly 390 real 1-minute candles")
    print("PASS: no overnight positions")


if __name__ == "__main__":
    audit()
