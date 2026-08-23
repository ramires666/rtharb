import json
import re
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "audit_output"


def test_sip_session_counts_match_official_calendar():
    sessions = pd.read_csv(OUT / "session_bar_audit.csv")
    assert len(sessions) == 501
    assert sessions["ok"].all()
    assert sessions.actual_bars.sum() == 194_490
    assert set(sessions.expected_bars) == {210, 390}


def test_latest_svg_really_contains_390_candles():
    svg = (OUT / "session_2026-08-21.svg").read_text(encoding="utf-8")
    declared = int(re.search(r'data-count="(\d+)"', svg).group(1))
    assert declared == 390
    assert svg.count('class="wick"') == 390
    assert svg.count('class="body"') == 390


def test_trade_accounting_reconciles():
    trades = pd.read_csv(OUT / "trades_full.csv")
    error = trades.net_pnl - (trades.gross_pnl - trades.commission - trades.slippage)
    assert error.abs().max() < 1e-7
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["data"]["feed"] == "Alpaca SIP"
