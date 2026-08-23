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


def test_verified_q95_overlays_preserve_winners_and_splits():
    summary = json.loads(
        (ROOT / "research_output" / "duration_stoploss_verified" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["data"] == {
        "sessions": 501, "development": 250, "validation": 125, "holdout": 126,
    }
    assert set(summary["winner_duration_quantiles_bars"]) == {"0.9", "0.95", "0.975", "0.99"}
    assert set(summary["winner_mae_pct_quantiles"]) == {"0.9", "0.95", "0.975", "0.99"}
    survival = summary["development_q95_survival"]
    assert survival["time_stop"]["still_net_profitable_pct"] >= 95.0
    assert survival["stop_loss"]["still_net_profitable_pct"] >= 95.0
    for family in ("base",):
        for split in ("development", "validation", "holdout", "full"):
            assert abs(summary[family][split]["reconciliation_error"]) < 1e-6
    for family in ("time_stop_only", "stop_loss_only"):
        for quantile in ("0.9", "0.95", "0.975", "0.99"):
            for split in ("development", "validation", "holdout", "full"):
                assert abs(summary[family][quantile][split]["reconciliation_error"]) < 1e-6


def test_vwap_rr_ratio_variants_reconcile_to_frozen_cohort():
    summary = json.loads(
        (ROOT / "research_output" / "risk_reward" / "vwap_rr_ratios_summary.json").read_text(encoding="utf-8")
    )
    assert summary["rr_ratios"] == [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    assert summary["fixed_stop_pct"] == 0.01
    for variant in summary["variants"].values():
        reconciliation = variant["reconciliation"]
        assert reconciliation["candidate_entries"] == 1656
        assert reconciliation["candidate_entries_accounted"]
        assert reconciliation["split_trade_rows_equal_full"]
        assert reconciliation["split_net_pnl_equal_full"]
        assert reconciliation["csv_net_pnl_equal_metrics"]
