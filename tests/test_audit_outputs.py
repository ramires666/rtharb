import json
import re
from pathlib import Path
import pandas as pd
import numpy as np

from research_synthetic_index import entry_events


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


def test_synthetic_index_uses_causal_basket_and_exact_sip_data():
    summary = json.loads(
        (ROOT / "research_output" / "synthetic_index" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["basket"]["official_snapshot_date"] == "2024-06-28"
    assert summary["basket"]["symbols"] == ["MSFT", "AAPL", "NVDA", "AMZN"]
    assert abs(summary["basket"]["combined_ndx_weight_pct"] - 30.1) < 1e-9
    assert summary["data"]["sessions"] == 501
    assert summary["data"]["bars"] == 194_490
    assert summary["splits"]["development"]["sessions"] == 250
    assert summary["splits"]["validation"]["sessions"] == 125
    assert summary["splits"]["holdout"]["sessions"] == 126
    assert summary["execution"]["traded_symbol"] == "QQQ"
    assert not summary["execution"]["rr_convergence_exit"]
    assert summary["reconciliation"]["all_trade_rows_exact"]
    assert summary["reconciliation"]["holdout_not_used_for_selection"]
    assert not summary["reconciliation"]["mock_or_interpolated_bars"]
    for model in ("convergence", "risk_reward"):
        for split in ("development", "validation", "holdout", "full"):
            assert abs(summary[model]["results"][split]["reconciliation_error"]) < 1e-8


def test_synthetic_hook_can_confirm_after_reentering_threshold():
    metric = np.array([np.nan, 2.0, 1.9, 1.7], dtype=float)
    events = entry_events(metric, np.zeros(len(metric), dtype=np.int8), threshold=2.0, hook=0.2)
    assert events.tolist() == [0, 0, 0, -1]


def test_absolute_vwap_brackets_are_honest_and_reconciled():
    summary = json.loads(
        (ROOT / "old" / "frozen_vwap_absolute" / "output" / "results" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["grid"]["combinations"] == 144
    assert summary["selected"] == {"stop_usd": 2.0, "target_usd": 1.25}
    assert summary["selection"]["holdout_opened_after_selection"]
    assert not summary["selection"]["no_confirmed_edge"]
    assert summary["mark_to_market"]["bars"] > 90_000
    assert all(summary["reconciliation"].values())


def test_event_driven_absolute_vwap_report_reconciles_and_supersedes_frozen_cohort():
    summary = json.loads(
        (ROOT / "research_output" / "vwap_absolute_event_driven" / "summary.json").read_text(encoding="utf-8")
    )
    payload = json.loads(
        (ROOT / "tradingview_vwap_absolute" / "report_data.json").read_text(encoding="utf-8")
    )
    assert summary["selected"] == {"stop_usd": 3.0, "target_usd": 1.25}
    assert summary["selection"]["holdout_opened_after_selection"]
    assert summary["selected_results"]["full"]["trades"] == 492
    assert summary["selected_results"]["full"]["generated_flat_signals"] == 492
    assert summary["selected_results"]["full"]["stops"] + summary["selected_results"]["full"]["targets"] + summary["selected_results"]["full"]["forced_eod"] == 492
    assert all(summary["reconciliation"].values())
    assert len(payload["bars"]["t"]) == 97_530
    assert len(payload["trades"]) == 492
    assert payload["meta"]["selected"] == summary["selected"]
    assert abs(payload["results"]["full"]["net_pnl"] - summary["selected_results"]["full"]["net_pnl"]) < 1e-8
    assert all(payload["meta"]["reconciliation"].values())
