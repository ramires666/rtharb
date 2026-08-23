"""Integrity checks for the wide base-strategy research."""

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_output"


def audit():
    required = ["BASE_STRATEGY_REPORT.html", "base_strategy_summary.json",
                "stage_a_signal_grid.csv", "stage_b_model_grid.csv",
                "stage_c_exit_lockout_grid.csv", "validation_finalists.csv",
                "selected_holdout_trades.csv", "exact_selected_check.json"]
    for name in required:
        p = OUT / name
        assert p.exists() and p.stat().st_size > 0, f"Missing/empty {p}"

    summary = json.loads((OUT / "base_strategy_summary.json").read_text(encoding="utf-8"))
    exact = json.loads((OUT / "exact_selected_check.json").read_text(encoding="utf-8"))
    assert summary["data"]["sessions"] == 501 and summary["data"]["bars"] == 194490
    assert summary["tested_configurations"] == {"stage_a":256,"stage_b":360,"stage_c":192,"validation":50}
    assert "stop_loss" not in summary["selected_parameters"]
    assert "max_holding" not in summary["selected_parameters"]
    for split in ["development", "validation", "holdout"]:
        assert exact[split]["total_trades"] == summary[split]["trades"]
        assert abs(exact[split]["total_pnl"] - summary[split]["net_pnl"]) < 1e-7

    trades = pd.read_csv(OUT / "selected_holdout_trades.csv")
    assert abs((trades.gross_pnl - trades.costs - trades.net_pnl)).max() < 1e-7
    assert len(trades) == summary["holdout"]["trades"]
    entry = pd.to_datetime(trades.entry_time, utc=True).dt.tz_convert("America/New_York")
    exit_ = pd.to_datetime(trades.exit_time, utc=True).dt.tz_convert("America/New_York")
    assert (entry.dt.date == exit_.dt.date).all()
    finalists = pd.read_csv(OUT / "validation_finalists.csv")
    assert len(finalists) == 50

    print("PASS: 808 development configurations + 50 frozen validation finalists")
    print("PASS: production state machine exactly matches fast research engine")
    print(f"PASS: holdout {len(trades)} trades reconcile gross - costs = net")
    print("PASS: base research contains no stop-loss or time-stop")


if __name__ == "__main__":
    audit()
