"""Integrity checks for the one-year absolute-dollar VWAP bracket study."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "research_output" / "vwap_absolute_brackets"


def main() -> None:
    required = (
        "summary.json", "full_grid.csv", "finalists.csv", "selected_full_trades.csv",
        "selected_full_equity.csv", "selected_full_closed_trade_equity.csv", "REPORT.html",
    )
    for name in required:
        path = OUT / name
        assert path.exists() and path.stat().st_size > 100, f"missing/empty {path}"
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["period"] == {"start": "2025-08-22", "end": "2026-08-21", "sessions": 251}
    assert summary["grid"]["combinations"] == 144
    assert summary["selected"] == {"stop_usd": 2.0, "target_usd": 1.25}
    assert summary["selection"]["holdout_opened_after_selection"]
    grid = pd.read_csv(OUT / "full_grid.csv")
    assert len(grid) == 144 and int(grid.selected.sum()) == 1
    assert grid.holdout_net_pnl.notna().sum() == 1 and grid.full_net_pnl.notna().sum() == 1
    trades = pd.read_csv(OUT / "selected_full_trades.csv")
    full = summary["selected_results"]["full"]
    assert len(trades) == full["trades"] == 490
    assert abs(trades.net_pnl.sum() - full["net_pnl"]) < 1e-7
    assert abs(trades.costs.sum() - full["costs"]) < 1e-7
    assert set(trades.exit_reason) <= {"STOP", "TAKE_PROFIT_BRACKET", "FORCED_EOD"}
    equity = pd.read_csv(OUT / "selected_full_equity.csv")
    mtm = summary["mark_to_market"]
    assert len(equity) == mtm["bars"]
    assert abs(equity.equity.iloc[-1] - (100_000 + full["net_pnl"])) < 1e-7
    assert abs(equity.drawdown_usd.max() - mtm["max_drawdown_usd"]) < 1e-7
    assert abs(equity.drawdown_pct.max() - mtm["max_drawdown_pct"]) < 1e-7
    assert all(summary["reconciliation"].values())
    html = (OUT / "REPORT.html").read_text(encoding="utf-8")
    for token in ("mark-to-market", "equity", "drawdown", "$0.0035", "2 bps", "Holdout"):
        assert token in html, token
    print(f"PASS absolute VWAP brackets: 144 combos, {len(trades)} trades, "
          f"net ${full['net_pnl']:.2f}, MTM MDD ${mtm['max_drawdown_usd']:.2f}")


if __name__ == "__main__":
    main()
