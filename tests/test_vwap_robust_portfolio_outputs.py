"""Regression gate for the guarded robust VWAP portfolio stage."""
from __future__ import annotations

import pytest

from rtharb.audit.vwap_robust_portfolio import OUT, audit


def test_vwap_robust_portfolio_reconciles_when_ready():
    if not (OUT / "manifest.json").is_file():
        pytest.skip("robust portfolio outputs are not built")
    summary = audit()
    assert summary["status"] == "COMPLETE"
    assert summary["confirmed_count"] >= 1
    assert set(summary["seen_veto_survivors"]).issubset(summary["confirmed"])
    if summary["confirmed_count"] == 1:
        assert summary["diversification_tested"] is False
        assert summary["single_asset_warning"]
