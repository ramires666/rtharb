"""Regression gates for completed all-nine walk-forward selection artifacts."""
from __future__ import annotations

import os

import pytest

from rtharb.audit.vwap_multi_asset_walkforward import OUT, UNIVERSE, audit_all, audit_symbol


def test_vwap_multi_asset_walkforward_outputs_when_ready():
    if not (OUT / "progress.json").is_file():
        pytest.skip("walk-forward outputs have not started")
    result = audit_all(raw_replay=False)
    assert result["status"] == "PASS"
    if result["production_status"] == "COMPLETE":
        assert result["completed"] == list(UNIVERSE)
        assert result["remaining"] == []


def test_vwap_multi_asset_deep_grid_from_raw_when_requested():
    symbol = os.environ.get("RTHARB_DEEP_AUDIT_SYMBOL")
    if not symbol:
        pytest.skip("set RTHARB_DEEP_AUDIT_SYMBOL for the expensive independent full-grid audit")
    result = audit_symbol(symbol.upper(), raw_replay=True, deep_grid=True)
    assert result["status"] == "PASS"
    assert result["deep_grid"] is True
    assert result["raw_replay"] is True
