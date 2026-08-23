"""Regression gates for the independently audited portfolio outputs."""
from __future__ import annotations

from pathlib import Path

import pytest

from rtharb.audit.vwap_absolute_portfolio import OUT, VARIANTS, audit


def _ready() -> None:
    if not (OUT / "manifest.json").is_file():
        pytest.skip("portfolio research outputs not built")


def test_portfolio_outputs_lightweight_reconciliation_and_report_schema():
    _ready()
    results = audit(raw_replay=False)
    assert tuple(results) == VARIANTS
    assert all(results[name]["full"]["trades"] > 0 for name in VARIANTS)
    assert results["uncapped_diagnostic"]["full"]["trades"] == results["equal_allocation"]["full"]["trades"]

