"""Regression gate for the independently audited NVDA tail-risk outputs."""
from __future__ import annotations

import pytest

from rtharb.audit.vwap_nvda_tail_risk import OUT, audit


def test_vwap_nvda_tail_risk_outputs_reconcile_when_ready():
    if not (OUT / "manifest.json").is_file():
        pytest.skip("NVDA tail-risk outputs are not built")
    result = audit(raw_replay=True, report=True)
    assert result["status"] == "PASS"
    assert result["verdict"] == "NO_OP_BASELINE"
    assert result["grid"]["pairs"] == 378
    assert result["grid"]["eligible_overlays"] == 0
    assert result["raw_replay"]["trades"] == 456
    assert result["report"]["status"] == "PASS"
    assert result["report"]["sessions"] == 251
    assert result["report"]["bars"] == 97_530
    assert result["report"]["trades"] == 456
    assert result["report"]["stop3_full_net"] == pytest.approx(6597.511202120431)
