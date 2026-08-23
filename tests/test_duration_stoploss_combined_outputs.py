"""Regression gate for combined duration/stop-loss research artifacts."""
from __future__ import annotations

import pytest

from rtharb.audit.duration_stoploss_combined import OUT, audit


def test_duration_stoploss_combined_outputs_reconcile_when_ready():
    if not (OUT / "manifest.json").is_file():
        pytest.skip("combined duration/stop-loss outputs are not built")
    result = audit(raw_replay=False)
    assert result["selection"]["selected_max_holding_bars"] == 61
    assert result["selection"]["holdout_opened_once_after_selection"] is True
    assert result["raw"]["eligible"] is False
