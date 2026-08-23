"""Regression gate for the independently audited nine-stock study."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from rtharb.audit.vwap_absolute_multi_asset import MANIFEST, SOURCE, _select_completed, audit


ROOT = Path(__file__).resolve().parents[1]


def _require_outputs() -> None:
    if not MANIFEST.is_file() or not (SOURCE / "manifest.json").is_file():
        pytest.skip(
            "multi-asset outputs are not built; run research and reporting launchers first"
        )


def test_multi_asset_outputs_pass_independent_raw_event_audit():
    _require_outputs()
    results = audit()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    stage = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    assert list(results) == stage["symbols_completed"]
    assert set(results).issubset(manifest["frozen_universe"])
    if stage["status"] == "COMPLETE":
        assert list(results) == manifest["frozen_universe"]
        assert len(results) == 9
    else:
        assert len(results) >= 1
    assert all(item["bars"] > 97_000 for item in results.values())
    assert all(item["trades"] >= 0 for item in results.values())


def test_partial_audit_symbol_selection_is_safe_and_keeps_frozen_order():
    completed = ["NVDA", "MSFT", "AAPL"]
    assert _select_completed(completed, None) == completed
    assert _select_completed(completed, ["aapl", "NVDA"]) == ["NVDA", "AAPL"]
    with pytest.raises(ValueError, match="not completed"):
        _select_completed(completed, ["AMZN"])
    with pytest.raises(ValueError, match="Duplicate"):
        _select_completed(completed, ["MSFT", "msft"])
