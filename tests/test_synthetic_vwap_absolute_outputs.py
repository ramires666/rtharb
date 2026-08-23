"""Regression gates for the independent synthetic VWAP absolute audit."""
from __future__ import annotations

import pytest

from rtharb.audit.synthetic_vwap_absolute import OUT, VARIANTS, audit


def test_synthetic_vwap_absolute_outputs_reconcile_when_ready():
    if not (OUT / "manifest.json").is_file():
        pytest.skip("synthetic VWAP absolute outputs are not built")
    results = audit(raw_replay=False)
    assert tuple(results) == VARIANTS
    assert all("full" in results[name] and "holdout" in results[name] for name in VARIANTS)
