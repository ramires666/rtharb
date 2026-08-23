"""Unit test for 4-scenario matrix comparator."""

import pandas as pd
import pytest
from rtharb.config import AppConfig
from rtharb.models.fair_value import FairValueModel
from rtharb.analysis.matrix_comparator import MatrixComparator
from tests.test_fair_value import create_dummy_pair


def test_matrix_comparator():
    df_lead, df_target = create_dummy_pair()
    fv = FairValueModel(beta_mode="fixed_1.0", rolling_window_w=30)
    df_metrics = fv.compute_intraday_metrics(df_lead, df_target)

    cfg = AppConfig()
    matrix_comp = MatrixComparator(cfg)
    matrix_out = matrix_comp.run_all_scenarios(df_metrics)

    assert "comparison_df" in matrix_out
    assert "equity_curves" in matrix_out
    assert len(matrix_out["comparison_df"]) == 4
    assert len(matrix_out["equity_curves"].columns) == 4
