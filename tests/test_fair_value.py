"""Unit tests for Fair Value, Beta, Spread, and Z-Score calculations."""

import numpy as np
import pandas as pd
import pytest
from rtharb.models.fair_value import FairValueModel


def create_dummy_pair():
    # 12 sessions provide enough prior-day observations for causal sigma.
    sessions = [
        pd.date_range(f"{day.date()} 09:30:00-04:00", f"{day.date()} 15:59:00-04:00", freq="1min")
        for day in pd.bdate_range("2026-08-03", periods=12)
    ]
    full_index = sessions[0]
    for session in sessions[1:]:
        full_index = full_index.append(session)

    df_lead = pd.DataFrame(index=full_index)
    df_target = pd.DataFrame(index=full_index)

    # Lead price random walk
    np.random.seed(42)
    lead_ret = np.random.normal(0, 0.0005, len(full_index))
    target_ret = 1.5 * lead_ret + np.random.normal(0, 0.0002, len(full_index))

    df_lead["open"] = 700.0 * np.exp(np.cumsum(lead_ret))
    df_lead["close"] = df_lead["open"] * 1.0001
    df_lead["high"] = df_lead[["open", "close"]].max(axis=1) + 0.1
    df_lead["low"] = df_lead[["open", "close"]].min(axis=1) - 0.1
    df_lead["volume"] = 10000

    df_target["open"] = 200.0 * np.exp(np.cumsum(target_ret))
    df_target["close"] = df_target["open"] * 1.0001
    df_target["high"] = df_target[["open", "close"]].max(axis=1) + 0.1
    df_target["low"] = df_target[["open", "close"]].min(axis=1) - 0.1
    df_target["volume"] = 20000

    return df_lead, df_target


def test_fair_value_model():
    df_lead, df_target = create_dummy_pair()
    fv = FairValueModel(beta_mode="fixed_1.0", rolling_window_w=10,
                        min_session_warmup_bars=15, min_sigma_history_days=5)
    metrics = fv.compute_intraday_metrics(df_lead, df_target)

    assert "z_score" in metrics.columns
    assert "spread" in metrics.columns
    assert "target_fair_price" in metrics.columns
    assert len(metrics) == len(df_lead)
    assert metrics.groupby("session_date")["spread"].first().abs().max() < 1e-12
    assert metrics.loc[metrics["bar_of_day"] < 15, "z_score"].isna().all()
    assert metrics.loc[metrics["session_date"] >= sorted(metrics.session_date.unique())[6], "z_score"].notna().any()


def test_session_vwap_is_typical_price_volume_weighted_and_resets():
    idx = pd.date_range("2026-08-03 09:30", periods=3, freq="1min", tz="America/New_York")
    idx = idx.append(pd.date_range("2026-08-04 09:30", periods=1, freq="1min", tz="America/New_York"))
    lead = pd.DataFrame(index=idx)
    target = pd.DataFrame(index=idx)
    for frame, base in ((lead, 100.0), (target, 200.0)):
        frame["open"] = base
        frame["close"] = [base, base + 3, base + 6, base + 30]
        frame["high"] = frame["close"] + 1
        frame["low"] = frame["close"] - 1
        frame["volume"] = [1, 2, 3, 4]

    model = FairValueModel(beta_mode="fixed_1.0", min_session_warmup_bars=0, vwap_mode="session")
    metrics = model.compute_intraday_metrics(lead, target)
    # Target typical prices on day one are 200, 203, 206 with volumes 1,2,3.
    expected = (200 * 1 + 203 * 2 + 206 * 3) / 6
    assert metrics.iloc[0]["target_vwap"] == 200.0
    assert metrics.iloc[2]["target_vwap"] == pytest.approx(expected)
    # The next session starts a fresh cumulative VWAP.
    assert metrics.iloc[3]["target_vwap"] == 230.0


def test_session_vwap_has_no_future_bar_lookahead():
    idx = pd.date_range("2026-08-03 09:30", periods=3, freq="1min", tz="America/New_York")
    def make(last_close):
        frame = pd.DataFrame(index=idx)
        frame["open"] = [100.0, 101.0, 102.0]
        frame["close"] = [100.0, 101.0, last_close]
        frame["high"] = frame["close"] + 1.0
        frame["low"] = frame["close"] - 1.0
        frame["volume"] = [1, 1, 1]
        return frame

    first = FairValueModel(beta_mode="fixed_1.0", min_session_warmup_bars=0, vwap_mode="session")
    second = FairValueModel(beta_mode="fixed_1.0", min_session_warmup_bars=0, vwap_mode="session")
    m1 = first.compute_intraday_metrics(make(102.0), make(202.0))
    m2 = second.compute_intraday_metrics(make(999.0), make(888.0))
    assert m1.iloc[:2]["lead_vwap"].tolist() == m2.iloc[:2]["lead_vwap"].tolist()
    assert m1.iloc[:2]["target_vwap"].tolist() == m2.iloc[:2]["target_vwap"].tolist()
