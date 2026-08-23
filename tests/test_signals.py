"""Unit tests for Signal Generation, Reversal Confirmation Hook, and 4-Sigma Protections."""

import pandas as pd
import numpy as np
import pytest
from rtharb.models.signals import SignalGenerator, SignalType


def test_reversal_hook_and_lockout():
    # 1 session day with 100 bars
    dates = pd.date_range("2026-08-17 09:30:00-04:00", periods=100, freq="1min")
    df = pd.DataFrame(index=dates)
    df["session_date"] = dates.date
    df["time_str"] = dates.strftime("%H:%M")
    df["bar_of_day"] = range(100)
    df["target_open"] = 200.0
    df["target_close"] = 200.0
    df["target_fair_price"] = 200.0

    # Synthetic Z-score:
    # Bars 0-19: 0.0
    # Bars 20-25: drops to -2.0 (Arming for Long)
    # Bar 26: rises from -2.0 to -1.8 (rebound of +0.2 > delta 0.15 -> Trigger BUY)
    # Bars 27-35: returns to 0.0 -> Take Profit
    # Bar 40: blows up to +4.5 (4-sigma breach -> Lockout)
    z_vals = [0.0] * 100
    z_vals[20] = -1.6
    z_vals[21] = -1.8
    z_vals[22] = -2.0 # minimum
    z_vals[23] = -1.8 # hook up by +0.2 -> Signal BUY
    z_vals[24] = -1.0
    z_vals[25] = 0.0  # Take Profit

    z_vals[40] = 4.5  # 4-sigma breach

    df["z_score"] = z_vals

    sig_gen = SignalGenerator(
        z_entry=1.5,
        reversal_type="z_score_hook",
        reversal_delta=0.15,
        reversal_timeout_bars=10,
        enable_extreme_entry_lockout=True,
        enable_extreme_emergency_exit=False,
        z_max_allowed=4.0,
        lockout_mode="day_lockout",
        z_exit=0.0,
        forced_close_time="15:55",
        min_session_warmup_bars=15
    )

    df_out = sig_gen.generate_signals(df)

    # Verify BUY occurred on bar 23 (reversal hook)
    assert df_out.iloc[23]["signal"] == SignalType.BUY_LONG
    # Verify Take profit occurred on bar 25
    assert df_out.iloc[25]["signal"] == SignalType.EXIT_TAKE_PROFIT
    # Verify day lockout triggered on bar 40
    assert df_out.iloc[40]["is_locked_out"] == True
    assert df_out.iloc[50]["is_locked_out"] == True
