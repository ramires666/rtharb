"""Regression tests for optional absolute-deviation entry filters.

These tests intentionally exercise signal generation only.  Execution timing
belongs to ``BacktestEngine`` and remains next-bar-open.
"""

import pandas as pd

from rtharb.models.signals import SignalGenerator, SignalType


def _bars(rows):
    """Build a minimal one-session metrics frame from (z, close, fair)."""
    idx = pd.date_range("2026-08-17 09:30", periods=len(rows), freq="1min", tz="America/New_York")
    frame = pd.DataFrame(index=idx)
    frame["session_date"] = idx.date
    frame["time_str"] = idx.strftime("%H:%M")
    frame["bar_of_day"] = range(len(rows))
    frame["target_open"] = [close for _, close, _ in rows]
    frame["target_close"] = [close for _, close, _ in rows]
    frame["target_fair_price"] = [fair for _, _, fair in rows]
    frame["p0_target"] = rows[0][1]
    frame["z_score"] = [z for z, _, _ in rows]
    return frame


def _generator(**kwargs):
    defaults = dict(
        z_entry=2.0,
        reversal_type="z_score_hook",
        reversal_delta=0.15,
        reversal_timeout_bars=10,
        enable_extreme_entry_lockout=False,
        enable_extreme_emergency_exit=False,
        z_max_allowed=4.0,
        lockout_mode="day_lockout",
        z_exit=0.0,
        forced_close_time="15:55",
        min_session_warmup_bars=0,
    )
    defaults.update(kwargs)
    return SignalGenerator(**defaults)


def test_abs_only_enters_long_and_short_on_price_deviation():
    long_df = _bars([(0.0, 99.0, 100.0), (0.2, 99.2, 100.0), (0.2, 99.2, 100.0), (0.2, 99.2, 100.0)])
    long_out = _generator(entry_mode="abs_only", entry_abs_deviation_usd=0.50).generate_signals(long_df)
    assert long_out.iloc[1]["signal"] == SignalType.BUY_LONG

    short_df = _bars([(0.0, 101.0, 100.0), (-0.2, 100.8, 100.0), (-0.2, 100.8, 100.0), (-0.2, 100.8, 100.0)])
    short_out = _generator(entry_mode="abs_only", entry_abs_deviation_usd=0.50).generate_signals(short_df)
    assert short_out.iloc[1]["signal"] == SignalType.SELL_SHORT


def test_anchor_filter_blocks_wrong_side_of_session_anchor():
    # Long dislocation is present, but price is above p0_target: reject.
    long_df = _bars([(0.0, 100.0, 100.0), (-1.0, 101.0, 102.0), (-0.8, 101.2, 102.0), (-0.8, 101.2, 102.0), (-0.8, 101.2, 102.0)])
    long_out = _generator(
        entry_mode="abs_only", entry_abs_deviation_usd=0.50,
        enable_open_anchor_filter=True,
    ).generate_signals(long_df)
    assert (long_out["signal"] == SignalType.BUY_LONG).sum() == 0

    # Short dislocation is present, but price is below p0_target: reject.
    short_df = _bars([(0.0, 100.0, 100.0), (1.0, 99.0, 98.0), (0.8, 98.8, 98.0), (0.8, 98.8, 98.0), (0.8, 98.8, 98.0)])
    short_out = _generator(
        entry_mode="abs_only", entry_abs_deviation_usd=0.50,
        enable_open_anchor_filter=True,
    ).generate_signals(short_df)
    assert (short_out["signal"] == SignalType.SELL_SHORT).sum() == 0


def test_z_only_hook_and_default_configuration_are_unchanged():
    frame = _bars([(0.0, 100.0, 100.0), (-2.0, 98.0, 100.0), (-1.8, 98.2, 100.0), (-1.8, 98.2, 100.0), (-1.8, 98.2, 100.0)])
    old_style = _generator().generate_signals(frame)
    explicit_defaults = _generator(
        entry_mode="z_only",
        entry_abs_deviation_usd=None,
        enable_open_anchor_filter=False,
    ).generate_signals(frame)
    assert old_style["signal"].tolist() == explicit_defaults["signal"].tolist()
    assert explicit_defaults.iloc[2]["signal"] == SignalType.BUY_LONG


def test_zero_hook_signals_on_the_threshold_bar():
    frame = _bars([(0.0, 100.0, 100.0), (-2.1, 98.0, 100.0), (-1.9, 98.2, 100.0), (-1.8, 98.3, 100.0), (-1.7, 98.4, 100.0)])
    out = _generator(reversal_delta=0.0, reversal_timeout_bars=0).generate_signals(frame)
    assert out.iloc[1]["signal"] == SignalType.BUY_LONG
