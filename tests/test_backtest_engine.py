"""Unit tests for Backtesting Engine and Next-Bar execution."""

import pandas as pd
import numpy as np
import pytest
from rtharb.models.signals import SignalType
from rtharb.backtest.engine import BacktestEngine
from rtharb.backtest.metrics import calculate_performance_metrics


def test_next_bar_execution():
    dates = pd.date_range("2026-08-17 09:30:00-04:00", periods=50, freq="1min")
    df = pd.DataFrame(index=dates)
    df["session_date"] = dates.date
    df["time_str"] = dates.strftime("%H:%M")
    df["bar_of_day"] = range(50)
    df["target_open"] = [100.0 + i for i in range(50)]
    df["target_close"] = [100.5 + i for i in range(50)]
    df["z_score"] = 0.0
    df["signal"] = SignalType.NONE

    # Signal on bar 10 close
    df.iloc[10, df.columns.get_loc("signal")] = SignalType.BUY_LONG
    # Exit signal on bar 20 close
    df.iloc[20, df.columns.get_loc("signal")] = SignalType.EXIT_TAKE_PROFIT

    engine = BacktestEngine(
        initial_capital=100000.0,
        position_size_usd=10000.0,
        commission_per_share=0.0035,
        slippage_pct=0.0,
        allow_short=True
    )

    res = engine.run(df, ticker_target="NVDA")
    trades = res["trades_df"]

    assert len(trades) == 1
    trade = trades.iloc[0]
    # Executed on bar 11 open ($111.0)
    assert trade["entry_time"] == dates[11]
    assert trade["entry_price"] == 111.0
    # Closed on bar 21 open ($121.0)
    assert trade["exit_time"] == dates[21]
    assert trade["exit_price"] == 121.0
    assert trade["gross_pnl"] > 0
