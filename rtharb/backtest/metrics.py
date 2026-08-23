"""Quantitative performance metrics calculation for stat-arb backtests."""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
    initial_capital: float
    final_equity: float
    total_pnl: float
    total_return_pct: float
    cagr_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    max_drawdown_usd: float
    total_trades: int
    win_rate_pct: float
    profit_factor: float
    avg_trade_pnl: float
    avg_win_pnl: float
    avg_loss_pnl: float
    win_loss_ratio: float
    avg_duration_mins: float
    total_commissions: float
    total_slippage: float
    exit_reasons_breakdown: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


def calculate_performance_metrics(
    df_results: pd.DataFrame,
    trades_df: pd.DataFrame,
    initial_capital: float = 100000.0
) -> PerformanceMetrics:
    if df_results.empty or "portfolio_equity" not in df_results.columns:
        return PerformanceMetrics(
            initial_capital=initial_capital,
            final_equity=initial_capital,
            total_pnl=0.0,
            total_return_pct=0.0,
            cagr_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown_pct=0.0,
            max_drawdown_usd=0.0,
            total_trades=0,
            win_rate_pct=0.0,
            profit_factor=0.0,
            avg_trade_pnl=0.0,
            avg_win_pnl=0.0,
            avg_loss_pnl=0.0,
            win_loss_ratio=0.0,
            avg_duration_mins=0.0,
            total_commissions=0.0,
            total_slippage=0.0,
            exit_reasons_breakdown={}
        )

    equity = df_results["portfolio_equity"]
    final_equity = equity.iloc[-1]
    total_pnl = final_equity - initial_capital
    total_return_pct = (total_pnl / initial_capital) * 100.0

    # Daily returns for Sharpe and Sortino
    daily_equity = df_results.groupby("session_date")["portfolio_equity"].last()
    previous_equity = daily_equity.shift(1).fillna(initial_capital)
    daily_returns = (daily_equity / previous_equity - 1.0).dropna()

    days_count = len(daily_equity)
    years = max(days_count / 252.0, 1.0 / 252.0)
    cagr_pct = (((final_equity / initial_capital) ** (1.0 / years)) - 1.0) * 100.0 if final_equity > 0 else -100.0

    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe_ratio = float(np.sqrt(252.0) * (daily_returns.mean() / daily_returns.std()))
        downside_dev = float(np.sqrt(np.mean(np.minimum(daily_returns.to_numpy(), 0.0) ** 2)))
        sortino_ratio = float(np.sqrt(252.0) * daily_returns.mean() / downside_dev) if downside_dev > 0 else 0.0
    else:
        sharpe_ratio = 0.0
        sortino_ratio = 0.0

    # Drawdowns
    equity_with_start = pd.concat([pd.Series([initial_capital]), equity.reset_index(drop=True)], ignore_index=True)
    running_max = np.maximum.accumulate(equity_with_start)
    drawdown_usd = running_max - equity_with_start
    drawdown_pct = (drawdown_usd / running_max) * 100.0
    max_drawdown_usd = float(drawdown_usd.max())
    max_drawdown_pct = float(drawdown_pct.max())

    # Trade statistics
    total_trades = len(trades_df)
    if total_trades > 0:
        wins = trades_df[trades_df["net_pnl"] > 0]
        losses = trades_df[trades_df["net_pnl"] <= 0]
        win_rate_pct = (len(wins) / total_trades) * 100.0
        
        gross_profit = wins["net_pnl"].sum()
        gross_loss = abs(losses["net_pnl"].sum())
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

        avg_trade_pnl = float(trades_df["net_pnl"].mean())
        avg_win_pnl = float(wins["net_pnl"].mean()) if len(wins) > 0 else 0.0
        avg_loss_pnl = float(losses["net_pnl"].mean()) if len(losses) > 0 else 0.0
        win_loss_ratio = abs(avg_win_pnl / avg_loss_pnl) if avg_loss_pnl != 0 else 0.0

        avg_duration_mins = float(trades_df["duration_bars"].mean())
        total_commissions = float(trades_df["commission"].sum())
        total_slippage = float(trades_df["slippage"].sum())
        exit_reasons = trades_df["exit_reason"].value_counts().to_dict()
    else:
        win_rate_pct = 0.0
        profit_factor = 0.0
        avg_trade_pnl = 0.0
        avg_win_pnl = 0.0
        avg_loss_pnl = 0.0
        win_loss_ratio = 0.0
        avg_duration_mins = 0.0
        total_commissions = 0.0
        total_slippage = 0.0
        exit_reasons = {}

    return PerformanceMetrics(
        initial_capital=initial_capital,
        final_equity=final_equity,
        total_pnl=total_pnl,
        total_return_pct=total_return_pct,
        cagr_pct=cagr_pct,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown_pct=max_drawdown_pct,
        max_drawdown_usd=max_drawdown_usd,
        total_trades=total_trades,
        win_rate_pct=win_rate_pct,
        profit_factor=profit_factor,
        avg_trade_pnl=avg_trade_pnl,
        avg_win_pnl=avg_win_pnl,
        avg_loss_pnl=avg_loss_pnl,
        win_loss_ratio=win_loss_ratio,
        avg_duration_mins=avg_duration_mins,
        total_commissions=total_commissions,
        total_slippage=total_slippage,
        exit_reasons_breakdown=exit_reasons
    )
