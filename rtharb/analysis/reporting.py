"""Reporting and summary generation module."""

from pathlib import Path
from typing import Optional
import pandas as pd
from ..backtest.metrics import PerformanceMetrics


def generate_summary_report(
    metrics: PerformanceMetrics,
    ticker_lead: str,
    ticker_target: str,
    trades_df: pd.DataFrame,
    output_path: Optional[str] = None
) -> str:
    """Generate a clean markdown summary report."""
    report = f"""# Intraday Stat-Arb Backtest Report: {ticker_target} vs {ticker_lead}

## Executive Summary
- **Target Asset (Traded):** {ticker_target}
- **Leader Asset (Signal):** {ticker_lead}
- **Initial Capital:** ${metrics.initial_capital:,.2f}
- **Final Equity:** ${metrics.final_equity:,.2f}
- **Total Net PnL:** ${metrics.total_pnl:,.2f} ({metrics.total_return_pct:+.2f}%)
- **CAGR:** {metrics.cagr_pct:+.2f}%
- **Sharpe Ratio:** {metrics.sharpe_ratio:.2f}
- **Sortino Ratio:** {metrics.sortino_ratio:.2f}
- **Max Drawdown:** {metrics.max_drawdown_pct:.2f}% (${metrics.max_drawdown_usd:,.2f})

## Trade Statistics
- **Total Trades:** {metrics.total_trades}
- **Win Rate:** {metrics.win_rate_pct:.1f}%
- **Profit Factor:** {metrics.profit_factor:.2f}
- **Average Trade PnL:** ${metrics.avg_trade_pnl:,.2f}
- **Average Win:** ${metrics.avg_win_pnl:,.2f}
- **Average Loss:** ${metrics.avg_loss_pnl:,.2f}
- **Win/Loss Ratio:** {metrics.win_loss_ratio:.2f}
- **Average Duration:** {metrics.avg_duration_mins:.1f} minutes
- **Total Commissions Paid:** ${metrics.total_commissions:,.2f}
- **Total Slippage Cost:** ${metrics.total_slippage:,.2f}

## Exit Reasons Breakdown
"""
    for reason, count in metrics.exit_reasons_breakdown.items():
        report += f"- **{reason}:** {count} trades ({(count / metrics.total_trades * 100):.1f}%)\n"

    if output_path:
        Path(output_path).write_text(report, encoding="utf-8")

    return report
