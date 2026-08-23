"""Matrix Comparator: A/B/C/D testing of 4-sigma Entry Lockout and Emergency Exit combinations."""

from typing import Dict, Any, List
import pandas as pd

from ..config import AppConfig, StrategyConfig
from ..models.signals import SignalGenerator
from ..backtest.engine import BacktestEngine
from ..backtest.metrics import calculate_performance_metrics, PerformanceMetrics


class MatrixComparator:
    def __init__(self, config: AppConfig):
        self.config = config

    def run_all_scenarios(self, df_metrics: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute 4 scenarios on the same dataset:
        - Scenario A: Pure Reversion (Entry Lockout: OFF, Emergency Exit: OFF)
        - Scenario B: Smart Lockout (Entry Lockout: ON, Emergency Exit: OFF)
        - Scenario C: Hard Emergency Stop (Entry Lockout: OFF, Emergency Exit: ON)
        - Scenario D: Conservative (Entry Lockout: ON, Emergency Exit: ON)
        """
        scenarios = {
            "A: Pure Reversion (No 4σ caps)": {
                "enable_extreme_entry_lockout": False,
                "enable_extreme_emergency_exit": False,
                "desc": "Trades all divergence reversals, holds until fair value or EOD."
            },
            "B: Entry Lockout Only (Recommended)": {
                "enable_extreme_entry_lockout": True,
                "enable_extreme_emergency_exit": False,
                "desc": "Blocks new entries on 4σ regime shifts, but does not cut active trades at the peak."
            },
            "C: Emergency Exit Only": {
                "enable_extreme_entry_lockout": False,
                "enable_extreme_emergency_exit": True,
                "desc": "Allows entries everywhere, but forces immediate liquidation if 4σ is breached."
            },
            "D: Conservative (Lockout + Exit)": {
                "enable_extreme_entry_lockout": True,
                "enable_extreme_emergency_exit": True,
                "desc": "Blocks new 4σ entries and liquidates active positions on 4σ breach."
            }
        }

        results = {}
        comparison_rows = []
        equity_curves = pd.DataFrame(index=df_metrics.index)

        for name, params in scenarios.items():
            # Build signal generator with specific flags
            sig_gen = SignalGenerator(
                z_entry=self.config.strategy.z_entry,
                reversal_type=self.config.strategy.reversal_type,
                reversal_delta=self.config.strategy.reversal_delta,
                reversal_timeout_bars=self.config.strategy.reversal_timeout_bars,
                enable_extreme_entry_lockout=params["enable_extreme_entry_lockout"],
                enable_extreme_emergency_exit=params["enable_extreme_emergency_exit"],
                z_max_allowed=self.config.strategy.z_max_allowed,
                lockout_mode=self.config.strategy.lockout_mode,
                z_exit=self.config.strategy.z_exit,
                forced_close_time=self.config.strategy.forced_close_time,
                min_session_warmup_bars=self.config.strategy.min_session_warmup_bars
            )

            df_signals = sig_gen.generate_signals(df_metrics)

            engine = BacktestEngine(
                initial_capital=self.config.backtest.initial_capital,
                position_size_usd=self.config.backtest.position_size_usd,
                commission_per_share=self.config.backtest.commission_per_share,
                slippage_pct=self.config.backtest.slippage_pct,
                allow_short=self.config.backtest.allow_short
            )

            bt_res = engine.run(df_signals, ticker_target=self.config.strategy.ticker_target)
            metrics = calculate_performance_metrics(
                df_results=bt_res["df_results"],
                trades_df=bt_res["trades_df"],
                initial_capital=self.config.backtest.initial_capital
            )

            equity_curves[name] = bt_res["df_results"]["portfolio_equity"]

            results[name] = {
                "params": params,
                "df_results": bt_res["df_results"],
                "trades_df": bt_res["trades_df"],
                "metrics": metrics
            }

            comparison_rows.append({
                "Scenario": name,
                "Total PnL ($)": f"${metrics.total_pnl:,.2f}",
                "Total Return (%)": f"{metrics.total_return_pct:.2f}%",
                "CAGR (%)": f"{metrics.cagr_pct:.2f}%",
                "Sharpe Ratio": f"{metrics.sharpe_ratio:.2f}",
                "Sortino Ratio": f"{metrics.sortino_ratio:.2f}",
                "Max Drawdown (%)": f"{metrics.max_drawdown_pct:.2f}%",
                "Max Drawdown ($)": f"${metrics.max_drawdown_usd:,.2f}",
                "Total Trades": metrics.total_trades,
                "Win Rate (%)": f"{metrics.win_rate_pct:.1f}%",
                "Profit Factor": f"{metrics.profit_factor:.2f}",
                "Avg Duration (min)": f"{metrics.avg_duration_mins:.1f}",
                "Commissions ($)": f"${metrics.total_commissions:,.2f}",
                "Emergency Exits": metrics.exit_reasons_breakdown.get("EMERGENCY_4SIGMA", 0)
            })

        comparison_df = pd.DataFrame(comparison_rows)

        return {
            "results": results,
            "equity_curves": equity_curves,
            "comparison_df": comparison_df
        }
