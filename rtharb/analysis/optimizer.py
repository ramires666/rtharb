"""Parameter Optimizer: Grid search and sensitivity analysis across key parameters."""

import itertools
from typing import Dict, Any, List, Optional
import pandas as pd
from tqdm import tqdm

from ..config import AppConfig
from ..models.signals import SignalGenerator
from ..backtest.engine import BacktestEngine
from ..backtest.metrics import calculate_performance_metrics


class ParameterOptimizer:
    def __init__(self, config: AppConfig):
        self.config = config

    def grid_search(
        self,
        df_metrics: pd.DataFrame,
        z_entries: Optional[List[float]] = None,
        reversal_deltas: Optional[List[float]] = None,
        z_max_alloweds: Optional[List[float]] = None,
        enable_lockouts: Optional[List[bool]] = None,
        enable_emergency_exits: Optional[List[bool]] = None
    ) -> pd.DataFrame:
        """Run grid search over strategy parameters and rank combinations."""
        z_entries = z_entries or [1.2, 1.5, 1.8, 2.0]
        reversal_deltas = reversal_deltas or [0.05, 0.10, 0.15, 0.20, 0.25]
        z_max_alloweds = z_max_alloweds or [3.5, 4.0, 4.5]
        enable_lockouts = enable_lockouts or [True, False]
        enable_emergency_exits = enable_emergency_exits or [False, True]

        combinations = list(itertools.product(
            z_entries, reversal_deltas, z_max_alloweds, enable_lockouts, enable_emergency_exits
        ))

        records = []

        for z_entry, rev_delta, z_max, lockout, emerg_exit in combinations:
            sig_gen = SignalGenerator(
                z_entry=z_entry,
                reversal_type=self.config.strategy.reversal_type,
                reversal_delta=rev_delta,
                reversal_timeout_bars=self.config.strategy.reversal_timeout_bars,
                enable_extreme_entry_lockout=lockout,
                enable_extreme_emergency_exit=emerg_exit,
                z_max_allowed=z_max,
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

            records.append({
                "z_entry": z_entry,
                "reversal_delta": rev_delta,
                "z_max_allowed": z_max,
                "entry_lockout": lockout,
                "emergency_exit": emerg_exit,
                "total_pnl": metrics.total_pnl,
                "total_return_pct": metrics.total_return_pct,
                "cagr_pct": metrics.cagr_pct,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "max_drawdown_pct": metrics.max_drawdown_pct,
                "win_rate_pct": metrics.win_rate_pct,
                "profit_factor": metrics.profit_factor,
                "total_trades": metrics.total_trades,
                "avg_duration_mins": metrics.avg_duration_mins
            })

        df_opt = pd.DataFrame(records)
        df_opt.sort_values(by="sharpe_ratio", ascending=False, inplace=True)
        df_opt.reset_index(drop=True, inplace=True)
        return df_opt
