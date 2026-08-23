"""Configuration schemas and loaders for RTH Stat-Arb strategy and backtesting."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal
import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class StrategyConfig:
    ticker_lead: str = "QQQ"
    ticker_target: str = "NVDA"
    data_source: str = "alpaca" # "alpaca", "yfinance", "csv"
    data_feed: str = "sip" # consolidated US tape; IEX is too sparse for this study
    
    # Session Times in New York timezone (America/New_York)
    session_start: str = "09:30"
    session_end: str = "16:00"
    forced_close_time: str = "15:55"
    
    # Fair Value / Beta modeling
    beta_mode: str = "dynamic_rolling" # "dynamic_rolling", "fixed_1.0", "historical_daily"
    beta_rolling_days: int = 10
    rolling_window_w: int = 30 # intraday rolling spread mean/std in minutes
    min_sigma_history_days: int = 10
    min_session_warmup_bars: int = 15 # wait 15 min after open
    
    # Signal Entry & Reversal Hook
    z_entry: float = 2.0
    reversal_type: str = "z_score_hook" # "z_score_hook", "price_pct_rebound"
    reversal_delta: float = 0.15 # delta to confirm reversion movement
    reversal_timeout_bars: int = 10
    
    # 4-Sigma Extreme Dislocation Filters (Independent A/B/C/D testable)
    enable_extreme_entry_lockout: bool = True
    enable_extreme_emergency_exit: bool = False
    z_max_allowed: float = 4.0
    lockout_mode: str = "day_lockout" # "day_lockout", "window_lockout"
    
    # Exit Parameters
    z_exit: float = 0.0
    stop_loss_pct: float = 0.015 # 1.5% fixed stop loss fallback


@dataclass
class BacktestConfig:
    initial_capital: float = 100000.0
    position_size_usd: float = 20000.0
    commission_per_share: float = 0.0035 # IBKR default $0.0035/share
    slippage_pct: float = 0.0002 # 0.02% (2 bps)
    allow_short: bool = True


@dataclass
class AppConfig:
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    cache_dir: str = str(Path(__file__).parent.parent / "data_cache")

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "AppConfig":
        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            strat_data = data.get("strategy", {})
            bt_data = data.get("backtest", {})
            return cls(
                strategy=StrategyConfig(**strat_data),
                backtest=BacktestConfig(**bt_data),
                cache_dir=data.get("cache_dir", str(Path(__file__).parent.parent / "data_cache"))
            )
        return cls()
