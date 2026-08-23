"""Position and Trade data models for realistic intraday backtesting."""

from dataclasses import dataclass
from typing import Optional
import pandas as pd


@dataclass
class Trade:
    trade_id: int
    ticker: str
    direction: int # 1 for Long, -1 for Short
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    shares: float
    position_value: float
    gross_pnl: float
    commission: float
    slippage: float
    net_pnl: float
    return_pct: float
    exit_reason: str
    duration_bars: int # minutes
    entry_z_score: float = 0.0
    exit_z_score: float = 0.0


class Position:
    def __init__(
        self,
        ticker: str,
        direction: int,
        entry_time: pd.Timestamp,
        entry_price: float,
        shares: float,
        position_value: float,
        entry_z_score: float = 0.0,
        entry_reference_price: Optional[float] = None,
        entry_commission: float = 0.0,
    ):
        self.ticker = ticker
        self.direction = direction
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.shares = shares
        self.position_value = position_value
        self.entry_z_score = entry_z_score
        self.entry_reference_price = entry_reference_price if entry_reference_price is not None else entry_price
        self.entry_commission = entry_commission

    def calculate_unrealized_pnl(self, current_price: float) -> float:
        if self.direction == 1:
            return (current_price - self.entry_price) * self.shares
        else:
            return (self.entry_price - current_price) * self.shares

    def close_position(
        self,
        exit_time: pd.Timestamp,
        exit_price: float,
        exit_reason: str,
        commission_per_share: float,
        slippage_pct: float,
        exit_z_score: float,
        trade_id: int,
        duration_bars: int
    ) -> Trade:
        # Apply slippage on exit
        if self.direction == 1:
            effective_exit_price = exit_price * (1.0 - slippage_pct)
        else:
            effective_exit_price = exit_price * (1.0 + slippage_pct)

        gross_pnl = self.direction * (exit_price - self.entry_reference_price) * self.shares
        total_commission = self.entry_commission + self.shares * commission_per_share
        entry_slippage = abs(self.entry_price - self.entry_reference_price) * self.shares
        exit_slippage = abs(effective_exit_price - exit_price) * self.shares
        total_slippage = entry_slippage + exit_slippage
        net_pnl = gross_pnl - total_commission - total_slippage
        return_pct = net_pnl / self.position_value if self.position_value > 0 else 0.0

        return Trade(
            trade_id=trade_id,
            ticker=self.ticker,
            direction=self.direction,
            entry_time=self.entry_time,
            entry_price=self.entry_price,
            exit_time=exit_time,
            exit_price=effective_exit_price,
            shares=self.shares,
            position_value=self.position_value,
            gross_pnl=gross_pnl,
            commission=total_commission,
            slippage=total_slippage,
            net_pnl=net_pnl,
            return_pct=return_pct,
            exit_reason=exit_reason,
            duration_bars=duration_bars,
            entry_z_score=self.entry_z_score,
            exit_z_score=exit_z_score
        )
