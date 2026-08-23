"""Event-driven Backtesting Engine with Next-Bar Open execution and zero lookahead bias."""

from typing import List, Dict, Any, Optional
import math
import numpy as np
import pandas as pd

from ..models.signals import SignalType
from .position import Position, Trade


class BacktestEngine:
    def __init__(
        self,
        initial_capital: float = 100000.0,
        position_size_usd: float = 20000.0,
        commission_per_share: float = 0.0035, # IBKR rate
        slippage_pct: float = 0.0002, # 2 bps
        allow_short: bool = True,
        max_holding_bars: Optional[int] = None,
        stop_loss_pct: Optional[float] = None,
    ):
        self.initial_capital = initial_capital
        self.position_size_usd = position_size_usd
        self.commission_per_share = commission_per_share
        self.slippage_pct = slippage_pct
        self.allow_short = allow_short
        self.max_holding_bars = max_holding_bars
        self.stop_loss_pct = stop_loss_pct

    def run(self, df_signals: pd.DataFrame, ticker_target: str = "NVDA") -> Dict[str, Any]:
        """
        Execute backtest chronologically:
        Signal generated on bar i (close) -> Executed on bar i+1 (open) with slippage.
        """
        df = df_signals.copy()
        n = len(df)
        timestamps = df.index
        open_values = df["target_open"].to_numpy(float)
        close_values = df["target_close"].to_numpy(float)
        low_values = df["target_low"].to_numpy(float) if "target_low" in df else close_values
        high_values = df["target_high"].to_numpy(float) if "target_high" in df else close_values
        z_values = df["z_score"].to_numpy(float)
        signal_values = df["signal"].to_numpy()
        session_dates = df["session_date"].to_numpy()

        equity_curve = np.zeros(n)
        cash_curve = np.zeros(n)
        position_curve = np.zeros(n)

        cash = self.initial_capital
        active_position: Optional[Position] = None
        trades: List[Trade] = []
        trade_counter = 0

        pending_action: Optional[Dict[str, Any]] = None
        entry_bar_idx = 0

        for i in range(n):
            ts = timestamps[i]
            open_price = open_values[i]
            close_price = close_values[i]
            z = z_values[i]
            signal = signal_values[i]

            # 1. EXECUTE PENDING ACTION FROM PREVIOUS BAR'S CLOSE
            if pending_action is not None:
                action = pending_action["action"]
                action_reason = pending_action["reason"]
                prev_z = pending_action["z_score"]

                if action == "CLOSE" and active_position is not None:
                    duration = i - entry_bar_idx
                    trade = active_position.close_position(
                        exit_time=ts,
                        exit_price=open_price,
                        exit_reason=action_reason,
                        commission_per_share=self.commission_per_share,
                        slippage_pct=self.slippage_pct,
                        exit_z_score=z,
                        trade_id=trade_counter,
                        duration_bars=duration
                    )
                    trades.append(trade)
                    cash += trade.net_pnl + active_position.entry_commission
                    active_position = None

                elif action == "OPEN_LONG" and active_position is None:
                    trade_counter += 1
                    entry_bar_idx = i
                    # Apply slippage on entry (buy slightly higher)
                    effective_entry_price = open_price * (1.0 + self.slippage_pct)
                    shares = math.floor(self.position_size_usd / effective_entry_price)
                    entry_commission = shares * self.commission_per_share
                    cash -= entry_commission
                    active_position = Position(
                        ticker=ticker_target,
                        direction=1,
                        entry_time=ts,
                        entry_price=effective_entry_price,
                        shares=shares,
                        position_value=shares * open_price,
                        entry_z_score=prev_z,
                        entry_reference_price=open_price,
                        entry_commission=entry_commission,
                    )

                elif action == "OPEN_SHORT" and active_position is None and self.allow_short:
                    trade_counter += 1
                    entry_bar_idx = i
                    # Apply slippage on entry (sell short slightly lower)
                    effective_entry_price = open_price * (1.0 - self.slippage_pct)
                    shares = math.floor(self.position_size_usd / effective_entry_price)
                    entry_commission = shares * self.commission_per_share
                    cash -= entry_commission
                    active_position = Position(
                        ticker=ticker_target,
                        direction=-1,
                        entry_time=ts,
                        entry_price=effective_entry_price,
                        shares=shares,
                        position_value=shares * open_price,
                        entry_z_score=prev_z,
                        entry_reference_price=open_price,
                        entry_commission=entry_commission,
                    )

                pending_action = None

            # Risk overlays are evaluated only after the next-open execution.
            # Stops use intrabar high/low and assume a worse gap-open fill.
            if active_position is not None:
                duration = i - entry_bar_idx
                filter_price = None
                filter_reason = None
                if self.stop_loss_pct is not None:
                    if active_position.direction == 1:
                        stop = active_position.entry_reference_price * (1.0 - self.stop_loss_pct)
                        if low_values[i] <= stop:
                            filter_price = min(open_price, stop)
                            filter_reason = "STOP_LOSS"
                    else:
                        stop = active_position.entry_reference_price * (1.0 + self.stop_loss_pct)
                        if high_values[i] >= stop:
                            filter_price = max(open_price, stop)
                            filter_reason = "STOP_LOSS"
                if filter_reason is None and self.max_holding_bars is not None and duration >= self.max_holding_bars:
                    filter_price = open_price
                    filter_reason = "TIME_STOP"
                if filter_reason is not None:
                    trade = active_position.close_position(
                        exit_time=ts, exit_price=filter_price,
                        exit_reason=filter_reason,
                        commission_per_share=self.commission_per_share,
                        slippage_pct=self.slippage_pct,
                        exit_z_score=z, trade_id=trade_counter,
                        duration_bars=duration,
                    )
                    trades.append(trade)
                    cash += trade.net_pnl + active_position.entry_commission
                    active_position = None
                    pending_action = None

            # 2. EVALUATE SIGNAL AT CURRENT BAR'S CLOSE -> QUEUE PENDING ACTION FOR NEXT BAR OPEN
            if signal == SignalType.BUY_LONG and active_position is None:
                pending_action = {"action": "OPEN_LONG", "reason": "BUY_REVERSAL_HOOK", "z_score": z}
            elif signal == SignalType.SELL_SHORT and active_position is None:
                pending_action = {"action": "OPEN_SHORT", "reason": "SHORT_REVERSAL_HOOK", "z_score": z}
            elif signal == SignalType.EXIT_TAKE_PROFIT and active_position is not None:
                pending_action = {"action": "CLOSE", "reason": "TAKE_PROFIT", "z_score": z}
            elif signal == SignalType.EXIT_EMERGENCY and active_position is not None:
                pending_action = {"action": "CLOSE", "reason": "EMERGENCY_4SIGMA", "z_score": z}
            elif signal == SignalType.EXIT_FORCED_EOD and active_position is not None:
                pending_action = {"action": "CLOSE", "reason": "FORCED_EOD", "z_score": z}

            # Absolute guard: never carry a position or pending entry overnight,
            # even if a source session has a missing penultimate bar.
            is_session_last = i == n - 1 or session_dates[i + 1] != session_dates[i]
            if is_session_last:
                if active_position is not None:
                    trade = active_position.close_position(
                        exit_time=ts, exit_price=close_price,
                        exit_reason="SESSION_END_FALLBACK",
                        commission_per_share=self.commission_per_share,
                        slippage_pct=self.slippage_pct,
                        exit_z_score=z, trade_id=trade_counter,
                        duration_bars=i - entry_bar_idx,
                    )
                    trades.append(trade)
                    cash += trade.net_pnl + active_position.entry_commission
                    active_position = None
                pending_action = None

            # 3. RECORD MARK-TO-MARKET EQUITY AT BAR CLOSE
            unrealized_pnl = active_position.calculate_unrealized_pnl(close_price) if active_position is not None else 0.0
            equity_curve[i] = cash + unrealized_pnl
            cash_curve[i] = cash
            position_curve[i] = active_position.direction if active_position is not None else 0

        # Close any remaining position on the final bar
        if active_position is not None:
            last_ts = df.index[-1]
            last_price = df.iloc[-1]["target_close"]
            trade = active_position.close_position(
                exit_time=last_ts,
                exit_price=last_price,
                exit_reason="BACKTEST_END",
                commission_per_share=self.commission_per_share,
                slippage_pct=self.slippage_pct,
                exit_z_score=z_values[-1],
                trade_id=trade_counter,
                duration_bars=n - entry_bar_idx
            )
            trades.append(trade)
            cash += trade.net_pnl + active_position.entry_commission
            equity_curve[-1] = cash

        df_results = df.copy()
        df_results["portfolio_equity"] = equity_curve
        df_results["portfolio_cash"] = cash_curve
        df_results["position_state"] = position_curve

        trades_df = pd.DataFrame([t.__dict__ for t in trades]) if trades else pd.DataFrame()

        return {
            "df_results": df_results,
            "trades_df": trades_df,
            "total_trades": len(trades),
            "final_equity": equity_curve[-1]
        }
