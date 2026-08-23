"""Chronological mean-reversion signal state machine."""

from dataclasses import dataclass
from enum import Enum
import math
import numpy as np
import pandas as pd


class SignalType(str, Enum):
    NONE = "NONE"
    BUY_LONG = "BUY_LONG"
    SELL_SHORT = "SELL_SHORT"
    EXIT_TAKE_PROFIT = "EXIT_TAKE_PROFIT"
    EXIT_EMERGENCY = "EXIT_EMERGENCY"
    EXIT_FORCED_EOD = "EXIT_FORCED_EOD"


@dataclass
class SignalEvent:
    timestamp: pd.Timestamp
    signal_type: SignalType
    z_score: float
    target_price: float
    target_fair_price: float
    reversal_delta_achieved: float = 0.0
    notes: str = ""


class SignalGenerator:
    def __init__(self, z_entry=2.0, reversal_type="z_score_hook", reversal_delta=0.15,
                 reversal_timeout_bars=10, enable_extreme_entry_lockout=True,
                 enable_extreme_emergency_exit=False, z_max_allowed=4.0,
                 lockout_mode="day_lockout", z_exit=0.0, forced_close_time="15:55",
                 min_session_warmup_bars=15, entry_abs_deviation_usd=None,
                 entry_mode="z_only", enable_open_anchor_filter=False):
        if entry_mode not in {"z_only", "abs_only", "z_or_abs"}:
            raise ValueError("entry_mode must be z_only, abs_only, or z_or_abs")
        if entry_mode != "z_only" and (entry_abs_deviation_usd is None or entry_abs_deviation_usd <= 0):
            raise ValueError("A positive entry_abs_deviation_usd is required for absolute entry modes")
        self.z_entry = z_entry
        self.reversal_type = reversal_type
        self.reversal_delta = reversal_delta
        self.reversal_timeout_bars = reversal_timeout_bars
        self.enable_extreme_entry_lockout = enable_extreme_entry_lockout
        self.enable_extreme_emergency_exit = enable_extreme_emergency_exit
        self.z_max_allowed = z_max_allowed
        self.lockout_mode = lockout_mode
        self.z_exit = z_exit
        self.forced_close_time = forced_close_time
        self.min_session_warmup_bars = min_session_warmup_bars
        self.entry_abs_deviation_usd = entry_abs_deviation_usd
        self.entry_mode = entry_mode
        self.enable_open_anchor_filter = enable_open_anchor_filter

    def generate_signals(self, df_metrics: pd.DataFrame) -> pd.DataFrame:
        df = df_metrics.copy()
        n = len(df)
        signals = [SignalType.NONE] * n
        notes = [""] * n
        armed_states = ["NONE"] * n
        entry_triggers = ["NONE"] * n
        lockouts = [False] * n
        dates = df["session_date"].to_numpy()
        times = df["time_str"].to_numpy()
        bars = df["bar_of_day"].to_numpy(int)
        z_values = df["z_score"].to_numpy(float)
        prices = df["target_close"].to_numpy(float)
        if "target_fair_price" in df:
            abs_deviations = prices - df["target_fair_price"].to_numpy(float)
        elif self.entry_mode != "z_only":
            raise ValueError("target_fair_price is required for absolute-deviation entry")
        else:
            abs_deviations = np.full(n, np.nan)
        if self.enable_open_anchor_filter:
            if "p0_target" not in df:
                raise ValueError("p0_target is required for the 09:30 open-anchor filter")
            open_anchors = df["p0_target"].to_numpy(float)
        else:
            open_anchors = np.full(n, np.nan)

        position = armed = armed_age = 0
        extreme_z = extreme_price = 0.0
        armed_trigger = "NONE"
        locked = False
        current_day = None

        for i in range(n):
            day, tm, bar = dates[i], times[i], bars[i]
            z, price = z_values[i], prices[i]
            if day != current_day:
                current_day = day
                position = armed = armed_age = 0
                armed_trigger = "NONE"
                locked = False

            penultimate = i + 2 >= n or dates[i + 2] != day
            if not math.isfinite(z):
                armed = 0
                if penultimate and position:
                    signals[i] = SignalType.EXIT_FORCED_EOD
                    position = 0
                continue

            if self.enable_extreme_entry_lockout and abs(z) >= self.z_max_allowed:
                if self.lockout_mode == "day_lockout":
                    locked, armed = True, 0
            lockouts[i] = locked

            if tm >= self.forced_close_time or penultimate:
                if position:
                    signals[i] = SignalType.EXIT_FORCED_EOD
                    notes[i] = "Close before official session end"
                    position = 0
                armed = 0
                continue

            if position and self.enable_extreme_emergency_exit and abs(z) >= self.z_max_allowed:
                signals[i] = SignalType.EXIT_EMERGENCY
                notes[i] = f"Extreme dislocation: Z={z:.2f}"
                position = armed = 0
                continue

            if position == 1:
                if z >= self.z_exit:
                    signals[i] = SignalType.EXIT_TAKE_PROFIT
                    position = 0
                continue
            if position == -1:
                if z <= self.z_exit:
                    signals[i] = SignalType.EXIT_TAKE_PROFIT
                    position = 0
                continue
            if bar < self.min_session_warmup_bars or locked:
                continue

            def anchor_allows(direction: int) -> bool:
                if not self.enable_open_anchor_filter:
                    return True
                anchor = open_anchors[i]
                if not math.isfinite(anchor):
                    return False
                return price < anchor if direction == 1 else price > anchor

            if not armed:
                abs_dev = abs_deviations[i]
                z_long, z_short = z <= -self.z_entry, z >= self.z_entry
                abs_long = math.isfinite(abs_dev) and abs_dev <= -self.entry_abs_deviation_usd if self.entry_abs_deviation_usd is not None else False
                abs_short = math.isfinite(abs_dev) and abs_dev >= self.entry_abs_deviation_usd if self.entry_abs_deviation_usd is not None else False
                if self.entry_mode == "z_only":
                    long_trigger, short_trigger = z_long, z_short
                elif self.entry_mode == "abs_only":
                    long_trigger, short_trigger = abs_long, abs_short
                elif abs_long or abs_short:
                    # In the hybrid mode a sufficiently large dollar dislocation
                    # owns the direction when rolling-Z and raw fair-value signs disagree.
                    long_trigger, short_trigger = abs_long, abs_short
                else:
                    long_trigger, short_trigger = z_long, z_short
                if long_trigger and anchor_allows(1):
                    armed, armed_age, extreme_z, extreme_price = 1, 0, z, price
                    armed_trigger = "Z+ABS" if z_long and abs_long else "Z" if z_long else "ABS_USD"
                    armed_states[i] = "ARMED_LONG"
                    entry_triggers[i] = armed_trigger
                    if self.reversal_delta <= 0:
                        signals[i], position, armed = SignalType.BUY_LONG, 1, 0
                elif short_trigger and anchor_allows(-1):
                    armed, armed_age, extreme_z, extreme_price = -1, 0, z, price
                    armed_trigger = "Z+ABS" if z_short and abs_short else "Z" if z_short else "ABS_USD"
                    armed_states[i] = "ARMED_SHORT"
                    entry_triggers[i] = armed_trigger
                    if self.reversal_delta <= 0:
                        signals[i], position, armed = SignalType.SELL_SHORT, -1, 0
                continue

            armed_age += 1
            armed_states[i] = ("ARMED_LONG" if armed == 1 else "ARMED_SHORT") + f" ({armed_age})"
            entry_triggers[i] = armed_trigger
            if not anchor_allows(armed):
                notes[i], armed, armed_trigger = "Open-anchor filter invalidated arming", 0, "NONE"
                continue
            if armed == 1:
                extreme_z, extreme_price = min(extreme_z, z), min(extreme_price, price)
                delta = z - extreme_z if self.reversal_type == "z_score_hook" else (price - extreme_price) / extreme_price
                if delta >= self.reversal_delta:
                    signals[i], position, armed = SignalType.BUY_LONG, 1, 0
            else:
                extreme_z, extreme_price = max(extreme_z, z), max(extreme_price, price)
                delta = extreme_z - z if self.reversal_type == "z_score_hook" else (extreme_price - price) / extreme_price
                if delta >= self.reversal_delta:
                    signals[i], position, armed = SignalType.SELL_SHORT, -1, 0
            if armed and armed_age >= self.reversal_timeout_bars:
                notes[i], armed, armed_trigger = "Arming timed out", 0, "NONE"

        df["signal"] = signals
        df["signal_note"] = notes
        df["armed_state"] = armed_states
        df["entry_trigger"] = entry_triggers
        df["abs_deviation_usd"] = abs_deviations
        df["is_locked_out"] = lockouts
        return df
