"""Causal session-anchored fair value and dislocation calculations."""

import numpy as np
import pandas as pd


class FairValueModel:
    """Reset dislocation at each RTH open and normalize inside each session."""

    def __init__(
        self,
        beta_mode: str = "dynamic_rolling",
        beta_rolling_days: int = 10,
        rolling_window_w: int = 20,
        min_session_warmup_bars: int = 15,
        min_sigma_history_days: int = 10,
        vwap_mode: str = "none",
    ):
        self.beta_mode = beta_mode
        self.beta_rolling_days = beta_rolling_days
        self.rolling_window_w = rolling_window_w
        self.min_session_warmup_bars = min_session_warmup_bars
        self.min_sigma_history_days = min_sigma_history_days
        if vwap_mode not in {"none", "session"}:
            raise ValueError("vwap_mode must be 'none' or 'session'")
        self.vwap_mode = vwap_mode

    def compute_daily_betas(self, df_lead: pd.DataFrame, df_target: pd.DataFrame) -> pd.Series:
        lead_daily = df_lead.groupby(df_lead.index.date)["close"].last().pct_change()
        target_daily = df_target.groupby(df_target.index.date)["close"].last().pct_change()
        pair = pd.concat({"lead": lead_daily, "target": target_daily}, axis=1).dropna()
        cov = pair["target"].rolling(self.beta_rolling_days, min_periods=self.beta_rolling_days).cov(pair["lead"])
        var = pair["lead"].rolling(self.beta_rolling_days, min_periods=self.beta_rolling_days).var()
        return (cov / var).shift(1).clip(0.2, 4.0)

    def compute_intraday_metrics(self, df_lead: pd.DataFrame, df_target: pd.DataFrame) -> pd.DataFrame:
        common = df_lead.index.intersection(df_target.index)
        combined = pd.DataFrame(index=common)
        combined["lead_open"] = df_lead.loc[common, "open"].astype(float)
        combined["lead_close"] = df_lead.loc[common, "close"].astype(float)
        combined["target_open"] = df_target.loc[common, "open"].astype(float)
        combined["target_high"] = df_target.loc[common, "high"].astype(float)
        combined["target_low"] = df_target.loc[common, "low"].astype(float)
        combined["target_close"] = df_target.loc[common, "close"].astype(float)
        if self.vwap_mode == "session":
            for frame, name in ((df_lead, "lead"), (df_target, "target")):
                if "volume" not in frame.columns:
                    raise ValueError(f"{name} bars require a volume column for session VWAP")
                combined[f"{name}_high"] = frame.loc[common, "high"].astype(float)
                combined[f"{name}_low"] = frame.loc[common, "low"].astype(float)
                combined[f"{name}_volume"] = frame.loc[common, "volume"].astype(float)
        combined["session_date"] = combined.index.date
        combined["time_str"] = combined.index.strftime("%H:%M")
        grouped = combined.groupby("session_date", sort=False)
        combined["bar_of_day"] = grouped.cumcount()

        if self.vwap_mode == "session":
            # Typical-price VWAP is cumulative through the current bar only;
            # therefore it is available causally when that bar closes.
            for name in ("lead", "target"):
                typical = (combined[f"{name}_high"] + combined[f"{name}_low"] + combined[f"{name}_close"]) / 3.0
                volume = combined[f"{name}_volume"]
                notional = (typical * volume).groupby(combined["session_date"], sort=False).cumsum()
                volume_sum = volume.groupby(combined["session_date"], sort=False).cumsum()
                combined[f"{name}_vwap"] = (notional / volume_sum).where(volume_sum > 0)

        if self.beta_mode.startswith("fixed_"):
            combined["beta"] = float(self.beta_mode.removeprefix("fixed_"))
        else:
            betas = self.compute_daily_betas(df_lead.loc[common], df_target.loc[common])
            combined["beta"] = combined["session_date"].map(betas).fillna(1.5)

        # A close-based anchor is consistent with close-generated signals and
        # makes r_lead, r_target, and spread exactly zero on the first bar.
        combined["p0_lead"] = grouped["lead_close"].transform("first")
        combined["p0_target"] = grouped["target_close"].transform("first")
        combined["r_lead"] = combined["lead_close"] / combined["p0_lead"] - 1.0
        combined["r_target"] = combined["target_close"] / combined["p0_target"] - 1.0
        combined["spread"] = combined["r_target"] - combined["beta"] * combined["r_lead"]
        combined["target_fair_price"] = combined["p0_target"] * (1.0 + combined["beta"] * combined["r_lead"])

        # This is the MD strategy: a causal rolling mean/std within the current
        # RTH session. The current close is known when its signal is generated.
        min_periods = min(self.rolling_window_w, self.min_session_warmup_bars)
        hist_mean = grouped["spread"].transform(
            lambda s: s.rolling(self.rolling_window_w, min_periods=min_periods).mean()
        )
        hist_std = grouped["spread"].transform(
            lambda s: s.rolling(self.rolling_window_w, min_periods=min_periods).std(ddof=1)
        ).where(lambda s: s > 1e-8)
        combined["spread_mean"] = hist_mean
        combined["spread_std"] = hist_std
        combined["z_score"] = ((combined["spread"] - hist_mean) / hist_std).replace([np.inf, -np.inf], np.nan)
        combined.loc[combined["bar_of_day"] < self.min_session_warmup_bars, "z_score"] = np.nan
        return combined
