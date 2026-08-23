"""Unified data loader and preprocessor for intraday pairs data."""

from datetime import datetime, time
from pathlib import Path
from typing import Optional, Tuple, Literal, Any
import pandas as pd
import numpy as np
import pytz

from .storage import DataStorage
from .alpaca_client import AlpacaClient
from .yfinance_client import YFinanceClient


class DataLoader:
    def __init__(self, cache_dir: str = "data_cache", source: str = "alpaca", data_feed: str = "sip"):
        self.storage = DataStorage(cache_dir)
        self.source = source.lower()
        self.data_feed = data_feed.lower()
        self.alpaca = AlpacaClient()
        self.yfinance = YFinanceClient()

    def _filter_official_rth(self, df: pd.DataFrame, session_start: str, session_end: str) -> pd.DataFrame:
        """Filter by the saved exchange calendar, including official half-days."""
        calendar_path = self.storage.cache_dir / "market_calendar.csv"
        if not calendar_path.exists():
            return df.between_time(session_start, session_end, inclusive="left")
        calendar = pd.read_csv(calendar_path, dtype=str)
        to_minute = lambda s: s.str.slice(0, 2).astype(int) * 60 + s.str.slice(3, 5).astype(int)
        calendar["open_minute"] = to_minute(calendar["open"])
        calendar["close_minute"] = to_minute(calendar["close"])
        opens = calendar.set_index("date")["open_minute"]
        closes = calendar.set_index("date")["close_minute"]
        dates = pd.Series(df.index.strftime("%Y-%m-%d"), index=df.index)
        minute = df.index.hour * 60 + df.index.minute
        open_minute = dates.map(opens).to_numpy(float)
        close_minute = dates.map(closes).to_numpy(float)
        mask = np.isfinite(open_minute) & (minute >= open_minute) & (minute < close_minute)
        return df.loc[mask]

    def get_symbol_bars(
        self,
        symbol: str,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
        days_back: int = 730, # fallback if start_date is None
        force_download: bool = False,
        source: Optional[str] = None
    ) -> pd.DataFrame:
        """Get 1m bars for symbol, loading from cache if available or fetching from API."""
        src = (source or self.source).lower()
        
        if not force_download:
            cached_df = self.storage.load_bars(symbol, interval="1m")
            if cached_df is not None and not cached_df.empty:
                df = cached_df
                if start_date is not None:
                    s_dt = pd.to_datetime(start_date).tz_localize("America/New_York") if pd.to_datetime(start_date).tzinfo is None else pd.to_datetime(start_date)
                    df = df[df.index >= s_dt]
                if end_date is not None:
                    e_dt = pd.to_datetime(end_date).tz_localize("America/New_York") if pd.to_datetime(end_date).tzinfo is None else pd.to_datetime(end_date)
                    df = df[df.index <= e_dt]
                if not df.empty:
                    return df

        # Fetch from remote source
        if src == "alpaca":
            df = self.alpaca.fetch_1m_bars(
                symbol, start_date=start_date, end_date=end_date,
                days_back=days_back, feed=self.data_feed
            )
        elif src == "yfinance":
            df = self.yfinance.fetch_1m_bars(symbol, days_back=days_back)
        else:
            raise ValueError(f"Unknown data source: {src}")

        if not df.empty:
            self.storage.save_bars(symbol, df, interval="1m")
            if start_date is not None:
                s_dt = pd.to_datetime(start_date).tz_localize("America/New_York") if pd.to_datetime(start_date).tzinfo is None else pd.to_datetime(start_date)
                df = df[df.index >= s_dt]
            if end_date is not None:
                e_dt = pd.to_datetime(end_date).tz_localize("America/New_York") if pd.to_datetime(end_date).tzinfo is None else pd.to_datetime(end_date)
                df = df[df.index <= e_dt]

        return df

    def load_csv_file(self, symbol: str, file_path: str) -> pd.DataFrame:
        """Load a custom CSV file with columns datetime, open, high, low, close, volume."""
        df = pd.read_csv(file_path)
        dt_col = next((c for c in df.columns if c.lower() in ["datetime", "date", "time", "timestamp"]), None)
        if dt_col is None:
            raise ValueError("CSV must contain a datetime/date column.")
        
        df["datetime"] = pd.to_datetime(df[dt_col])
        df.set_index("datetime", inplace=True)
        
        if df.index.tz is None:
            df.index = df.index.tz_localize("America/New_York")
        else:
            df.index = df.index.tz_convert("America/New_York")

        df.columns = [c.lower() for c in df.columns]
        df.sort_index(inplace=True)
        self.storage.save_bars(symbol, df, interval="1m")
        return df

    def get_synchronized_pair(
        self,
        ticker_lead: str = "QQQ",
        ticker_target: str = "NVDA",
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
        days_back: int = 730,
        session_start: str = "09:30",
        session_end: str = "16:00",
        force_download: bool = False,
        source: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch, synchronize, and filter both lead and target to RTH session hours."""
        df_lead = self.get_symbol_bars(ticker_lead, start_date=start_date, end_date=end_date, days_back=days_back, force_download=force_download, source=source)
        df_target = self.get_symbol_bars(ticker_target, start_date=start_date, end_date=end_date, days_back=days_back, force_download=force_download, source=source)

        if df_lead.empty or df_target.empty:
            raise ValueError(f"Failed to load data for {ticker_lead} or {ticker_target}.")

        # Convert to America/New_York if not already
        for df in [df_lead, df_target]:
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
            else:
                df.index = df.index.tz_convert("America/New_York")

        # Filter to RTH (Regular Trading Hours)
        t_start = datetime.strptime(session_start, "%H:%M").time()
        t_end = datetime.strptime(session_end, "%H:%M").time()

        # Alpaca timestamps bars by their opening minute. The upper bound is
        # exclusive; the saved calendar also handles 13:00 official half-days.
        lead_rth = self._filter_official_rth(df_lead, session_start, session_end)
        target_rth = self._filter_official_rth(df_target, session_start, session_end)

        # Synchronize indices: inner join to keep only matching timestamps
        common_index = lead_rth.index.intersection(target_rth.index)
        lead_sync = lead_rth.loc[common_index].copy()
        target_sync = target_rth.loc[common_index].copy()

        # Add helper date/time columns
        lead_sync["session_date"] = lead_sync.index.date
        lead_sync["time_str"] = lead_sync.index.strftime("%H:%M")
        
        target_sync["session_date"] = target_sync.index.date
        target_sync["time_str"] = target_sync.index.strftime("%H:%M")

        return lead_sync, target_sync
