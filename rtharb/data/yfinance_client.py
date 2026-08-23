"""Yahoo Finance client for downloading 1-minute historical data (up to 30 days)."""

from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import pytz
import yfinance as yf


class YFinanceClient:
    def fetch_1m_bars(
        self,
        symbol: str,
        days_back: int = 7 # max 30 days for 1m
    ) -> pd.DataFrame:
        """Fetch 1-minute historical bars from Yahoo Finance."""
        days = min(days_back, 29)
        period = f"{days}d"
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval="1m", auto_adjust=False, prepost=False)

        if df.empty:
            return pd.DataFrame()

        df.columns = [c.lower() for c in df.columns]
        
        # Ensure timezone America/New_York
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        
        df.index = df.index.tz_convert("America/New_York")
        df.index.name = "datetime"
        df.sort_index(inplace=True)
        return df
