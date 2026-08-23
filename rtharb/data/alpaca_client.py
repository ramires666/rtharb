"""Alpaca Markets client for downloading multi-year 1-minute historical intraday data."""

import os
from datetime import datetime, timedelta
from typing import Optional, List, Any
import pandas as pd
import pytz
from dotenv import load_dotenv

load_dotenv()


class AlpacaClient:
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not self.api_key or not self.secret_key:
                raise ValueError(
                    "Alpaca API credentials missing. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file."
                )
            from alpaca.data.historical import StockHistoricalDataClient
            self._client = StockHistoricalDataClient(self.api_key, self.secret_key)
        return self._client

    def fetch_1m_bars(
        self,
        symbol: str,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
        days_back: int = 730, # fallback if start_date not provided
        feed: str = "sip",
    ) -> pd.DataFrame:
        """Fetch 1-minute historical bars for a symbol from Alpaca."""
        client = self._get_client()
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed

        if end_date is None:
            end_date = datetime.now(pytz.UTC)
        elif isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).tz_localize("America/New_York").tz_convert("UTC")
        elif getattr(end_date, "tzinfo", None) is None:
            end_date = pytz.timezone("America/New_York").localize(end_date).astimezone(pytz.UTC)

        if start_date is None:
            start_date = end_date - timedelta(days=days_back)
        elif isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).tz_localize("America/New_York").tz_convert("UTC")
        elif getattr(start_date, "tzinfo", None) is None:
            start_date = pytz.timezone("America/New_York").localize(start_date).astimezone(pytz.UTC)

        feed_name = feed.lower()
        if feed_name not in {"sip", "iex"}:
            raise ValueError(f"Unsupported Alpaca stock feed: {feed}")

        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start_date,
            end=end_date,
            feed=DataFeed.SIP if feed_name == "sip" else DataFeed.IEX
        )

        bars = client.get_stock_bars(req)
        df = bars.df

        if df.empty:
            return pd.DataFrame()

        # Handle MultiIndex ('symbol', 'timestamp') if present
        if isinstance(df.index, pd.MultiIndex):
            if "symbol" in df.index.names:
                df = df.xs(symbol, level="symbol") if symbol in df.index.get_level_values("symbol") else df.droplevel(0)

        # Standardize column names to lowercase
        df.columns = [c.lower() for c in df.columns]
        
        # Ensure DateTime index with timezone
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        
        df.index = df.index.tz_convert("America/New_York")
        df.index.name = "datetime"
        df.sort_index(inplace=True)
        return df
