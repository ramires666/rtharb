"""Local storage manager for caching intraday market data in Parquet format."""

import os
from pathlib import Path
import pandas as pd
from typing import Optional


class DataStorage:
    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, symbol: str, interval: str = "1m") -> Path:
        safe_sym = symbol.replace("=", "_").replace("^", "_").replace("/", "_")
        return self.cache_dir / f"{safe_sym}_{interval}.parquet"

    def save_bars(self, symbol: str, df: pd.DataFrame, interval: str = "1m") -> None:
        """Save or merge new bars into local parquet cache."""
        if df.empty:
            return
        
        file_path = self._get_file_path(symbol, interval)
        if file_path.exists():
            try:
                existing_df = pd.read_parquet(file_path)
                combined = pd.concat([existing_df, df])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined.sort_index(inplace=True)
                combined.to_parquet(file_path, engine="pyarrow", compression="snappy")
                return
            except Exception:
                pass
        
        df_to_save = df[~df.index.duplicated(keep="last")].sort_index()
        df_to_save.to_parquet(file_path, engine="pyarrow", compression="snappy")

    def load_bars(self, symbol: str, interval: str = "1m") -> Optional[pd.DataFrame]:
        """Load bars from local cache if exists."""
        file_path = self._get_file_path(symbol, interval)
        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                if not df.empty:
                    return df
            except Exception:
                return None
        return None

    def clear_cache(self, symbol: Optional[str] = None, interval: str = "1m") -> None:
        """Clear cache for specific symbol or all symbols."""
        if symbol:
            file_path = self._get_file_path(symbol, interval)
            if file_path.exists():
                file_path.unlink()
        else:
            for p in self.cache_dir.glob("*.parquet"):
                p.unlink()
