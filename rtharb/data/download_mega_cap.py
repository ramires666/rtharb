"""Download and audit the frozen mega-cap universe from Alpaca SIP."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rtharb.data.loader import DataLoader


ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data_cache"
MANIFEST = CACHE / "mega_cap_sip_manifest.json"
LEAD = "QQQ"
UNIVERSE = ("NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "AMD")
PREHISTORY_START = "2024-08-22"
DOWNLOAD_END = "2026-08-22"
STUDY_START = pd.Timestamp("2025-08-22").date()
STUDY_END = pd.Timestamp("2026-08-21").date()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_frame(symbol: str, frame: pd.DataFrame) -> dict:
    if frame.empty:
        raise AssertionError(f"{symbol}: empty cache")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise AssertionError(f"{symbol}: timestamps must be timezone-aware")
    frame = frame.sort_index()
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise AssertionError(f"{symbol}: missing OHLCV columns {sorted(missing)}")
    o, h, l, c = (frame[name].to_numpy(float) for name in ("open", "high", "low", "close"))
    invalid_ohlc = int(np.sum((h < np.maximum.reduce([o, l, c])) | (l > np.minimum.reduce([o, h, c]))))
    duplicates = int(frame.index.duplicated().sum())
    nonpositive = int(np.sum((o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)))
    if invalid_ohlc or duplicates or nonpositive:
        raise AssertionError(
            f"{symbol}: invalid_ohlc={invalid_ohlc}, duplicates={duplicates}, nonpositive={nonpositive}"
        )
    return {
        "rows": int(len(frame)),
        "first_timestamp": frame.index[0].isoformat(),
        "last_timestamp": frame.index[-1].isoformat(),
        "duplicate_timestamps": duplicates,
        "invalid_ohlc_rows": invalid_ohlc,
        "nonpositive_ohlc_rows": nonpositive,
    }


def build_manifest(loader: DataLoader) -> dict:
    calendar = pd.read_csv(CACHE / "market_calendar.csv", dtype=str)
    calendar = calendar[
        (pd.to_datetime(calendar["date"]).dt.date >= STUDY_START)
        & (pd.to_datetime(calendar["date"]).dt.date <= STUDY_END)
    ].copy()
    expected_by_date = {
        row.date: int((int(row.close[:2]) * 60 + int(row.close[3:5]))
                      - (int(row.open[:2]) * 60 + int(row.open[3:5])))
        for row in calendar.itertuples(index=False)
    }
    qqq = loader.storage.load_bars(LEAD, "1m")
    if qqq is None:
        raise FileNotFoundError(CACHE / "QQQ_1m.parquet")
    qqq = loader._filter_official_rth(qqq, "09:30", "16:00")
    qqq = qqq[(qqq.index.date >= STUDY_START) & (qqq.index.date <= STUDY_END)]

    symbols = {}
    for symbol in (LEAD, *UNIVERSE):
        path = CACHE / f"{symbol}_1m.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        stats = validate_frame(symbol, frame)
        rth = loader._filter_official_rth(frame, "09:30", "16:00")
        rth = rth[(rth.index.date >= STUDY_START) & (rth.index.date <= STUDY_END)]
        counts = pd.Series(1, index=rth.index).groupby(rth.index.strftime("%Y-%m-%d")).sum()
        expected = pd.Series(expected_by_date, dtype=float)
        aligned = counts.reindex(expected.index, fill_value=0)
        pair_rows = len(rth.index.intersection(qqq.index))
        expected_rows = int(expected.sum())
        stats.update({
            "file": str(path.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(path),
            "bytes": int(path.stat().st_size),
            "study_rth_rows": int(len(rth)),
            "study_sessions_present": int((aligned > 0).sum()),
            "study_sessions_expected": int(len(expected)),
            "study_expected_rth_rows": expected_rows,
            "study_rth_coverage_pct": float(100 * len(rth) / expected_rows),
            "missing_official_minutes": int(np.maximum(expected.to_numpy() - aligned.to_numpy(), 0).sum()),
            "pairwise_rows_with_qqq": int(pair_rows),
            "pairwise_coverage_pct": float(100 * pair_rows / expected_rows),
        })
        symbols[symbol] = stats

    return {
        "schema_version": 1,
        "frozen_universe": list(UNIVERSE),
        "lead": LEAD,
        "roles": "QQQ is reference only; each target stock is traded separately",
        "universe_policy": (
            "Exploratory fixed mega-cap technology universe declared before per-symbol P&L inspection; "
            "GOOGL is the only Alphabet share class. Existing synthetic report previously used only "
            "MSFT/AAPL/NVDA/AMZN, not nine symbols."
        ),
        "survivorship_warning": "Current mega-cap membership may create survivorship/look-ahead bias.",
        "provider": "Alpaca",
        "feed": "SIP",
        "timeframe": "1 minute",
        "adjustment": "RAW",
        "download_request": {"start": PREHISTORY_START, "end_exclusive": DOWNLOAD_END},
        "study_period": {"start": str(STUDY_START), "end": str(STUDY_END)},
        "pairwise_intersection": True,
        "no_resampling_or_fill": True,
        "symbols": symbols,
    }


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--force-all", action="store_true")
    args = parser.parse_args()
    loader = DataLoader(str(CACHE), "alpaca", "sip")
    for symbol in (LEAD, *UNIVERSE):
        path = CACHE / f"{symbol}_1m.parquet"
        should_download = args.force_all or (args.download_missing and not path.is_file())
        if should_download:
            print(f"DOWNLOAD {symbol}: Alpaca SIP RAW {PREHISTORY_START}..{DOWNLOAD_END}", flush=True)
            frame = loader.get_symbol_bars(
                symbol, start_date=PREHISTORY_START, end_date=DOWNLOAD_END,
                force_download=True, source="alpaca",
            )
            print(f"SAVED {symbol}: {len(frame):,} requested-range rows", flush=True)
    manifest = build_manifest(loader)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS {len(UNIVERSE)} targets; manifest {MANIFEST}")
    for symbol in UNIVERSE:
        item = manifest["symbols"][symbol]
        print(
            f"{symbol}: raw={item['rows']:,}, RTH={item['study_rth_rows']:,}, "
            f"pairwise QQQ={item['pairwise_rows_with_qqq']:,}, coverage={item['pairwise_coverage_pct']:.3f}%"
        )


if __name__ == "__main__":
    main()
