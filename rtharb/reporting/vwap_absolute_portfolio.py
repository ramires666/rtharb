"""Build the interactive portfolio report from audited stage-3 outputs.

The browser loads a small ``manifest.js`` first and one variant JSON on demand.
Required research layout is documented in ``vwap_absolute_portfolio`` research:
three ``variants/<name>`` directories, root daily constituent P&L, and exact
Pearson/Spearman matrices for every split.  A few harmless column aliases are
accepted, but raw combined MTM equity and its provenance are mandatory.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "research_output" / "vwap_absolute_portfolio"
OUT = ROOT / "tradingview_vwap_absolute_portfolio"
DATA_OUT = OUT / "data"
VARIANTS = ("equal_allocation", "shared_cap", "uncapped_diagnostic")
VARIANT_LABELS = {
    "equal_allocation": "Equal allocation · $11,111 на акцию",
    "shared_cap": "Shared cap · общий лимит $100k",
    "uncapped_diagnostic": "Uncapped diagnostic · leverage до 179%",
}
SPLITS = ("development", "validation", "holdout", "full")
SYMBOLS = ("NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "AMD")
NY = "America/New_York"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source(path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha(path), "bytes": path.stat().st_size}


def _number(value: Any) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    value = float(value)
    return value if math.isfinite(value) else None


def _epoch(value: Any, *, date_close: bool = False) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        if date_close and timestamp.hour == timestamp.minute == timestamp.second == 0:
            timestamp += pd.Timedelta(hours=16)
        timestamp = timestamp.tz_localize(NY)
    return int(timestamp.tz_convert("UTC").timestamp())


def _first(paths: Iterable[Path], label: str) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Не найден {label}: " + ", ".join(map(str, paths)))


def _variant_file(variant: str, kind: str) -> Path:
    folder = SRC / "variants" / variant
    aliases = {
        "summary": ("summary.json",), "equity": ("equity.csv", "combined_equity.csv"),
        "daily": ("daily_equity.csv",), "trades": ("trades.csv", "portfolio_trades.csv"),
        "audit": ("audit.json",),
    }[kind]
    return _first((folder / name for name in aliases), f"{variant}/{kind}")


def _column(frame: pd.DataFrame, *names: str, required: bool = True, default: float = 0.0) -> pd.Series:
    for name in names:
        if name in frame:
            return frame[name]
    if required:
        raise KeyError(f"Нет обязательной колонки; ожидалась одна из {names}")
    return pd.Series(default, index=frame.index)


def _read_time_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    time_name = next((name for name in ("timestamp", "time", "datetime") if name in frame), None)
    if time_name is None:
        raise KeyError(f"{path}: нет timestamp")
    frame[time_name] = pd.to_datetime(frame[time_name], format="mixed", utc=True).dt.tz_convert(NY)
    if time_name != "timestamp":
        frame = frame.rename(columns={time_name: "timestamp"})
    if frame.timestamp.duplicated().any() or not frame.timestamp.is_monotonic_increasing:
        raise AssertionError(f"{path}: timestamps должны быть unique/sorted")
    return frame


def _periods(summary: dict) -> dict:
    raw = summary.get("periods", summary.get("results", summary.get("selected_results", {})))
    if not isinstance(raw, dict) or not all(split in raw for split in SPLITS):
        raise KeyError("Variant summary обязан содержать periods development/validation/holdout/full")
    return raw


def _split_dates(summary: dict, manifest: dict, equity: pd.DataFrame) -> dict:
    explicit = summary.get("splits", manifest.get("splits", {}))
    if not explicit:
        period_metrics = _periods(summary)
        if all(period_metrics.get(split, {}).get("start") and period_metrics.get(split, {}).get("end")
               for split in SPLITS[:-1]):
            explicit = {split: {"start": period_metrics[split]["start"],
                                "end": period_metrics[split]["end"],
                                "sessions": period_metrics[split]["sessions"]}
                        for split in SPLITS[:-1]}
    if isinstance(explicit, dict) and all(split in explicit for split in SPLITS[:-1]):
        result = {}
        for split in SPLITS[:-1]:
            item = explicit[split]
            start, end = pd.Timestamp(item["start"]).date(), pd.Timestamp(item["end"]).date()
            subset = equity[(equity.timestamp.dt.date >= start) & (equity.timestamp.dt.date <= end)]
            if subset.empty:
                raise AssertionError(f"{split}: split range не пересекает equity")
            result[split] = {**item, "start_epoch": _epoch(subset.timestamp.iloc[0]),
                             "end_epoch": _epoch(subset.timestamp.iloc[-1])}
        return result
    # Canonical research has fixed 125/63/63 RTH sessions.  Infer boundaries
    # only when the summary omitted cosmetic split dates.
    dates = list(pd.unique(equity.timestamp.dt.date))
    counts = [int(_periods(summary)[name]["sessions"]) for name in SPLITS[:-1]]
    if sum(counts) != len(dates):
        raise AssertionError("Нельзя однозначно восстановить split dates из sessions")
    result, cursor = {}, 0
    for split, count in zip(SPLITS[:-1], counts):
        selected = dates[cursor:cursor + count]
        subset = equity[equity.timestamp.dt.date.astype(str).isin(map(str, selected))]
        result[split] = {"start": str(selected[0]), "end": str(selected[-1]), "sessions": count,
                         "start_epoch": _epoch(subset.timestamp.iloc[0]), "end_epoch": _epoch(subset.timestamp.iloc[-1])}
        cursor += count
    return result


def _metric_block(raw: dict) -> dict:
    aliases = {"return_pct": ("return_pct", "net_return_pct"),
               "net_sharpe": ("net_sharpe", "sharpe"),
               "net_sortino": ("net_sortino", "sortino")}
    out = dict(raw)
    for target, names in aliases.items():
        out[target] = next((raw[name] for name in names if raw.get(name) is not None), 0.0)
    for key in ("sessions", "trades"):
        out[key] = int(raw.get(key, 0))
    for key in ("gross_pnl", "commissions", "slippage", "costs", "net_pnl", "return_pct",
                "win_rate_pct", "profit_factor", "net_sharpe", "net_sortino",
                "max_drawdown_usd_mtm", "max_drawdown_pct_mtm", "final_equity_rebased"):
        out[key] = float(out.get(key, 0.0))
    if not math.isclose(out["commissions"] + out["slippage"], out["costs"], abs_tol=1e-6):
        raise AssertionError("Portfolio costs != commission + slippage")
    if not math.isclose(out["gross_pnl"] - out["costs"], out["net_pnl"], abs_tol=1e-6):
        raise AssertionError("Portfolio gross - costs != net")
    return out


def _matrix(path: Path) -> tuple[list[str], list[list[float | None]]]:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str).str.upper()
    frame.columns = frame.columns.astype(str).str.upper()
    names = [symbol for symbol in SYMBOLS if symbol in frame.index and symbol in frame.columns]
    if not names:
        raise AssertionError(f"{path}: пустая correlation matrix")
    square = frame.loc[names, names].to_numpy(float)
    if square.shape[0] != square.shape[1] or not np.allclose(square, square.T, atol=1e-8, equal_nan=True):
        raise AssertionError(f"{path}: correlation matrix не symmetric")
    return names, [[_number(value) for value in row] for row in square]


def _constituent_curves(variant: str, trades_path: Path, daily_path: Path,
                        capital: float) -> tuple[list[int], dict[str, list[float]]]:
    """Build accepted-trade constituent contribution curves on daily closes."""
    daily = pd.read_csv(daily_path)
    date_col = next((name for name in ("date", "session_date", "timestamp") if name in daily), None)
    if date_col is None:
        raise KeyError(f"{daily_path}: нет date")
    dates = pd.to_datetime(daily[date_col]).dt.date
    trades = pd.read_csv(trades_path)
    symbol_col = next((name for name in ("symbol", "target", "ticker") if name in trades), None)
    exit_col = next((name for name in ("exit_time", "timestamp", "date") if name in trades), None)
    pnl_col = next((name for name in ("net_pnl", "pnl_net") if name in trades), None)
    if symbol_col and exit_col and pnl_col:
        trades[exit_col] = pd.to_datetime(trades[exit_col], format="mixed", utc=True).dt.tz_convert(NY)
        trades["_date"] = trades[exit_col].dt.date
        grouped = trades.groupby(["_date", trades[symbol_col].astype(str).str.upper()])[pnl_col].sum()
        curves = {}
        for symbol in SYMBOLS:
            values = np.asarray([float(grouped.get((date, symbol), 0.0)) for date in dates])
            curves[symbol] = [float(value) for value in capital + np.cumsum(values)]
    else:
        # Tolerant fallback for a wide daily file, used only if it publishes
        # symbol-level accepted net P&L explicitly.
        curves = {}
        for symbol in SYMBOLS:
            candidates = (symbol, f"{symbol}_net_pnl", f"net_pnl_{symbol}")
            column = next((name for name in candidates if name in daily), None)
            if column:
                curves[symbol] = [float(value) for value in capital + daily[column].fillna(0).cumsum()]
        if not curves:
            raise KeyError(f"{variant}: нет symbol/net_pnl в trades и нет constituent columns в daily equity")
    return [_epoch(date, date_close=True) for date in dates], curves


def _build_variant(variant: str, manifest: dict, correlations: dict) -> tuple[dict, dict]:
    summary_path, equity_path = _variant_file(variant, "summary"), _variant_file(variant, "equity")
    daily_path, trades_path, audit_path = (_variant_file(variant, name) for name in ("daily", "trades", "audit"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if str(audit.get("status", "PASS")).upper() != "PASS":
        raise AssertionError(f"{variant}: audit status != PASS")
    equity = _read_time_csv(equity_path)
    capital_model = summary.get("capital_model", {})
    capital = float(capital_model.get("starting_capital_usd", capital_model.get("capital_usd", 100_000.0)))
    eq = _column(equity, "equity", "portfolio_equity").to_numpy(float)
    peak = _column(equity, "running_peak", required=False, default=np.nan).to_numpy(float)
    if np.isnan(peak).all(): peak = np.maximum.accumulate(eq)
    dd = _column(equity, "drawdown_usd", required=False, default=np.nan).to_numpy(float)
    if np.isnan(dd).all(): dd = peak - eq
    dd_pct = _column(equity, "drawdown_pct", required=False, default=np.nan).to_numpy(float)
    if np.isnan(dd_pct).all(): dd_pct = np.divide(dd, peak, out=np.zeros_like(dd), where=peak != 0) * 100
    arrays = {
        "t": [_epoch(value) for value in equity.timestamp], "equity": [_number(x) for x in eq],
        "drawdown": [_number(x) for x in dd], "drawdown_pct": [_number(x) for x in dd_pct],
        "active_positions": [_number(x) for x in _column(equity, "active_positions", "concurrent_positions")],
        "gross_entry_exposure": [_number(x) for x in _column(equity, "gross_entry_exposure", required=False)],
        "gross_mtm_exposure": [_number(x) for x in _column(equity, "gross_mtm_exposure", "gross_exposure_usd")],
        "signed_mtm_exposure": [_number(x) for x in _column(equity, "signed_mtm_exposure", required=False)],
        "utilization_pct": [_number(x) for x in _column(equity, "utilization_pct", "capital_utilization_pct")],
    }
    if set(map(len, arrays.values())) != {len(equity)}:
        raise AssertionError(f"{variant}: combined MTM arrays не aligned")
    periods = {split: _metric_block(_periods(summary)[split]) for split in SPLITS}
    if not math.isclose(eq[-1], periods["full"]["final_equity_rebased"], abs_tol=1e-6):
        raise AssertionError(f"{variant}: final minute equity != full final_equity_rebased")
    constituent_t, constituents = _constituent_curves(variant, trades_path, daily_path, capital)
    splits = _split_dates(summary, manifest, equity)
    payload = {
        "meta": {"schema_version": 1, "variant": variant,
                 "label": summary.get("label", VARIANT_LABELS[variant]),
                 "lead": "QQQ", "targets": list(constituents), "capital_model": capital_model,
                 "global_calendar": manifest.get("global_calendar", {}),
                 "execution": summary.get("execution", {}), "admission_statistics": summary.get("admission_statistics", {}),
                 "exposure_statistics": summary.get("exposure_statistics", {}), "splits": splits,
                 "source": "Exact combined raw-minute MTM from admitted event-driven target trades; QQQ is reference only and never a portfolio leg",
                 "warning": "Portfolio construction and three capital variants add another multiple-testing layer; holdout is diagnostic, not proof of live profitability. Uncapped diagnostic is a common-calendar replay, not an exact sum of standalone pairwise histories.",
                 "sources": {"manifest": _source(SRC / "manifest.json"), "summary": _source(summary_path),
                             "equity": _source(equity_path), "daily_equity": _source(daily_path),
                             "trades": _source(trades_path), "audit": _source(audit_path)}},
        "bars": arrays, "constituents": {"t": constituent_t, "equity": constituents},
        "correlations": correlations, "results": periods,
    }
    full, holdout = periods["full"], periods["holdout"]
    comparison = {"variant": variant, "label": payload["meta"]["label"], "data": f"data/{variant}.json",
                  "capital_model": capital_model, "trades": full["trades"], "net_pnl": full["net_pnl"],
                  "net_sharpe": full["net_sharpe"], "return_pct": full["return_pct"],
                  "max_drawdown_usd": full["max_drawdown_usd_mtm"], "max_drawdown_pct": full["max_drawdown_pct_mtm"],
                  "costs": full["costs"], "holdout_net_pnl": holdout["net_pnl"],
                  "holdout_sharpe": holdout["net_sharpe"], "holdout_max_drawdown_pct": holdout["max_drawdown_pct_mtm"],
                  "max_active_positions": max(x or 0 for x in arrays["active_positions"]),
                  "max_utilization_pct": max(x or 0 for x in arrays["utilization_pct"])}
    return payload, comparison


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
    manifest_path = SRC / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Сначала завершите portfolio research: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = str(manifest.get("status", "COMPLETE")).upper()
    if status != "COMPLETE":
        raise AssertionError(f"Portfolio research ещё не завершён: status={status}")
    variants = [name for name in VARIANTS if (SRC / "variants" / name / "summary.json").is_file()]
    if variants != list(VARIANTS):
        raise AssertionError(f"Ожидались варианты {VARIANTS}; готовы {variants}")
    correlations: dict[str, dict[str, Any]] = {}
    correlation_sources: dict[str, Any] = {}
    for split in SPLITS:
        p_path = _first((SRC / f"correlation_pearson_{split}.csv", SRC / f"pearson_{split}.csv"), f"Pearson {split}")
        s_path = _first((SRC / f"correlation_spearman_{split}.csv", SRC / f"spearman_{split}.csv"), f"Spearman {split}")
        p_names, pearson = _matrix(p_path); s_names, spearman = _matrix(s_path)
        if p_names != s_names: raise AssertionError(f"{split}: Pearson/Spearman symbols differ")
        correlations[split] = {"symbols": p_names, "pearson": pearson, "spearman": spearman}
        correlation_sources[split] = {"pearson": _source(p_path), "spearman": _source(s_path)}
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    assets = []
    for variant in variants:
        payload, item = _build_variant(variant, manifest, correlations)
        destination = DATA_OUT / f"{variant}.json"
        destination.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
        check = json.loads(destination.read_text(encoding="utf-8"))
        if len(check["bars"]["t"]) != len(check["bars"]["equity"]):
            raise AssertionError(f"{variant}: JSON readback failed")
        item.update({"bytes": destination.stat().st_size, "sha256": _sha(destination)})
        assets.append(item)
        print(f"BUILT {variant}: {item['trades']} trades, net {item['net_pnl']:+,.2f}, MTM DD {item['max_drawdown_usd']:,.2f}")
    report_manifest = {"schema_version": 1, "source": _source(manifest_path), "lead": "QQQ",
                       "targets": list(SYMBOLS), "variants": variants, "default_variant": "equal_allocation",
                       "global_calendar": manifest.get("global_calendar", {}),
                       "frozen_stop_target": manifest.get("frozen_stop_target", {}),
                       "assets": assets, "correlation_sources": correlation_sources,
                       "warning": "Exploratory portfolio combination across nine separately optimized strategies and three capital variants; multiple testing and survivorship bias remain."}
    compact = json.dumps(report_manifest, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.js").write_text("window.VWAP_PORTFOLIO_MANIFEST=" + compact + ";\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT / "index.html"), "variants": variants,
                      "lazy_payload_bytes": sum(x["bytes"] for x in assets)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
