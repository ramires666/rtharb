"""Build the detailed QQQ-vs-synthetic-VWAP report from exact raw SIP bars.

QQQ is the only traded instrument.  MSFT/AAPL/NVDA/AMZN are reference
constituents whose causal session-VWAP deviations form fair QQQ.  The builder
publishes one lazy JSON per frozen normal/reverse × convergence/bracket variant;
it never manufactures a synthetic OHLC candle series.
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

from rtharb.research.synthetic_index import load_raw


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "research_output" / "synthetic_vwap_absolute"
OUT = ROOT / "tradingview_synthetic_vwap_absolute"
DATA_OUT = OUT / "data"
VARIANTS = ("normal_convergence", "reverse_convergence", "normal_dollar_bracket", "reverse_dollar_bracket")
SPLITS = ("development", "validation", "holdout", "full")
CONSTITUENTS = ("MSFT", "AAPL", "NVDA", "AMZN")
OFFICIAL_WEIGHTS_PCT = {"MSFT": 8.6, "AAPL": 8.4, "NVDA": 7.9, "AMZN": 5.2}
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
    result = float(value)
    return result if math.isfinite(result) else None


def _vector(values: Any) -> list[Any]:
    """Convert a numeric vector to strict-JSON scalars without per-cell Python work."""
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.integer):
        return array.tolist()
    array = array.astype(float, copy=False)
    if np.isfinite(array).all():
        return array.tolist()
    return [None if not math.isfinite(float(value)) else float(value) for value in array]


def _epoch(value: Any) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(NY)
    return int(timestamp.tz_convert("UTC").timestamp())


def _first(paths: Iterable[Path], label: str) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Не найден {label}: " + ", ".join(map(str, paths)))


def _variant_file(variant: str, kind: str, split: str | None = None) -> Path:
    folder = SRC / "variants" / variant
    if kind in {"summary", "audit"}:
        return _first((folder / f"{kind}.json", SRC / variant / f"{kind}.json"), f"{variant}/{kind}")
    if split is None:
        raise ValueError("split required")
    noun = "trades" if kind == "trades" else "equity"
    names = (f"selected_{split}_{noun}.csv", f"{split}_{noun}.csv")
    return _first(tuple(folder / name for name in names) + tuple(SRC / variant / name for name in names),
                  f"{variant}/{split}/{noun}")


def _session_slices(day_code: np.ndarray) -> list[tuple[int, int]]:
    starts = np.r_[0, np.flatnonzero(np.diff(day_code)) + 1]
    ends = np.r_[starts[1:], len(day_code)]
    return list(zip(starts.tolist(), ends.tolist()))


def _session_vwap(frame: pd.DataFrame, common: pd.DatetimeIndex, day_code: np.ndarray) -> np.ndarray:
    frame = frame.loc[common]
    typical = (frame.high.to_numpy(float) + frame.low.to_numpy(float) + frame.close.to_numpy(float)) / 3.0
    volume = frame.volume.to_numpy(float)
    result = np.full(len(common), np.nan)
    for low, high in _session_slices(day_code):
        cumulative_volume = np.cumsum(volume[low:high])
        result[low:high] = np.divide(np.cumsum(typical[low:high] * volume[low:high]), cumulative_volume,
                                     out=np.full(high - low, np.nan), where=cumulative_volume > 0)
    return result


def _rolling_z(residual: np.ndarray, day_code: np.ndarray, window: int) -> np.ndarray:
    z = np.full(len(residual), np.nan)
    for low_bound, high_bound in _session_slices(day_code):
        values = residual[low_bound:high_bound]
        cumulative = np.r_[0.0, np.cumsum(values)]
        cumulative_sq = np.r_[0.0, np.cumsum(values * values)]
        for local in range(window - 1, len(values)):
            low, high = local - window + 1, local + 1
            total = cumulative[high] - cumulative[low]
            total_sq = cumulative_sq[high] - cumulative_sq[low]
            mean = total / window
            variance = (total_sq - total * total / window) / (window - 1)
            std = math.sqrt(max(variance, 0.0))
            if std > 1e-12:
                z[low_bound + local] = (values[local] - mean) / std
    return z


def _market(manifest: dict) -> tuple[pd.DatetimeIndex, dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    # Reuse the research project's read-only raw loader so the report and engine
    # share the identical exchange calendar, period bounds, and five-way inner
    # intersection.  This still reads the cached Alpaca SIP parquets directly.
    frames, common, day_code = load_raw()
    period = manifest.get("period", manifest.get("study_period", {}))
    if common[0].date() != pd.Timestamp(period["start"]).date() or common[-1].date() != pd.Timestamp(period["end"]).date():
        raise AssertionError("Raw loader period differs from completed manifest")
    if common.has_duplicates or not common.is_monotonic_increasing or not len(common):
        raise AssertionError("Invalid five-way common raw calendar")
    expected = manifest.get("data", manifest).get("bars", manifest.get("period", {}).get("raw_bars",
               manifest.get("period", {}).get("raw_minutes")))
    if expected is not None and len(common) != int(expected):
        raise AssertionError(f"Raw common bars {len(common)} != manifest {expected}")
    print("PHASE causal VWAP/fair arrays", flush=True)
    vwap = {symbol: _session_vwap(frame, common, day_code) for symbol, frame in frames.items()}
    normalized = {symbol: OFFICIAL_WEIGHTS_PCT[symbol] / sum(OFFICIAL_WEIGHTS_PCT.values()) for symbol in CONSTITUENTS}
    deviations = {symbol: frames[symbol].close.to_numpy(float) / vwap[symbol] - 1.0 for symbol in CONSTITUENTS}
    basket_dev = sum(normalized[symbol] * deviations[symbol] for symbol in CONSTITUENTS)
    fair = vwap["QQQ"] * (1.0 + basket_dev)
    residual = (frames["QQQ"].close.to_numpy(float) - fair) / fair
    arrays = {"qvwap": vwap["QQQ"], "fair": fair, "residual": residual, "basket_dev": basket_dev,
              "_day_code": day_code, "_t": common.as_unit("s").asi8.astype(np.int64),
              **{f"dev_{symbol}": deviations[symbol] for symbol in CONSTITUENTS}}
    if int(arrays["_t"][0]) != _epoch(common[0]) or int(arrays["_t"][-1]) != _epoch(common[-1]):
        raise AssertionError("Unit-safe epoch conversion failed at raw calendar boundaries")
    return common, frames, arrays


def _parse_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ("timestamp", "signal_time", "entry_signal_time", "entry_time", "exit_time"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], format="mixed", utc=True).dt.tz_convert(NY)
    return frame


def _col(row: pd.Series, *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row.index and not pd.isna(row[name]):
            return row[name]
    return default


def _parameters(summary: dict) -> dict:
    raw = summary.get("signal_parameters", summary.get("parameters", {}))
    selected = summary.get("selected", {})
    return {
        "window": int(selected.get("window", raw.get("window", raw.get("z_window", 60)))),
        "z_entry": float(selected.get("z_entry", raw.get("z_entry", 2.5))),
        "hook_delta": float(selected.get("hook_delta", raw.get("hook_delta", 0.0))),
        "hook_timeout": selected.get("hook_timeout", raw.get("hook_timeout")),
        "exit_residual": float(selected.get("exit_residual", raw.get("exit_residual", 0.0))),
    }


def _split_for(timestamp: pd.Timestamp, summary: dict) -> str:
    for split in SPLITS[:-1]:
        item = summary.get("splits", {}).get(split, {})
        if item.get("start") and item.get("end"):
            if pd.Timestamp(item["start"]).date() <= timestamp.date() <= pd.Timestamp(item["end"]).date():
                return split
    return "full"


def _trades(frame: pd.DataFrame, variant: str, summary: dict) -> list[dict]:
    items = []
    for i, row in frame.iterrows():
        signal_time = _col(row, "signal_time", "entry_signal_time")
        entry_time, exit_time = _col(row, "entry_time"), _col(row, "exit_time")
        direction_text = str(_col(row, "direction", "side", default="LONG")).upper()
        direction = 1 if direction_text in {"LONG", "BUY", "1", "+1"} else -1
        entry_ref = float(_col(row, "entry_reference", "entry_raw", "raw_entry", "entry_price"))
        exit_ref = float(_col(row, "exit_reference", "exit_raw", "raw_exit", "exit_price"))
        entry_price = float(_col(row, "entry_price", "entry_effective", default=entry_ref))
        exit_price = float(_col(row, "exit_price", "exit_effective", default=exit_ref))
        shares = int(_col(row, "shares", "qty", default=0))
        commissions = float(_col(row, "commissions", "commission", default=0.0))
        slippage = float(_col(row, "slippage", "slippage_cost", default=0.0))
        costs = float(_col(row, "costs", default=commissions + slippage))
        gross = float(_col(row, "gross_pnl", default=direction * (exit_ref - entry_ref) * shares))
        net = float(_col(row, "net_pnl", default=gross - costs))
        stop_usd = _col(row, "stop_usd_per_share", "stop_usd")
        target_usd = _col(row, "target_usd_per_share", "target_usd")
        stop_price, target_price = _col(row, "stop_price"), _col(row, "target_price")
        entry_timestamp = pd.Timestamp(entry_time)
        items.append({
            "id": i + 1, "variant": variant, "split": _split_for(entry_timestamp, summary),
            "side": "LONG" if direction == 1 else "SHORT", "direction": direction,
            "entry_signal_time": _epoch(signal_time), "entry_time": _epoch(entry_time), "exit_time": _epoch(exit_time),
            "signal_z": _number(_col(row, "signal_z", "entry_z", "z")),
            "source_sign": _number(_col(row, "source_sign", "armed_sign")),
            "signal_residual": _number(_col(row, "signal_residual", "residual")),
            "signal_fair_qqq": _number(_col(row, "signal_fair_qqq", "fair_qqq")),
            "signal_qqq_vwap": _number(_col(row, "signal_qqq_vwap", "qqq_vwap")),
            "signal_qqq_close": _number(_col(row, "signal_qqq_close", "qqq_close")),
            "armed_z": _number(_col(row, "armed_z")), "extreme_z": _number(_col(row, "extreme_z")),
            "hook_delta": _number(_col(row, "hook_delta")), "hook_bars": int(_col(row, "hook_bars", default=0)),
            "entry_reference": entry_ref, "entry_price": entry_price,
            "exit_reference": exit_ref, "exit_price": exit_price, "shares": shares,
            "stop_usd_per_share": _number(stop_usd), "target_usd_per_share": _number(target_usd),
            "stop_price": _number(stop_price), "target_price": _number(target_price),
            "risk_reward_ratio": _number(_col(row, "risk_reward_ratio", "reward_to_risk",
                                               default=float(target_usd) / float(stop_usd) if stop_usd else None)),
            "gross_risk_usd": _number(_col(row, "gross_risk_usd",
                                            default=float(stop_usd) * shares if stop_usd else None)),
            "gross_reward_usd": _number(_col(row, "gross_reward_usd",
                                              default=float(target_usd) * shares if target_usd else None)),
            "exit_signal_residual": _number(_col(row, "exit_signal_residual", "exit_residual")),
            "exit_signal_time": None if _col(row, "exit_signal_time") is None else _epoch(_col(row, "exit_signal_time")),
            "exit_reason": str(_col(row, "exit_reason", default="UNKNOWN")),
            "duration_minutes": int(_col(row, "duration_minutes", "duration_bars", default=0)),
            "gross_pnl": gross, "commissions": commissions, "slippage": slippage, "costs": costs, "net_pnl": net,
        })
    return items


def _results(summary: dict) -> dict:
    raw = summary.get("selected_results", summary.get("results", {}))
    if not all(split in raw for split in SPLITS):
        raise KeyError("summary missing four selected_results splits")
    out = {}
    for split in SPLITS:
        item = dict(raw[split])
        for key in ("sessions", "trades", "stops", "targets", "forced_eod", "convergence_exits"):
            item[key] = int(item.get(key, 0))
        for key in ("gross_pnl", "commissions", "slippage", "costs", "net_pnl", "net_return_pct", "return_pct",
                    "win_rate_pct", "profit_factor", "net_sharpe", "net_sortino", "max_drawdown_usd_mtm",
                    "max_drawdown_pct_mtm", "final_equity"):
            if key in item: item[key] = float(item[key])
        item["return_pct"] = float(item.get("return_pct", item.get("net_return_pct", 0.0)))
        if not math.isclose(float(item.get("commissions", 0)) + float(item.get("slippage", 0)),
                            float(item.get("costs", 0)), abs_tol=1e-6):
            raise AssertionError(f"{split}: commission + slippage != costs")
        out[split] = item
    return out


def _split_meta(summary: dict, common: pd.DatetimeIndex) -> dict:
    result = {}
    for split in SPLITS[:-1]:
        item = summary.get("splits", {}).get(split, {})
        start, end = pd.Timestamp(item["start"]).date(), pd.Timestamp(item["end"]).date()
        times = common[(common.date >= start) & (common.date <= end)]
        result[split] = {**item, "start_epoch": _epoch(times[0]), "end_epoch": _epoch(times[-1])}
    return result


def _build_variant(variant: str, manifest: dict, common: pd.DatetimeIndex,
                   frames: dict[str, pd.DataFrame], common_arrays: dict[str, np.ndarray]) -> tuple[dict, dict]:
    summary_path, audit_path = _variant_file(variant, "summary"), _variant_file(variant, "audit")
    trades_path, equity_path = _variant_file(variant, "trades", "full"), _variant_file(variant, "equity", "full")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if str(audit.get("status", summary.get("audit", {}).get("status", "PASS"))).upper() != "PASS":
        raise AssertionError(f"{variant}: audit failed")
    params = _parameters(summary)
    z = _rolling_z(common_arrays["residual"], common_arrays["_day_code"], params["window"])
    equity = _parse_csv(equity_path)
    if "timestamp" not in equity or "equity" not in equity:
        raise KeyError(f"{variant}: equity requires timestamp/equity")
    equity = equity.set_index("timestamp").reindex(common)
    if equity.equity.isna().any(): raise AssertionError(f"{variant}: MTM equity does not cover raw common bars")
    peak = equity["running_peak"] if "running_peak" in equity else equity.equity.cummax()
    dd = equity["drawdown_usd"] if "drawdown_usd" in equity else peak - equity.equity
    dd_pct = equity["drawdown_pct"] if "drawdown_pct" in equity else dd / peak * 100
    qqq = frames["QQQ"]
    bars = {
        "t": _vector(common_arrays["_t"]), "qo": _vector(qqq.open),
        "qh": _vector(qqq.high), "ql": _vector(qqq.low),
        "qc": _vector(qqq.close), "qvolume": _vector(qqq.volume),
        "qvwap": _vector(common_arrays["qvwap"]),
        "fair": _vector(common_arrays["fair"]),
        "residual": _vector(common_arrays["residual"]),
        "basket_dev": _vector(common_arrays["basket_dev"]), "z": _vector(z),
        "equity": _vector(equity.equity), "drawdown": _vector(dd),
        "drawdown_pct": _vector(dd_pct),
        **{f"dev_{symbol}": _vector(common_arrays[f'dev_{symbol}']) for symbol in CONSTITUENTS},
    }
    if set(map(len, bars.values())) != {len(common)}: raise AssertionError(f"{variant}: payload arrays not aligned")
    items, results = _trades(_parse_csv(trades_path), variant, summary), _results(summary)
    if len(items) != int(results["full"]["trades"]): raise AssertionError(f"{variant}: trades count mismatch")
    if not math.isclose(sum(x["net_pnl"] for x in items), float(results["full"]["net_pnl"]), abs_tol=1e-6):
        raise AssertionError(f"{variant}: trades net mismatch")
    by_time = {timestamp: index for index, timestamp in enumerate(bars["t"])}
    for trade in items:
        signal_i, entry_i = by_time.get(trade["entry_signal_time"]), by_time.get(trade["entry_time"])
        if signal_i is None or entry_i != signal_i + 1:
            raise AssertionError(f"{variant} trade {trade['id']}: entry is not exact next common raw open")
        for key, expected in (("signal_z", bars["z"][signal_i]),
                              ("signal_residual", bars["residual"][signal_i]),
                              ("signal_fair_qqq", bars["fair"][signal_i]),
                              ("signal_qqq_vwap", bars["qvwap"][signal_i]),
                              ("signal_qqq_close", bars["qc"][signal_i])):
            actual = trade[key]
            if actual is None or expected is None or not math.isclose(float(actual), float(expected), abs_tol=1e-9):
                raise AssertionError(f"{variant} trade {trade['id']}: {key} != causal raw array")
    payload = {
        "meta": {"schema_version": 1, "variant": variant, "direction": summary.get("direction"),
                 "exit_model": summary.get("exit_model"), "lead": "QQQ", "traded": "QQQ",
                 "reference_constituents": list(CONSTITUENTS), "basket": summary.get("basket", manifest.get("basket", {})),
                 "official_weights_pct": OFFICIAL_WEIGHTS_PCT,
                 "normalized_weights": {symbol: OFFICIAL_WEIGHTS_PCT[symbol] / sum(OFFICIAL_WEIGHTS_PCT.values()) for symbol in CONSTITUENTS},
                 "formula": "fair_QQQ = QQQ_VWAP × [1 + Σ normalized_weight_i × (constituent_i / constituent_VWAP_i − 1)]; residual=(QQQ−fair)/fair",
                 "signal_parameters": params, "selected": summary.get("selected", {}),
                 "selection": summary.get("selection", {}), "execution": summary.get("execution", {}),
                 "splits": _split_meta(summary, common), "period": {**manifest.get("period", {}), "raw_bars": len(common), "sessions": len(pd.unique(common.date))},
                 "source": "Exact five-way raw Alpaca SIP 1-minute RTH inner intersection; only QQQ OHLC candles are rendered; no synthetic/mock candles, fill, resampling, or interpolation",
                 "warning": "Four direction/exit variants and parameter grids create multiple-testing risk; holdout is diagnostic and does not prove a live edge.",
                 "sources": {"manifest": _source(SRC / "manifest.json"), "summary": _source(summary_path),
                             "trades": _source(trades_path), "equity": _source(equity_path), "audit": _source(audit_path)}},
        "bars": bars, "trades": items, "results": results,
    }
    full, holdout = results["full"], results["holdout"]
    item = {"variant": variant, "label": summary.get("label", variant), "data": f"data/{variant}.json",
            "direction": summary.get("direction"), "exit_model": summary.get("exit_model"),
            "selected": summary.get("selected", {}), "trades": full["trades"], "net_pnl": full["net_pnl"],
            "net_sharpe": full.get("net_sharpe", 0), "win_rate_pct": full.get("win_rate_pct", 0),
            "profit_factor": full.get("profit_factor", 0), "max_drawdown_usd": full.get("max_drawdown_usd_mtm", 0),
            "max_drawdown_pct": full.get("max_drawdown_pct_mtm", 0), "costs": full.get("costs", 0),
            "holdout_net_pnl": holdout["net_pnl"], "holdout_sharpe": holdout.get("net_sharpe", 0),
            "holdout_max_drawdown_pct": holdout.get("max_drawdown_pct_mtm", 0)}
    return payload, item


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
    manifest_path = SRC / "manifest.json"
    if not manifest_path.is_file(): raise FileNotFoundError(f"Сначала завершите research: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("status", "COMPLETE")).upper() != "COMPLETE":
        raise AssertionError(f"Research status={manifest.get('status')}")
    print("PHASE exact raw Parquet load", flush=True)
    common, frames, common_arrays = _market(manifest)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    assets = []
    for variant in VARIANTS:
        print(f"PHASE build {variant}", flush=True)
        payload, item = _build_variant(variant, manifest, common, frames, common_arrays)
        destination = DATA_OUT / f"{variant}.json"
        destination.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
        check = json.loads(destination.read_text(encoding="utf-8"))
        if len(check["bars"]["t"]) != len(common) or len(check["trades"]) != item["trades"]:
            raise AssertionError(f"{variant}: JSON readback failed")
        item.update({"bytes": destination.stat().st_size, "sha256": _sha(destination)})
        assets.append(item)
        print(f"BUILT {variant}: {item['trades']} trades, full {item['net_pnl']:+,.2f}, holdout {item['holdout_net_pnl']:+,.2f}")
    report_manifest = {"schema_version": 1, "source": _source(manifest_path), "variants": list(VARIANTS),
                       "default_variant": "normal_convergence", "assets": assets, "lead": "QQQ", "traded": "QQQ",
                       "constituents": list(CONSTITUENTS), "official_weights_pct": OFFICIAL_WEIGHTS_PCT,
                       "warning": "Exploratory normal/reverse × convergence/dollar-bracket comparison; multiple testing and historical holdout limitations apply."}
    compact = json.dumps(report_manifest, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.js").write_text("window.SYNTHETIC_VWAP_ABSOLUTE_MANIFEST=" + compact + ";\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT / "index.html"), "bars": len(common),
                      "lazy_payload_bytes": sum(x["bytes"] for x in assets)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
