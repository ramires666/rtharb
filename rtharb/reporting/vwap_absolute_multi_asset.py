"""Publish the nine-stock VWAP absolute-bracket study as lazy report payloads.

The report intentionally does *not* embed all minute bars in one huge script.
``manifest.js`` contains only the comparison table and points the browser to
one JSON file per target under ``data/``.  Every JSON payload is built from the
exact pairwise intersection of raw Alpaca SIP one-minute QQQ/target bars.

The reader is deliberately tolerant of both the preferred output layout
(``<SYMBOL>/summary.json`` plus ``selected_full_*.csv``) and flat filename
aliases.  That keeps report generation independent from cosmetic research
folder changes while the reconciliation checks still fail closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from rtharb.config import AppConfig
from rtharb.data.loader import DataLoader
from rtharb.research.vwap_strategy import vwap_arrays


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "research_output" / "vwap_absolute_multi_asset"
OUT = ROOT / "tradingview_vwap_absolute_multi_asset"
DATA_OUT = OUT / "data"
DEFAULT_TARGETS = ("NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "AMD")
NY = "America/New_York"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: Any) -> int | float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    value = float(value)
    if not math.isfinite(value):
        return None
    return value


def _epoch(value: Any) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(NY)
    return int(timestamp.tz_convert("UTC").timestamp())


def _get(mapping: dict, *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _find_first(paths: Iterable[Path], label: str) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Не найден {label}; проверены: " + ", ".join(map(str, paths)))


def _symbol_file(symbol: str, kind: str) -> Path:
    """Resolve per-symbol research artifacts in directory or flat layouts."""
    symbol_lower = symbol.lower()
    names = {
        "summary": ("summary.json",),
        "trades": ("selected_full_trades.csv", "full_trades.csv", "trades_full.csv"),
        "equity": ("selected_full_equity.csv", "full_equity.csv", "equity_full.csv"),
    }[kind]
    candidates: list[Path] = []
    for folder in (SRC / symbol, SRC / symbol_lower):
        candidates.extend(folder / name for name in names)
    if kind == "summary":
        candidates.extend((SRC / f"{symbol}_summary.json", SRC / f"summary_{symbol}.json",
                           SRC / f"{symbol_lower}_summary.json"))
    else:
        noun = "trades" if kind == "trades" else "equity"
        candidates.extend((
            SRC / f"{symbol}_selected_full_{noun}.csv",
            SRC / f"selected_{symbol}_full_{noun}.csv",
            SRC / f"selected_full_{symbol}_{noun}.csv",
            SRC / f"{symbol_lower}_selected_full_{noun}.csv",
        ))
    return _find_first(candidates, f"{kind} для {symbol}")


def _targets(global_summary: dict) -> list[str]:
    value = _get(global_summary, "frozen_universe", "traded_separately", "targets", "symbols", default=None)
    if isinstance(value, dict):
        value = _get(value, "targets", "traded", "universe", default=None)
    if value is None and isinstance(global_summary.get("universe"), dict):
        value = _get(global_summary["universe"], "targets", "symbols", default=None)
    if value is None and isinstance(global_summary.get("per_symbol"), dict):
        value = list(global_summary["per_symbol"])
    targets = [str(x).upper() for x in (value or DEFAULT_TARGETS) if str(x).upper() != "QQQ"]
    # Fixed order makes browser selections and audit output deterministic.
    return [x for x in DEFAULT_TARGETS if x in targets] + [x for x in targets if x not in DEFAULT_TARGETS]


def _embedded_symbol_summary(global_summary: dict, symbol: str) -> dict:
    for key in ("per_symbol", "symbol_results", "assets", "results_by_symbol"):
        group = global_summary.get(key)
        if isinstance(group, dict) and isinstance(group.get(symbol), dict):
            return group[symbol]
    return {}


def _load_symbol_summary(global_summary: dict, symbol: str) -> tuple[dict, Path | None]:
    embedded = _embedded_symbol_summary(global_summary, symbol)
    try:
        path = _symbol_file(symbol, "summary")
    except FileNotFoundError:
        if not embedded:
            raise
        return embedded, None
    disk = json.loads(path.read_text(encoding="utf-8"))
    return ({**embedded, **disk} if embedded else disk), path


def _selected(summary: dict) -> dict:
    selected = _get(summary, "selected", "selected_parameters", "best", default={})
    if not isinstance(selected, dict):
        selected = {}
    stop = _get(selected, "stop_usd", "stop", "stop_usd_per_share",
                default=_get(summary, "stop_usd", "selected_stop_usd"))
    target = _get(selected, "target_usd", "target", "take_profit_usd", "target_usd_per_share",
                  default=_get(summary, "target_usd", "selected_target_usd"))
    if stop is None or target is None:
        raise KeyError("В summary отсутствуют выбранные stop_usd/target_usd")
    return {**selected, "stop_usd": float(stop), "target_usd": float(target)}


def _results(summary: dict) -> dict:
    results = _get(summary, "selected_results", "results", "metrics", default={})
    if isinstance(results, dict) and "full" in results:
        return results
    # Some compact summaries put the four split objects directly at top level.
    direct = {key: summary[key] for key in ("development", "validation", "holdout", "full")
              if isinstance(summary.get(key), dict)}
    if "full" not in direct:
        raise KeyError("В summary отсутствуют selected_results.full")
    return direct


def _strategy(summary: dict, global_summary: dict) -> dict:
    raw = _get(summary, "entry_parameters", "strategy", "parameters", default={})
    if not raw:
        raw = _get(global_summary, "entry_parameters", "strategy", "parameters", default={})
    return {
        "beta_days": int(_get(raw, "beta_days", default=5)),
        "window": int(_get(raw, "window", "z_window", default=60)),
        "warmup_bars": int(_get(raw, "warmup_bars", "warmup", default=30)),
        "z_entry": float(_get(raw, "z_entry", "entry_z", default=2.5)),
        "hook_delta": float(_get(raw, "hook_delta", default=0.0)),
        "z_lockout": _get(raw, "z_lockout", default=None),
    }


def _period(summary: dict, global_summary: dict, equity: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp, dict]:
    raw = _get(summary, "period", "study_period", default={}) or _get(
        global_summary, "period", "study_period", default={}) or {}
    start = pd.Timestamp(_get(raw, "start", default=equity["timestamp"].min())).date()
    end = pd.Timestamp(_get(raw, "end", default=equity["timestamp"].max())).date()
    return start, end, {**raw, "start": str(start), "end": str(end)}


def _read_csv_times(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ("timestamp", "signal_time", "entry_signal_time", "entry_time", "exit_time"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], format="mixed", utc=True).dt.tz_convert(NY)
    return frame


def _col(row: pd.Series, names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in row.index and not pd.isna(row[name]):
            return row[name]
    return default


def _split_for_time(timestamp: pd.Timestamp, summary: dict) -> str:
    splits = summary.get("splits", {})
    for key in ("development", "validation", "holdout"):
        item = splits.get(key, {})
        if item.get("start") and item.get("end"):
            if pd.Timestamp(item["start"]).date() <= timestamp.date() <= pd.Timestamp(item["end"]).date():
                return key
    return "full"


def _trade_items(frame: pd.DataFrame, symbol: str, summary: dict) -> list[dict]:
    target = symbol.lower()
    items: list[dict] = []
    for index, row in frame.iterrows():
        signal_time = _col(row, ("signal_time", "entry_signal_time"))
        entry_time, exit_time = _col(row, ("entry_time",)), _col(row, ("exit_time",))
        if signal_time is None or entry_time is None or exit_time is None:
            raise KeyError(f"{symbol} trade row {index}: отсутствуют signal/entry/exit timestamps")
        direction_text = str(_col(row, ("direction", "side"), "LONG")).upper()
        direction = 1 if direction_text in {"LONG", "BUY", "1", "+1"} else -1
        entry_ref = float(_col(row, ("entry_reference", "entry_raw", "raw_entry", "entry_price")))
        exit_ref = float(_col(row, ("exit_reference", "exit_raw", "raw_exit", "exit_price")))
        entry_price = float(_col(row, ("entry_price", "entry_effective"), entry_ref))
        exit_price = float(_col(row, ("exit_price", "exit_effective"), exit_ref))
        shares = int(_col(row, ("shares", "qty", "quantity"), 0))
        stop_usd = float(_col(row, ("stop_usd_per_share", "stop_usd", "stop_distance"), 0.0))
        target_usd = float(_col(row, ("target_usd_per_share", "target_usd", "target_distance"), 0.0))
        commissions = float(_col(row, ("commissions", "commission"), 0.0))
        slippage = float(_col(row, ("slippage", "slippage_cost"), 0.0))
        costs = float(_col(row, ("costs", "total_costs"), commissions + slippage))
        gross = float(_col(row, ("gross_pnl", "pnl_gross"), direction * (exit_ref - entry_ref) * shares))
        net = float(_col(row, ("net_pnl", "pnl_net"), gross - costs))
        stop_price = float(_col(row, ("stop_price",), entry_ref - direction * stop_usd))
        target_price = float(_col(row, ("target_price",), entry_ref + direction * target_usd))
        entry_timestamp = pd.Timestamp(entry_time)
        items.append({
            "id": index + 1, "symbol": symbol,
            "split": str(_col(row, ("split",), _split_for_time(entry_timestamp, summary))),
            "side": "LONG" if direction == 1 else "SHORT", "direction": direction,
            "entry_signal_time": _epoch(signal_time), "entry_time": _epoch(entry_time), "exit_time": _epoch(exit_time),
            "entry_z": _number(_col(row, ("entry_z", "signal_z", "z"))),
            "signal_target_close": _number(_col(row, ("signal_target_close", f"signal_{target}_close", "signal_nvda_close"))),
            "signal_target_vwap": _number(_col(row, ("signal_target_vwap", f"signal_{target}_vwap", "signal_nvda_vwap"))),
            "signal_qqq_vwap": _number(_col(row, ("signal_qqq_vwap", "signal_lead_vwap"))),
            "signal_fair_target": _number(_col(row, ("signal_fair_target", f"signal_fair_{target}", "signal_fair_nvda"))),
            "entry_reference": entry_ref, "entry_price": entry_price,
            "exit_reference": exit_ref, "exit_price": exit_price, "shares": shares,
            "stop_usd_per_share": stop_usd, "target_usd_per_share": target_usd,
            "stop_price": stop_price, "target_price": target_price,
            "gross_risk_usd": _number(_col(row, ("gross_risk_usd",), stop_usd * shares)),
            "gross_reward_usd": _number(_col(row, ("gross_reward_usd",), target_usd * shares)),
            "risk_reward_ratio": _number(_col(row, ("risk_reward_ratio",), target_usd / stop_usd if stop_usd else None)),
            "gross_pnl": gross, "commissions": commissions, "slippage": slippage,
            "costs": costs, "net_pnl": net, "exit_reason": str(_col(row, ("exit_reason", "reason"), "UNKNOWN")),
            "duration_minutes": int(_col(row, ("duration_minutes", "duration_bars", "bars_held"), 0)),
        })
    return items


def _metric(metric: dict, *names: str, default: Any = 0.0) -> Any:
    return _get(metric, *names, default=default)


def _ui_results(results: dict, trades: list[dict]) -> dict:
    out = {}
    for split in ("development", "validation", "holdout", "full"):
        raw = dict(results.get(split, {}))
        chosen = trades if split == "full" else [trade for trade in trades if trade["split"] == split]
        gross = sum(float(x["gross_pnl"]) for x in chosen)
        commissions = sum(float(x["commissions"]) for x in chosen)
        slippage = sum(float(x["slippage"]) for x in chosen)
        net = sum(float(x["net_pnl"]) for x in chosen)
        raw.update({
            "trades": int(_metric(raw, "trades", default=len(chosen))),
            "gross_pnl": float(_metric(raw, "gross_pnl", default=gross)),
            "commissions": float(_metric(raw, "commissions", default=commissions)),
            "slippage": float(_metric(raw, "slippage", default=slippage)),
            "costs": float(_metric(raw, "costs", default=commissions + slippage)),
            "net_pnl": float(_metric(raw, "net_pnl", default=net)),
            "net_sharpe": float(_metric(raw, "net_sharpe", "sharpe", default=0.0)),
            "win_rate_pct": float(_metric(raw, "win_rate_pct", "win_rate", default=0.0)),
            "profit_factor": float(_metric(raw, "profit_factor", default=0.0)),
            "max_drawdown_usd_mtm": float(_metric(raw, "max_drawdown_usd_mtm", "max_drawdown_usd", "max_drawdown_usd_daily", default=0.0)),
            "max_drawdown_pct_mtm": float(_metric(raw, "max_drawdown_pct_mtm", "max_drawdown_pct", "max_drawdown_pct_daily", default=0.0)),
            "stops": int(_metric(raw, "stops", default=sum(t["exit_reason"] == "STOP" for t in chosen))),
            "targets": int(_metric(raw, "targets", default=sum(t["exit_reason"] in {"TAKE_PROFIT_BRACKET", "TARGET"} for t in chosen))),
            "forced_eod": int(_metric(raw, "forced_eod", default=sum(t["exit_reason"] in {"FORCED_EOD", "EOD"} for t in chosen))),
        })
        out[split] = raw
    return out


def _split_meta(summary: dict, common: pd.DatetimeIndex) -> dict:
    result = {}
    for split, item in summary.get("splits", {}).items():
        if not isinstance(item, dict) or not item.get("start") or not item.get("end"):
            continue
        first, last = pd.Timestamp(item["start"]).date(), pd.Timestamp(item["end"]).date()
        times = common[(common.date >= first) & (common.date <= last)]
        if len(times):
            result[split] = {**item, "start_epoch": _epoch(times[0]), "end_epoch": _epoch(times[-1])}
    return result


def _build_symbol(loader: DataLoader, global_summary: dict, symbol: str, coverage: dict) -> tuple[dict, dict]:
    summary, summary_path = _load_symbol_summary(global_summary, symbol)
    trades_path, equity_path = _symbol_file(symbol, "trades"), _symbol_file(symbol, "equity")
    trades_frame, equity = _read_csv_times(trades_path), _read_csv_times(equity_path)
    if "timestamp" not in equity or "equity" not in equity:
        raise KeyError(f"{symbol}: equity CSV обязан иметь timestamp/equity")
    if "drawdown_usd" not in equity:
        peak = equity["equity"].cummax()
        equity["drawdown_usd"] = peak - equity["equity"]
    start, end, period = _period(summary, global_summary, equity)
    strategy = _strategy(summary, global_summary)
    selected = _selected(summary)

    # Compute beta/VWAP/fair/Z on full prehistory, then slice.  This preserves
    # strictly prior-session beta information at the first study date.
    lead_all, target_all = loader.get_synchronized_pair("QQQ", symbol)
    common_all = lead_all.index.intersection(target_all.index)
    lead_all, target_all = lead_all.loc[common_all], target_all.loc[common_all]
    arrays_all = vwap_arrays(lead_all, target_all, strategy["beta_days"], strategy["window"], strategy["warmup_bars"])
    mask = np.fromiter((start <= timestamp.date() <= end for timestamp in common_all), bool, len(common_all))
    common = common_all[mask]
    lead, target = lead_all.loc[common], target_all.loc[common]
    arrays = {key: value[mask] for key, value in arrays_all.items()
              if isinstance(value, np.ndarray) and len(value) == len(common_all)}
    if not len(common) or not common.equals(target.index):
        raise AssertionError(f"{symbol}: пустая или рассинхронизированная pairwise выборка")

    equity = equity.set_index("timestamp").reindex(common)
    if equity["equity"].isna().any():
        missing = int(equity["equity"].isna().sum())
        raise AssertionError(f"{symbol}: equity не покрывает {missing} pairwise минут")
    items = _trade_items(trades_frame, symbol, summary)
    results = _ui_results(_results(summary), items)
    if results["full"]["trades"] != len(items):
        raise AssertionError(f"{symbol}: число full trades не совпадает с CSV")
    net_csv = sum(float(x["net_pnl"]) for x in items)
    if not math.isclose(net_csv, float(results["full"]["net_pnl"]), abs_tol=1e-6):
        raise AssertionError(f"{symbol}: full net не совпадает с реестром сделок")

    bars = {
        "t": [_epoch(x) for x in common],
        "qo": [_number(x) for x in lead.open], "qh": [_number(x) for x in lead.high],
        "ql": [_number(x) for x in lead.low], "qc": [_number(x) for x in lead.close],
        "qv": [_number(x) for x in lead.volume],
        "to": [_number(x) for x in target.open], "th": [_number(x) for x in target.high],
        "tl": [_number(x) for x in target.low], "tc": [_number(x) for x in target.close],
        "tv": [_number(x) for x in target.volume],
        "qvwap": [_number(x) for x in arrays_all["vwap_lead"][mask]],
        "tvwap": [_number(x) for x in arrays_all["vwap_target"][mask]],
        "fair": [_number(x) for x in arrays_all["fair_price"][mask]],
        "z": [_number(x) for x in arrays_all["z"][mask]],
        "equity": [_number(x) for x in equity["equity"]],
        "drawdown": [_number(x) for x in equity["drawdown_usd"]],
    }
    if set(map(len, bars.values())) != {len(common)}:
        raise AssertionError(f"{symbol}: массивы payload рассинхронизированы")

    execution = _get(summary, "execution", default={}) or _get(global_summary, "execution", default={}) or {}
    commission = float(_get(execution, "commission_usd_per_share_per_side", default=0.0035))
    slip_fraction = float(_get(execution, "slippage_fraction_per_execution", default=0.0002))
    payload = {
        "meta": {
            "schema_version": 1, "lead": "QQQ", "target": symbol,
            "roles": {"QQQ": "reference only — not traded", symbol: "the only traded instrument"},
            "source": f"Exact pairwise synchronized raw Alpaca SIP 1-minute QQQ/{symbol} RTH bars; no resampling, fill, interpolation, or synthetic quotes",
            "period": {**period, "raw_bars": len(common), "sessions": int(len(pd.unique(common.date)))},
            "strategy": strategy, "selected": selected,
            "execution": {**execution, "commission_usd_per_share_per_side": commission,
                          "slippage_fraction_per_execution": slip_fraction,
                          "slippage_bps_per_execution": slip_fraction * 10_000,
                          "stop_usd_per_share": selected["stop_usd"],
                          "target_usd_per_share": selected["target_usd"],
                          "reward_risk_ratio": selected["target_usd"] / selected["stop_usd"],
                          "convergence_exit": False},
            "splits": _split_meta(summary, common), "coverage": coverage,
            "selection": _get(summary, "selection", default={}),
            "warning": "Exploratory per-symbol optimization with multiple testing and current-universe survivorship bias; historical holdout is not proof of a live edge.",
            "sources": {
                "summary": None if summary_path is None else {"path": summary_path.relative_to(ROOT).as_posix(), "sha256": _sha256(summary_path)},
                "trades": {"path": trades_path.relative_to(ROOT).as_posix(), "sha256": _sha256(trades_path)},
                "equity": {"path": equity_path.relative_to(ROOT).as_posix(), "sha256": _sha256(equity_path)},
                "lead_raw": "data_cache/QQQ_1m.parquet", "target_raw": f"data_cache/{symbol}_1m.parquet",
            },
        },
        "bars": bars, "trades": items, "results": results,
    }
    comparison = {
        "symbol": symbol, "data": f"data/{symbol}.json",
        "stop_usd": selected["stop_usd"], "target_usd": selected["target_usd"],
        "reward_risk_ratio": selected["target_usd"] / selected["stop_usd"],
        "trades": results["full"]["trades"], "net_pnl": results["full"]["net_pnl"],
        "net_sharpe": results["full"]["net_sharpe"], "win_rate_pct": results["full"]["win_rate_pct"],
        "profit_factor": results["full"]["profit_factor"],
        "max_drawdown_usd": results["full"]["max_drawdown_usd_mtm"],
        "max_drawdown_pct": results["full"]["max_drawdown_pct_mtm"],
        "holdout_net_pnl": results.get("holdout", {}).get("net_pnl", 0.0),
        "holdout_sharpe": results.get("holdout", {}).get("net_sharpe", 0.0),
        "raw_bars": len(common), "sessions": int(len(pd.unique(common.date))),
        "coverage_pct": _get(coverage, "pairwise_coverage_pct", "study_rth_coverage_pct", default=None),
    }
    return payload, comparison


def main() -> None:
    global SRC, OUT, DATA_OUT
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SRC)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    SRC, OUT, DATA_OUT = args.source.resolve(), args.output.resolve(), args.output.resolve() / "data"
    global_path = next((path for path in (SRC / "summary.json", SRC / "manifest.json") if path.is_file()),
                       SRC / "manifest.json")
    if not global_path.is_file():
        raise FileNotFoundError(f"Сначала завершите research: {SRC / 'manifest.json'}")
    global_summary = json.loads(global_path.read_text(encoding="utf-8"))
    research_status = global_summary.get("status", "COMPLETE")
    if global_path.name == "manifest.json" and research_status not in {"PARTIAL", "COMPLETE"}:
        raise AssertionError(f"Multi-asset research ещё не публикуем: status={research_status}")
    if research_status == "PARTIAL":
        completed = {str(symbol).upper() for symbol in global_summary.get("symbols_completed", [])}
        targets = [symbol for symbol in DEFAULT_TARGETS if symbol in completed]
        if not targets:
            raise AssertionError("PARTIAL research не содержит завершённых символов")
    else:
        targets = _targets(global_summary)
    if research_status == "COMPLETE" and set(DEFAULT_TARGETS) - set(targets):
        raise AssertionError(f"Ожидались все девять акций: {DEFAULT_TARGETS}; получено {targets}")
    coverage_path = ROOT / "data_cache" / "mega_cap_sip_manifest.json"
    coverage_manifest = json.loads(coverage_path.read_text(encoding="utf-8")) if coverage_path.is_file() else {}
    loader = DataLoader(AppConfig.load(str(ROOT / "configs" / "default_config.yaml")).cache_dir, "alpaca", "sip")
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    comparisons = []
    for symbol in targets:
        payload, comparison = _build_symbol(loader, global_summary, symbol,
                                            coverage_manifest.get("symbols", {}).get(symbol, {}))
        destination = DATA_OUT / f"{symbol}.json"
        destination.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
        # Readback is cheap insurance against a truncated multi-megabyte file.
        check = json.loads(destination.read_text(encoding="utf-8"))
        if len(check["bars"]["t"]) != comparison["raw_bars"] or len(check["trades"]) != comparison["trades"]:
            raise AssertionError(f"{symbol}: JSON readback failed")
        comparison["bytes"] = destination.stat().st_size
        comparison["sha256"] = _sha256(destination)
        comparisons.append(comparison)
        print(f"{symbol}: {comparison['raw_bars']:,} bars, {comparison['trades']} trades, net {comparison['net_pnl']:+.2f}", flush=True)
    manifest = {
        "schema_version": 1, "research_status": research_status,
        "generated_from": global_path.relative_to(ROOT).as_posix(),
        "source_sha256": _sha256(global_path), "lead": "QQQ", "targets": targets,
        "default_symbol": "NVDA" if "NVDA" in targets else targets[0], "assets": comparisons,
        "roles": "QQQ is reference only and never traded; selected target is the only traded leg",
        "warning": "Exploratory nine-asset grid with multiple testing and survivorship bias; compare holdout, costs, and drawdown—not only full-sample net P&L.",
    }
    compact = json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    (OUT / "manifest.js").write_text("window.VWAP_MULTI_ASSET_MANIFEST=" + compact + ";\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT / "index.html"), "targets": targets,
                      "lazy_payload_bytes": sum(x["bytes"] for x in comparisons)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
