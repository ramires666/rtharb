"""Build the standalone duration + stop-loss interactive report.

The two lazy variants share the exact 501-session QQQ/NVDA raw SIP calendar.
QQQ is reference-only and NVDA is the only traded instrument.  The builder
reconstructs the frozen classic fair/Z series and never creates quote candles.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rtharb.config import AppConfig
from rtharb.data.loader import DataLoader
from rtharb.models.fair_value import FairValueModel


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "research_output" / "duration_stoploss_combined"
OUT = ROOT / "tradingview_duration_stoploss_combined"
DATA_OUT = OUT / "data"
VARIANTS = ("raw_q95_q95", "selected")
SPLITS = ("development", "validation", "holdout", "full")
NY = "America/New_York"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source(path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}


def epoch(value: Any) -> int:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize(NY)
    return int(stamp.tz_convert("UTC").timestamp())


def vector(values: Any) -> list[Any]:
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.integer):
        return array.tolist()
    array = array.astype(float, copy=False)
    if np.isfinite(array).all():
        return array.tolist()
    return [None if not math.isfinite(float(value)) else float(value) for value in array]


def number(value: Any) -> float | int | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    result = float(value)
    return result if math.isfinite(result) else None


def parse_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ("timestamp", "entry_time", "exit_time"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], format="mixed", utc=True).dt.tz_convert(NY)
    return frame


def load_market(shared: dict[str, Any]) -> tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = AppConfig.load(str(ROOT / "configs" / "default_config.yaml"))
    frozen = shared["frozen_parameters"]
    lead, target = DataLoader(cfg.cache_dir, "alpaca", "sip").get_synchronized_pair("QQQ", "NVDA")
    if len(lead) != 194_490 or len(target) != 194_490 or not lead.index.equals(target.index):
        raise AssertionError(f"Expected exact 194,490 synchronized bars, got {len(lead)}/{len(target)}")
    model = FairValueModel(
        frozen["beta_mode"], int(frozen["beta_days"]), int(frozen["window"]),
        cfg.strategy.min_session_warmup_bars, cfg.strategy.min_sigma_history_days,
    )
    metrics = model.compute_intraday_metrics(lead, target)
    common = metrics.index
    expected = shared["data"]
    if len(common) != int(expected["raw_bars"]) or len(pd.unique(common.date)) != int(expected["sessions"]):
        raise AssertionError("Reconstructed market coverage differs from completed summary")
    if common[0].isoformat() != expected["first_timestamp"] or common[-1].isoformat() != expected["last_timestamp"]:
        raise AssertionError("Reconstructed market boundaries differ from completed summary")
    return common, lead.loc[common], target.loc[common], metrics


def split_meta(common: pd.DatetimeIndex) -> dict[str, dict[str, Any]]:
    days = pd.unique(common.date)
    bounds = {"development": (0, 250), "validation": (250, 375), "holdout": (375, 501), "full": (0, 501)}
    result = {}
    dates = np.asarray(common.date)
    for name, (low, high) in bounds.items():
        mask = (dates >= days[low]) & (dates <= days[high - 1])
        times = common[mask]
        result[name] = {
            "start": str(days[low]), "end": str(days[high - 1]), "sessions": high - low,
            "bars": len(times), "start_epoch": epoch(times[0]), "end_epoch": epoch(times[-1]),
        }
    return result


def split_for(timestamp: pd.Timestamp, splits: dict[str, dict[str, Any]]) -> str:
    day = timestamp.date()
    for name in SPLITS[:-1]:
        if pd.Timestamp(splits[name]["start"]).date() <= day <= pd.Timestamp(splits[name]["end"]).date():
            return name
    raise AssertionError(f"Trade timestamp outside splits: {timestamp}")


def results(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out = {}
    for split in SPLITS:
        item = dict(raw[split])
        for key in ("sessions", "raw_bars", "trades"):
            item[key] = int(item[key])
        item["exit_reasons"] = {str(k): int(v) for k, v in item.get("exit_reasons", {}).items()}
        for key, value in list(item.items()):
            if key not in {"exit_reasons", "sessions", "raw_bars", "trades"} and isinstance(value, (int, float)):
                item[key] = float(value)
        if not math.isclose(item["commissions"] + item["slippage"], item["costs"], abs_tol=1e-6):
            raise AssertionError(f"{split}: commission + slippage != costs")
        if not math.isclose(item["gross_pnl"] - item["costs"], item["net_pnl"], abs_tol=1e-6):
            raise AssertionError(f"{split}: gross - costs != net")
        out[split] = item
    return out


def trade_items(frame: pd.DataFrame, common: pd.DatetimeIndex, metrics: pd.DataFrame,
                splits: dict[str, dict[str, Any]], hold: int, stop_pct: float) -> list[dict[str, Any]]:
    indexer = {timestamp: i for i, timestamp in enumerate(common)}
    session_last: dict[object, int] = {}
    for i, day in enumerate(common.date):
        session_last[day] = i
    items = []
    for position, row in frame.iterrows():
        entry_time, exit_time = pd.Timestamp(row.entry_time), pd.Timestamp(row.exit_time)
        entry_i, exit_i = indexer.get(entry_time), indexer.get(exit_time)
        if entry_i is None or exit_i is None or entry_i == 0 or common[entry_i - 1].date() != entry_time.date():
            raise AssertionError(f"Trade {position + 1}: entry/exit absent or no same-session signal bar")
        direction = int(row.direction)
        entry_reference = float(row.entry_reference_price)
        stop_price = entry_reference * (1.0 - stop_pct if direction == 1 else 1.0 + stop_pct)
        expiry_i = min(entry_i + hold, session_last[entry_time.date()])
        reason = str(row.exit_reason)
        signal_z = float(metrics.z_score.iloc[entry_i - 1])
        if not math.isclose(signal_z, float(row.entry_z_score), abs_tol=1e-9):
            raise AssertionError(f"Trade {position + 1}: recorded entry Z != previous causal raw bar")
        if reason == "TIME_STOP" and (exit_i != entry_i + hold or int(row.duration_bars) != hold):
            raise AssertionError(f"Trade {position + 1}: invalid time-stop expiry")
        if reason == "STOP_LOSS":
            raw_open = float(metrics.target_open.iloc[exit_i])
            expected_fill = min(raw_open, stop_price) if direction == 1 else max(raw_open, stop_price)
            if not math.isclose(expected_fill, float(row.exit_reference_price), abs_tol=1e-8):
                raise AssertionError(f"Trade {position + 1}: stop fill is not stop/adverse gap open")
        commission, slippage = float(row.commission), float(row.slippage)
        items.append({
            "id": int(row.trade_id), "split": split_for(entry_time, splits),
            "direction": direction, "side": "LONG" if direction == 1 else "SHORT",
            "entry_signal_time": epoch(common[entry_i - 1]), "entry_time": epoch(entry_time), "exit_time": epoch(exit_time),
            "entry_reference": entry_reference, "entry_price": float(row.entry_price),
            "exit_reference": float(row.exit_reference_price), "exit_price": float(row.exit_price),
            "shares": int(row.shares), "position_value": float(row.position_value),
            "entry_z": float(row.entry_z_score), "exit_z": number(row.exit_z_score),
            "max_holding_bars": hold, "stop_loss_pct": stop_pct, "stop_price": stop_price,
            "expiry_time": epoch(common[expiry_i]), "expiry_capped_eod": expiry_i != entry_i + hold,
            "expiry_reached": reason == "TIME_STOP", "exit_reason": reason,
            "duration_bars": int(row.duration_bars), "gross_pnl": float(row.gross_pnl),
            "commissions": commission, "slippage": slippage, "costs": commission + slippage,
            "net_pnl": float(row.net_pnl), "return_pct": float(row.return_pct),
        })
    return items


def build_variant(variant: str, shared: dict[str, Any], raw_summary: dict[str, Any],
                  common: pd.DatetimeIndex, lead: pd.DataFrame, target: pd.DataFrame,
                  metrics: pd.DataFrame, splits: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if variant == "selected":
        summary_path = SRC / "selected_summary.json"
        hold = int(shared["selection"]["selected_max_holding_bars"])
        stop_pct = float(shared["selection"]["selected_stop_loss_pct"])
        survival = float(shared["selection"]["development_winner_survival_pct"])
        eligible, role = True, "selected"
        result_map = shared["selected_results"]
    else:
        summary_path = SRC / "raw_q95_q95_summary.json"
        hold = int(raw_summary["max_holding_bars"]); stop_pct = float(raw_summary["stop_loss_pct"])
        survival = float(raw_summary["development_winner_survival"]["still_net_profitable_pct"])
        eligible, role = bool(raw_summary["eligible"]), "diagnostic_only"
        result_map = raw_summary["results"]
    trades_path = SRC / f"{variant}_full_trades.csv"
    equity_path = SRC / f"{variant}_full_equity.csv"
    audit_path, manifest_path = SRC / "audit.json", SRC / "manifest.json"
    result_map = results(result_map)
    equity = parse_csv(equity_path).set_index("timestamp").reindex(common)
    if equity.equity.isna().any():
        raise AssertionError(f"{variant}: full minute equity does not cover common raw calendar")
    trades = trade_items(parse_csv(trades_path), common, metrics, splits, hold, stop_pct)
    if len(trades) != result_map["full"]["trades"]:
        raise AssertionError(f"{variant}: trade count mismatch")
    if not math.isclose(sum(x["net_pnl"] for x in trades), result_map["full"]["net_pnl"], abs_tol=1e-6):
        raise AssertionError(f"{variant}: trade net mismatch")
    if not math.isclose(float(equity.equity.iloc[-1]), result_map["full"]["final_equity"], abs_tol=1e-6):
        raise AssertionError(f"{variant}: final equity mismatch")
    if not math.isclose(float(equity.drawdown_usd.max()), result_map["full"]["max_drawdown_usd_mtm"], abs_tol=1e-6):
        raise AssertionError(f"{variant}: MTM drawdown mismatch")
    timestamps = common.as_unit("s").asi8.astype(np.int64)
    if int(timestamps[0]) != epoch(common[0]) or int(timestamps[-1]) != epoch(common[-1]):
        raise AssertionError("Unit-safe epoch conversion failed")
    bars = {
        "t": vector(timestamps),
        "no": vector(target.open), "nh": vector(target.high), "nl": vector(target.low), "nc": vector(target.close),
        "qo": vector(lead.open), "qh": vector(lead.high), "ql": vector(lead.low), "qc": vector(lead.close),
        "target_fair": vector(metrics.target_fair_price), "spread": vector(metrics.spread),
        "z": vector(metrics.z_score), "beta": vector(metrics.beta),
        "equity": vector(equity.equity), "drawdown": vector(equity.drawdown_usd),
        "drawdown_pct": vector(equity.drawdown_pct),
    }
    if set(map(len, bars.values())) != {len(common)}:
        raise AssertionError(f"{variant}: payload arrays misaligned")
    payload = {
        "meta": {
            "schema_version": 1, "variant": variant, "role": role, "eligible": eligible,
            "lead": "QQQ", "traded": "NVDA", "winner_survival_pct": survival,
            "eligibility_threshold_pct": 95.0, "max_holding_bars": hold,
            "stop_loss_pct": stop_pct, "stop_loss_percent": stop_pct * 100.0,
            "frozen_parameters": shared["frozen_parameters"], "execution": shared["execution"],
            "selection": shared.get("selection", {}), "candidate_fit": shared.get("candidate_fit", {}),
            "splits": splits, "period": {"start": str(common[0].date()), "end": str(common[-1].date()),
                                        "sessions": 501, "raw_bars": len(common)},
            "source": "Exact synchronized raw Alpaca SIP QQQ/NVDA 1-minute RTH; NVDA only traded; no fill, resampling, interpolation, or synthetic candles",
            "warning": "Raw q95/q95 is diagnostic and ineligible at 93.94% winner survival. Selected passes 95%, but its untouched holdout is negative; this is not live evidence.",
            "sources": {"manifest": source(manifest_path), "audit": source(audit_path), "summary": source(summary_path),
                        "trades": source(trades_path), "equity": source(equity_path)},
        },
        "bars": bars, "trades": trades, "results": result_map,
    }
    full, holdout = result_map["full"], result_map["holdout"]
    asset = {
        "variant": variant, "label": "Raw q95 + q95 · diagnostic" if variant.startswith("raw") else "Selected combined · eligible",
        "data": f"data/{variant}.json", "eligible": eligible, "survival_pct": survival,
        "max_holding_bars": hold, "stop_loss_pct": stop_pct, "trades": full["trades"],
        "net_pnl": full["net_pnl"], "net_sharpe": full["net_sharpe"],
        "win_rate_pct": full["win_rate_pct"], "profit_factor": full["profit_factor"],
        "max_drawdown_usd": full["max_drawdown_usd_mtm"], "max_drawdown_pct": full["max_drawdown_pct_mtm"],
        "costs": full["costs"], "holdout_net_pnl": holdout["net_pnl"], "holdout_sharpe": holdout["net_sharpe"],
    }
    return payload, asset


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
    manifest_path, audit_path = SRC / "manifest.json", SRC / "audit.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or audit.get("status") != "PASS":
        raise AssertionError("Research must be COMPLETE and audit PASS")
    shared = json.loads((SRC / "selected_summary.json").read_text(encoding="utf-8"))
    raw_summary = json.loads((SRC / "raw_q95_q95_summary.json").read_text(encoding="utf-8"))
    print("PHASE exact raw QQQ/NVDA load + frozen fair/Z", flush=True)
    common, lead, target, metrics = load_market(shared)
    splits = split_meta(common)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    assets = []
    for variant in VARIANTS:
        print(f"PHASE build {variant}", flush=True)
        payload, asset = build_variant(variant, shared, raw_summary, common, lead, target, metrics, splits)
        path = DATA_OUT / f"{variant}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")
        check = json.loads(path.read_text(encoding="utf-8"))
        if len(check["bars"]["t"]) != 194_490 or len(check["trades"]) != asset["trades"]:
            raise AssertionError(f"{variant}: JSON readback mismatch")
        asset.update({"bytes": path.stat().st_size, "sha256": sha256(path)})
        assets.append(asset)
        print(f"BUILT {variant}: {asset['trades']} trades, full {asset['net_pnl']:+,.2f}, holdout {asset['holdout_net_pnl']:+,.2f}", flush=True)
    report_manifest = {
        "schema_version": 1, "variants": list(VARIANTS), "default_variant": "selected",
        "assets": assets, "lead": "QQQ", "traded": "NVDA", "source": source(manifest_path),
        "audit": source(audit_path),
        "warning": "Exploratory overlay study: selected full result is positive but untouched holdout is negative.",
    }
    compact = json.dumps(report_manifest, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    (OUT / "manifest.js").write_text("window.DURATION_STOPLOSS_COMBINED_MANIFEST=" + compact + ";\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT / "index.html"), "bars": len(common),
                      "lazy_payload_bytes": sum(x["bytes"] for x in assets)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
