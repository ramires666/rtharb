"""Build the lazy nine-symbol walk-forward VWAP selection dashboard.

The generator refuses to run before the research progress artifact is
COMPLETE.  Each target receives one JSON payload; the HTML and manifest stay
small and never embed the full minute history.
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
from rtharb.research.vwap_strategy import vwap_arrays


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "research_output" / "vwap_all_assets_robust_selection"
OUT = ROOT / "tradingview_vwap_multi_asset_walkforward"
DATA_OUT = OUT / "data"
UNIVERSE = ("NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "AMD")
NY = "America/New_York"
START = pd.Timestamp("2025-08-22").date()
END = pd.Timestamp("2026-08-21").date()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source(path: Path) -> dict[str, Any]:
    return {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha(path)}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def number(value: Any) -> int | float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    result = float(value)
    return result if math.isfinite(result) else None


def epoch(value: Any) -> int:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(NY)
    return int(timestamp.tz_convert("UTC").timestamp())


def read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ("timestamp", "signal_time", "entry_time", "exit_time"):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], format="mixed", utc=True).dt.tz_convert(NY)
    return frame


def require_complete() -> dict[str, Any]:
    progress_path = SRC / "progress.json"
    if not progress_path.is_file():
        raise FileNotFoundError("Сначала запустите research robust-selection: progress.json отсутствует")
    progress = read_json(progress_path)
    completed = [str(item.get("symbol")) for item in progress.get("completed", [])]
    if progress.get("status") != "COMPLETE" or completed != list(UNIVERSE) or progress.get("remaining"):
        raise RuntimeError(
            f"Research ещё не COMPLETE: status={progress.get('status')}, "
            f"готово {len(completed)}/9. Генератор не будет публиковать partial report."
        )
    for symbol in UNIVERSE:
        needed = ("summary.json", "pre_seen_freeze.json", "pre_seen_freeze.sha256",
                  "development_grid.csv", "block_metrics.csv", "folds.csv", "audit.json",
                  "current_full_trades.csv", "current_full_equity.csv",
                  "selected_full_trades.csv", "selected_full_equity.csv")
        missing = [name for name in needed if not (SRC / symbol / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{symbol}: COMPLETE artifacts отсутствуют: {missing}")
    return progress


def market(loader: DataLoader, symbol: str) -> tuple[pd.DatetimeIndex, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    qqq, target = loader.get_synchronized_pair("QQQ", symbol)
    common = qqq.index.intersection(target.index)
    qqq, target = qqq.loc[common], target.loc[common]
    arrays = vwap_arrays(qqq, target, beta_days=5, window=60, warmup=30)
    mask = np.fromiter((START <= item.date() <= END for item in common), bool, len(common))
    common = common[mask]; qqq = qqq.loc[common]; target = target.loc[common]
    display = {key: value[mask] for key, value in arrays.items()
               if isinstance(value, np.ndarray) and len(value) == len(mask)}
    if len(pd.unique(common.date)) != 251 or common.has_duplicates:
        raise AssertionError(f"{symbol}: expected exact 251-session no-fill pairwise raw clock")
    return common, qqq, target, display


def exact_row(grid: pd.DataFrame, params: dict[str, Any] | None) -> dict[str, Any] | None:
    if params is None:
        return None
    hit = grid[np.isclose(grid.stop_usd, float(params["stop_usd"]), atol=1e-9, rtol=0) &
               np.isclose(grid.target_usd, float(params["target_usd"]), atol=1e-9, rtol=0)]
    if len(hit) != 1:
        raise AssertionError(f"Grid row not unique for {params}")
    row = hit.iloc[0]
    return {key: number(value) if not isinstance(value, str) else value for key, value in row.items()}


def trades(frame: pd.DataFrame, variant: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        direction = 1 if str(row.direction).upper() == "LONG" else -1
        out.append({
            "id": index + 1, "variant": variant, "direction": direction,
            "side": "LONG" if direction == 1 else "SHORT",
            "signal_time": epoch(row.signal_time), "entry_time": epoch(row.entry_time),
            "exit_time": epoch(row.exit_time), "entry_z": number(row.entry_z),
            "entry_reference": number(row.entry_reference), "entry_price": number(row.entry_price),
            "exit_reference": number(row.exit_reference), "exit_price": number(row.exit_price),
            "shares": int(row.shares), "stop_price": number(row.stop_price),
            "target_price": number(row.target_price), "stop_usd": number(row.stop_usd_per_share),
            "target_usd": number(row.target_usd_per_share), "exit_reason": str(row.exit_reason),
            "duration_bars": int(row.duration_bars), "gross_pnl": number(row.gross_pnl),
            "commissions": number(row.commissions), "slippage": number(row.slippage),
            "costs": number(row.costs), "net_pnl": number(row.net_pnl),
        })
    return out


def equity_on_clock(path: Path, common: pd.DatetimeIndex) -> pd.DataFrame:
    frame = read_csv(path)
    if len(frame) != len(common) or not frame.timestamp.equals(pd.Series(common)):
        left = pd.DatetimeIndex(frame.timestamp)
        if len(left) != len(common) or not left.equals(common):
            raise AssertionError(f"{path}: equity clock differs from exact raw pair")
    return frame


def compact_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{key: (value if isinstance(value, str) else number(value)) for key, value in row.items()}
            for row in frame.to_dict(orient="records")]


def build_symbol(loader: DataLoader, symbol: str) -> tuple[dict[str, Any], dict[str, Any]]:
    folder = SRC / symbol
    summary = read_json(folder / "summary.json")
    freeze = read_json(folder / "pre_seen_freeze.json")
    audit = read_json(folder / "audit.json")
    if audit.get("status") != "PASS" or summary.get("audit", {}).get("status") != "PASS":
        raise AssertionError(f"{symbol}: source audit is not PASS")
    frozen_hash = (folder / "pre_seen_freeze.sha256").read_text(encoding="utf-8").strip()
    if frozen_hash != sha(folder / "pre_seen_freeze.json") or frozen_hash != summary["pre_seen_freeze_sha256"]:
        raise AssertionError(f"{symbol}: immutable pre-seen freeze hash differs")
    grid = pd.read_csv(folder / "development_grid.csv")
    blocks = pd.read_csv(folder / "block_metrics.csv")
    folds = pd.read_csv(folder / "folds.csv")
    current_row = exact_row(grid, freeze["current"])
    candidate_row = exact_row(grid, freeze.get("candidate"))
    selected_row = exact_row(grid, freeze.get("selected"))
    common, qqq, target, arrays = market(loader, symbol)
    current_equity = equity_on_clock(folder / "current_full_equity.csv", common)
    selected_equity = equity_on_clock(folder / "selected_full_equity.csv", common)
    current_trades = read_csv(folder / "current_full_trades.csv")
    selected_trades = read_csv(folder / "selected_full_trades.csv")
    if len(current_trades) != int(summary["current_results"]["full"]["trades"]):
        raise AssertionError(f"{symbol}: current trade count differs")
    if len(selected_trades) != int(summary["selected_results"]["full"]["trades"]):
        raise AssertionError(f"{symbol}: selected trade count differs")

    neighbor_rows: list[dict[str, Any]] = []
    if candidate_row is not None:
        chosen = grid[np.isclose(grid.stop_usd, freeze["candidate"]["stop_usd"]) &
                      np.isclose(grid.target_usd, freeze["candidate"]["target_usd"])].index[0]
        ids = str(grid.at[chosen, "neighbor_ids"]).split("|") if "neighbor_ids" in grid else []
        for value in ids:
            if value and value != "nan":
                row = grid.iloc[int(value)]
                neighbor_rows.append({key: number(row[key]) for key in
                                      ("stop_usd", "target_usd", "viable", "total_pnl", "pnl_over_dd",
                                       "cvar5_loss_usd", "worst_loss_usd") if key in row})
    block_rows = []
    for label, params in (("CURRENT", freeze["current"]), ("CANDIDATE", freeze.get("candidate")),
                          ("SELECTED", freeze.get("selected"))):
        if params is None:
            continue
        hit = blocks[np.isclose(blocks.stop_usd, params["stop_usd"]) &
                     np.isclose(blocks.target_usd, params["target_usd"])]
        for row in hit.itertuples(index=False):
            block_rows.append({"variant": label, "block": int(row.block),
                               "net_pnl": number(row.net_pnl), "trades": int(row.trades),
                               "pnl_over_dd": number(row.pnl_over_dd),
                               "cvar5_loss_usd": number(row.cvar5_loss_usd)})

    q = qqq; n = target
    bars = {
        "t": (common.as_unit("ns").asi8 // 1_000_000_000).astype(int).tolist(),
        "qo": q.open.astype(float).tolist(), "qh": q.high.astype(float).tolist(),
        "ql": q.low.astype(float).tolist(), "qc": q.close.astype(float).tolist(),
        "no": n.open.astype(float).tolist(), "nh": n.high.astype(float).tolist(),
        "nl": n.low.astype(float).tolist(), "nc": n.close.astype(float).tolist(),
        "qqq_vwap": arrays["vwap_lead"].astype(float).tolist(),
        "target_vwap": arrays["vwap_target"].astype(float).tolist(),
        "fair": arrays["fair_price"].astype(float).tolist(),
        "z": [number(value) for value in arrays["z"]],
        "current_equity": current_equity.equity.astype(float).tolist(),
        "current_drawdown": current_equity.drawdown_usd.astype(float).tolist(),
        "selected_equity": selected_equity.equity.astype(float).tolist(),
        "selected_drawdown": selected_equity.drawdown_usd.astype(float).tolist(),
        "cash_equity": [100_000.0] * len(common), "cash_drawdown": [0.0] * len(common),
    }
    provenance_names = (
        "summary.json", "pre_seen_freeze.json", "pre_seen_freeze.sha256", "audit.json",
        "development_grid.csv", "block_metrics.csv", "folds.csv",
        "current_full_trades.csv", "current_full_equity.csv",
        "selected_full_trades.csv", "selected_full_equity.csv",
    )
    raw_manifest_path = ROOT / "data_cache" / "mega_cap_sip_manifest.json"
    raw_manifest = read_json(raw_manifest_path)
    declared_raw = raw_manifest["symbols"]
    provenance = {name: source(folder / name) for name in provenance_names}
    provenance["raw_input_manifest"] = source(raw_manifest_path)
    for key, raw_symbol in (("raw_qqq", "QQQ"), ("raw_target", symbol)):
        item = declared_raw[raw_symbol]
        raw_path = ROOT / item["file"]
        if not raw_path.is_file() or raw_path.stat().st_size != int(item["bytes"]):
            raise AssertionError(f"{symbol}: declared raw {raw_symbol} file/size differs")
        provenance[key] = {"path": item["file"], "bytes": int(item["bytes"]),
                           "sha256": item["sha256"], "feed": "Alpaca SIP",
                           "pairwise_rows": int(item["pairwise_rows_with_qqq"])}
    payload = {
        "meta": {
            "schema_version": 1, "symbol": symbol, "lead": "QQQ", "traded": symbol,
            "verdict": summary["verdict"], "current": summary["current"],
            "candidate": freeze.get("candidate"), "selected": summary.get("selected"),
            "entry": {"z_long": -2.5, "z_short": 2.5, "execution": "next raw open"},
            "execution": {"stop_first": True, "adverse_gap": True, "forced_eod": True,
                          "commission_per_share_side": 0.0035, "slippage_bps_each_execution": 2.0,
                          "notional_usd": 20_000.0, "capital_usd": 100_000.0},
            "period": {**summary["data"],
                       "seen_start_epoch": epoch(common[np.flatnonzero(common.date == pd.unique(common.date)[188])[0]])},
            "selection": freeze,
            "warning": summary["seen_warning"],
            "old_selection_warning": "Старый выбор оптимизировал один development/validation разрез. Новый verdict использует 9 блоков, walk-forward folds, Pareto, соседей, dominance и boundary gate.",
            "sources": provenance,
            "source": "Exact pairwise raw Alpaca SIP 1-minute RTH; no fill/resample/interpolation",
        },
        "bars": bars,
        "trades": {"current": trades(current_trades, "current"),
                   "selected": trades(selected_trades, "selected")},
        "results": {"current": summary["current_results"], "selected": summary["selected_results"]},
        "diagnostics": {
            "current_grid": current_row, "candidate_grid": candidate_row,
            "selected_grid": selected_row, "folds": compact_rows(folds),
            "blocks": block_rows, "neighbors": neighbor_rows,
        },
    }
    full_current = summary["current_results"]["full"]
    full_selected = summary["selected_results"]["full"]
    asset = {
        "symbol": symbol, "data": f"data/{symbol}.json", "verdict": summary["verdict"],
        "current": summary["current"], "candidate": freeze.get("candidate"),
        "selected": summary.get("selected"), "current_full_net_pnl": full_current["net_pnl"],
        "selected_full_net_pnl": full_selected["net_pnl"],
        "current_full_dd": full_current["max_drawdown_usd_mtm"],
        "selected_full_dd": full_selected["max_drawdown_usd_mtm"],
        "current_pre_seen_net_pnl": summary["current_results"]["pre_seen"]["net_pnl"],
        "selected_pre_seen_net_pnl": summary["selected_results"]["pre_seen"]["net_pnl"],
        "current_seen_net_pnl": summary["current_results"]["seen"]["net_pnl"],
        "selected_seen_net_pnl": summary["selected_results"]["seen"]["net_pnl"],
    }
    return payload, asset


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    progress = require_complete()
    OUT.mkdir(parents=True, exist_ok=True); DATA_OUT.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(str(ROOT / "data_cache"), "alpaca", "sip")
    assets = []
    for symbol in UNIVERSE:
        payload, asset = build_symbol(loader, symbol)
        path = DATA_OUT / f"{symbol}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"),
                                   allow_nan=False), encoding="utf-8")
        check = read_json(path)
        if len(check["bars"]["t"]) != int(payload["meta"]["period"]["raw_bars"]):
            raise AssertionError(f"{symbol}: JSON readback failed")
        asset.update({"bytes": path.stat().st_size, "sha256": sha(path)})
        assets.append(asset)
        print(f"BUILT {symbol}: {asset['verdict']}", flush=True)
    comparison = read_json(SRC / "cross_asset_summary.json")
    report = {
        "schema_version": 1, "generated_from": source(SRC / "progress.json"),
        "cross_asset_source": source(SRC / "cross_asset_summary.json"),
        "targets": list(UNIVERSE), "default_symbol": "NVDA", "assets": assets,
        "comparison": comparison, "roles": {"lead": "QQQ reference-only", "traded": "selected target only"},
        "warning": "SEEN 63-session period is diagnostic only and never ranks or gates candidates.",
    }
    (OUT / "manifest.js").write_text(
        "window.VWAP_MULTI_ASSET_WALKFORWARD_MANIFEST=" +
        json.dumps(report, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + ";\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "COMPLETE", "assets": len(assets),
                      "bytes": sum(item["bytes"] for item in assets)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
