"""Incrementally publish completed research variants into TradingView Lite."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import pandas as pd

from build_tradingview_lite_report import build_trade_items
from research_vwap_strategy import vwap_arrays
from rtharb.backtest.engine import BacktestEngine
from rtharb.config import AppConfig
from rtharb.data.loader import DataLoader
from rtharb.models.fair_value import FairValueModel
from rtharb.models.signals import SignalGenerator


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "tradingview_lite"


def finite(value, digits=5):
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def epoch(value) -> int:
    return int(pd.Timestamp(value).tz_convert("UTC").timestamp())


def publish_classic_rr(payload: dict) -> None:
    folder = ROOT / "research_output" / "risk_reward"
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    trades = pd.read_csv(folder / "selected_full_trades.csv")
    actual_start = int(payload["bars"]["t"][0])
    base_by_entry = {int(t["entry_time"]): t for t in payload["variants"]["base"]["trades"]}
    z_by_time = dict(zip(payload["bars"]["t"], payload["bars"]["z"]))
    items = []
    for row in trades.itertuples(index=False):
        entry_time, exit_time = epoch(row.entry_time), epoch(row.exit_time)
        if entry_time < actual_start:
            continue
        if entry_time not in base_by_entry:
            raise ValueError(f"RR entry is absent from the frozen base report: {row.entry_time}")
        item = deepcopy(base_by_entry[entry_time])
        item.update({
            "entry_reason": f"{item['entry_reason']}_RR_ONLY",
            "exit_time": exit_time,
            "exit_signal_time": exit_time,
            "exit_price": finite(row.exit_price),
            "exit_reference_price": finite(
                float(row.exit_price) / (1 - payload["meta"]["slippage_bps_per_execution"] / 10_000)
                if item["direction"] == 1
                else float(row.exit_price) / (1 + payload["meta"]["slippage_bps_per_execution"] / 10_000)
            ),
            "exit_reason": str(row.exit_reason),
            "exit_signal_z": finite(z_by_time.get(exit_time, float("nan"))),
            "exit_execution_z": finite(z_by_time.get(exit_time, float("nan"))),
            "shares": int(row.shares),
            "duration_minutes": int(row.duration_bars),
            "gross_pnl": finite(row.gross_pnl, 4),
            "commission": finite(row.commissions, 4),
            "slippage": finite(row.slippage, 4),
            "net_pnl": finite(row.net_pnl, 4),
            "return_pct": finite(float(row.net_pnl) / (float(row.entry_reference) * int(row.shares)) * 100),
            "stop_pct": finite(row.stop_pct, 6),
            "rr": finite(row.rr, 3),
            "stop_price": finite(row.stop_price),
            "target_price": finite(row.target_price),
        })
        items.append(item)
    net = sum(t["net_pnl"] for t in items)
    gross = sum(t["gross_pnl"] for t in items)
    wins = sum(t["net_pnl"] > 0 for t in items)
    selected = summary["selected"]
    payload["variants"]["rr_classic"] = {
        "label": f"Classic Z · stop {selected['stop_pct'] * 100:g}% · target {selected['rr']:g}R",
        "entry_mode": "z_only", "abs_threshold_usd": None,
        "anchor_filter": False, "inverse": False, "rr_only": True,
        "stop_pct": selected["stop_pct"], "rr": selected["rr"],
        "trades_count": len(items), "gross_pnl": round(gross, 4),
        "net_pnl": round(net, 4), "win_rate_pct": round(100 * wins / len(items), 3) if items else 0,
        "trades": items,
    }
    payload["meta"]["risk_reward_research"] = summary


def publish_vwap_z(payload: dict) -> None:
    summary = json.loads(
        (ROOT / "research_output" / "vwap_strategy" / "summary.json").read_text(encoding="utf-8")
    )
    selected = summary["selected"]
    cfg = AppConfig.load(str(ROOT / "configs" / "default_config.yaml"))
    lead, target = DataLoader(cfg.cache_dir, "alpaca", "sip").get_synchronized_pair("QQQ", "NVDA")
    common = lead.index.intersection(target.index)
    arrays = vwap_arrays(
        lead, target, int(selected["beta_days"]), int(selected["window"]),
        int(selected["warmup_bars"]),
    )
    metrics = pd.DataFrame(index=common)
    for column in ("open", "high", "low", "close"):
        metrics[f"target_{column}"] = target.loc[common, column].astype(float)
    metrics["target_fair_price"] = arrays["fair_price"]
    metrics["p0_target"] = arrays["p0_target"]
    metrics["z_score"] = arrays["z"]
    metrics["session_date"] = common.date
    metrics["time_str"] = common.strftime("%H:%M")
    metrics["bar_of_day"] = metrics.groupby("session_date", sort=False).cumcount()
    signals = SignalGenerator(
        z_entry=float(selected["z_entry"]), reversal_delta=float(selected["hook_delta"]),
        reversal_timeout_bars=int(selected["hook_timeout"]),
        enable_extreme_entry_lockout=selected["z_lockout"] is not None,
        enable_extreme_emergency_exit=False,
        z_max_allowed=float(selected["z_lockout"] or 99.0),
        z_exit=float(selected["exit_band"]), forced_close_time="15:55",
        min_session_warmup_bars=int(selected["warmup_bars"]),
    ).generate_signals(metrics)
    exact = BacktestEngine(
        cfg.backtest.initial_capital, cfg.backtest.position_size_usd,
        cfg.backtest.commission_per_share, cfg.backtest.slippage_pct, True,
    ).run(signals, "NVDA")["trades_df"]
    actual_start = pd.Timestamp(payload["bars"]["t"][0], unit="s", tz="UTC").tz_convert("America/New_York")
    items, filtered = build_trade_items(signals, exact, actual_start, target)
    net = float(filtered["net_pnl"].sum()) if not filtered.empty else 0.0
    gross = float(filtered["gross_pnl"].sum()) if not filtered.empty else 0.0
    wins = int((filtered["net_pnl"] > 0).sum()) if not filtered.empty else 0
    start_pos = common.get_indexer([actual_start])[0]
    view_slice = slice(start_pos, None)
    payload["bars"]["vz"] = [finite(x) for x in arrays["z"][view_slice]]
    payload["bars"]["vfair"] = [finite(x, 4) for x in arrays["fair_price"][view_slice]]
    payload["variants"]["vwap_z"] = {
        "label": "VWAP-Z · convergence", "entry_mode": "z_only",
        "abs_threshold_usd": None, "anchor_filter": False, "inverse": False,
        "rr_only": False, "z_basis": "vwap", "strategy": selected,
        "trades_count": len(items), "gross_pnl": round(gross, 4),
        "net_pnl": round(net, 4), "win_rate_pct": round(100 * wins / len(items), 3) if items else 0,
        "trades": items,
    }
    payload["meta"]["vwap_z_research"] = summary


def publish_vwap_rr(payload: dict) -> None:
    folder = ROOT / "research_output" / "risk_reward"
    all_summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    family = all_summary["vwap_z"]
    trades = pd.read_csv(folder / "vwap_z_selected_full_trades.csv")
    actual_start = int(payload["bars"]["t"][0])
    base_by_entry = {int(t["entry_time"]): t for t in payload["variants"]["vwap_z"]["trades"]}
    z_by_time = dict(zip(payload["bars"]["t"], payload["bars"]["vz"]))
    items = []
    for row in trades.itertuples(index=False):
        entry_time, exit_time = epoch(row.entry_time), epoch(row.exit_time)
        if entry_time < actual_start:
            continue
        if entry_time not in base_by_entry:
            raise ValueError(f"VWAP RR entry is absent from VWAP-Z report: {row.entry_time}")
        item = deepcopy(base_by_entry[entry_time])
        item.update({
            "entry_reason": f"{item['entry_reason']}_RR_ONLY",
            "exit_time": exit_time, "exit_signal_time": exit_time,
            "exit_price": finite(row.exit_price),
            "exit_reference_price": finite(
                float(row.exit_price) / (1 - payload["meta"]["slippage_bps_per_execution"] / 10_000)
                if item["direction"] == 1 else
                float(row.exit_price) / (1 + payload["meta"]["slippage_bps_per_execution"] / 10_000)
            ),
            "exit_reason": str(row.exit_reason),
            "exit_signal_z": finite(z_by_time.get(exit_time, float("nan"))),
            "exit_execution_z": finite(z_by_time.get(exit_time, float("nan"))),
            "shares": int(row.shares), "duration_minutes": int(row.duration_bars),
            "gross_pnl": finite(row.gross_pnl, 4), "commission": finite(row.commissions, 4),
            "slippage": finite(row.slippage, 4), "net_pnl": finite(row.net_pnl, 4),
            "return_pct": finite(float(row.net_pnl) / (float(row.entry_reference) * int(row.shares)) * 100),
            "stop_pct": finite(row.stop_pct, 6), "rr": finite(row.rr, 3),
            "stop_price": finite(row.stop_price), "target_price": finite(row.target_price),
        })
        items.append(item)
    net, gross = sum(t["net_pnl"] for t in items), sum(t["gross_pnl"] for t in items)
    wins, selected = sum(t["net_pnl"] > 0 for t in items), family["selected"]
    payload["variants"]["rr_vwap"] = {
        "label": f"VWAP-Z · stop {selected['stop_pct'] * 100:g}% · target {selected['rr']:g}R",
        "entry_mode": "z_only", "abs_threshold_usd": None,
        "anchor_filter": False, "inverse": False, "rr_only": True,
        "z_basis": "vwap", "strategy": payload["variants"]["vwap_z"]["strategy"],
        "stop_pct": selected["stop_pct"], "rr": selected["rr"],
        "trades_count": len(items), "gross_pnl": round(gross, 4), "net_pnl": round(net, 4),
        "win_rate_pct": round(100 * wins / len(items), 3) if items else 0, "trades": items,
    }


def publish_duration_stoploss(payload: dict) -> None:
    """Publish the independently evaluated q95 time and price-risk overlays."""
    folder = ROOT / "research_output" / "duration_stoploss_verified"
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    selected = summary["frozen_parameters"]
    cfg = AppConfig.load(str(ROOT / "configs" / "default_config.yaml"))
    lead, target = DataLoader(cfg.cache_dir, "alpaca", "sip").get_synchronized_pair("QQQ", "NVDA")
    metrics = FairValueModel(
        selected["beta_mode"], int(selected["beta_days"]), int(selected["window"]), 15,
    ).compute_intraday_metrics(lead, target)
    signals = SignalGenerator(
        z_entry=float(selected["z_entry"]), reversal_delta=float(selected["hook_delta"]),
        reversal_timeout_bars=int(selected["hook_timeout"]),
        enable_extreme_entry_lockout=True, enable_extreme_emergency_exit=False,
        z_max_allowed=float(selected["z_lockout"]), lockout_mode="day_lockout",
        z_exit=float(selected["exit_band"]), forced_close_time="15:55",
        min_session_warmup_bars=15,
    ).generate_signals(metrics)
    actual_start = pd.Timestamp(payload["bars"]["t"][0], unit="s", tz="UTC").tz_convert("America/New_York")
    qkey = "0.95"
    selected_stop = float(summary["selected_q95_overlays"]["stop_loss"]["selected_threshold_pct"])
    definitions = {
        "time_stop_q95": {
            "label": f"Base Z · time-stop {summary['winner_duration_quantiles_bars'][qkey]} мин (95%)",
            "max_holding_bars": int(summary["winner_duration_quantiles_bars"][qkey]),
            "stop_loss_pct": None,
        },
        "stop_loss_q95": {
            "label": f"Base Z · stop {selected_stop * 100:.3f}% (≥95% winners)",
            "max_holding_bars": None,
            "stop_loss_pct": selected_stop,
        },
    }
    for key, definition in definitions.items():
        exact = BacktestEngine(
            cfg.backtest.initial_capital, cfg.backtest.position_size_usd,
            cfg.backtest.commission_per_share, cfg.backtest.slippage_pct, True,
            max_holding_bars=definition["max_holding_bars"],
            stop_loss_pct=definition["stop_loss_pct"],
        ).run(signals, "NVDA")["trades_df"]
        items, filtered = build_trade_items(signals, exact, actual_start, target)
        for item in items:
            if item["exit_reason"] in {"TIME_STOP", "STOP_LOSS"}:
                item["exit_signal_time"] = item["exit_time"]
                item["exit_signal_z"] = item["exit_execution_z"]
                if item["exit_reason"] == "STOP_LOSS":
                    slip = payload["meta"]["slippage_bps_per_execution"] / 10_000
                    item["exit_reference_price"] = finite(
                        item["exit_price"] / (1 - slip) if item["direction"] == 1
                        else item["exit_price"] / (1 + slip)
                    )
        net = float(filtered["net_pnl"].sum()) if not filtered.empty else 0.0
        gross = float(filtered["gross_pnl"].sum()) if not filtered.empty else 0.0
        wins = int((filtered["net_pnl"] > 0).sum()) if not filtered.empty else 0
        payload["variants"][key] = {
            "label": definition["label"], "entry_mode": "z_only",
            "abs_threshold_usd": None, "anchor_filter": False, "inverse": False,
            "rr_only": False, "risk_overlay": key,
            "max_holding_bars": definition["max_holding_bars"],
            "stop_loss_pct": definition["stop_loss_pct"],
            "strategy": selected, "trades_count": len(items),
            "gross_pnl": round(gross, 4), "net_pnl": round(net, 4),
            "win_rate_pct": round(100 * wins / len(items), 3) if items else 0,
            "trades": items,
        }
    payload["meta"]["duration_stoploss_research"] = summary


def main() -> None:
    payload = json.loads((OUT / "report_data.json").read_text(encoding="utf-8"))
    publish_classic_rr(payload)
    publish_vwap_z(payload)
    publish_vwap_rr(payload)
    publish_duration_stoploss(payload)
    compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    (OUT / "data.js").write_text("window.RT_HARB_DATA=" + compact + ";\n", encoding="utf-8")
    (OUT / "report_data.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps({
        "published": {
            name: {"trades": payload["variants"][name]["trades_count"], "net_pnl": payload["variants"][name]["net_pnl"]}
            for name in ("rr_classic", "vwap_z", "rr_vwap", "time_stop_q95", "stop_loss_q95")
        }
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
