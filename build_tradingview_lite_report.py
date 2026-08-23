"""Build a raw 1-minute TradingView Lightweight Charts report for the latest two months."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rtharb.backtest.engine import BacktestEngine
from rtharb.config import AppConfig
from rtharb.data.loader import DataLoader
from rtharb.models.fair_value import FairValueModel
from rtharb.models.signals import SignalGenerator, SignalType


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "tradingview_lite"
NY = "America/New_York"


def finite(value: float, digits: int = 5):
    return round(float(value), digits) if math.isfinite(float(value)) else None


def epoch_seconds(ts: pd.Timestamp) -> int:
    return int(pd.Timestamp(ts).tz_convert("UTC").timestamp())


def previous_bar(index: pd.DatetimeIndex, timestamp: pd.Timestamp) -> tuple[int, pd.Timestamp]:
    pos = int(index.get_indexer([timestamp])[0])
    if pos <= 0:
        raise ValueError(f"No previous bar for {timestamp}")
    return pos - 1, index[pos - 1]


def hook_details(signals: pd.DataFrame, signal_pos: int, direction: int) -> dict:
    prefix = "ARMED_LONG" if direction == 1 else "ARMED_SHORT"
    session = signals.iloc[signal_pos]["session_date"]
    first = signal_pos
    while first > 0:
        prev = signals.iloc[first - 1]
        if prev["session_date"] != session or not str(prev["armed_state"]).startswith(prefix):
            break
        first -= 1
    segment = signals.iloc[first : signal_pos + 1]
    if segment.empty or not str(segment.iloc[-1]["armed_state"]).startswith(prefix):
        segment = signals.iloc[signal_pos : signal_pos + 1]
    z = segment["z_score"].astype(float)
    extreme = float(z.min() if direction == 1 else z.max())
    signal_z = float(signals.iloc[signal_pos]["z_score"])
    delta = signal_z - extreme if direction == 1 else extreme - signal_z
    return {
        "armed_time": epoch_seconds(segment.index[0]),
        "armed_z": finite(z.iloc[0]),
        "extreme_z": finite(extreme),
        "signal_z": finite(signal_z),
        "hook_delta": finite(delta),
        "hook_bars": int(signal_pos - first),
        "entry_trigger": str(signals.iloc[signal_pos].get("entry_trigger", "Z")),
        "armed_abs_deviation_usd": finite(signals.iloc[first].get("abs_deviation_usd", np.nan), 4),
        "signal_abs_deviation_usd": finite(signals.iloc[signal_pos].get("abs_deviation_usd", np.nan), 4),
        "signal_anchor_price": finite(signals.iloc[signal_pos].get("p0_target", np.nan), 4),
        "signal_target_price": finite(signals.iloc[signal_pos].get("target_close", np.nan), 4),
    }


def build_trade_items(signals: pd.DataFrame, trades: pd.DataFrame,
                      actual_start: pd.Timestamp, target: pd.DataFrame,
                      source_direction_multiplier: int = 1) -> tuple[list[dict], pd.DataFrame]:
    if trades.empty:
        return [], trades.copy()
    entry_times = pd.to_datetime(trades["entry_time"], utc=True).dt.tz_convert(NY)
    trade_view = trades.loc[entry_times >= actual_start].copy()
    items = []
    for _, tr in trade_view.iterrows():
        entry_ts = pd.Timestamp(tr["entry_time"])
        exit_ts = pd.Timestamp(tr["exit_time"])
        entry_pos = int(signals.index.get_indexer([entry_ts])[0])
        exit_pos = int(signals.index.get_indexer([exit_ts])[0])
        if entry_pos < 1 or exit_pos < 0:
            raise ValueError(f"Trade timestamp missing from synchronized bars: #{tr['trade_id']}")
        signal_pos, signal_ts = previous_bar(signals.index, entry_ts)
        direction = int(tr["direction"])
        source_signal_direction = direction * source_direction_multiplier
        hook = hook_details(signals, signal_pos, source_signal_direction)
        exit_signal_pos = max(exit_pos - 1, 0)
        exit_signal_ts = signals.index[exit_signal_pos]
        entry_reference = float(target.loc[entry_ts, "open"])
        if tr["exit_reason"] in {"SESSION_END_FALLBACK", "BACKTEST_END"}:
            exit_reference = float(target.loc[exit_ts, "close"])
            exit_signal_ts = exit_ts
            exit_signal_pos = exit_pos
        else:
            exit_reference = float(target.loc[exit_ts, "open"])
        action = "BUY" if direction == 1 else "SHORT"
        inverse_prefix = "INVERSE_OF_" if source_direction_multiplier == -1 else ""
        reason_code = f"{action}_{inverse_prefix}{hook['entry_trigger']}_REVERSAL_HOOK"
        items.append({
            "id": int(tr["trade_id"]), "side": "LONG" if direction == 1 else "SHORT",
            "direction": direction, "source_signal_direction": source_signal_direction,
            "entry_time": epoch_seconds(entry_ts),
            "entry_signal_time": epoch_seconds(signal_ts), "entry_price": finite(tr["entry_price"], 5),
            "entry_reference_price": finite(entry_reference, 5), "entry_reason": reason_code,
            **hook, "exit_time": epoch_seconds(exit_ts), "exit_signal_time": epoch_seconds(exit_signal_ts),
            "exit_price": finite(tr["exit_price"], 5), "exit_reference_price": finite(exit_reference, 5),
            "exit_reason": str(tr["exit_reason"]), "exit_signal_z": finite(signals.iloc[exit_signal_pos]["z_score"]),
            "exit_execution_z": finite(signals.iloc[exit_pos]["z_score"]), "shares": int(tr["shares"]),
            "duration_minutes": int(tr["duration_bars"]), "gross_pnl": finite(tr["gross_pnl"], 4),
            "commission": finite(tr["commission"], 4), "slippage": finite(tr["slippage"], 4),
            "net_pnl": finite(tr["net_pnl"], 4), "return_pct": finite(float(tr["return_pct"]) * 100, 5),
        })
    return items, trade_view


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(exist_ok=True)

    selected = json.loads(
        (ROOT / "research_output" / "base_strategy_summary.json").read_text(encoding="utf-8")
    )["selected_parameters"]
    cfg = AppConfig.load(str(ROOT / "configs" / "default_config.yaml"))
    lead, target = DataLoader(cfg.cache_dir, "alpaca", "sip").get_synchronized_pair("QQQ", "NVDA")

    metrics = FairValueModel(
        selected["beta_mode"], selected["beta_days"], selected["window"], 15,
        vwap_mode="session",
    ).compute_intraday_metrics(lead, target)
    absolute_summary = json.loads(
        (ROOT / "research_output" / "absolute_filters" / "summary.json").read_text(encoding="utf-8")
    )
    chosen_abs = absolute_summary["chosen_after_development_validation"]["abs_threshold_usd"] or 7.5
    variant_defs = {
        "base": {"label": "Base Z", "entry_mode": "z_only", "abs_threshold_usd": None, "anchor_filter": False, "inverse": False},
        "base_anchor": {"label": "Base Z + 09:30 anchor", "entry_mode": "z_only", "abs_threshold_usd": None, "anchor_filter": True, "inverse": False},
        "hybrid_abs": {"label": f"Z OR |Δ$|≥${chosen_abs:g}", "entry_mode": "z_or_abs", "abs_threshold_usd": chosen_abs, "anchor_filter": False, "inverse": False},
        "hybrid_abs_anchor": {"label": f"Z OR |Δ$|≥${chosen_abs:g} + anchor", "entry_mode": "z_or_abs", "abs_threshold_usd": chosen_abs, "anchor_filter": True, "inverse": False},
    }
    variant_runs = {}
    for key, definition in variant_defs.items():
        signals_variant = SignalGenerator(
            z_entry=selected["z_entry"], reversal_delta=selected["hook_delta"],
            reversal_timeout_bars=selected["hook_timeout"],
            enable_extreme_entry_lockout=True, enable_extreme_emergency_exit=False,
            z_max_allowed=selected["z_lockout"], z_exit=selected["exit_band"],
            forced_close_time="15:55", min_session_warmup_bars=15,
            entry_mode=definition["entry_mode"],
            entry_abs_deviation_usd=definition["abs_threshold_usd"],
            enable_open_anchor_filter=definition["anchor_filter"],
        ).generate_signals(metrics)
        bt_variant = BacktestEngine(
            cfg.backtest.initial_capital, cfg.backtest.position_size_usd,
            cfg.backtest.commission_per_share, cfg.backtest.slippage_pct, True,
        ).run(signals_variant, "NVDA")
        variant_runs[key] = {"signals": signals_variant, "trades": bt_variant["trades_df"].copy()}
    inverse_signals = variant_runs["base"]["signals"].copy()
    inverse_signals["signal"] = inverse_signals["signal"].replace({
        SignalType.BUY_LONG: SignalType.SELL_SHORT,
        SignalType.SELL_SHORT: SignalType.BUY_LONG,
    })
    inverse_bt = BacktestEngine(
        cfg.backtest.initial_capital, cfg.backtest.position_size_usd,
        cfg.backtest.commission_per_share, cfg.backtest.slippage_pct, True,
    ).run(inverse_signals, "NVDA")
    variant_defs["base_inverse"] = {
        "label": "Base Z · направление наоборот", "entry_mode": "z_only",
        "abs_threshold_usd": None, "anchor_filter": False, "inverse": True,
    }
    variant_runs["base_inverse"] = {
        "signals": variant_runs["base"]["signals"],
        "trades": inverse_bt["trades_df"].copy(),
    }

    signals = variant_runs["base"]["signals"]
    data_end = metrics.index.max()
    requested_start = data_end.normalize() - pd.DateOffset(months=2)
    view_index = signals.index[signals.index >= requested_start]
    actual_start = view_index.min()
    view = signals.loc[view_index]
    lead_view = lead.loc[view_index]
    target_view = target.loc[view_index]

    arrays = {
        "t": [epoch_seconds(ts) for ts in view_index],
        "qo": [finite(x, 4) for x in lead_view["open"]],
        "qh": [finite(x, 4) for x in lead_view["high"]],
        "ql": [finite(x, 4) for x in lead_view["low"]],
        "qc": [finite(x, 4) for x in lead_view["close"]],
        "no": [finite(x, 4) for x in target_view["open"]],
        "nh": [finite(x, 4) for x in target_view["high"]],
        "nl": [finite(x, 4) for x in target_view["low"]],
        "nc": [finite(x, 4) for x in target_view["close"]],
        "z": [finite(x, 5) for x in view["z_score"]],
        "fair": [finite(x, 4) for x in view["target_fair_price"]],
        "absd": [finite(x, 4) for x in view["target_close"] - view["target_fair_price"]],
        "qvwap": [finite(x, 4) for x in view["lead_vwap"]],
        "nvwap": [finite(x, 4) for x in view["target_vwap"]],
    }
    variants = {}
    for key, definition in variant_defs.items():
        items, filtered = build_trade_items(
            variant_runs[key]["signals"], variant_runs[key]["trades"], actual_start, target,
            source_direction_multiplier=-1 if definition["inverse"] else 1,
        )
        net = float(filtered["net_pnl"].sum()) if not filtered.empty else 0.0
        gross = float(filtered["gross_pnl"].sum()) if not filtered.empty else 0.0
        wins = int((filtered["net_pnl"] > 0).sum()) if not filtered.empty else 0
        variants[key] = {
            **definition, "trades_count": len(items), "gross_pnl": round(gross, 4),
            "net_pnl": round(net, 4), "win_rate_pct": round(100 * wins / len(filtered), 3) if len(filtered) else 0,
            "trades": items,
        }
    trade_items = variants["base"]["trades"]
    payload = {
        "meta": {
            "source": "Alpaca SIP raw 1-minute OHLC, synchronized QQQ/NVDA",
            "timezone": NY,
            "requested_start": str(requested_start.date()),
            "actual_start": str(actual_start),
            "end": str(data_end),
            "bars": len(view_index),
            "sessions": int(view["session_date"].nunique()),
            "trades": len(trade_items),
            "gross_pnl": variants["base"]["gross_pnl"],
            "net_pnl": variants["base"]["net_pnl"],
            "win_rate_pct": variants["base"]["win_rate_pct"],
            "position_size_usd": cfg.backtest.position_size_usd,
            "commission_per_share_per_side": cfg.backtest.commission_per_share,
            "slippage_bps_per_execution": cfg.backtest.slippage_pct * 10000,
            "strategy": selected,
            "absolute_filter_research": {
                "chosen": absolute_summary["chosen_after_development_validation"],
                "baseline_holdout": absolute_summary["baseline"]["holdout"],
                "chosen_holdout": absolute_summary["chosen_holdout"],
            },
            "reverse_research": json.loads(
                (ROOT / "research_output" / "reverse_strategy" / "summary.json").read_text(encoding="utf-8")
            )["periods"],
        },
        "bars": arrays,
        "trades": trade_items,
        "variants": variants,
    }
    compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    (OUT / "data.js").write_text("window.RT_HARB_DATA=" + compact + ";\n", encoding="utf-8")
    (OUT / "report_data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUT),
                "actual_start": str(actual_start),
                "end": str(data_end),
                "bars": len(view_index),
                "sessions": payload["meta"]["sessions"],
                "trades": len(trade_items),
                "variants": {k: {"trades": v["trades_count"], "net_pnl": v["net_pnl"]} for k, v in variants.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
