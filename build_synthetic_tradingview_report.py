"""Build the standalone interactive QQQ-versus-synthetic-basket report."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from research_synthetic_index import load_raw, market_arrays, rolling_z


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "tradingview_synthetic"
SRC = ROOT / "research_output" / "synthetic_index"


def finite(value, digits=5):
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def epoch(value) -> int:
    return int(pd.Timestamp(value).tz_convert("UTC").timestamp())


def selected_metric(selected: dict, arrays: dict) -> np.ndarray:
    if selected["basis"] == "z":
        return rolling_z(arrays["spread"], arrays["starts"], arrays["ends"], int(selected["window"]))
    if selected["basis"] == "absolute_qqq_points":
        return arrays["points"].copy()
    raise ValueError(f"Unsupported synthetic basis: {selected['basis']}")


def trade_items(path: Path, strategy: str, common: pd.DatetimeIndex) -> list[dict]:
    raw = pd.read_csv(path)
    by_time = {ts: i for i, ts in enumerate(common)}
    items = []
    for number, row in enumerate(raw.itertuples(index=False), 1):
        entry_ts, exit_ts = pd.Timestamp(row.entry_time), pd.Timestamp(row.exit_time)
        entry_i, exit_i = by_time.get(entry_ts), by_time.get(exit_ts)
        if entry_i is None or exit_i is None or entry_i < 1:
            raise ValueError(f"Trade timestamp absent from common SIP bars: {entry_ts} / {exit_ts}")
        direction = 1 if str(row.direction).upper() == "LONG" else -1
        commission, slippage = float(row.commission), float(row.slippage)
        gross, net = float(row.gross_pnl), float(row.net_pnl)
        item = {
            "id": number, "strategy": strategy,
            "side": "LONG" if direction == 1 else "SHORT", "direction": direction,
            "entry_signal_time": epoch(common[entry_i - 1]), "entry_time": epoch(entry_ts),
            "exit_time": epoch(exit_ts), "entry_reference": finite(row.entry_reference),
            "entry_price": finite(row.entry_price), "exit_reference": finite(row.exit_reference),
            "exit_price": finite(row.exit_price), "shares": int(row.shares),
            "exit_reason": str(row.exit_reason), "duration_minutes": int(row.duration_minutes),
            "entry_basis": str(row.entry_basis), "entry_threshold": finite(row.entry_threshold),
            "entry_hook": finite(row.entry_hook), "entry_window": int(row.entry_window),
            "signal_metric": finite(row.signal_metric), "signal_spread": finite(row.signal_spread, 7),
            "signal_points": finite(row.signal_points), "gross_pnl": finite(gross, 4),
            "commission": finite(commission, 4), "slippage": finite(slippage, 4),
            "net_pnl": finite(net, 4), "costs": finite(commission + slippage, 4),
            "reconciliation_error": finite(gross - commission - slippage - net, 8),
        }
        if strategy == "risk_reward":
            item.update({"stop_pct": finite(row.stop_pct, 6), "rr": finite(row.rr, 3),
                         "stop_price": finite(row.stop_price), "target_price": finite(row.target_price)})
        items.append(item)
    return items


def variant(name: str, summary: dict, trades: list[dict]) -> dict:
    return {
        "label": "Схождение" if name == "convergence" else "RR без схождения",
        "selected": summary[name]["selected"], "selection": summary[name]["selection"],
        "results": summary[name]["results"], "trades_count": len(trades),
        "net_pnl": round(sum(t["net_pnl"] for t in trades), 4),
        "gross_pnl": round(sum(t["gross_pnl"] for t in trades), 4),
        "win_rate_pct": round(100 * sum(t["net_pnl"] > 0 for t in trades) / len(trades), 3) if trades else 0,
        "trades": trades,
    }


def main() -> None:
    OUT.mkdir(exist_ok=True)
    summary = json.loads((SRC / "summary.json").read_text(encoding="utf-8"))
    frames, common, day = load_raw()
    arrays = market_arrays(frames, common, day)
    qqq_anchor = arrays["close"][arrays["starts"]][day]
    basket_equivalent = qqq_anchor * (1.0 + arrays["basket_return"])
    conv_trades = trade_items(SRC / "convergence_selected_full_trades.csv", "convergence", common)
    rr_trades = trade_items(SRC / "risk_reward_selected_full_trades.csv", "risk_reward", common)
    variants = {"convergence": variant("convergence", summary, conv_trades),
                "risk_reward": variant("risk_reward", summary, rr_trades)}
    payload = {
        "meta": {"source": summary["data"]["source"], "data": summary["data"],
                 "basket": summary["basket"], "execution": summary["execution"],
                 "splits": summary["splits"]},
        "bars": {
            "t": [epoch(ts) for ts in common], "qo": [finite(x, 4) for x in arrays["open"]],
            "qh": [finite(x, 4) for x in arrays["high"]], "ql": [finite(x, 4) for x in arrays["low"]],
            "qc": [finite(x, 4) for x in arrays["close"]],
            "basket": [finite(x, 4) for x in basket_equivalent],
            "points": [finite(x, 5) for x in arrays["points"]],
            "spread": [finite(x, 7) for x in arrays["spread"]],
            "conv_metric": [finite(x) for x in selected_metric(summary["convergence"]["selected"], arrays)],
            "rr_metric": [finite(x) for x in selected_metric(summary["risk_reward"]["selected"], arrays)],
        },
        "variants": variants,
    }
    compact = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    (OUT / "data.js").write_text("window.SYNTHETIC_DATA=" + compact + ";\n", encoding="utf-8")
    (OUT / "report_data.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"bars": len(common), "sessions": summary["data"]["sessions"],
                      "variants": {k: {"trades": v["trades_count"], "net_pnl": v["net_pnl"]}
                                   for k, v in variants.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
