"""Independent integrity audit for the event-driven VWAP absolute report.

This module deliberately does not import the report builder or its research
simulator.  It reconstructs raw synchronized bars, session VWAP, fair value,
rolling Z, every published execution, and minute mark-to-market equity from
the saved source artifacts.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rtharb.config import AppConfig
from rtharb.data.loader import DataLoader


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "tradingview_vwap_absolute"
SOURCE_CANDIDATES = (
    ROOT / "research_output" / "vwap_absolute_event_driven",
    ROOT / "research_output" / "vwap_absolute_brackets_event_driven",
)
NY = "America/New_York"
ATOL = 1e-8


def close(actual: Any, expected: Any, label: str, atol: float = ATOL) -> None:
    a, e = float(actual), float(expected)
    if not (math.isfinite(a) and math.isfinite(e)) or not math.isclose(a, e, abs_tol=atol, rel_tol=1e-10):
        raise AssertionError(f"{label}: {a!r} != {e!r}")


def epoch(value: Any) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise AssertionError(f"Naive source timestamp: {value!r}")
    return int(ts.tz_convert("UTC").timestamp())


def source_dir() -> Path:
    required = ("summary.json", "selected_full_trades.csv", "selected_full_equity.csv")
    for path in SOURCE_CANDIDATES:
        if all((path / name).is_file() for name in required):
            return path
    searched = ", ".join(str(path) for path in SOURCE_CANDIDATES)
    raise FileNotFoundError(f"Event-driven artifacts are not ready; searched: {searched}")


def load_payload() -> dict[str, Any]:
    json_path, js_path = REPORT / "report_data.json", REPORT / "data.js"
    if not json_path.is_file() or not js_path.is_file():
        raise FileNotFoundError("Run python -m rtharb.reporting.vwap_absolute_trading first")
    raw = json_path.read_text(encoding="utf-8").strip()
    payload = json.loads(raw)
    prefix = "window.VWAP_ABSOLUTE_DATA="
    js = js_path.read_text(encoding="utf-8").strip()
    if not js.startswith(prefix) or not js.endswith(";"):
        raise AssertionError("data.js wrapper is malformed")
    if json.loads(js[len(prefix):-1]) != payload:
        raise AssertionError("data.js and report_data.json payloads differ")
    return payload


def independent_arrays(qqq: pd.DataFrame, nvda: pd.DataFrame, beta_days: int,
                       window: int, warmup: int) -> dict[str, np.ndarray]:
    common = qqq.index.intersection(nvda.index)
    qqq, nvda = qqq.loc[common], nvda.loc[common]
    dates = np.asarray(common.date)
    unique_days = pd.unique(dates)
    day = pd.factorize(dates, sort=False)[0]
    starts = np.r_[0, np.flatnonzero(day[1:] != day[:-1]) + 1]
    ends = np.r_[starts[1:] - 1, len(common) - 1]

    def session_vwap(frame: pd.DataFrame) -> np.ndarray:
        typical = (frame.high.to_numpy(float) + frame.low.to_numpy(float) + frame.close.to_numpy(float)) / 3.0
        volume = frame.volume.to_numpy(float)
        out = np.full(len(frame), np.nan)
        for start, end in zip(starts, ends):
            v = volume[start:end + 1]
            cv = np.cumsum(v)
            out[start:end + 1] = np.divide(
                np.cumsum(typical[start:end + 1] * v), cv,
                out=np.full(len(v), np.nan), where=cv > 0,
            )
            close(out[start], typical[start], f"VWAP session reset {common[start]}")
        return out

    qvwap, nvwap = session_vwap(qqq), session_vwap(nvda)
    q_daily = qqq.close.to_numpy(float)[ends]
    n_daily = nvda.close.to_numpy(float)[ends]
    qr, nr = pd.Series(q_daily).pct_change(), pd.Series(n_daily).pct_change()
    beta = (nr.rolling(beta_days, min_periods=beta_days).cov(qr) /
            qr.rolling(beta_days, min_periods=beta_days).var()).shift(1).clip(0.2, 4.0).fillna(1.5).to_numpy()
    spread = (nvda.close.to_numpy(float) / nvwap - 1.0 -
              beta[day] * (qqq.close.to_numpy(float) / qvwap - 1.0))
    fair = nvwap * (1.0 + beta[day] * (qqq.close.to_numpy(float) / qvwap - 1.0))
    z = np.full(len(spread), np.nan)
    for start, end in zip(starts, ends):
        x = spread[start:end + 1]
        count = np.minimum(np.arange(1, len(x) + 1), window)
        roll_start = np.maximum(0, np.arange(len(x)) - window + 1)
        cs, cs2 = np.r_[0.0, np.cumsum(x)], np.r_[0.0, np.cumsum(x * x)]
        total = cs[np.arange(1, len(x) + 1)] - cs[roll_start]
        total2 = cs2[np.arange(1, len(x) + 1)] - cs2[roll_start]
        variance = np.divide(total2 - total * total / count, count - 1,
                             out=np.full(len(x), np.nan), where=count > 1)
        std = np.sqrt(np.maximum(variance, 0.0))
        values = np.divide(x - total / count, std, out=np.full(len(x), np.nan), where=std > 1e-8)
        values[:warmup] = np.nan
        z[start:end + 1] = values
    return {"qvwap": qvwap, "nvwap": nvwap, "fair": fair, "z": z,
            "starts": starts, "ends": ends, "days": np.asarray(unique_days)}


def payload_value(trade: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in trade:
            return trade[name]
    raise AssertionError(f"Trade payload is missing aliases {names}")


def audit_trade(row: Any, item: dict[str, Any], number: int, common: pd.DatetimeIndex,
                nvda: pd.DataFrame, z: np.ndarray, selected: dict[str, Any],
                commission: float, slip: float, by_epoch: dict[int, int],
                day_end_by_date: dict[Any, int]) -> None:
    signal_ts, entry_ts, exit_ts = pd.Timestamp(row.signal_time), pd.Timestamp(row.entry_time), pd.Timestamp(row.exit_time)
    si, ei, xi = by_epoch[epoch(signal_ts)], by_epoch[epoch(entry_ts)], by_epoch[epoch(exit_ts)]
    if ei != si + 1 or signal_ts.date() != entry_ts.date():
        raise AssertionError(f"Trade {number}: entry is not the next synchronized open after signal close")
    if int(item["entry_signal_time"]) != epoch(signal_ts) or int(item["entry_time"]) != epoch(entry_ts) or int(item["exit_time"]) != epoch(exit_ts):
        raise AssertionError(f"Trade {number}: report/source timestamps differ")
    close(z[si], row.entry_z, f"trade {number} signal Z", 1e-10)
    close(item["entry_z"], row.entry_z, f"trade {number} payload Z", 1e-10)
    direction = 1 if str(row.direction).upper() == "LONG" else -1
    if (direction == 1 and z[si] > -float(selected.get("z_entry", 2.5))) or (direction == -1 and z[si] < float(selected.get("z_entry", 2.5))):
        raise AssertionError(f"Trade {number}: direction contradicts VWAP-Z")
    raw_open = float(nvda.open.iloc[ei])
    close(raw_open, row.entry_reference, f"trade {number} raw entry open")
    entry_eff = raw_open * (1 + slip if direction == 1 else 1 - slip)
    close(row.entry_price, entry_eff, f"trade {number} entry execution")
    if int(row.shares) != math.floor(float(selected["position_notional_usd"]) / entry_eff):
        raise AssertionError(f"Trade {number}: position size mismatch")
    stop_usd, target_usd = float(row.stop_usd_per_share), float(row.target_usd_per_share)
    close(stop_usd, selected["stop_usd"], f"trade {number} stop distance")
    close(target_usd, selected["target_usd"], f"trade {number} target distance")
    stop = raw_open - stop_usd if direction == 1 else raw_open + stop_usd
    target = raw_open + target_usd if direction == 1 else raw_open - target_usd
    close(row.stop_price, stop, f"trade {number} stop price")
    close(row.target_price, target, f"trade {number} target price")

    # Independently prove that the saved exit is the first permissible raw-bar exit.
    expected_i, expected_raw, expected_reason = None, None, None
    day = entry_ts.date()
    day_end = day_end_by_date[day]
    for i in range(ei, day_end + 1):
        op, hi, lo = float(nvda.open.iloc[i]), float(nvda.high.iloc[i]), float(nvda.low.iloc[i])
        stop_hit = (op <= stop or lo <= stop) if direction == 1 else (op >= stop or hi >= stop)
        target_hit = hi >= target if direction == 1 else lo <= target
        if stop_hit:
            gap = op <= stop if direction == 1 else op >= stop
            expected_i, expected_raw, expected_reason = i, (op if gap else stop), "STOP"
            break
        if target_hit:
            expected_i, expected_raw, expected_reason = i, target, "TAKE_PROFIT_BRACKET"
            break
    if expected_i is None:
        expected_i, expected_raw, expected_reason = day_end, float(nvda.close.iloc[day_end]), "FORCED_EOD"
    if xi != expected_i or str(row.exit_reason) != expected_reason:
        raise AssertionError(f"Trade {number}: first raw bracket exit mismatch")
    close(row.exit_reference, expected_raw, f"trade {number} raw exit")
    exit_eff = expected_raw * (1 - slip if direction == 1 else 1 + slip)
    close(row.exit_price, exit_eff, f"trade {number} exit execution")
    shares = int(row.shares)
    gross = direction * (expected_raw - raw_open) * shares
    slippage = (abs(entry_eff - raw_open) + abs(exit_eff - expected_raw)) * shares
    commissions = 2 * shares * commission
    close(row.gross_pnl, gross, f"trade {number} gross")
    close(row.slippage, slippage, f"trade {number} slippage")
    close(row.commissions, commissions, f"trade {number} commissions")
    close(row.costs, slippage + commissions, f"trade {number} costs")
    close(row.net_pnl, gross - slippage - commissions, f"trade {number} net")
    for source_name, aliases in {
        "entry_price": ("entry_price", "entry_fill_price"),
        "exit_price": ("exit_price", "exit_fill_price"),
        "entry_reference": ("entry_reference",), "exit_reference": ("exit_reference",),
        "stop_price": ("stop_price",), "target_price": ("target_price",),
        "gross_pnl": ("gross_pnl",), "slippage": ("slippage",),
        "commissions": ("commissions", "commission"), "costs": ("costs",), "net_pnl": ("net_pnl",),
    }.items():
        close(payload_value(item, *aliases), getattr(row, source_name), f"trade {number} payload {source_name}")


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    src = source_dir()
    summary = json.loads((src / "summary.json").read_text(encoding="utf-8"))
    payload = load_payload()
    bars = payload["bars"]
    if set(payload) != {"meta", "bars", "trades", "results"}:
        raise AssertionError(f"Unexpected top-level payload keys: {set(payload)}")

    period = summary["period"]
    start, end = pd.Timestamp(period["start"]).date(), pd.Timestamp(period["end"]).date()
    cfg = AppConfig.load(str(ROOT / "configs" / "default_config.yaml"))
    qqq_all, nvda_all = DataLoader(cfg.cache_dir, "alpaca", "sip").get_synchronized_pair("QQQ", "NVDA")
    full_common = qqq_all.index.intersection(nvda_all.index)
    qqq_all, nvda_all = qqq_all.loc[full_common], nvda_all.loc[full_common]
    keep = np.asarray([(start <= ts.date() <= end) for ts in full_common], dtype=bool)
    common = full_common[keep]
    qqq, nvda = qqq_all.loc[common], nvda_all.loc[common]
    if len(common) != int(period["raw_bars"]) or len(common) != len(bars["t"]):
        raise AssertionError("Raw synchronized bar count mismatch")
    if [epoch(ts) for ts in common] != bars["t"]:
        raise AssertionError("Payload raw timestamps differ from Alpaca SIP archive")
    for key, expected in {"qo": qqq.open, "qh": qqq.high, "ql": qqq.low, "qc": qqq.close, "qv": qqq.volume,
                          "no": nvda.open, "nh": nvda.high, "nl": nvda.low, "nc": nvda.close, "nv": nvda.volume}.items():
        if not np.allclose(np.asarray(bars[key], float), expected.to_numpy(float), atol=0.0, rtol=0.0):
            raise AssertionError(f"Payload {key} differs from exact raw archive")

    entry = summary["entry_parameters"]
    # Beta legitimately uses sessions preceding the displayed year.  Rebuild
    # the full causal history first and only then slice the published period.
    full_arrays = independent_arrays(qqq_all, nvda_all, int(entry["beta_days"]),
                                     int(entry["window"]), int(entry["warmup_bars"]))
    arrays = {key: value[keep] for key, value in full_arrays.items()
              if key in {"qvwap", "nvwap", "fair", "z"}}
    period_day = pd.factorize(np.asarray(common.date), sort=False)[0]
    arrays["starts"] = np.r_[0, np.flatnonzero(period_day[1:] != period_day[:-1]) + 1]
    arrays["ends"] = np.r_[arrays["starts"][1:] - 1, len(common) - 1]
    for key in ("qvwap", "nvwap", "fair"):
        if not np.allclose(np.asarray(bars[key], float), arrays[key], atol=1e-9, rtol=1e-11, equal_nan=True):
            raise AssertionError(f"Payload causal series {key} differs from independent reconstruction")
    payload_z = np.asarray([np.nan if value is None else value for value in bars["z"]], float)
    if not np.allclose(payload_z, arrays["z"], atol=1e-9, rtol=1e-10, equal_nan=True):
        raise AssertionError("Payload Z differs from independent causal rolling reconstruction")
    for start_i in arrays["starts"]:
        typical = (float(nvda.high.iloc[start_i]) + float(nvda.low.iloc[start_i]) + float(nvda.close.iloc[start_i])) / 3
        close(bars["nvwap"][start_i], typical, f"NVDA VWAP reset {common[start_i]}", 1e-9)
        if np.any(np.isfinite(payload_z[start_i:start_i + int(entry["warmup_bars"])])):
            raise AssertionError(f"Warmup contains finite Z at {common[start_i]}")

    trades = pd.read_csv(src / "selected_full_trades.csv")
    for col in ("signal_time", "entry_time", "exit_time"):
        trades[col] = pd.to_datetime(trades[col], format="mixed", utc=True).dt.tz_convert(NY)
    if len(trades) != len(payload["trades"]):
        raise AssertionError("Payload/source trade count mismatch")
    selected = {
        "stop_usd": float(summary["selected"]["stop_usd"]),
        "target_usd": float(summary["selected"]["target_usd"]),
        "z_entry": float(entry["z_entry"]),
        "position_notional_usd": float(summary["execution"]["position_notional_usd"]),
    }
    commission = float(summary["execution"]["commission_usd_per_share_per_side"])
    slip = float(summary["execution"]["slippage_fraction_per_execution"])
    by_epoch = {epoch(ts): i for i, ts in enumerate(common)}
    day_end_by_date = {common[int(i)].date(): int(i) for i in arrays["ends"]}
    for number, (row, item) in enumerate(zip(trades.itertuples(index=False), payload["trades"]), 1):
        audit_trade(row, item, number, common, nvda, arrays["z"], selected,
                    commission, slip, by_epoch, day_end_by_date)

    # Source and payload minute MTM must be identical, including the open trade path.
    equity = pd.read_csv(src / "selected_full_equity.csv")
    equity["timestamp"] = pd.to_datetime(equity.timestamp, format="mixed", utc=True).dt.tz_convert(NY)
    if [epoch(ts) for ts in equity.timestamp] != bars["t"]:
        raise AssertionError("MTM timestamps do not match raw report bars")
    for key, column in (("equity", "equity"), ("drawdown", "drawdown_usd")):
        if not np.allclose(np.asarray(bars[key], float), equity[column].to_numpy(float), atol=1e-8, rtol=1e-11):
            raise AssertionError(f"Payload {key} differs from event-driven MTM source")
    full = summary["selected_results"]["full"]
    close(equity.equity.iloc[-1], float(summary["execution"]["starting_capital_usd"]) + float(full["net_pnl"]), "final MTM equity")
    close(equity.drawdown_usd.max(), full["max_drawdown_usd_mtm"], "MTM max drawdown USD")
    close(equity.drawdown_pct.max(), full["max_drawdown_pct_mtm"], "MTM max drawdown percent")
    results = payload["results"]
    if "full" in results:
        published_full = results["full"]
    elif "splits" in results:
        published_full = results["splits"]["full"]
    elif "selected_results" in results:
        published_full = results["selected_results"]["full"]
    else:
        raise AssertionError("Payload results do not publish full-period metrics")
    for key, expected in full.items():
        if key in published_full and isinstance(expected, (int, float)) and not isinstance(expected, bool):
            close(published_full[key], expected, f"published full.{key}")

    html = (REPORT / "index.html").read_text(encoding="utf-8")
    required_tokens = (
        "Торгуется только NVDA", "QQQ", "не торгуется", "синтетическ", "VWAP",
        "Fair NVDA", "±2.5", "следующ", "open", "stop-first", "Equity", "Drawdown",
        "$0.0035", "2 bps", "borrow", "event-driven", "не гарантия",
    )
    folded = html.casefold()
    missing = [token for token in required_tokens if token.casefold() not in folded]
    if missing:
        raise AssertionError(f"HTML is missing required explanations: {missing}")
    for path in (REPORT / "index.html", REPORT / "data.js", REPORT / "report_data.json"):
        if not path.is_file() or path.stat().st_size < 1000:
            raise AssertionError(f"Report artifact is absent or implausibly small: {path}")
    if "vwap_absolute_event_driven" not in json.dumps(payload["meta"], ensure_ascii=False):
        raise AssertionError("Payload provenance does not identify the event-driven source")

    print(f"PASS event-driven VWAP absolute report: {len(common):,} raw SIP bars, "
          f"{len(trades):,} trades, net ${float(full['net_pnl']):,.2f}, "
          f"MTM MDD ${float(full['max_drawdown_usd_mtm']):,.2f}")


if __name__ == "__main__":
    main()
