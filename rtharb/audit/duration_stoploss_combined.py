"""Independent audit of the combined winner-preserving duration/stop overlay.

This module intentionally imports neither the research implementation nor the
report builder.  It rebuilds the frozen classic signal and execution state
machines directly from raw Alpaca SIP minute bars.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rtharb.data.loader import DataLoader


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_output" / "duration_stoploss_combined"
REPORT = ROOT / "tradingview_duration_stoploss_combined"
VARIANTS = ("raw_q95_q95", "selected")
SPLITS = {"development": (0, 250), "validation": (250, 375),
          "holdout": (375, 501), "full": (0, 501)}
FROZEN = {"beta_mode": "dynamic_rolling", "beta_days": 5, "window": 60,
          "z_entry": 3.0, "hook_delta": 0.15, "hook_timeout": 5,
          "exit_band": 0.0, "z_lockout": 3.5, "direction": "both"}
CAPITAL = 100_000.0
NOTIONAL = 20_000.0
COMMISSION = 0.0035
SLIP = 0.0002
QUANTILES = (0.95, 0.975, 0.99, 1.0)
INDEPENDENT_HOLD = 61
INDEPENDENT_STOP = 0.00721284703320633
NY = "America/New_York"
ATOL = 3e-7


class ArtifactsNotReady(FileNotFoundError):
    pass


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _close(actual: Any, expected: Any, label: str, atol: float = ATOL) -> None:
    a, e = float(actual), float(expected)
    if not math.isclose(a, e, abs_tol=atol, rel_tol=1e-10):
        raise AssertionError(f"{label}: {a!r} != {e!r}")


def _time_frame(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in columns:
        frame[column] = pd.to_datetime(frame[column], format="mixed", utc=True).dt.tz_convert(NY)
    return frame


@dataclass
class Market:
    index: pd.DatetimeIndex
    qqq: pd.DataFrame
    nvda: pd.DataFrame
    dates: np.ndarray
    day: np.ndarray
    bar: np.ndarray
    last: np.ndarray
    beta: np.ndarray
    fair: np.ndarray
    spread: np.ndarray
    z: np.ndarray
    signals: np.ndarray


def _load_raw() -> Market:
    loader = DataLoader(str(ROOT / "data_cache"), "alpaca", "sip")
    qqq, nvda = loader.get_synchronized_pair("QQQ", "NVDA")
    if (len(qqq), len(nvda)) != (194_490, 194_490) or not qqq.index.equals(nvda.index):
        raise AssertionError("Raw QQQ/NVDA clock is not the exact 194,490-minute inner intersection")
    if qqq.index.tz is None or qqq.index.has_duplicates:
        raise AssertionError("Raw pair timestamps are not unique/timezone-aware")
    index = qqq.index
    dates = np.asarray(index.date)
    day = pd.factorize(dates, sort=False)[0].astype(np.int64)
    if len(pd.unique(dates)) != 501:
        raise AssertionError("Raw pair does not contain exactly 501 RTH sessions")
    bar = pd.Series(day).groupby(day, sort=False).cumcount().to_numpy(np.int64)
    last = np.r_[day[1:] != day[:-1], True]

    # Five completed daily closes are used for the next day's beta; the shift
    # is the independent causality gate.
    q_daily = qqq.groupby(qqq.index.date).close.last().pct_change()
    n_daily = nvda.groupby(nvda.index.date).close.last().pct_change()
    pair = pd.concat({"lead": q_daily, "target": n_daily}, axis=1).dropna()
    cov = pair.target.rolling(5, min_periods=5).cov(pair.lead)
    var = pair.lead.rolling(5, min_periods=5).var()
    daily_beta = (cov / var).shift(1).clip(0.2, 4.0)
    beta = pd.Series(dates).map(daily_beta).fillna(1.5).to_numpy(float)

    q_close = qqq.close.to_numpy(float); n_close = nvda.close.to_numpy(float)
    first = np.r_[0, np.flatnonzero(day[1:] != day[:-1]) + 1]
    q0 = q_close[first][day]; n0 = n_close[first][day]
    q_return = q_close / q0 - 1.0; n_return = n_close / n0 - 1.0
    spread = n_return - beta * q_return
    fair = n0 * (1.0 + beta * q_return)
    spread_s = pd.Series(spread)
    grouped = spread_s.groupby(day, sort=False)
    mean = grouped.transform(lambda value: value.rolling(60, min_periods=15).mean()).to_numpy(float)
    std = grouped.transform(lambda value: value.rolling(60, min_periods=15).std(ddof=1)).to_numpy(float).copy()
    std[std <= 1e-8] = np.nan
    z = (spread - mean) / std
    z[bar < 15] = np.nan
    signals = _signals(index, dates, bar, z)
    return Market(index, qqq, nvda, dates, day, bar, last, beta, fair, spread, z, signals)


def _signals(index: pd.DatetimeIndex, dates: np.ndarray, bar: np.ndarray,
             z: np.ndarray) -> np.ndarray:
    signals = np.full(len(index), "NONE", dtype=object)
    position = armed = armed_age = 0
    extreme = 0.0; locked = False; current_day: Any = None
    times = index.strftime("%H:%M").to_numpy()
    for i in range(len(index)):
        day = dates[i]
        if day != current_day:
            current_day = day; position = armed = armed_age = 0
            extreme = 0.0; locked = False
        penultimate = i + 2 >= len(index) or dates[i + 2] != day
        value = float(z[i])
        if not math.isfinite(value):
            armed = 0
            if penultimate and position:
                signals[i] = "EXIT_FORCED_EOD"; position = 0
            continue
        if abs(value) >= 3.5:
            locked = True; armed = 0
        if times[i] >= "15:55" or penultimate:
            if position:
                signals[i] = "EXIT_FORCED_EOD"; position = 0
            armed = 0
            continue
        if position == 1:
            if value >= 0.0:
                signals[i] = "EXIT_TAKE_PROFIT"; position = 0
            continue
        if position == -1:
            if value <= 0.0:
                signals[i] = "EXIT_TAKE_PROFIT"; position = 0
            continue
        if bar[i] < 15 or locked:
            continue
        if not armed:
            if value <= -3.0:
                armed = 1; armed_age = 0; extreme = value
            elif value >= 3.0:
                armed = -1; armed_age = 0; extreme = value
            continue
        armed_age += 1
        if armed == 1:
            extreme = min(extreme, value)
            if value - extreme >= 0.15:
                signals[i] = "BUY_LONG"; position = 1; armed = 0
        else:
            extreme = max(extreme, value)
            if extreme - value >= 0.15:
                signals[i] = "SELL_SHORT"; position = -1; armed = 0
        if armed and armed_age >= 5:
            armed = 0
    return signals


def _mask(market: Market, first_day: int, last_day: int) -> np.ndarray:
    return np.flatnonzero((market.day >= first_day) & (market.day < last_day))


def _simulate(market: Market, first_day: int, last_day: int,
              hold: int | None, stop_pct: float | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = _mask(market, first_day, last_day)
    op = market.nvda.open.to_numpy(float); hi = market.nvda.high.to_numpy(float)
    lo = market.nvda.low.to_numpy(float); close = market.nvda.close.to_numpy(float)
    cash = CAPITAL; position: dict[str, Any] | None = None
    pending: dict[str, Any] | None = None; trade_id = 0
    trades: list[dict[str, Any]] = []; equity_rows: list[dict[str, Any]] = []

    def close_trade(i: int, raw_exit: float, reason: str) -> None:
        nonlocal cash, position
        assert position is not None
        direction = position["direction"]; shares = position["shares"]
        exit_price = raw_exit * (1.0 - SLIP if direction == 1 else 1.0 + SLIP)
        gross = direction * (raw_exit - position["entry_reference_price"]) * shares
        commission = 2.0 * shares * COMMISSION
        slippage = (abs(position["entry_price"] - position["entry_reference_price"]) +
                    abs(exit_price - raw_exit)) * shares
        net = gross - commission - slippage
        row = {**position, "exit_time": market.index[i], "exit_reference_price": raw_exit,
               "exit_price": exit_price, "gross_pnl": gross, "commission": commission,
               "slippage": slippage, "net_pnl": net,
               "return_pct": net / position["position_value"], "exit_reason": reason,
               "duration_bars": i - position["entry_i"], "exit_z_score": market.z[i],
               "max_holding_bars": hold, "stop_loss_pct": stop_pct}
        row.pop("entry_i"); row.pop("entry_commission")
        trades.append(row)
        cash += net + position["entry_commission"]
        position = None

    for i in idx:
        if pending is not None:
            action = pending["action"]
            if action == "CLOSE" and position is not None:
                close_trade(i, float(op[i]), pending["reason"])
            elif action in ("OPEN_LONG", "OPEN_SHORT") and position is None:
                direction = 1 if action == "OPEN_LONG" else -1
                trade_id += 1
                entry_price = op[i] * (1.0 + SLIP if direction == 1 else 1.0 - SLIP)
                shares = math.floor(NOTIONAL / entry_price)
                entry_commission = shares * COMMISSION; cash -= entry_commission
                position = {"trade_id": trade_id, "ticker": "NVDA", "direction": direction,
                            "entry_time": market.index[i], "entry_reference_price": float(op[i]),
                            "entry_price": entry_price, "shares": shares,
                            "position_value": shares * float(op[i]), "entry_i": i,
                            "entry_commission": entry_commission,
                            "entry_z_score": pending["z_score"]}
            pending = None
        if position is not None:
            reason: str | None = None; raw_exit = math.nan
            if stop_pct is not None:
                if position["direction"] == 1:
                    stop = position["entry_reference_price"] * (1.0 - stop_pct)
                    if lo[i] <= stop:
                        raw_exit = min(op[i], stop); reason = "STOP_LOSS"
                else:
                    stop = position["entry_reference_price"] * (1.0 + stop_pct)
                    if hi[i] >= stop:
                        raw_exit = max(op[i], stop); reason = "STOP_LOSS"
            if reason is None and hold is not None and i - position["entry_i"] >= hold:
                raw_exit = op[i]; reason = "TIME_STOP"
            if reason is not None:
                close_trade(i, float(raw_exit), reason); pending = None
        signal = market.signals[i]
        if signal == "BUY_LONG" and position is None:
            pending = {"action": "OPEN_LONG", "reason": "BUY_REVERSAL_HOOK", "z_score": market.z[i]}
        elif signal == "SELL_SHORT" and position is None:
            pending = {"action": "OPEN_SHORT", "reason": "SHORT_REVERSAL_HOOK", "z_score": market.z[i]}
        elif signal == "EXIT_TAKE_PROFIT" and position is not None:
            pending = {"action": "CLOSE", "reason": "TAKE_PROFIT", "z_score": market.z[i]}
        elif signal == "EXIT_FORCED_EOD" and position is not None:
            pending = {"action": "CLOSE", "reason": "FORCED_EOD", "z_score": market.z[i]}
        if market.last[i]:
            if position is not None:
                close_trade(i, float(close[i]), "SESSION_END_FALLBACK")
            pending = None
        unrealized = 0.0 if position is None else (
            position["direction"] * (close[i] - position["entry_price"]) * position["shares"])
        equity_rows.append({"timestamp": market.index[i], "equity": cash + unrealized})
    if position is not None or pending is not None:
        raise AssertionError("Live state leaked from split replay")
    equity = pd.DataFrame(equity_rows)
    peak = np.maximum.accumulate(np.r_[CAPITAL, equity.equity.to_numpy(float)])[1:]
    equity["running_peak"] = peak
    equity["drawdown_usd"] = peak - equity.equity
    equity["drawdown_pct"] = np.divide(
        equity.drawdown_usd, peak, out=np.zeros(len(equity)), where=peak != 0) * 100.0
    return pd.DataFrame(trades), equity


def _metrics(trades: pd.DataFrame, equity: pd.DataFrame, market: Market,
             first_day: int, last_day: int) -> dict[str, Any]:
    nets = trades.net_pnl.to_numpy(float) if len(trades) else np.array([], float)
    daily_last = equity.groupby(np.asarray(market.dates[_mask(market, first_day, last_day)]), sort=False).equity.last()
    prior = daily_last.shift(1).fillna(CAPITAL)
    returns = daily_last / prior - 1.0
    sharpe = (math.sqrt(252.0) * returns.mean() / returns.std(ddof=1)
              if len(returns) > 1 and returns.std(ddof=1) > 0 else 0.0)
    downside = math.sqrt(float(np.mean(np.minimum(returns.to_numpy(float), 0.0) ** 2)))
    sortino = math.sqrt(252.0) * returns.mean() / downside if downside else 0.0
    wins = nets[nets > 0]; losses = nets[nets <= 0]
    profit_factor = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() else (math.inf if len(wins) else 0.0)
    return {"sessions": last_day - first_day, "raw_bars": len(equity), "trades": len(trades),
            "gross_pnl": float(trades.gross_pnl.sum()), "commissions": float(trades.commission.sum()),
            "slippage": float(trades.slippage.sum()),
            "costs": float(trades.commission.sum() + trades.slippage.sum()),
            "net_pnl": float(equity.equity.iloc[-1] - CAPITAL),
            "net_return_pct": float((equity.equity.iloc[-1] - CAPITAL) / CAPITAL * 100),
            "net_sharpe": float(sharpe), "net_sortino": float(sortino),
            "max_drawdown_usd_mtm": float(equity.drawdown_usd.max()),
            "max_drawdown_pct_mtm": float(equity.drawdown_pct.max()),
            "win_rate_pct": float(len(wins) / len(nets) * 100 if len(nets) else 0),
            "profit_factor": float(profit_factor), "avg_net_trade": float(nets.mean() if len(nets) else 0),
            "avg_duration_bars": float(trades.duration_bars.mean() if len(trades) else 0),
            "final_equity": float(equity.equity.iloc[-1]),
            "exit_reasons": {str(key): int(value) for key, value in trades.exit_reason.value_counts().items()}}


def _survival(winners: pd.DataFrame, overlay: pd.DataFrame) -> dict[str, Any]:
    by_entry = {row.entry_time: row for row in overlay.itertuples(index=False)}
    matched = profitable = not_early = direction_match = 0
    for row in winners.itertuples(index=False):
        candidate = by_entry.get(row.entry_time)
        if candidate is None:
            continue
        matched += 1; direction_match += int(candidate.direction == row.direction)
        profitable += int(candidate.net_pnl > 0); not_early += int(candidate.exit_time >= row.exit_time)
    total = len(winners)
    return {"baseline_development_winners": total, "matched_entry_events": matched,
            "matched_entry_pct": 100 * matched / total, "direction_match_pct": 100 * direction_match / total,
            "still_net_profitable_count": profitable, "still_net_profitable_pct": 100 * profitable / total,
            "not_prematurely_closed_pct": 100 * not_early / total}


def _audit_trade_frame(label: str, expected: pd.DataFrame, saved: pd.DataFrame) -> None:
    if len(expected) != len(saved):
        raise AssertionError(f"{label}: trade count differs {len(expected)} != {len(saved)}")
    for column in ("entry_time", "exit_time"):
        if not expected[column].equals(saved[column]):
            raise AssertionError(f"{label}: {column} differs")
    for column in ("trade_id", "ticker", "direction", "shares", "exit_reason", "duration_bars"):
        if expected[column].astype(str).tolist() != saved[column].astype(str).tolist():
            raise AssertionError(f"{label}: {column} differs")
    numeric = ("entry_reference_price", "entry_price", "exit_reference_price", "exit_price",
               "position_value", "gross_pnl", "commission", "slippage", "net_pnl", "return_pct",
               "entry_z_score", "exit_z_score", "max_holding_bars", "stop_loss_pct")
    for column in numeric:
        if not np.allclose(expected[column], saved[column], atol=ATOL, rtol=1e-10, equal_nan=True):
            raise AssertionError(f"{label}: {column} differs")


def _audit_equity(label: str, expected: pd.DataFrame, saved: pd.DataFrame,
                  metrics: dict[str, Any]) -> None:
    if len(expected) != len(saved) or not expected.timestamp.equals(saved.timestamp):
        raise AssertionError(f"{label}: equity raw clock differs")
    for column in ("equity", "running_peak", "drawdown_usd", "drawdown_pct"):
        if not np.allclose(expected[column], saved[column], atol=ATOL, rtol=1e-10):
            raise AssertionError(f"{label}: minute MTM {column} differs")
    _close(saved.equity.iloc[-1], metrics["final_equity"], f"{label} final equity")
    _close(saved.drawdown_usd.max(), metrics["max_drawdown_usd_mtm"], f"{label} MTM DD")


def _audit_report(manifest: dict[str, Any], summary: dict[str, Any],
                  market: Market | None = None) -> None:
    path = REPORT / "manifest.js"
    if not path.is_file():
        return
    html = REPORT / "index.html"
    if not html.is_file() or html.stat().st_size < 1000:
        raise AssertionError("Combined q95 report HTML missing")
    text = path.read_text(encoding="utf-8").strip()
    prefix = "window.DURATION_STOPLOSS_COMBINED_MANIFEST="
    if not text.startswith(prefix) or not text.endswith(";"):
        raise AssertionError("Combined q95 report manifest wrapper malformed")
    report = json.loads(text[len(prefix):-1])
    if tuple(report.get("variants", ())) != VARIANTS:
        raise AssertionError("Combined q95 report variants differ")
    source = report.get("source", {})
    if source.get("sha256") != _sha(OUT / "manifest.json"):
        raise AssertionError("Combined q95 report source hash differs")
    assets = report.get("assets", [])
    if [item.get("variant") for item in assets] != list(VARIANTS):
        raise AssertionError("Combined q95 report asset order differs")
    for asset in assets:
        payload_path = REPORT / asset["data"]
        if (not payload_path.is_file() or payload_path.stat().st_size != int(asset["bytes"])
                or _sha(payload_path) != asset["sha256"]):
            raise AssertionError(f"{asset['variant']}: lazy payload hash/size differs")
        payload = _read(payload_path)
        if set(payload) != {"meta", "bars", "trades", "results"}:
            raise AssertionError(f"{asset['variant']}: lazy payload schema differs")
        if {len(value) for value in payload["bars"].values()} != {194_490}:
            raise AssertionError(f"{asset['variant']}: lazy arrays not aligned")
        result_key = "raw_q95_q95_diagnostic" if asset["variant"] == "raw_q95_q95" else "selected_results"
        source_results = (summary[result_key]["results"] if asset["variant"] == "raw_q95_q95"
                          else summary[result_key])
        _close(payload["results"]["full"]["net_pnl"], source_results["full"]["net_pnl"],
               f"{asset['variant']} report full net")
        if len(payload["trades"]) != int(source_results["full"]["trades"]):
            raise AssertionError(f"{asset['variant']}: report trade count differs")
        for provenance in payload["meta"].get("sources", {}).values():
            source_path = ROOT / provenance["path"]
            if (not source_path.is_file() or _sha(source_path) != provenance["sha256"]
                    or source_path.stat().st_size != int(provenance["bytes"])):
                raise AssertionError(f"{asset['variant']}: report provenance differs")
        if market is not None:
            bars = payload["bars"]
            raw = {"qo": market.qqq.open, "qh": market.qqq.high, "ql": market.qqq.low,
                   "qc": market.qqq.close, "no": market.nvda.open, "nh": market.nvda.high,
                   "nl": market.nvda.low, "nc": market.nvda.close,
                   "target_fair": market.fair, "spread": market.spread,
                   "beta": market.beta, "z": market.z}
            epochs = (market.index.as_unit("ns").asi8 // 1_000_000_000).tolist()
            if bars["t"] != epochs:
                raise AssertionError(f"{asset['variant']}: report timestamps differ")
            for key, values in raw.items():
                published = np.asarray([np.nan if value is None else value for value in bars[key]], float)
                if not np.allclose(published, np.asarray(values, float), atol=1e-9,
                                   rtol=1e-10, equal_nan=True):
                    raise AssertionError(f"{asset['variant']}: report {key} differs")
            source_equity = pd.read_csv(OUT / f"{asset['variant']}_full_equity.csv")
            for published_key, source_key in (("equity", "equity"), ("drawdown", "drawdown_usd"),
                                               ("drawdown_pct", "drawdown_pct")):
                if not np.allclose(bars[published_key], source_equity[source_key], atol=ATOL, rtol=1e-10):
                    raise AssertionError(f"{asset['variant']}: report {published_key} differs")
            source_trades = _time_frame(
                OUT / f"{asset['variant']}_full_trades.csv", ("entry_time", "exit_time"))
            day_end = np.empty(len(market.index), dtype=np.int64)
            start = 0
            for end in np.flatnonzero(market.last):
                day_end[start:end + 1] = end; start = end + 1
            for report_trade, source_trade in zip(payload["trades"], source_trades.itertuples(index=False)):
                entry_i = int(market.index.searchsorted(source_trade.entry_time))
                exit_i = int(market.index.searchsorted(source_trade.exit_time))
                expected_times = (int(market.index[entry_i - 1].tz_convert("UTC").timestamp()),
                                  int(source_trade.entry_time.tz_convert("UTC").timestamp()),
                                  int(source_trade.exit_time.tz_convert("UTC").timestamp()))
                actual_times = (int(report_trade["entry_signal_time"]), int(report_trade["entry_time"]),
                                int(report_trade["exit_time"]))
                if actual_times != expected_times:
                    raise AssertionError(f"{asset['variant']}: report trade timing/next-open differs")
                stop = (source_trade.entry_reference_price * (1.0 - source_trade.stop_loss_pct)
                        if source_trade.direction == 1 else
                        source_trade.entry_reference_price * (1.0 + source_trade.stop_loss_pct))
                expiry_i = min(entry_i + int(source_trade.max_holding_bars), int(day_end[entry_i]))
                _close(report_trade["stop_price"], stop, f"{asset['variant']} report stop", 1e-9)
                if int(report_trade["expiry_time"]) != int(market.index[expiry_i].tz_convert("UTC").timestamp()):
                    raise AssertionError(f"{asset['variant']}: report expiry timestamp differs")
                if bool(report_trade["expiry_reached"]) != (source_trade.exit_reason == "TIME_STOP"):
                    raise AssertionError(f"{asset['variant']}: report expiry flag differs")
                for report_key, source_key in (("entry_reference", "entry_reference_price"),
                                               ("entry_price", "entry_price"),
                                               ("exit_reference", "exit_reference_price"),
                                               ("exit_price", "exit_price"), ("gross_pnl", "gross_pnl"),
                                               ("commissions", "commission"), ("slippage", "slippage"),
                                               ("net_pnl", "net_pnl")):
                    _close(report_trade[report_key], getattr(source_trade, source_key),
                           f"{asset['variant']} report trade {report_key}")


def _gate() -> tuple[dict[str, Any], dict[str, Any]]:
    path = OUT / "manifest.json"
    if not path.is_file():
        raise ArtifactsNotReady("Run python -m rtharb.research.duration_stoploss_combined first")
    manifest = _read(path); summary = _read(OUT / "summary.json"); audit = _read(OUT / "audit.json")
    if manifest.get("status") != "COMPLETE" or manifest.get("audit", {}).get("status") != "PASS":
        raise ArtifactsNotReady("Combined q95 stage is not COMPLETE/PASS")
    if tuple(manifest.get("variants", ())) != VARIANTS or tuple(manifest.get("splits", ())) != tuple(SPLITS):
        raise AssertionError("Combined q95 manifest variants/splits differ")
    if summary.get("frozen_parameters") != FROZEN:
        raise AssertionError("Frozen classic signal tuple differs")
    data = summary["data"]
    if (data["lead"], data["traded"], data["raw_bars"], data["sessions"]) != ("QQQ", "NVDA", 194_490, 501):
        raise AssertionError("Combined q95 raw roles/calendar differ")
    if audit.get("status") != "PASS" or audit.get("holdout_used_in_selection") is not False:
        raise AssertionError("Combined q95 audit/holdout isolation differs")
    execution = summary["execution"]
    for key, expected in (("starting_capital_usd", CAPITAL), ("position_notional_usd", NOTIONAL),
                          ("commission_usd_per_share_per_side", COMMISSION),
                          ("slippage_fraction_per_execution", SLIP)):
        _close(execution[key], expected, key)
    return manifest, summary


def audit(*, raw_replay: bool = True) -> dict[str, Any]:
    manifest, summary = _gate()
    grid = pd.read_csv(OUT / "development_grid.csv")
    finalists = pd.read_csv(OUT / "validation_finalists.csv")
    if any("holdout" in column.casefold() for column in grid.columns) or any(
            "holdout" in column.casefold() for column in finalists.columns):
        raise AssertionError("Holdout leaked into selection artifacts")
    if raw_replay:
        market = _load_raw()
        base_trades, _ = _simulate(market, 0, 250, None, None)
        winners = base_trades[base_trades.net_pnl > 0].copy()
        if len(winners) != 132:
            raise AssertionError(f"Frozen development winner count differs: {len(winners)}")
        # Independently reconstruct MAE on entry-inclusive/exit-exclusive raw paths.
        maes: list[float] = []
        for row in winners.itertuples(index=False):
            start = market.index.searchsorted(row.entry_time); end = market.index.searchsorted(row.exit_time)
            if row.direction == 1:
                mae = max(0.0, (row.entry_reference_price - market.nvda.low.iloc[start:end].min()) /
                          row.entry_reference_price)
            else:
                mae = max(0.0, (market.nvda.high.iloc[start:end].max() - row.entry_reference_price) /
                          row.entry_reference_price)
            maes.append(mae)
        winners["mae_pct"] = maes
        raw_duration = {str(q): int(winners.duration_bars.quantile(q, interpolation="higher")) for q in QUANTILES}
        raw_stop = {str(q): float(winners.mae_pct.quantile(q, interpolation="higher")) for q in QUANTILES}
        candidate_fit = summary["candidate_fit"]
        if raw_duration != candidate_fit["winner_duration_quantiles_bars"]:
            raise AssertionError("Winner duration quantiles differ")
        for q in map(str, QUANTILES):
            _close(raw_stop[q], candidate_fit["winner_mae_pct_quantiles"][q], f"winner MAE q{q}", 1e-12)
        holds = sorted(set(raw_duration.values()) | {INDEPENDENT_HOLD})
        stops = sorted(set(raw_stop.values()) | {INDEPENDENT_STOP})
        if len(holds) * len(stops) != len(grid):
            raise AssertionError("Development candidate axes/grid size differ")
        replay_rows: list[dict[str, Any]] = []
        dev_cache: dict[tuple[int, float], tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]] = {}
        for hold in holds:
            for stop in stops:
                trades, equity = _simulate(market, 0, 250, hold, stop)
                metrics = _metrics(trades, equity, market, 0, 250)
                dev_cache[(hold, stop)] = (trades, equity, metrics)
                replay_rows.append({"max_holding_bars": hold, "stop_loss_pct": stop,
                                    "eligible": (_survival(winners, trades)["matched_entry_pct"] == 100 and
                                                 _survival(winners, trades)["direction_match_pct"] == 100 and
                                                 _survival(winners, trades)["still_net_profitable_pct"] >= 95),
                                    **_survival(winners, trades),
                                    **{f"development_{key}": value for key, value in metrics.items()}})
        replay = pd.DataFrame(replay_rows).sort_values(
            ["eligible", "development_net_sharpe", "development_net_pnl", "max_holding_bars", "stop_loss_pct"],
            ascending=[False, False, False, True, True], kind="mergesort").reset_index(drop=True)
        for column in ("max_holding_bars", "eligible", "baseline_development_winners", "matched_entry_events",
                       "still_net_profitable_count", "development_trades"):
            if replay[column].astype(str).tolist() != grid[column].astype(str).tolist():
                raise AssertionError(f"Development grid {column} differs")
        for column in ("stop_loss_pct", "matched_entry_pct", "direction_match_pct",
                       "still_net_profitable_pct", "not_prematurely_closed_pct",
                       "development_net_pnl", "development_net_sharpe", "development_costs"):
            if not np.allclose(replay[column], grid[column], atol=ATOL, rtol=1e-10):
                raise AssertionError(f"Development grid {column} differs")
        top = replay[replay.eligible].head(10)
        top_pairs = top[["max_holding_bars", "stop_loss_pct"]].sort_values(
            ["max_holding_bars", "stop_loss_pct"]).to_numpy(float)
        finalist_pairs = finalists[["max_holding_bars", "stop_loss_pct"]].sort_values(
            ["max_holding_bars", "stop_loss_pct"]).to_numpy(float)
        if not np.allclose(top_pairs, finalist_pairs, atol=1e-12, rtol=0):
            raise AssertionError("Validation finalists are not exact eligible development top 10")
        validation_rows = []
        for row in top.itertuples(index=False):
            trades, equity = _simulate(market, 250, 375, int(row.max_holding_bars), float(row.stop_loss_pct))
            metrics = _metrics(trades, equity, market, 250, 375)
            validation_rows.append({"max_holding_bars": int(row.max_holding_bars),
                                    "stop_loss_pct": float(row.stop_loss_pct),
                                    "development_net_sharpe": row.development_net_sharpe,
                                    "development_net_pnl": row.development_net_pnl,
                                    "validation_net_sharpe": metrics["net_sharpe"],
                                    "validation_net_pnl": metrics["net_pnl"],
                                    "validation_trades": metrics["trades"],
                                    "robust_score": min(row.development_net_sharpe, metrics["net_sharpe"])})
        validation = pd.DataFrame(validation_rows).sort_values(
            ["robust_score", "validation_net_pnl", "development_net_sharpe", "max_holding_bars", "stop_loss_pct"],
            ascending=[False, False, False, True, True], kind="mergesort").reset_index(drop=True)
        for column in validation:
            if not np.allclose(validation[column], finalists[column], atol=ATOL, rtol=1e-10):
                raise AssertionError(f"Validation selection {column} differs")
        selected_hold = int(validation.iloc[0].max_holding_bars)
        selected_stop = float(validation.iloc[0].stop_loss_pct)
        selection = summary["selection"]
        _close(selected_hold, selection["selected_max_holding_bars"], "selected holding")
        _close(selected_stop, selection["selected_stop_loss_pct"], "selected stop", 1e-12)
        variants = {"raw_q95_q95": (raw_duration["0.95"], raw_stop["0.95"]),
                    "selected": (selected_hold, selected_stop)}
        for variant, (hold, stop) in variants.items():
            results = (summary["raw_q95_q95_diagnostic"]["results"] if variant == "raw_q95_q95"
                       else summary["selected_results"])
            for split, (first_day, last_day) in SPLITS.items():
                expected_trades, expected_equity = _simulate(market, first_day, last_day, int(hold), float(stop))
                saved_trades = _time_frame(OUT / f"{variant}_{split}_trades.csv", ("entry_time", "exit_time"))
                saved_equity = _time_frame(OUT / f"{variant}_{split}_equity.csv", ("timestamp",))
                label = f"{variant} {split}"
                _audit_trade_frame(label, expected_trades, saved_trades)
                _audit_equity(label, expected_equity, saved_equity, results[split])
                metrics = _metrics(expected_trades, expected_equity, market, first_day, last_day)
                for key in ("trades", "gross_pnl", "commissions", "slippage", "costs", "net_pnl",
                            "net_sharpe", "max_drawdown_usd_mtm", "max_drawdown_pct_mtm"):
                    _close(metrics[key], results[split][key], f"{label} {key}")
        raw_survival = _survival(winners, dev_cache[(raw_duration["0.95"], raw_stop["0.95"])][0])
        if raw_survival["still_net_profitable_pct"] >= 95 or summary["raw_q95_q95_diagnostic"]["eligible"]:
            raise AssertionError("Raw q95/q95 diagnostic eligibility differs")
        _audit_report(manifest, summary, market)
    else:
        _audit_report(manifest, summary)
    return {"selection": summary["selection"], "raw": summary["raw_q95_q95_diagnostic"],
            "selected_results": summary["selected_results"]}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        result = audit()
    except ArtifactsNotReady as exc:
        print(f"NOT READY: {exc}", file=sys.stderr); raise SystemExit(2) from exc
    selected = result["selection"]; full = result["selected_results"]["full"]
    print("PASS combined duration+stop-loss independent raw replay")
    print(f"selected {selected['selected_max_holding_bars']} bars / {selected['selected_stop_loss_pct']:.12f}; "
          f"full {full['trades']} trades, net ${full['net_pnl']:,.2f}, MTM DD ${full['max_drawdown_usd_mtm']:,.2f}")


if __name__ == "__main__":
    main()
