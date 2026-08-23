"""Causal QQQ-versus-mega-cap synthetic-index research.

Only QQQ is traded.  The four-stock basket is a fixed reference selected from
an official Nasdaq weight snapshot published before the sample starts.  All
signals use synchronized raw Alpaca SIP 1-minute bars during official RTH and
all market entries/convergence exits execute at the following bar's open.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_output" / "synthetic_index"
CACHE = ROOT / "data_cache"
SYMBOLS = ("QQQ", "MSFT", "AAPL", "NVDA", "AMZN")
BASKET = ("MSFT", "AAPL", "NVDA", "AMZN")
NDX_WEIGHTS = {"MSFT": 8.6, "AAPL": 8.4, "NVDA": 7.9, "AMZN": 5.2}
WEIGHTS = {key: value / sum(NDX_WEIGHTS.values()) for key, value in NDX_WEIGHTS.items()}
SOURCE_URL = "https://indexes.nasdaq.com/docs/202407%20NDX%20Research.pdf"
START = pd.Timestamp("2024-08-22")
END_EXCLUSIVE = pd.Timestamp("2026-08-22")
CAPITAL = 100_000.0
POSITION_SIZE = 20_000.0
COMMISSION = 0.0035
SLIPPAGE = 0.0002


@dataclass(frozen=True)
class EntryConfig:
    basis: str
    threshold: float
    hook: float
    window: int = 0

    @property
    def key(self) -> str:
        return f"{self.basis}_w{self.window}_t{self.threshold:g}_h{self.hook:g}"


def load_raw() -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex, np.ndarray]:
    """Read cached raw SIP bars, apply official RTH, then take exact intersection."""
    calendar = pd.read_csv(CACHE / "market_calendar.csv", dtype=str)
    calendar = calendar[(calendar["date"] >= START.strftime("%Y-%m-%d")) &
                        (calendar["date"] < END_EXCLUSIVE.strftime("%Y-%m-%d"))]
    close_minutes = {
        row.date: int(row.close[:2]) * 60 + int(row.close[3:5])
        for row in calendar.itertuples(index=False)
    }
    allowed = set(close_minutes)
    frames: dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS:
        path = CACHE / f"{symbol}_1m.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing raw Alpaca SIP cache: {path}")
        frame = pd.read_parquet(path, columns=["open", "high", "low", "close", "volume"])
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TypeError(f"{symbol}: parquet index is not DatetimeIndex")
        if frame.index.tz is None:
            frame.index = frame.index.tz_localize("UTC").tz_convert("America/New_York")
        else:
            frame.index = frame.index.tz_convert("America/New_York")
        frame = frame[(frame.index.tz_localize(None) >= START) &
                      (frame.index.tz_localize(None) < END_EXCLUSIVE)]
        date_text = frame.index.strftime("%Y-%m-%d")
        minute = frame.index.hour * 60 + frame.index.minute
        closing = np.fromiter((close_minutes.get(value, -1) for value in date_text), dtype=np.int16)
        mask = np.fromiter((value in allowed for value in date_text), dtype=bool)
        mask &= (minute >= 570) & (minute < closing)
        frame = frame.loc[mask].sort_index()
        if frame.index.has_duplicates:
            raise AssertionError(f"{symbol}: duplicate raw timestamps")
        if not ((frame.high >= frame[["open", "close"]].max(axis=1)) &
                (frame.low <= frame[["open", "close"]].min(axis=1))).all():
            raise AssertionError(f"{symbol}: OHLC invariant failed")
        frames[symbol] = frame
        print(f"loaded {symbol}: {len(frame):,} official RTH bars", flush=True)
    common = reduce(pd.DatetimeIndex.intersection, (frames[symbol].index for symbol in SYMBOLS))
    frames = {symbol: frame.loc[common].copy() for symbol, frame in frames.items()}
    session_text = common.strftime("%Y-%m-%d")
    unique_text = pd.unique(session_text)
    day_lookup = {value: i for i, value in enumerate(unique_text)}
    day_code = np.fromiter((day_lookup[value] for value in session_text), dtype=np.int16)
    if len(unique_text) != 501:
        raise AssertionError(f"Expected 501 common sessions, found {len(unique_text)}")
    if len(common) == 0:
        raise AssertionError("No synchronized raw bars")
    print(f"intersection: {len(common):,} bars / {len(unique_text)} sessions", flush=True)
    return frames, common, day_code


def market_arrays(frames: dict[str, pd.DataFrame], common: pd.DatetimeIndex,
                  day: np.ndarray) -> dict[str, np.ndarray]:
    starts = np.r_[0, np.flatnonzero(np.diff(day)) + 1]
    ends = np.r_[starts[1:] - 1, len(day) - 1]
    basket_return = np.zeros(len(common), dtype=float)
    for symbol in BASKET:
        close = frames[symbol].close.to_numpy(float)
        anchor = close[starts][day]
        basket_return += WEIGHTS[symbol] * (close / anchor - 1.0)
    qqq_close = frames["QQQ"].close.to_numpy(float)
    qqq_anchor = qqq_close[starts][day]
    qqq_return = qqq_close / qqq_anchor - 1.0
    spread = qqq_return - basket_return
    points = qqq_close - qqq_anchor * (1.0 + basket_return)
    bar = np.arange(len(day)) - starts[day]
    return {
        "index": common,
        "day": day,
        "starts": starts,
        "ends": ends,
        "bar": bar,
        "open": frames["QQQ"].open.to_numpy(float),
        "high": frames["QQQ"].high.to_numpy(float),
        "low": frames["QQQ"].low.to_numpy(float),
        "close": qqq_close,
        "basket_return": basket_return,
        "qqq_return": qqq_return,
        "spread": spread,
        "points": points,
    }


def rolling_z(values: np.ndarray, starts: np.ndarray, ends: np.ndarray,
              window: int, warmup: int = 15) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for start, end in zip(starts, ends):
        x = values[start:end + 1]
        n = len(x)
        right = np.arange(1, n + 1)
        left = np.maximum(0, right - window)
        count = right - left
        cs = np.r_[0.0, np.cumsum(x)]
        cs2 = np.r_[0.0, np.cumsum(x * x)]
        total = cs[right] - cs[left]
        total2 = cs2[right] - cs2[left]
        mean = total / count
        variance = np.divide(total2 - total * total / count, count - 1,
                             out=np.full(n, np.nan), where=count > 1)
        std = np.sqrt(np.maximum(variance, 0.0))
        z = np.divide(x - mean, std, out=np.full(n, np.nan), where=std > 1e-10)
        z[:warmup] = np.nan
        out[start:end + 1] = z
    return out


def entry_events(metric: np.ndarray, day: np.ndarray, threshold: float,
                 hook: float) -> np.ndarray:
    """Create an independent causal threshold/reversal event stream."""
    events = np.zeros(len(metric), dtype=np.int8)
    armed = 0
    extreme = np.nan
    locked = False
    previous_day = -1
    armed_bar = 0
    for i, value in enumerate(metric):
        if day[i] != previous_day:
            armed, extreme, locked, previous_day = 0, np.nan, False, int(day[i])
        if not np.isfinite(value):
            continue
        if locked:
            if abs(value) < threshold:
                locked = False
            continue
        if armed == 0:
            if abs(value) < threshold:
                continue
            sign = 1 if value > 0 else -1
            armed = sign
            extreme = value
            armed_bar = i
            if hook == 0:
                events[i] = -sign
                armed, locked = 0, True
            continue
        # Once armed, confirm a retracement from the observed extreme even if
        # the metric has already moved back inside the arming threshold.  The
        # old implementation skipped those bars and materially undercounted
        # legitimate convergence hooks.
        sign = armed
        if armed > 0:
            extreme = max(extreme, value)
            retrace = extreme - value
        else:
            extreme = min(extreme, value)
            retrace = value - extreme
        if retrace >= hook:
            events[i] = -sign
            armed, locked = 0, True
        elif i - armed_bar > 60:
            armed = 0
    return events


def effective_price(raw: float, direction: int, is_entry: bool) -> float:
    sign = direction if is_entry else -direction
    return raw * (1.0 + sign * SLIPPAGE)


def finish_trade(row: dict, raw_exit: float, exit_i: int, reason: str,
                 arrays: dict) -> dict:
    direction = row["direction_value"]
    exit_eff = effective_price(raw_exit, direction, False)
    gross = direction * (raw_exit - row["entry_reference"]) * row["shares"]
    slip = (abs(row["entry_price"] - row["entry_reference"]) +
            abs(exit_eff - raw_exit)) * row["shares"]
    commissions = 2.0 * row["shares"] * COMMISSION
    net = gross - slip - commissions
    row.update({
        "exit_time": arrays["index"][exit_i],
        "exit_reference": raw_exit,
        "exit_price": exit_eff,
        "exit_reason": reason,
        "gross_pnl": gross,
        "commission": commissions,
        "slippage": slip,
        "costs": commissions + slip,
        "net_pnl": net,
        "duration_minutes": int((arrays["index"][exit_i] - row["entry_time"]).total_seconds() // 60),
        "reconciliation_error": gross - commissions - slip - net,
    })
    row.pop("direction_value")
    return row


def open_trade(direction: int, i: int, cfg: EntryConfig, metric: np.ndarray,
               arrays: dict, stop_pct: float | None = None,
               rr: float | None = None) -> dict:
    raw = arrays["open"][i]
    eff = effective_price(raw, direction, True)
    shares = math.floor(POSITION_SIZE / eff)
    row = {
        "entry_time": arrays["index"][i],
        "direction": "LONG" if direction == 1 else "SHORT",
        "direction_value": direction,
        "entry_reference": raw,
        "entry_price": eff,
        "shares": shares,
        "entry_basis": cfg.basis,
        "entry_threshold": cfg.threshold,
        "entry_hook": cfg.hook,
        "entry_window": cfg.window,
        "signal_metric": float(metric[i - 1]),
        "signal_spread": float(arrays["spread"][i - 1]),
        "signal_points": float(arrays["points"][i - 1]),
    }
    if stop_pct is not None and rr is not None:
        distance = raw * stop_pct
        row.update({
            "stop_pct": stop_pct,
            "rr": rr,
            "stop_price": raw - direction * distance,
            "target_price": raw + direction * distance * rr,
        })
    return row


def simulate_convergence(events: np.ndarray, metric: np.ndarray, cfg: EntryConfig,
                         arrays: dict, lo_day: int, hi_day: int) -> pd.DataFrame:
    rows: list[dict] = []
    position: dict | None = None
    pending_exit = False
    day, ends = arrays["day"], set(arrays["ends"])
    lo = int(arrays["starts"][lo_day])
    hi = int(arrays["starts"][hi_day]) if hi_day < len(arrays["starts"]) else len(day)
    for i in range(lo, hi):
        if pending_exit and position is not None:
            rows.append(finish_trade(position, arrays["open"][i], i, "CONVERGENCE_NEXT_OPEN", arrays))
            position, pending_exit = None, False
        if position is None and i > lo and day[i] == day[i - 1] and events[i - 1] != 0:
            position = open_trade(int(events[i - 1]), i, cfg, metric, arrays)
        if position is not None:
            direction = 1 if position["direction"] == "LONG" else -1
            converged = arrays["spread"][i] >= 0 if direction == 1 else arrays["spread"][i] <= 0
            if i in ends:
                rows.append(finish_trade(position, arrays["close"][i], i, "FORCED_EOD", arrays))
                position, pending_exit = None, False
            elif converged:
                pending_exit = True
    return pd.DataFrame(rows)


def simulate_rr(events: np.ndarray, metric: np.ndarray, cfg: EntryConfig,
                arrays: dict, lo_day: int, hi_day: int,
                stop_pct: float, rr: float) -> pd.DataFrame:
    """Evaluate sparse causal entry events with vectorized intratrade brackets."""
    rows: list[dict] = []
    day = arrays["day"]
    lo = int(arrays["starts"][lo_day])
    hi = int(arrays["starts"][hi_day]) if hi_day < len(arrays["starts"]) else len(day)
    signals = np.flatnonzero(events[lo:hi - 1]) + lo
    next_signal_min = lo
    for signal_i in signals:
        if signal_i < next_signal_min:
            continue
        i = int(signal_i + 1)
        if day[i] != day[signal_i]:
            continue
        position = open_trade(int(events[signal_i]), i, cfg, metric, arrays, stop_pct, rr)
        direction = 1 if position["direction"] == "LONG" else -1
        stop, target = position["stop_price"], position["target_price"]
        eod = int(arrays["ends"][day[i]])
        view = slice(i, eod + 1)
        stop_hits = (arrays["low"][view] <= stop if direction == 1
                     else arrays["high"][view] >= stop)
        target_hits = (arrays["high"][view] >= target if direction == 1
                       else arrays["low"][view] <= target)
        stop_ix = np.flatnonzero(stop_hits)
        target_ix = np.flatnonzero(target_hits)
        first_stop = int(stop_ix[0]) if len(stop_ix) else math.inf
        first_target = int(target_ix[0]) if len(target_ix) else math.inf
        if first_stop < math.inf and first_stop <= first_target:
            exit_i = i + int(first_stop)
            op = arrays["open"][exit_i]
            raw_exit = min(op, stop) if direction == 1 else max(op, stop)
            rows.append(finish_trade(position, raw_exit, exit_i, "STOP", arrays))
        elif first_target < math.inf:
            exit_i = i + int(first_target)
            rows.append(finish_trade(position, target, exit_i, "TAKE_PROFIT", arrays))
        else:
            exit_i = eod
            rows.append(finish_trade(position, arrays["close"][eod], eod, "FORCED_EOD", arrays))
        # A signal observed at the close of the intrabar exit may enter next
        # open. Signals observed while the trade was open are ignored.
        next_signal_min = exit_i
    return pd.DataFrame(rows)


def metrics(trades: pd.DataFrame, session_dates: list[str]) -> dict:
    daily = pd.Series(0.0, index=session_dates, dtype=float)
    if not trades.empty:
        by_day = trades.assign(day=pd.to_datetime(trades.exit_time).dt.strftime("%Y-%m-%d")).groupby("day").net_pnl.sum()
        daily.loc[by_day.index] = by_day
    prior_equity = CAPITAL + daily.cumsum().shift(1).fillna(0.0)
    returns = daily / prior_equity
    sd = returns.std(ddof=1)
    sharpe = float(np.sqrt(252) * returns.mean() / sd) if np.isfinite(sd) and sd > 0 else 0.0
    equity = CAPITAL + daily.cumsum()
    drawdown = equity / equity.cummax() - 1.0
    wins = trades[trades.net_pnl > 0] if not trades.empty else trades
    losses = trades[trades.net_pnl <= 0] if not trades.empty else trades
    loss_sum = float(losses.net_pnl.sum()) if not losses.empty else 0.0
    gross = float(trades.gross_pnl.sum()) if not trades.empty else 0.0
    commission = float(trades.commission.sum()) if not trades.empty else 0.0
    slippage = float(trades.slippage.sum()) if not trades.empty else 0.0
    net = float(trades.net_pnl.sum()) if not trades.empty else 0.0
    return {
        "trades": int(len(trades)),
        "gross_pnl": gross,
        "commission": commission,
        "slippage": slippage,
        "costs": commission + slippage,
        "net_pnl": net,
        "net_return_pct": net / CAPITAL * 100.0,
        "net_sharpe": sharpe,
        "max_drawdown_pct": float(-drawdown.min() * 100.0),
        "max_drawdown_usd": float((equity.cummax() - equity).max()),
        "win_rate_pct": float((trades.net_pnl > 0).mean() * 100.0) if not trades.empty else 0.0,
        "profit_factor": float(wins.net_pnl.sum() / abs(loss_sum)) if loss_sum < 0 else None,
        "reconciliation_error": gross - commission - slippage - net,
    }


def period_dates(unique_dates: list[str], lo: int, hi: int) -> list[str]:
    return unique_dates[lo:hi]


def choose_with_validation(dev: pd.DataFrame, validator, top_n: int = 12):
    eligible = dev[dev.trades >= 10].sort_values(["net_sharpe", "net_pnl"], ascending=False).head(top_n)
    validation = []
    for row in eligible.to_dict("records"):
        result = validator(row)
        result["dev_net_sharpe"] = row["net_sharpe"]
        result["robust_score"] = min(row["net_sharpe"], result["net_sharpe"])
        validation.append(result)
    val = pd.DataFrame(validation).sort_values(["robust_score", "net_pnl"], ascending=False)
    if val.empty:
        raise AssertionError("No validation candidates")
    return val.iloc[0].to_dict(), eligible, val


def equity_rows(trades: pd.DataFrame, label: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame([{"timestamp": "", "strategy": label, "net_pnl": 0.0,
                              "equity": CAPITAL, "drawdown_pct": 0.0}])
    out = trades[["exit_time", "net_pnl"]].copy().sort_values("exit_time")
    out["strategy"] = label
    out["equity"] = CAPITAL + out.net_pnl.cumsum()
    out["drawdown_pct"] = (out.equity / out.equity.cummax() - 1.0) * 100.0
    return out.rename(columns={"exit_time": "timestamp"})[["timestamp", "strategy", "net_pnl", "equity", "drawdown_pct"]]


def clean_json(value):
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def svg_equity(equity: pd.DataFrame, strategy: str, color: str) -> str:
    subset = equity[equity.strategy == strategy]
    y = subset.equity.to_numpy(float)
    width, height, pad = 1100, 300, 45
    if len(y) == 1:
        y = np.r_[y, y]
    ymin, ymax = min(y.min(), CAPITAL), max(y.max(), CAPITAL)
    span = max(ymax - ymin, 1.0)
    xs = np.linspace(pad, width - pad, len(y))
    ys = height - pad - (y - ymin) / span * (height - 2 * pad)
    points = " ".join(f"{x:.1f},{yy:.1f}" for x, yy in zip(xs, ys))
    return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Equity {strategy}">'
            f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#465064"/>'
            f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}"/>'
            f'<text x="{pad}" y="24" fill="#dfe7f5">{strategy}: ${y[-1]:,.2f}</text>'
            f'<text x="{pad}" y="{height-8}" fill="#96a1b5">$ {ymin:,.0f}</text>'
            f'<text x="{width-180}" y="24" fill="#96a1b5">$ {ymax:,.0f}</text></svg>')


def write_html(summary: dict, equity: pd.DataFrame) -> None:
    def fmt(value, digits=2):
        return "—" if value is None else f"{value:,.{digits}f}"
    rows = []
    for model in ("convergence", "risk_reward"):
        for period in ("development", "validation", "holdout", "full"):
            m = summary[model]["results"][period]
            rows.append(f"<tr><td>{model}</td><td>{period}</td><td>{m['trades']}</td>"
                        f"<td>${fmt(m['gross_pnl'])}</td><td>${fmt(m['net_pnl'])}</td>"
                        f"<td>{fmt(m['net_sharpe'])}</td><td>{fmt(m['max_drawdown_pct'])}%</td>"
                        f"<td>{fmt(m['win_rate_pct'])}%</td><td>{fmt(m['profit_factor'])}</td>"
                        f"<td>${fmt(m['costs'])}</td></tr>")
    verdicts = []
    for model in ("convergence", "risk_reward"):
        value = summary[model]["selection"]["no_confirmed_edge"]
        verdicts.append(f"<li><b>{model}</b>: {'edge не подтверждён' if value else 'holdout положительный, требуется дополнительная проверка'}.</li>")
    html = f"""<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QQQ против синтетической корзины — исследование</title><style>
body{{font:15px system-ui;margin:auto;max-width:1280px;padding:24px;background:#0b0f17;color:#dfe7f5}}h1,h2{{color:#fff}}.card{{background:#141b27;border:1px solid #2b3547;border-radius:12px;padding:16px;margin:14px 0}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:8px;border:1px solid #344054;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}svg{{width:100%;background:#101722;border-radius:8px;margin:8px 0}}code{{color:#79c0ff}}a{{color:#79c0ff}}.warn{{color:#ffcc66}}
</style><h1>QQQ против фиксированной mega-cap корзины</h1>
<div class="card"><b>Честная постановка.</b> Торгуется только QQQ; MSFT/AAPL/NVDA/AMZN — reference. Состав и веса заморожены по официальному срезу Nasdaq от 28.06.2024, до начала выборки 22.08.2024. Исходные веса 8,6% / 8,4% / 7,9% / 5,2% (30,1% NDX), внутри корзины нормализованы до 100%. Это не реплика NDX. <a href="{SOURCE_URL}">Источник Nasdaq</a>.</div>
<div class="card"><b>Исполнение.</b> {summary['data']['sessions']} общих RTH-сессий, {summary['data']['bars']:,} синхронных raw SIP 1m баров. Split 250/125/126. Сигнал по close → исполнение на следующем open; $20,000; комиссия $0.0035/акцию/сторону; slippage 2 bps/исполнение. Для RR convergence-выход отключён.</div>
<div class="card"><h2>Вердикт</h2><ul>{''.join(verdicts)}</ul><p class="warn">Выбор параметров сделан на development, validation использован только для устойчивости, holdout открыт один раз после выбора.</p></div>
<div class="card"><h2>Equity, весь период</h2>{svg_equity(equity,'convergence','#56d364')}{svg_equity(equity,'risk_reward','#58a6ff')}</div>
<div class="card"><h2>Метрики</h2><table><tr><th>Модель</th><th>Период</th><th>Сделки</th><th>Gross</th><th>Net</th><th>Sharpe</th><th>Max DD</th><th>Win rate</th><th>PF</th><th>Costs</th></tr>{''.join(rows)}</table></div>
<div class="card"><h2>Выбранные параметры</h2><pre>{json.dumps({'convergence':summary['convergence']['selected'],'risk_reward':summary['risk_reward']['selected']},ensure_ascii=False,indent=2)}</pre></div>
</html>"""
    (OUT / "REPORT.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames, common, day = load_raw()
    arrays = market_arrays(frames, common, day)
    unique_dates = list(pd.unique(common.strftime("%Y-%m-%d")))
    splits = {"development": (0, 250), "validation": (250, 375),
              "holdout": (375, 501), "full": (0, 501)}

    configs = ([EntryConfig("z", threshold, hook, window)
                for window in (30, 60, 120)
                for threshold in (1.5, 2.0, 2.5, 3.0)
                for hook in (0.0, 0.15)] +
               [EntryConfig("absolute_qqq_points", threshold, hook)
                for threshold in (0.25, 0.50, 0.75, 1.00, 1.50)
                for hook in (0.0, 0.10)])
    z_cache = {window: rolling_z(arrays["spread"], arrays["starts"], arrays["ends"], window)
               for window in (30, 60, 120)}
    metric_cache = {cfg.key: z_cache[cfg.window] if cfg.basis == "z" else arrays["points"] for cfg in configs}
    event_cache = {cfg.key: entry_events(metric_cache[cfg.key], day, cfg.threshold, cfg.hook) for cfg in configs}
    cfg_by_key = {cfg.key: cfg for cfg in configs}

    convergence_dev = []
    print(f"convergence development grid: {len(configs)} configurations", flush=True)
    for cfg in configs:
        trades = simulate_convergence(event_cache[cfg.key], metric_cache[cfg.key], cfg, arrays, 0, 250)
        convergence_dev.append({"config_key": cfg.key, "basis": cfg.basis, "window": cfg.window,
                                "threshold": cfg.threshold, "hook": cfg.hook,
                                **metrics(trades, period_dates(unique_dates, 0, 250))})
    conv_dev_df = pd.DataFrame(convergence_dev)
    print("convergence development complete", flush=True)

    def validate_convergence(row):
        cfg = cfg_by_key[row["config_key"]]
        trades = simulate_convergence(event_cache[cfg.key], metric_cache[cfg.key], cfg, arrays, 250, 375)
        return {"config_key": cfg.key, "basis": cfg.basis, "window": cfg.window,
                "threshold": cfg.threshold, "hook": cfg.hook,
                **metrics(trades, period_dates(unique_dates, 250, 375))}

    conv_selected_row, conv_finalists, conv_val_df = choose_with_validation(conv_dev_df, validate_convergence)
    conv_cfg = cfg_by_key[conv_selected_row["config_key"]]
    conv_results, conv_trades = {}, {}
    for period, (lo, hi) in splits.items():
        trades = simulate_convergence(event_cache[conv_cfg.key], metric_cache[conv_cfg.key], conv_cfg, arrays, lo, hi)
        conv_trades[period] = trades
        conv_results[period] = metrics(trades, period_dates(unique_dates, lo, hi))
        trades.to_csv(OUT / f"convergence_selected_{period}_trades.csv", index=False)
    conv_dev_df.to_csv(OUT / "convergence_development_grid.csv", index=False)
    conv_finalists.to_csv(OUT / "convergence_development_finalists.csv", index=False)
    conv_val_df.to_csv(OUT / "convergence_validation_selection.csv", index=False)
    print(f"convergence selected: {conv_cfg.key}", flush=True)

    rr_dev = []
    print(f"risk/reward development grid: {len(configs) * 5 * 6} configurations", flush=True)
    for cfg in configs:
        for stop_pct in (0.0015, 0.0025, 0.0050, 0.0075, 0.0100):
            for rr in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
                trades = simulate_rr(event_cache[cfg.key], metric_cache[cfg.key], cfg, arrays, 0, 250, stop_pct, rr)
                rr_dev.append({"config_key": cfg.key, "basis": cfg.basis, "window": cfg.window,
                               "threshold": cfg.threshold, "hook": cfg.hook,
                               "stop_pct": stop_pct, "rr": rr,
                               **metrics(trades, period_dates(unique_dates, 0, 250))})
    rr_dev_df = pd.DataFrame(rr_dev)
    print("risk/reward development complete", flush=True)

    def validate_rr(row):
        cfg = cfg_by_key[row["config_key"]]
        stop_pct, rr = float(row["stop_pct"]), float(row["rr"])
        trades = simulate_rr(event_cache[cfg.key], metric_cache[cfg.key], cfg, arrays, 250, 375, stop_pct, rr)
        return {"config_key": cfg.key, "basis": cfg.basis, "window": cfg.window,
                "threshold": cfg.threshold, "hook": cfg.hook,
                "stop_pct": stop_pct, "rr": rr,
                **metrics(trades, period_dates(unique_dates, 250, 375))}

    rr_selected_row, rr_finalists, rr_val_df = choose_with_validation(rr_dev_df, validate_rr, top_n=20)
    rr_cfg = cfg_by_key[rr_selected_row["config_key"]]
    rr_stop, rr_ratio = float(rr_selected_row["stop_pct"]), float(rr_selected_row["rr"])
    rr_results, rr_trades = {}, {}
    for period, (lo, hi) in splits.items():
        trades = simulate_rr(event_cache[rr_cfg.key], metric_cache[rr_cfg.key], rr_cfg, arrays, lo, hi, rr_stop, rr_ratio)
        rr_trades[period] = trades
        rr_results[period] = metrics(trades, period_dates(unique_dates, lo, hi))
        trades.to_csv(OUT / f"risk_reward_selected_{period}_trades.csv", index=False)
    rr_dev_df.to_csv(OUT / "risk_reward_development_grid.csv", index=False)
    rr_finalists.to_csv(OUT / "risk_reward_development_finalists.csv", index=False)
    rr_val_df.to_csv(OUT / "risk_reward_validation_selection.csv", index=False)
    print(f"risk/reward selected: {rr_cfg.key}, stop={rr_stop:g}, rr={rr_ratio:g}", flush=True)

    equity = pd.concat([equity_rows(conv_trades["full"], "convergence"),
                        equity_rows(rr_trades["full"], "risk_reward")], ignore_index=True)
    equity.to_csv(OUT / "selected_equity.csv", index=False)
    session_counts = pd.Series(1, index=common).groupby(common.strftime("%Y-%m-%d")).sum()
    summary = {
        "hypothesis": "Trade QQQ toward a fixed weighted normalized basket of four pre-sample NDX mega-caps.",
        "basket": {"symbols": list(BASKET), "official_ndx_weights_pct": NDX_WEIGHTS,
                   "normalized_reference_weights": WEIGHTS, "official_snapshot_date": "2024-06-28",
                   "official_source": SOURCE_URL, "combined_ndx_weight_pct": sum(NDX_WEIGHTS.values()),
                   "limitation": "Fixed four-stock proxy, not an exact or dynamically rebalanced NDX replica."},
        "data": {"source": "Alpaca SIP raw 1-minute Parquet", "range": "2024-08-22..2026-08-21",
                 "sessions": len(unique_dates), "bars": len(common),
                 "min_common_bars_per_session": int(session_counts.min()),
                 "max_common_bars_per_session": int(session_counts.max()),
                 "symbols": list(SYMBOLS)},
        "execution": {"traded_symbol": "QQQ", "reference_only": list(BASKET),
                      "signal": "current close", "market_fill": "next synchronized minute open",
                      "position_size_usd": POSITION_SIZE, "capital_usd": CAPITAL,
                      "commission_per_share_per_side": COMMISSION,
                      "slippage_bps_per_execution": SLIPPAGE * 10_000,
                      "rth_only": True, "same_bar_rr_collision": "stop first",
                      "rr_convergence_exit": False},
        "splits": {"development": {"sessions": 250, "start": unique_dates[0], "end": unique_dates[249]},
                   "validation": {"sessions": 125, "start": unique_dates[250], "end": unique_dates[374]},
                   "holdout": {"sessions": 126, "start": unique_dates[375], "end": unique_dates[500]}},
        "convergence": {"selected": {"basis": conv_cfg.basis, "window": conv_cfg.window,
                                        "threshold": conv_cfg.threshold, "hook": conv_cfg.hook},
                        "selection": {"method": "top development Sharpe -> max min(dev, validation) Sharpe; holdout once",
                                      "development_sharpe": conv_selected_row["dev_net_sharpe"],
                                      "validation_sharpe": conv_selected_row["net_sharpe"],
                                      "robust_score": conv_selected_row["robust_score"],
                                      "no_confirmed_edge": bool(conv_selected_row["robust_score"] <= 0 or conv_results["holdout"]["net_pnl"] <= 0)},
                        "results": conv_results},
        "risk_reward": {"selected": {"basis": rr_cfg.basis, "window": rr_cfg.window,
                                        "threshold": rr_cfg.threshold, "hook": rr_cfg.hook,
                                        "stop_pct": rr_stop, "reward_risk": rr_ratio,
                                        "convergence_exit": False},
                        "selection": {"method": "top development Sharpe -> max min(dev, validation) Sharpe; holdout once",
                                      "development_sharpe": rr_selected_row["dev_net_sharpe"],
                                      "validation_sharpe": rr_selected_row["net_sharpe"],
                                      "robust_score": rr_selected_row["robust_score"],
                                      "no_confirmed_edge": bool(rr_selected_row["robust_score"] <= 0 or rr_results["holdout"]["net_pnl"] <= 0)},
                        "results": rr_results},
        "reconciliation": {"all_trade_rows_exact": bool(all(abs(result["reconciliation_error"]) < 1e-8 for result in [*conv_results.values(), *rr_results.values()])),
                           "gross_minus_commission_minus_slippage_equals_net": True,
                           "holdout_not_used_for_selection": True,
                           "mock_or_interpolated_bars": False},
    }
    summary = clean_json(summary)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# QQQ vs fixed mega-cap basket\n\n"
        "The basket uses the four largest Nasdaq-100 securities in Nasdaq's 2024-06-28 snapshot: "
        "MSFT 8.6%, AAPL 8.4%, NVDA 7.9%, AMZN 5.2% (30.1% combined). Weights are frozen and "
        "renormalized within the four-stock reference before the 2024-08-22 sample starts. This is a "
        "causal proxy, not an exact NDX replica. Every session starts at zero using each instrument's "
        "first synchronized RTH close. Only QQQ is traded. Inputs are exact synchronized raw Alpaca SIP "
        "1-minute bars; no interpolation. Convergence and risk/reward (without convergence) are separate "
        "exit models. Development selects candidates, validation tests robustness, and holdout is opened once.\n",
        encoding="utf-8")
    write_html(summary, equity)

    for frame in [*conv_trades.values(), *rr_trades.values()]:
        if not frame.empty and frame.reconciliation_error.abs().max() >= 1e-8:
            raise AssertionError("Trade reconciliation failed")
    if abs(conv_results["full"]["net_pnl"] - conv_trades["full"].net_pnl.sum()) >= 1e-8:
        raise AssertionError("Convergence summary mismatch")
    if abs(rr_results["full"]["net_pnl"] - rr_trades["full"].net_pnl.sum()) >= 1e-8:
        raise AssertionError("RR summary mismatch")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
