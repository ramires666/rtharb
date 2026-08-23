"""Recalculate the strategy from SIP minute bars and build an auditable report."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from rtharb.config import AppConfig
from rtharb.data.loader import DataLoader
from rtharb.models.fair_value import FairValueModel
from rtharb.models.signals import SignalGenerator
from rtharb.backtest.engine import BacktestEngine
from rtharb.backtest.metrics import calculate_performance_metrics


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "audit_output"


def run_case(metrics: pd.DataFrame, cfg: AppConfig, z_entry: float, reversal_delta: float,
             max_holding_bars=None, stop_loss_pct=None):
    sig = SignalGenerator(
        z_entry=z_entry, reversal_delta=reversal_delta,
        reversal_timeout_bars=cfg.strategy.reversal_timeout_bars,
        enable_extreme_entry_lockout=cfg.strategy.enable_extreme_entry_lockout,
        enable_extreme_emergency_exit=cfg.strategy.enable_extreme_emergency_exit,
        z_max_allowed=cfg.strategy.z_max_allowed,
        lockout_mode=cfg.strategy.lockout_mode, z_exit=cfg.strategy.z_exit,
        forced_close_time=cfg.strategy.forced_close_time,
        min_session_warmup_bars=cfg.strategy.min_session_warmup_bars,
    ).generate_signals(metrics)
    bt = BacktestEngine(
        initial_capital=cfg.backtest.initial_capital,
        position_size_usd=cfg.backtest.position_size_usd,
        commission_per_share=cfg.backtest.commission_per_share,
        slippage_pct=cfg.backtest.slippage_pct,
        allow_short=cfg.backtest.allow_short,
        max_holding_bars=max_holding_bars,
        stop_loss_pct=stop_loss_pct,
    ).run(sig, ticker_target=cfg.strategy.ticker_target)
    perf = calculate_performance_metrics(bt["df_results"], bt["trades_df"], cfg.backtest.initial_capital)
    return sig, bt, perf


def calibrate_winner_preserving_filters(metrics: pd.DataFrame, trades: pd.DataFrame, quantile=0.95):
    """Fit both overlays on winners only at the requested survival quantile."""
    winners = trades[trades["net_pnl"] > 0].copy()
    if winners.empty:
        raise ValueError("Cannot calibrate filters without profitable training trades")
    maes = []
    for trade in winners.itertuples(index=False):
        path = metrics[(metrics.index >= trade.entry_time) & (metrics.index < trade.exit_time)]
        if trade.direction == 1:
            mae = max(0.0, (trade.entry_price - float(path["target_low"].min())) / trade.entry_price)
        else:
            mae = max(0.0, (float(path["target_high"].max()) - trade.entry_price) / trade.entry_price)
        maes.append(mae)
    winners["mae_pct"] = maes
    max_holding = int(winners["duration_bars"].quantile(quantile, interpolation="higher"))
    stop_loss = float(winners["mae_pct"].quantile(quantile, interpolation="higher"))
    return max_holding, stop_loss, winners


def audit_sessions(lead: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    calendar = pd.read_csv(ROOT / "data_cache" / "market_calendar.csv", dtype=str)
    common = lead.index.intersection(target.index)
    actual = pd.Series(1, index=common).groupby(common.strftime("%Y-%m-%d")).sum()
    rows = []
    for row in calendar.itertuples(index=False):
        expected = int((pd.Timestamp(row.close) - pd.Timestamp(row.open)).total_seconds() // 60)
        rows.append({"date": row.date, "open": row.open, "close": row.close,
                     "expected_bars": expected, "actual_bars": int(actual.get(row.date, 0))})
    result = pd.DataFrame(rows)
    result["ok"] = result["expected_bars"] == result["actual_bars"]
    if not result["ok"].all():
        bad = result.loc[~result["ok"]].to_dict("records")
        raise AssertionError(f"Incomplete synchronized SIP sessions: {bad[:10]}")
    return result


def _xy(values, x0, x1, y0, y1):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    lo, hi = np.nanmin(arr[finite]), np.nanmax(arr[finite])
    pad = max((hi - lo) * 0.04, 1e-9)
    lo, hi = lo - pad, hi + pad
    xs = np.linspace(x0, x1, len(arr))
    ys = y1 - (arr - lo) / (hi - lo) * (y1 - y0)
    return xs, ys, lo, hi


def make_session_svg(metrics: pd.DataFrame, trades: pd.DataFrame, date: str, path: Path):
    day = metrics[metrics["session_date"].astype(str) == date]
    n = len(day)
    xs, _, lo, hi = _xy(pd.concat([day.target_low, day.target_high, day.target_fair_price]), 90, 1910, 90, 650)
    xs = np.linspace(90, 1910, n)
    py = lambda v: 650 - (np.asarray(v) - lo) / (hi - lo) * 560
    width = max(1.2, 1820 / n * 0.72)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="2000" height="1000" viewBox="0 0 2000 1000">',
             '<rect width="2000" height="1000" fill="#0b0e14"/>',
             f'<text x="90" y="42" fill="white" font-size="25">NVDA {date}: {n} настоящих минутных свечей SIP</text>',
             f'<text x="90" y="70" fill="#9aa4b2" font-size="16">RTH America/New_York; каждая линия+прямоугольник ниже = одна строка Parquet</text>',
             f'<g id="candles-1m" data-count="{n}">']
    for x, row in zip(xs, day.itertuples()):
        color = "#19c37d" if row.target_close >= row.target_open else "#ff5c5c"
        yh, yl, yo, yc = py([row.target_high, row.target_low, row.target_open, row.target_close])
        top, height = min(yo, yc), max(abs(yc - yo), 1.0)
        parts.append(f'<line class="wick" x1="{x:.2f}" y1="{yh:.2f}" x2="{x:.2f}" y2="{yl:.2f}" stroke="{color}"/>')
        parts.append(f'<rect class="body" x="{x-width/2:.2f}" y="{top:.2f}" width="{width:.2f}" height="{height:.2f}" fill="{color}"/>')
    parts.append('</g>')
    fair = ' '.join(f'{x:.2f},{y:.2f}' for x, y in zip(xs, py(day.target_fair_price)))
    parts.append(f'<polyline points="{fair}" fill="none" stroke="#ffd43b" stroke-width="2"/>')
    z = day.z_score.to_numpy(float)
    z_clip = np.clip(z, -5, 5)
    zy = 900 - (z_clip + 5) / 10 * 200
    zpoints = ' '.join(f'{x:.2f},{y:.2f}' for x, y, ok in zip(xs, zy, np.isfinite(z)) if ok)
    parts += ['<rect x="90" y="700" width="1820" height="200" fill="#121722"/>',
              '<line x1="90" y1="800" x2="1910" y2="800" stroke="#667085"/>',
              '<line x1="90" y1="760" x2="1910" y2="760" stroke="#ff5c5c" stroke-dasharray="6 4"/>',
              '<line x1="90" y1="840" x2="1910" y2="840" stroke="#19c37d" stroke-dasharray="6 4"/>',
              f'<polyline points="{zpoints}" fill="none" stroke="#36c5f0" stroke-width="2"/>',
              '<text x="20" y="765" fill="#ff5c5c">+2σ</text><text x="20" y="845" fill="#19c37d">−2σ</text>',
              f'<text x="90" y="950" fill="#9aa4b2">Цена: ${lo:.2f}…${hi:.2f}; жёлтая линия — fair value QQQ×β</text>',
              '</svg>']
    path.write_text('\n'.join(parts), encoding="utf-8")


def make_equity_svg(results: pd.DataFrame, path: Path):
    daily = results.groupby("session_date")["portfolio_equity"].last()
    xs, ys, lo, hi = _xy(daily, 80, 1520, 70, 520)
    points = ' '.join(f'{x:.2f},{y:.2f}' for x, y in zip(xs, ys))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="600">
<rect width="1600" height="600" fill="#0b0e14"/><text x="80" y="38" fill="white" font-size="24">Out-of-sample equity, daily closes ({len(daily)} points)</text>
<polyline points="{points}" fill="none" stroke="#36c5f0" stroke-width="2"/>
<text x="80" y="570" fill="#9aa4b2">{daily.index[0]} — {daily.index[-1]} | ${lo:,.0f}…${hi:,.0f}</text></svg>'''
    path.write_text(svg, encoding="utf-8")


def main():
    OUT.mkdir(exist_ok=True)
    cfg = AppConfig.load(str(ROOT / "configs" / "default_config.yaml"))
    loader = DataLoader(cfg.cache_dir, cfg.strategy.data_source, cfg.strategy.data_feed)
    lead, target = loader.get_synchronized_pair(cfg.strategy.ticker_lead, cfg.strategy.ticker_target)
    session_audit = audit_sessions(lead, target)
    model = FairValueModel(cfg.strategy.beta_mode, cfg.strategy.beta_rolling_days,
                           cfg.strategy.rolling_window_w, cfg.strategy.min_session_warmup_bars,
                           cfg.strategy.min_sigma_history_days)
    metrics = model.compute_intraday_metrics(lead, target)
    assert metrics.groupby("session_date")["spread"].first().abs().max() < 1e-12

    dates = sorted(metrics.session_date.unique())
    development_end = dates[int(len(dates) * 0.40)]
    validation_end = dates[int(len(dates) * 0.60)]
    development = metrics[metrics.session_date < development_end]
    validation = metrics[(metrics.session_date >= development_end) & (metrics.session_date < validation_end)]
    holdout = metrics[metrics.session_date >= validation_end]
    grid = []
    for z in [1.5, 1.75, 2.0, 2.25, 2.5, 2.75]:
        for delta in [0.05, 0.15, 0.25]:
            _, _, perf = run_case(development, cfg, z, delta)
            grid.append({"z_entry": z, "reversal_delta": delta, **asdict(perf)})
    grid_df = pd.DataFrame(grid).sort_values(["sharpe_ratio", "total_pnl"], ascending=False)
    eligible = grid_df[grid_df.total_trades >= 50]
    best = (eligible if not eligible.empty else grid_df).iloc[0]
    z, delta = float(best.z_entry), float(best.reversal_delta)
    _, dev_base_bt, dev_base_perf = run_case(development, cfg, z, delta)
    _, val_base_bt, val_base_perf = run_case(validation, cfg, z, delta)
    _, holdout_base_bt, holdout_base_perf = run_case(holdout, cfg, z, delta)
    _, _, winning_paths = calibrate_winner_preserving_filters(development, dev_base_bt["trades_df"])
    filter_grid = []
    quantiles = [0.90, 0.925, 0.95, 0.975, 0.99]
    for time_q in quantiles:
        max_holding = int(winning_paths["duration_bars"].quantile(time_q, interpolation="higher"))
        for stop_q in quantiles:
            stop_loss = float(winning_paths["mae_pct"].quantile(stop_q, interpolation="higher"))
            _, _, perf = run_case(validation, cfg, z, delta, max_holding, stop_loss)
            joint_survival = float(((winning_paths["duration_bars"] <= max_holding) &
                                    (winning_paths["mae_pct"] <= stop_loss)).mean())
            filter_grid.append({"time_winner_quantile": time_q, "stop_winner_quantile": stop_q,
                                "max_holding_bars": max_holding, "stop_loss_pct": stop_loss,
                                "development_joint_winner_survival": joint_survival,
                                **asdict(perf)})
    filter_grid_df = pd.DataFrame(filter_grid).sort_values(
        ["sharpe_ratio", "total_pnl"], ascending=False
    )
    survival_eligible = filter_grid_df[filter_grid_df.development_joint_winner_survival >= 0.95]
    chosen_filter = (survival_eligible if not survival_eligible.empty else filter_grid_df).iloc[0]
    max_holding = int(chosen_filter.max_holding_bars)
    stop_loss = float(chosen_filter.stop_loss_pct)
    time_q = float(chosen_filter.time_winner_quantile)
    stop_q = float(chosen_filter.stop_winner_quantile)
    joint_survival = float(chosen_filter.development_joint_winner_survival)
    _, dev_bt, dev_perf = run_case(development, cfg, z, delta, max_holding, stop_loss)
    _, val_bt, val_perf = run_case(validation, cfg, z, delta, max_holding, stop_loss)
    _, holdout_bt, holdout_perf = run_case(holdout, cfg, z, delta, max_holding, stop_loss)
    _, full_bt, full_perf = run_case(metrics, cfg, z, delta, max_holding, stop_loss)

    session_audit.to_csv(OUT / "session_bar_audit.csv", index=False)
    grid_df.to_csv(OUT / "training_parameter_grid.csv", index=False)
    filter_grid_df.to_csv(OUT / "training_filter_grid.csv", index=False)
    winning_paths.to_csv(OUT / "training_winner_paths.csv", index=False)
    dev_bt["trades_df"].to_csv(OUT / "trades_development.csv", index=False)
    val_bt["trades_df"].to_csv(OUT / "trades_validation.csv", index=False)
    holdout_bt["trades_df"].to_csv(OUT / "trades_holdout.csv", index=False)
    full_bt["trades_df"].to_csv(OUT / "trades_full.csv", index=False)
    latest = str(dates[-1])
    make_session_svg(metrics, full_bt["trades_df"], latest, OUT / f"session_{latest}.svg")
    make_equity_svg(holdout_bt["df_results"], OUT / "equity_holdout.svg")

    summary = {
        "data": {"feed": "Alpaca SIP", "timezone": "America/New_York",
                 "sessions": len(dates), "bars": len(metrics), "first": str(dates[0]), "last": str(dates[-1]),
                 "all_session_counts_match_calendar": bool(session_audit.ok.all())},
        "model": {"lead": "QQQ", "target": "NVDA", "session_anchor": "first 1m close",
                  "selected_before_holdout": {"z_entry": z, "reversal_delta": delta,
                    "max_holding_bars": max_holding,
                    "stop_loss_pct": stop_loss,
                    "selected_time_winner_quantile": time_q,
                    "selected_stop_winner_quantile": stop_q,
                    "development_joint_winner_survival": joint_survival},
                    "development_end": str(development_end), "validation_end": str(validation_end)},
        "baseline_without_last_two_filters": {"development": asdict(dev_base_perf),
                                                "validation": asdict(val_base_perf),
                                                "holdout": asdict(holdout_base_perf)},
        "with_time_and_stop_filters": {"development": asdict(dev_perf),
                                         "validation": asdict(val_perf),
                                         "holdout": asdict(holdout_perf)},
        "full_descriptive": asdict(full_perf),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    def row(label, p):
        return f'<tr><td>{label}</td><td>{p.total_trades}</td><td>{p.total_return_pct:.2f}%</td><td>{p.sharpe_ratio:.2f}</td><td>{p.sortino_ratio:.2f}</td><td>{p.max_drawdown_pct:.2f}%</td><td>{p.win_rate_pct:.1f}%</td><td>{p.profit_factor:.2f}</td></tr>'
    report = f'''<!doctype html><meta charset="utf-8"><title>Проверенный пересчёт QQQ→NVDA</title>
<style>body{{font:16px system-ui;max-width:1200px;margin:auto;padding:30px;background:#0b0e14;color:#e6edf3}}table{{border-collapse:collapse}}td,th{{padding:9px;border:1px solid #344054}}img{{width:100%}}code{{color:#36c5f0}}</style>
<h1>Проверенный пересчёт QQQ → NVDA</h1><p>Данные: Alpaca SIP, только официальная RTH America/New_York, {len(metrics):,} минут, {len(dates)} сессий. Все количества баров сверены с биржевым календарём.</p>
<p>Отклонение обнуляется на первой минуте каждой сессии. Сигналы выбраны на development: <code>|Z| ≥ {z:g}</code>, hook <code>{delta:g}σ</code>. Перцентили фильтров 90/92.5/95/97.5/99 проверены на отдельном validation при условии совместного сохранения ≥95% исходных development winners. Выбрано: время q={time_q*100:g}% → <code>{max_holding} мин</code>; стоп q={stop_q*100:g}% MAE → <code>{stop_loss*100:.2f}%</code>; совместное сохранение {joint_survival*100:.1f}%. Последние 40% истории — нетронутый holdout.</p>
<table><tr><th>Период</th><th>Сделки</th><th>Доходность</th><th>Sharpe</th><th>Sortino</th><th>Max DD</th><th>Win rate</th><th>PF</th></tr>{row('Development без фильтров',dev_base_perf)}{row('Validation без фильтров',val_base_perf)}{row('Holdout без фильтров',holdout_base_perf)}{row('Development + time/stop',dev_perf)}{row('Validation + time/stop',val_perf)}{row('Holdout + time/stop',holdout_perf)}{row('Full + time/stop (описательно)',full_perf)}</table>
<h2>Equity holdout</h2><img src="equity_holdout.svg"><h2>Последняя сессия: реально 1m</h2><img src="session_{latest}.svg">
<h2>Ограничения</h2><p>Это одноногая симуляция NVDA по лидеру QQQ, не парный хедж. Учтены комиссия и проскальзывание; не учтены borrow fee, влияние объёма, налоги и задержка получения данных. Результат не является обещанием будущей доходности.</p>'''
    (OUT / "REPORT.html").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
