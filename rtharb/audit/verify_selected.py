"""Re-run the frozen research winner through the production state machine."""

import json
from dataclasses import asdict
from pathlib import Path

from rtharb.config import AppConfig
from rtharb.data.loader import DataLoader
from rtharb.models.fair_value import FairValueModel
from rtharb.models.signals import SignalGenerator
from rtharb.backtest.engine import BacktestEngine
from rtharb.backtest.metrics import calculate_performance_metrics


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_output"


def run_exact(metrics, cfg, p):
    signals = SignalGenerator(
        z_entry=p["z_entry"], reversal_delta=p["hook_delta"],
        reversal_timeout_bars=p["hook_timeout"],
        enable_extreme_entry_lockout=p["z_lockout"] is not None,
        enable_extreme_emergency_exit=False,
        z_max_allowed=p["z_lockout"] or 99.0,
        z_exit=p["exit_band"], forced_close_time="15:55",
        min_session_warmup_bars=15,
    ).generate_signals(metrics)
    bt = BacktestEngine(
        initial_capital=cfg.backtest.initial_capital,
        position_size_usd=cfg.backtest.position_size_usd,
        commission_per_share=cfg.backtest.commission_per_share,
        slippage_pct=cfg.backtest.slippage_pct,
        allow_short=True,
    ).run(signals, ticker_target="NVDA")
    perf = calculate_performance_metrics(bt["df_results"], bt["trades_df"], cfg.backtest.initial_capital)
    return bt, asdict(perf)


def main():
    fast = json.loads((OUT / "base_strategy_summary.json").read_text(encoding="utf-8"))
    p = fast["selected_parameters"]
    cfg = AppConfig.load(str(ROOT / "configs" / "default_config.yaml"))
    lead, target = DataLoader(cfg.cache_dir, "alpaca", "sip").get_synchronized_pair("QQQ", "NVDA")
    metrics = FairValueModel(
        beta_mode=p["beta_mode"], beta_rolling_days=p["beta_days"],
        rolling_window_w=p["window"], min_session_warmup_bars=15,
    ).compute_intraday_metrics(lead, target)
    dates = sorted(metrics.session_date.unique())
    dev_end, val_end = len(dates)//2, len(dates)*3//4
    parts = {
        "development": metrics[metrics.session_date < dates[dev_end]],
        "validation": metrics[(metrics.session_date >= dates[dev_end]) & (metrics.session_date < dates[val_end])],
        "holdout": metrics[metrics.session_date >= dates[val_end]],
    }
    result = {"parameters": p}
    for name, frame in parts.items():
        bt, perf = run_exact(frame, cfg, p)
        bt["trades_df"].to_csv(OUT / f"exact_{name}_trades.csv", index=False)
        result[name] = perf
        result[name]["fast_trade_count"] = fast[name]["trades"]
        result[name]["fast_net_pnl"] = fast[name]["net_pnl"]
    (OUT / "exact_selected_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
