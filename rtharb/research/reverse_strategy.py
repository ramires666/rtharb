"""Exact causal entry-direction inversion of the frozen base strategy."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from rtharb.config import AppConfig
from rtharb.data.loader import DataLoader
from rtharb.models.fair_value import FairValueModel
from rtharb.models.signals import SignalGenerator, SignalType
from rtharb.backtest.engine import BacktestEngine
from rtharb.backtest.metrics import calculate_performance_metrics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_output" / "reverse_strategy"

def invert_signals(df):
    out = df.copy()
    out["signal"] = out["signal"].map({SignalType.BUY_LONG: SignalType.SELL_SHORT, SignalType.SELL_SHORT: SignalType.BUY_LONG}).fillna(out["signal"])
    return out

def run(df, cfg, inverse):
    sig = invert_signals(df) if inverse else df
    bt = BacktestEngine(initial_capital=cfg.backtest.initial_capital, position_size_usd=cfg.backtest.position_size_usd,
                        commission_per_share=cfg.backtest.commission_per_share, slippage_pct=cfg.backtest.slippage_pct,
                        allow_short=cfg.backtest.allow_short).run(sig, cfg.strategy.ticker_target)
    return bt, calculate_performance_metrics(bt["df_results"], bt["trades_df"], cfg.backtest.initial_capital)

def event_check(base, inv):
    if len(base) != len(inv): return {"same_count": False, "same_events": False, "base_count": len(base), "inverse_count": len(inv)}
    same = True
    for c in ["entry_time", "exit_time"]:
        same &= bool((pd.to_datetime(base[c]).reset_index(drop=True).to_numpy() == pd.to_datetime(inv[c]).reset_index(drop=True).to_numpy()).all())
    same &= bool((base["duration_bars"].to_numpy() == inv["duration_bars"].to_numpy()).all())
    return {"same_count": True, "same_events": same, "base_count": len(base), "inverse_count": len(inv)}

def row(perf, trades):
    gross = float(trades["gross_pnl"].sum()) if not trades.empty else 0.0
    d = {"trades": perf.total_trades, "gross_pnl": gross, "net_pnl": perf.total_pnl, "net_return_pct": perf.total_return_pct,
         "net_sharpe": perf.sharpe_ratio, "net_sortino": perf.sortino_ratio, "max_drawdown_pct": perf.max_drawdown_pct,
         "win_rate_pct": perf.win_rate_pct, "profit_factor": perf.profit_factor, "avg_duration": perf.avg_duration_mins,
         "commissions": perf.total_commissions, "slippage": perf.total_slippage}
    d["reconciliation_error"] = d["gross_pnl"] - d["commissions"] - d["slippage"] - d["net_pnl"]
    return d

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = AppConfig.load(str(ROOT / "configs" / "default_config.yaml"))
    frozen = json.loads((ROOT / "research_output" / "base_strategy_summary.json").read_text(encoding="utf-8"))["selected_parameters"]
    cfg.strategy.beta_mode, cfg.strategy.beta_rolling_days = frozen["beta_mode"], int(frozen["beta_days"])
    cfg.strategy.rolling_window_w, cfg.strategy.z_entry = int(frozen["window"]), float(frozen["z_entry"])
    cfg.strategy.reversal_delta, cfg.strategy.reversal_timeout_bars = float(frozen["hook_delta"]), int(frozen["hook_timeout"])
    cfg.strategy.z_exit, cfg.strategy.z_max_allowed = float(frozen["exit_band"]), float(frozen["z_lockout"])
    lead, target = DataLoader(cfg.cache_dir, "alpaca", "sip").get_synchronized_pair("QQQ", "NVDA")
    model = FairValueModel(cfg.strategy.beta_mode, cfg.strategy.beta_rolling_days, cfg.strategy.rolling_window_w,
                           cfg.strategy.min_session_warmup_bars, cfg.strategy.min_sigma_history_days)
    metrics = model.compute_intraday_metrics(lead, target)
    source = SignalGenerator(z_entry=cfg.strategy.z_entry, reversal_delta=cfg.strategy.reversal_delta,
                             reversal_timeout_bars=cfg.strategy.reversal_timeout_bars, enable_extreme_entry_lockout=True,
                             enable_extreme_emergency_exit=False, z_max_allowed=cfg.strategy.z_max_allowed,
                             lockout_mode="day_lockout", z_exit=cfg.strategy.z_exit, forced_close_time=cfg.strategy.forced_close_time,
                             min_session_warmup_bars=cfg.strategy.min_session_warmup_bars).generate_signals(metrics)
    dates = sorted(source.session_date.unique()); n = len(dates); dev, val = n//2, n*3//4
    masks = {"development": source.session_date < dates[dev], "validation": (source.session_date >= dates[dev]) & (source.session_date < dates[val]),
             "holdout": source.session_date >= dates[val], "full": source.session_date >= dates[0],
             "last_two_months": source.session_date >= pd.Timestamp("2026-06-22").date()}
    summary = {"frozen_parameters": frozen, "data": {"sessions": n, "development": dev, "validation": val-dev, "holdout": n-val}, "periods": {}}
    for name, mask in masks.items():
        frame = source.loc[mask]; base_bt, base_perf = run(frame, cfg, False); inv_bt, inv_perf = run(frame, cfg, True)
        btr, itr = base_bt["trades_df"], inv_bt["trades_df"]
        btr.to_csv(OUT / f"{name}_base_trades.csv", index=False); itr.to_csv(OUT / f"{name}_inverse_trades.csv", index=False)
        summary["periods"][name] = {"actual_start": str(frame.session_date.iloc[0]), "base": row(base_perf, btr), "inverse": row(inv_perf, itr), "event_pair_check": event_check(btr, itr)}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (OUT / "README.md").write_text("# Exact reverse entry research\n\nFrozen parameters are read from `research_output/base_strategy_summary.json`: dynamic beta 5d, window 60, z-entry 3.0, hook 0.15/5, exit 0, lockout 3.5. The original SignalGenerator creates source signals; only BUY_LONG and SELL_SHORT are swapped. Exits remain unchanged. Both runs use the same BacktestEngine next-open fills, integer shares, commissions, slippage and forced EOD. Every period includes base/inverse trade CSVs, event timestamp/duration equality checks, and gross-cost-net reconciliation. Last-two-months starts at 2026-06-22.\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

if __name__ == "__main__": main()
