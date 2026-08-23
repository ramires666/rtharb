"""CLI Entrypoint for RTH Stat-Arb Backtesting and 4-Scenario Analysis."""

import argparse
import sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from rtharb.config import AppConfig
from rtharb.data.loader import DataLoader
from rtharb.models.fair_value import FairValueModel
from rtharb.models.signals import SignalGenerator
from rtharb.backtest.engine import BacktestEngine
from rtharb.backtest.metrics import calculate_performance_metrics
from rtharb.analysis.matrix_comparator import MatrixComparator
from rtharb.analysis.optimizer import ParameterOptimizer
from rtharb.analysis.reporting import generate_summary_report

load_dotenv()

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="RTH Intraday Stat-Arb (NVDA vs QQQ) Research Platform")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--lead", type=str, default="QQQ", help="Leader symbol (e.g. QQQ)")
    parser.add_argument("--target", type=str, default="NVDA", help="Target symbol (e.g. NVDA)")
    parser.add_argument("--source", type=str, default="alpaca", choices=["alpaca", "yfinance"], help="Data source")
    parser.add_argument("--days", type=int, default=730, help="Lookback period in days (up to 730 for 2 years)")
    parser.add_argument("--start_date", type=str, default=None, help="Start date (e.g. 2025-01-01)")
    parser.add_argument("--end_date", type=str, default=None, help="End date (e.g. 2026-08-22)")
    parser.add_argument("--compare_matrix", action="store_true", help="Run 4-scenario comparative matrix (A/B/C/D)")
    parser.add_argument("--optimize", action="store_true", help="Run parameter grid search")
    parser.add_argument("--z_entry", type=float, default=None, help="Override Z_entry threshold")
    parser.add_argument("--reversal_delta", type=float, default=None, help="Override reversal confirmation delta")
    parser.add_argument("--z_max", type=float, default=None, help="Override Z_max threshold")
    parser.add_argument("--export_csv", type=str, default=None, help="Path to export trades CSV")

    args = parser.parse_args()

    # Load configuration
    cfg = AppConfig.load(args.config)
    if args.lead:
        cfg.strategy.ticker_lead = args.lead
    if args.target:
        cfg.strategy.ticker_target = args.target
    if args.source:
        cfg.strategy.data_source = args.source
    if args.z_entry is not None:
        cfg.strategy.z_entry = args.z_entry
    if args.reversal_delta is not None:
        cfg.strategy.reversal_delta = args.reversal_delta
    if args.z_max is not None:
        cfg.strategy.z_max_allowed = args.z_max

    print(f"\n{'='*70}")
    print(f"🎯 RTH Stat-Arb Backtester: {cfg.strategy.ticker_target} vs {cfg.strategy.ticker_lead}")
    print(f"Data Source: {cfg.strategy.data_source} | Period: {args.start_date or f'{args.days} days'} to {args.end_date or 'latest'}")
    print(f"{'='*70}\n")

    # 1. Load Data
    print("⏳ Loading and synchronizing 1-minute market data...")
    loader = DataLoader(
        cache_dir=cfg.cache_dir,
        source=cfg.strategy.data_source,
        data_feed=cfg.strategy.data_feed,
    )
    try:
        df_lead, df_target = loader.get_synchronized_pair(
            ticker_lead=cfg.strategy.ticker_lead,
            ticker_target=cfg.strategy.ticker_target,
            start_date=args.start_date,
            end_date=args.end_date,
            days_back=args.days,
            session_start=cfg.strategy.session_start,
            session_end=cfg.strategy.session_end,
            source=cfg.strategy.data_source
        )
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)

    print(f"✅ Synchronized {len(df_target):,} bars across {df_target['session_date'].nunique()} trading days.")
    print(f"   Date range: {df_target.index[0].date()} to {df_target.index[-1].date()}\n")

    # 2. Compute Fair Value & Metrics
    print("⏳ Computing Fair Value, Beta, Intraday Spreads & Z-Scores...")
    fv_model = FairValueModel(
        beta_mode=cfg.strategy.beta_mode,
        beta_rolling_days=cfg.strategy.beta_rolling_days,
        rolling_window_w=cfg.strategy.rolling_window_w,
        min_session_warmup_bars=cfg.strategy.min_session_warmup_bars,
        min_sigma_history_days=cfg.strategy.min_sigma_history_days,
    )
    df_metrics = fv_model.compute_intraday_metrics(df_lead, df_target)

    # 3. Mode Selection
    if args.compare_matrix:
        print("\n🔬 RUNNING 4-SCENARIO MATRIX COMPARISON (A/B/C/D):\n")
        matrix_comp = MatrixComparator(cfg)
        matrix_res = matrix_comp.run_all_scenarios(df_metrics)
        print(matrix_res["comparison_df"].to_string(index=False))
        return

    if args.optimize:
        print("\n⚡ RUNNING PARAMETER GRID SEARCH OPTIMIZATION:\n")
        opt = ParameterOptimizer(cfg)
        opt_df = opt.grid_search(df_metrics)
        print(opt_df.head(15).to_string(index=False))
        return

    # Standard Single Backtest
    print(f"⏳ Running backtest with Z_entry={cfg.strategy.z_entry}, Reversal_δ={cfg.strategy.reversal_delta}, Z_max={cfg.strategy.z_max_allowed}...")
    sig_gen = SignalGenerator(
        z_entry=cfg.strategy.z_entry,
        reversal_type=cfg.strategy.reversal_type,
        reversal_delta=cfg.strategy.reversal_delta,
        reversal_timeout_bars=cfg.strategy.reversal_timeout_bars,
        enable_extreme_entry_lockout=cfg.strategy.enable_extreme_entry_lockout,
        enable_extreme_emergency_exit=cfg.strategy.enable_extreme_emergency_exit,
        z_max_allowed=cfg.strategy.z_max_allowed,
        lockout_mode=cfg.strategy.lockout_mode,
        z_exit=cfg.strategy.z_exit,
        forced_close_time=cfg.strategy.forced_close_time,
        min_session_warmup_bars=cfg.strategy.min_session_warmup_bars
    )
    df_signals = sig_gen.generate_signals(df_metrics)

    engine = BacktestEngine(
        initial_capital=cfg.backtest.initial_capital,
        position_size_usd=cfg.backtest.position_size_usd,
        commission_per_share=cfg.backtest.commission_per_share,
        slippage_pct=cfg.backtest.slippage_pct,
        allow_short=cfg.backtest.allow_short
    )
    bt_out = engine.run(df_signals, ticker_target=cfg.strategy.ticker_target)
    metrics = calculate_performance_metrics(bt_out["df_results"], bt_out["trades_df"], cfg.backtest.initial_capital)

    report = generate_summary_report(
        metrics=metrics,
        ticker_lead=cfg.strategy.ticker_lead,
        ticker_target=cfg.strategy.ticker_target,
        trades_df=bt_out["trades_df"]
    )
    print("\n" + report)

    if args.export_csv and not bt_out["trades_df"].empty:
        bt_out["trades_df"].to_csv(args.export_csv, index=False)
        print(f"📁 Trades exported to {args.export_csv}")


if __name__ == "__main__":
    main()
