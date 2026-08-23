"""Staged research for absolute-dollar and 09:30-anchor entry filters."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from research_base_strategy import model_arrays, prepare_market, simulate, slice_arrays
from rtharb.backtest.engine import BacktestEngine
from rtharb.config import AppConfig
from rtharb.data.loader import DataLoader
from rtharb.models.fair_value import FairValueModel
from rtharb.models.signals import SignalGenerator


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "research_output" / "absolute_filters"


def candidate_params(base: dict, mode: str, threshold: float | None, anchor: bool) -> dict:
    return {
        **base,
        "entry_mode": mode,
        "abs_threshold_usd": threshold,
        "anchor_filter": anchor,
    }


def exact_case(metrics: pd.DataFrame, cfg: AppConfig, base: dict, params: dict):
    signals = SignalGenerator(
        z_entry=base["z_entry"], reversal_delta=base["hook_delta"],
        reversal_timeout_bars=base["hook_timeout"],
        enable_extreme_entry_lockout=True, enable_extreme_emergency_exit=False,
        z_max_allowed=base["z_lockout"], z_exit=base["exit_band"],
        forced_close_time="15:55", min_session_warmup_bars=15,
        entry_mode=params["entry_mode"],
        entry_abs_deviation_usd=params["abs_threshold_usd"],
        enable_open_anchor_filter=params["anchor_filter"],
    ).generate_signals(metrics)
    return BacktestEngine(
        cfg.backtest.initial_capital, cfg.backtest.position_size_usd,
        cfg.backtest.commission_per_share, cfg.backtest.slippage_pct, True,
    ).run(signals, "NVDA")


def slim(result: dict) -> dict:
    keep = [
        "entry_mode", "abs_threshold_usd", "anchor_filter", "trades", "net_pnl",
        "gross_pnl", "costs", "net_return_pct", "net_sharpe", "net_sortino",
        "max_drawdown_pct", "win_rate_pct", "profit_factor", "avg_net_trade",
        "median_net_trade", "avg_duration", "long_net_pnl", "short_net_pnl",
    ]
    return {k: result[k] for k in keep}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = AppConfig.load(str(ROOT / "configs" / "default_config.yaml"))
    base_selected = json.loads(
        (ROOT / "research_output" / "base_strategy_summary.json").read_text(encoding="utf-8")
    )["selected_parameters"]
    base = {
        "beta_mode": base_selected["beta_mode"], "beta_days": base_selected["beta_days"],
        "window": base_selected["window"], "z_entry": base_selected["z_entry"],
        "hook_delta": base_selected["hook_delta"], "hook_timeout": base_selected["hook_timeout"],
        "exit_band": base_selected["exit_band"], "z_lockout": base_selected["z_lockout"],
        "direction": base_selected["direction"],
    }
    lead, target = DataLoader(cfg.cache_dir, "alpaca", "sip").get_synchronized_pair("QQQ", "NVDA")
    market = prepare_market(lead, target)
    arrays = model_arrays(market, base["beta_mode"], base["beta_days"], base["window"])
    sessions = len(arrays["unique_days"])
    dev_end, val_end = sessions // 2, int(sessions * 0.75)
    parts = {
        "development": slice_arrays(arrays, 0, dev_end),
        "validation": slice_arrays(arrays, dev_end, val_end),
        "holdout": slice_arrays(arrays, val_end, sessions),
    }

    # Covers common bars through the extreme tail of the full-sample USD
    # dislocation distribution (about $4.56 at q95 and $10.02 at q99.9).
    thresholds = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00, 6.00, 7.50, 10.00]
    candidates = [candidate_params(base, "z_only", None, anchor) for anchor in (False, True)]
    candidates += [
        candidate_params(base, mode, threshold, anchor)
        for mode in ("abs_only", "z_or_abs")
        for threshold in thresholds
        for anchor in (False, True)
    ]

    dev_rows = [slim(simulate(parts["development"], p)) for p in candidates]
    dev = pd.DataFrame(dev_rows)
    dev["eligible"] = dev["trades"] >= 40
    dev = dev.sort_values(
        ["eligible", "net_sharpe", "profit_factor", "net_pnl"], ascending=False
    )
    dev.to_csv(OUT / "candidate_results_development.csv", index=False)

    shortlist_params = []
    for row in dev.head(10).to_dict("records"):
        shortlist_params.append(candidate_params(
            base, row["entry_mode"],
            None if pd.isna(row["abs_threshold_usd"]) else float(row["abs_threshold_usd"]),
            bool(row["anchor_filter"]),
        ))
    val_rows = [slim(simulate(parts["validation"], p)) for p in shortlist_params]
    validation = pd.DataFrame(val_rows)
    validation["eligible"] = validation["trades"] >= 20
    validation = validation.sort_values(
        ["eligible", "net_sharpe", "profit_factor", "net_pnl"], ascending=False
    )
    validation.to_csv(OUT / "shortlist_results_validation.csv", index=False)

    winner = validation.iloc[0]
    chosen = {
        "entry_mode": str(winner["entry_mode"]),
        "abs_threshold_usd": None if pd.isna(winner["abs_threshold_usd"]) else float(winner["abs_threshold_usd"]),
        "anchor_filter": bool(winner["anchor_filter"]),
    }
    chosen_params = candidate_params(
        base, chosen["entry_mode"], chosen["abs_threshold_usd"], chosen["anchor_filter"]
    )
    holdout_result = simulate(parts["holdout"], chosen_params, collect=True)
    full_result = simulate(arrays, chosen_params, collect=True)
    pd.DataFrame([slim(holdout_result)]).to_csv(OUT / "chosen_results_holdout.csv", index=False)
    holdout_result["trades_df"].to_csv(OUT / "chosen_trades_holdout.csv", index=False)
    full_result["trades_df"].to_csv(OUT / "chosen_trades_full.csv", index=False)

    baseline = {}
    baseline_params = candidate_params(base, "z_only", None, False)
    for period, frame in {**parts, "full_descriptive": arrays}.items():
        baseline[period] = slim(simulate(frame, baseline_params))

    # One exact production-engine reconciliation on the frozen holdout choice.
    metrics = FairValueModel(base["beta_mode"], base["beta_days"], base["window"], 15).compute_intraday_metrics(lead, target)
    holdout_first_day = arrays["unique_days"][val_end]
    holdout_metrics = metrics[metrics["session_date"] >= holdout_first_day]
    exact = exact_case(holdout_metrics, cfg, base, chosen)
    exact_trades = exact["trades_df"]
    exact_net = float(exact_trades["net_pnl"].sum()) if not exact_trades.empty else 0.0
    reconciliation = {
        "fast_trades": int(holdout_result["trades"]), "exact_trades": len(exact_trades),
        "fast_net_pnl": float(holdout_result["net_pnl"]), "exact_net_pnl": exact_net,
        "net_pnl_error": float(holdout_result["net_pnl"] - exact_net),
    }
    if reconciliation["fast_trades"] != reconciliation["exact_trades"] or abs(reconciliation["net_pnl_error"]) > 0.01:
        raise AssertionError(f"Fast/exact mismatch: {reconciliation}")

    summary = {
        "data": {
            "bars": len(arrays["z"]), "sessions": sessions,
            "validation_starts": str(arrays["unique_days"][dev_end]),
            "holdout_starts": str(holdout_first_day),
        },
        "base_parameters": base_selected,
        "thresholds_tested_usd": thresholds,
        "selection_rule": "top 10 eligible development by net Sharpe/PF/PnL; winner on validation by same ordering; holdout once",
        "chosen_after_development_validation": chosen,
        "baseline": baseline,
        "chosen_holdout": slim(holdout_result),
        "chosen_full_descriptive": slim(full_result),
        "exact_engine_reconciliation": reconciliation,
        "warning": "Holdout is a final diagnostic and was not used to retune the choice.",
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (OUT / "README.md").write_text(
        "# Absolute-entry filter research\n\nAbsolute deviation is `NVDA close - fair NVDA` in USD. "
        "The optional 09:30 anchor is the first one-minute close; LONG is allowed only below it and SHORT only above it. "
        "All signals execute on the next open with the production commission/slippage model.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
