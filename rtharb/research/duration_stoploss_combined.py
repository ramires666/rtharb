"""Combined winner-preserving time-stop and stop-loss research.

The classic QQQ -> NVDA signal tuple is frozen before this study.  Candidate
thresholds come only from profitable development trades.  Validation selects
one eligible pair and holdout is opened exactly once afterwards.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rtharb.backtest.engine import BacktestEngine
from rtharb.backtest.metrics import calculate_performance_metrics
from rtharb.config import AppConfig
from rtharb.data.loader import DataLoader
from rtharb.models.fair_value import FairValueModel
from rtharb.models.signals import SignalGenerator


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_output" / "duration_stoploss_combined"
Q = (0.95, 0.975, 0.99, 1.0)
INDEPENDENT_HOLD = 61
INDEPENDENT_STOP = 0.00721284703320633
TOP_DEV = 10
MIN_WINNER_SURVIVAL_PCT = 95.0
SPLITS = ("development", "validation", "holdout", "full")


def _default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_default) + "\n",
        encoding="utf-8",
    )


def frozen_frame() -> tuple[AppConfig, pd.DataFrame, pd.DataFrame, list[object], dict[str, pd.Series], dict[str, Any]]:
    cfg = AppConfig.load(str(ROOT / "configs" / "default_config.yaml"))
    source = json.loads((ROOT / "research_output" / "base_strategy_summary.json").read_text(encoding="utf-8"))
    frozen = source["selected_parameters"]
    cfg.strategy.beta_mode = frozen["beta_mode"]
    cfg.strategy.beta_rolling_days = int(frozen["beta_days"])
    cfg.strategy.rolling_window_w = int(frozen["window"])
    cfg.strategy.z_entry = float(frozen["z_entry"])
    cfg.strategy.reversal_delta = float(frozen["hook_delta"])
    cfg.strategy.reversal_timeout_bars = int(frozen["hook_timeout"])
    cfg.strategy.z_exit = float(frozen["exit_band"])
    cfg.strategy.z_max_allowed = float(frozen["z_lockout"])

    lead, target = DataLoader(cfg.cache_dir, "alpaca", "sip").get_synchronized_pair("QQQ", "NVDA")
    if len(lead) != 194_490 or len(target) != 194_490 or not lead.index.equals(target.index):
        raise AssertionError(f"Expected exact synchronized 194,490 bars, got {len(lead)}/{len(target)}")
    model = FairValueModel(
        cfg.strategy.beta_mode, cfg.strategy.beta_rolling_days, cfg.strategy.rolling_window_w,
        cfg.strategy.min_session_warmup_bars, cfg.strategy.min_sigma_history_days,
    )
    metrics = model.compute_intraday_metrics(lead, target)
    frame = SignalGenerator(
        z_entry=cfg.strategy.z_entry,
        reversal_delta=cfg.strategy.reversal_delta,
        reversal_timeout_bars=cfg.strategy.reversal_timeout_bars,
        enable_extreme_entry_lockout=True,
        enable_extreme_emergency_exit=False,
        z_max_allowed=cfg.strategy.z_max_allowed,
        lockout_mode="day_lockout",
        z_exit=cfg.strategy.z_exit,
        forced_close_time=cfg.strategy.forced_close_time,
        min_session_warmup_bars=cfg.strategy.min_session_warmup_bars,
    ).generate_signals(metrics)
    dates = sorted(frame.session_date.unique())
    if len(dates) != 501:
        raise AssertionError(f"Expected 501 sessions, got {len(dates)}")
    masks = {
        "development": frame.session_date < dates[250],
        "validation": (frame.session_date >= dates[250]) & (frame.session_date < dates[375]),
        "holdout": frame.session_date >= dates[375],
        "full": frame.session_date >= dates[0],
    }
    data = {
        "feed": "Alpaca SIP", "lead": "QQQ", "traded": "NVDA", "raw_bars": len(frame),
        "sessions": len(dates), "first_timestamp": frame.index[0].isoformat(),
        "last_timestamp": frame.index[-1].isoformat(), "no_fill_resample_or_interpolation": True,
        "splits": {name: {"sessions": int(pd.Series(frame.loc[mask, "session_date"]).nunique()),
                           "bars": int(mask.sum())} for name, mask in masks.items()},
    }
    return cfg, metrics, frame, dates, masks, {"parameters": frozen, "data": data}


def run_case(frame: pd.DataFrame, cfg: AppConfig, hold: int | None = None,
             stop: float | None = None) -> dict[str, Any]:
    engine = BacktestEngine(
        initial_capital=cfg.backtest.initial_capital,
        position_size_usd=cfg.backtest.position_size_usd,
        commission_per_share=cfg.backtest.commission_per_share,
        slippage_pct=cfg.backtest.slippage_pct,
        allow_short=cfg.backtest.allow_short,
        max_holding_bars=hold,
        stop_loss_pct=stop,
    )
    bt = engine.run(frame, cfg.strategy.ticker_target)
    perf = calculate_performance_metrics(bt["df_results"], bt["trades_df"], cfg.backtest.initial_capital)
    return {"bt": bt, "metrics": summarize(bt, perf, cfg.backtest.initial_capital)}


def summarize(bt: dict[str, Any], perf: Any, capital: float) -> dict[str, Any]:
    trades = bt["trades_df"]
    gross = float(trades.gross_pnl.sum()) if not trades.empty else 0.0
    commissions = float(trades.commission.sum()) if not trades.empty else 0.0
    slippage = float(trades.slippage.sum()) if not trades.empty else 0.0
    equity = bt["df_results"].portfolio_equity.to_numpy(float)
    peak = np.maximum.accumulate(np.r_[capital, equity])[1:]
    dd = peak - equity
    return {
        "sessions": int(bt["df_results"].session_date.nunique()), "raw_bars": len(bt["df_results"]),
        "trades": len(trades), "gross_pnl": gross, "commissions": commissions,
        "slippage": slippage, "costs": commissions + slippage, "net_pnl": float(perf.total_pnl),
        "net_return_pct": float(perf.total_return_pct), "net_sharpe": float(perf.sharpe_ratio),
        "net_sortino": float(perf.sortino_ratio), "max_drawdown_usd_mtm": float(dd.max()),
        "max_drawdown_pct_mtm": float(np.max(np.divide(dd, peak, out=np.zeros_like(dd), where=peak != 0)) * 100.0),
        "win_rate_pct": float(perf.win_rate_pct), "profit_factor": float(perf.profit_factor),
        "avg_net_trade": float(perf.avg_trade_pnl), "avg_duration_bars": float(perf.avg_duration_mins),
        "final_equity": float(perf.final_equity),
        "exit_reasons": {str(k): int(v) for k, v in perf.exit_reasons_breakdown.items()},
    }


def add_mae(trades: pd.DataFrame, metrics: pd.DataFrame, slip: float) -> pd.DataFrame:
    result = trades.copy(); maes: list[float] = []
    for row in result.itertuples(index=False):
        direction = int(row.direction)
        entry_ref = float(row.entry_price) / (1.0 + slip if direction == 1 else 1.0 - slip)
        start = int(metrics.index.searchsorted(pd.Timestamp(row.entry_time), side="left"))
        end = int(metrics.index.searchsorted(pd.Timestamp(row.exit_time), side="left"))
        path = metrics.iloc[start:end]
        if path.empty:
            mae = 0.0
        elif direction == 1:
            mae = max(0.0, (entry_ref - float(path.target_low.min())) / entry_ref)
        else:
            mae = max(0.0, (float(path.target_high.max()) - entry_ref) / entry_ref)
        maes.append(mae)
    result["mae_pct"] = maes
    result["mae_usd"] = result.mae_pct * result.entry_price
    return result


def survival(base_winners: pd.DataFrame, overlay: pd.DataFrame) -> dict[str, Any]:
    by_entry = {pd.Timestamp(row.entry_time): row for row in overlay.itertuples(index=False)}
    matched = profitable = not_early = direction_match = 0
    for row in base_winners.itertuples(index=False):
        candidate = by_entry.get(pd.Timestamp(row.entry_time))
        if candidate is None:
            continue
        matched += 1
        direction_match += int(int(candidate.direction) == int(row.direction))
        profitable += int(float(candidate.net_pnl) > 0.0)
        not_early += int(pd.Timestamp(candidate.exit_time) >= pd.Timestamp(row.exit_time))
    total = len(base_winners)
    return {
        "baseline_development_winners": total, "matched_entry_events": matched,
        "matched_entry_pct": 100.0 * matched / total if total else 0.0,
        "direction_match_pct": 100.0 * direction_match / total if total else 0.0,
        "still_net_profitable_count": profitable,
        "still_net_profitable_pct": 100.0 * profitable / total if total else 0.0,
        "not_prematurely_closed_pct": 100.0 * not_early / total if total else 0.0,
    }


def quantile_axes(winners: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    raw_duration = {str(q): int(winners.duration_bars.quantile(q, interpolation="higher")) for q in Q}
    raw_stop = {str(q): float(winners.mae_pct.quantile(q, interpolation="higher")) for q in Q}

    def merge(items: list[tuple[float, str]], integer: bool) -> list[dict[str, Any]]:
        grouped: dict[float, list[str]] = {}
        for value, source in items:
            key = float(int(value)) if integer else float(value)
            grouped.setdefault(key, []).append(source)
        return [{"value": int(key) if integer else key, "sources": sources}
                for key, sources in sorted(grouped.items())]

    duration = merge([(v, f"winner_q{q}") for q, v in raw_duration.items()] +
                     [(INDEPENDENT_HOLD, "independently_fitted_95pct")], True)
    stops = merge([(v, f"winner_q{q}") for q, v in raw_stop.items()] +
                  [(INDEPENDENT_STOP, "independently_fitted_95pct")], False)
    return duration, stops, {
        "winner_duration_quantiles_bars": raw_duration,
        "winner_mae_pct_quantiles": raw_stop,
        "independently_fitted": {"max_holding_bars": INDEPENDENT_HOLD, "stop_loss_pct": INDEPENDENT_STOP},
    }


def export_case(prefix: str, case: dict[str, Any], cfg: AppConfig,
                hold: int, stop: float) -> dict[str, Any]:
    bt = case["bt"]; trades = bt["trades_df"].copy()
    if not trades.empty:
        trades.insert(trades.columns.get_loc("entry_price"), "entry_reference_price",
                      np.where(trades.direction == 1, trades.entry_price / (1.0 + cfg.backtest.slippage_pct),
                               trades.entry_price / (1.0 - cfg.backtest.slippage_pct)))
        trades.insert(trades.columns.get_loc("exit_price"), "exit_reference_price",
                      np.where(trades.direction == 1, trades.exit_price / (1.0 - cfg.backtest.slippage_pct),
                               trades.exit_price / (1.0 + cfg.backtest.slippage_pct)))
    trades["max_holding_bars"] = hold; trades["stop_loss_pct"] = stop
    trades.to_csv(OUT / f"{prefix}_trades.csv", index=False, float_format="%.12f")
    equity = bt["df_results"].portfolio_equity.to_numpy(float)
    peak = np.maximum.accumulate(np.r_[cfg.backtest.initial_capital, equity])[1:]
    eq = pd.DataFrame({
        "timestamp": bt["df_results"].index, "equity": equity, "running_peak": peak,
        "drawdown_usd": peak - equity,
        "drawdown_pct": np.divide(peak - equity, peak, out=np.zeros_like(peak), where=peak != 0) * 100.0,
    })
    eq.to_csv(OUT / f"{prefix}_equity.csv", index=False, float_format="%.12f")
    return case["metrics"]


def finalize_existing_outputs() -> dict[str, Any]:
    """Finish deterministic audits after all heavy replay artifacts exist."""
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    raw_results = summary["raw_q95_q95_diagnostic"]["results"]
    selected_results = summary["selected_results"]
    data = summary["data"]; selection = summary["selection"]
    grid = pd.read_csv(OUT / "development_grid.csv")
    selected_row = grid[(grid.max_holding_bars == selection["selected_max_holding_bars"]) &
                        np.isclose(grid.stop_loss_pct, selection["selected_stop_loss_pct"], rtol=0, atol=1e-11)].iloc[0]
    checks: dict[str, bool] = {
        "exact_raw_bars": data["raw_bars"] == 194_490,
        "exact_sessions": data["sessions"] == 501,
        "development_validation_holdout_sessions": [data["splits"][x]["sessions"] for x in SPLITS[:3]] == [250, 125, 126],
        "development_winners_frozen": summary["candidate_fit"]["development_base_winners"] == 132,
        "raw_q95_diagnostic_published": (OUT / "raw_q95_q95_summary.json").is_file(),
        "selected_is_eligible": bool(selected_row.eligible),
        "selected_winner_entry_match_100pct": float(selected_row.matched_entry_pct) == 100.0,
        "selected_winner_survival_at_least_95pct": float(selected_row.still_net_profitable_pct) >= 95.0,
        "holdout_opened_after_selection": bool(selection["holdout_opened_once_after_selection"]),
        "selected_split_trade_additivity": sum(selected_results[x]["trades"] for x in SPLITS[:3]) == selected_results["full"]["trades"],
        "selected_split_net_additivity": abs(sum(selected_results[x]["net_pnl"] for x in SPLITS[:3]) - selected_results["full"]["net_pnl"]) <= 1e-8,
        "selected_gross_cost_net": abs(selected_results["full"]["gross_pnl"] - selected_results["full"]["costs"] - selected_results["full"]["net_pnl"]) <= 1e-8,
        "selected_final_equity": abs(selected_results["full"]["final_equity"] - (100_000.0 + selected_results["full"]["net_pnl"])) <= 1e-8,
        "all_exact_trade_equity_files": all((OUT / f"{variant}_{split}_{kind}.csv").is_file()
                                              for variant in ("raw_q95_q95", "selected")
                                              for split in SPLITS for kind in ("trades", "equity")),
    }
    for variant in ("raw_q95_q95", "selected"):
        results = raw_results if variant == "raw_q95_q95" else selected_results
        for split in SPLITS:
            trades_csv = pd.read_csv(OUT / f"{variant}_{split}_trades.csv")
            equity_csv = pd.read_csv(OUT / f"{variant}_{split}_equity.csv")
            checks[f"{variant}_{split}_trade_rows"] = len(trades_csv) == results[split]["trades"]
            checks[f"{variant}_{split}_equity_rows"] = len(equity_csv) == results[split]["raw_bars"]
            checks[f"{variant}_{split}_no_overnight"] = bool(
                trades_csv.empty or (pd.to_datetime(trades_csv.entry_time, utc=True).dt.date ==
                                     pd.to_datetime(trades_csv.exit_time, utc=True).dt.date).all()
            )
    audit = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
             "selection_inputs": ["development", "validation"], "holdout_used_in_selection": False}
    write_json(OUT / "audit.json", audit)
    if audit["status"] != "PASS": raise AssertionError(audit)
    manifest = {
        "schema_version": 1, "status": "COMPLETE", "module": "rtharb.research.duration_stoploss_combined",
        "outputs": {"summary": "summary.json", "audit": "audit.json", "development_grid": "development_grid.csv",
                    "validation_finalists": "validation_finalists.csv", "raw_q95_diagnostic": "raw_q95_q95_summary.json"},
        "variants": ["raw_q95_q95", "selected"], "splits": list(SPLITS),
        "audit": {"status": "PASS", "file": "audit.json"},
    }
    write_json(OUT / "manifest.json", manifest)
    return {"audit": audit, "manifest": manifest}


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    if "--audit-only" in sys.argv[1:]:
        finalized = finalize_existing_outputs()
        print(json.dumps({"status": finalized["manifest"]["status"], "audit": finalized["audit"]["status"]},
                         ensure_ascii=False, indent=2), flush=True)
        return
    cfg, metrics, frame, dates, masks, provenance = frozen_frame()
    print(f"loaded exact classic frame: {len(frame):,} bars / {len(dates)} sessions", flush=True)

    # Holdout is deliberately not replayed until the combined pair is frozen.
    base_cases = {"development": run_case(frame.loc[masks["development"]], cfg)}
    base_dev = add_mae(base_cases["development"]["bt"]["trades_df"], metrics, cfg.backtest.slippage_pct)
    winners = base_dev[base_dev.net_pnl > 0.0].copy()
    if len(winners) != 132:
        raise AssertionError(f"Frozen development winners changed: expected 132, got {len(winners)}")
    duration_axis, stop_axis, fitted = quantile_axes(winners)
    print(f"candidate axes: {len(duration_axis)} durations x {len(stop_axis)} stops", flush=True)

    rows: list[dict[str, Any]] = []
    for duration in duration_axis:
        for stop in stop_axis:
            hold, stop_pct = int(duration["value"]), float(stop["value"])
            case = run_case(frame.loc[masks["development"]], cfg, hold, stop_pct)
            surv = survival(winners, case["bt"]["trades_df"])
            eligible = (surv["matched_entry_pct"] == 100.0 and surv["direction_match_pct"] == 100.0 and
                        surv["still_net_profitable_pct"] >= MIN_WINNER_SURVIVAL_PCT)
            rows.append({
                "max_holding_bars": hold, "stop_loss_pct": stop_pct,
                "duration_sources": "+".join(duration["sources"]), "stop_sources": "+".join(stop["sources"]),
                "eligible": eligible, **surv,
                **{f"development_{key}": value for key, value in case["metrics"].items()},
            })
    grid = pd.DataFrame(rows).sort_values(
        ["eligible", "development_net_sharpe", "development_net_pnl", "max_holding_bars", "stop_loss_pct"],
        ascending=[False, False, False, True, True], kind="mergesort",
    ).reset_index(drop=True)
    grid.to_csv(OUT / "development_grid.csv", index=False, float_format="%.12f")
    eligible = grid[grid.eligible].head(TOP_DEV)
    if eligible.empty:
        raise RuntimeError("No combined candidate preserves at least 95% of development winners")

    finalists: list[dict[str, Any]] = []
    for row in eligible.itertuples(index=False):
        case = run_case(frame.loc[masks["validation"]], cfg, int(row.max_holding_bars), float(row.stop_loss_pct))
        finalists.append({
            "max_holding_bars": int(row.max_holding_bars), "stop_loss_pct": float(row.stop_loss_pct),
            "duration_sources": row.duration_sources, "stop_sources": row.stop_sources,
            "development_net_sharpe": float(row.development_net_sharpe),
            "development_net_pnl": float(row.development_net_pnl),
            "development_still_net_profitable_pct": float(row.still_net_profitable_pct),
            "validation_net_sharpe": case["metrics"]["net_sharpe"],
            "validation_net_pnl": case["metrics"]["net_pnl"],
            "validation_trades": case["metrics"]["trades"],
        })
    final = pd.DataFrame(finalists)
    final["robust_score"] = np.minimum(final.development_net_sharpe, final.validation_net_sharpe)
    final = final.sort_values(
        ["robust_score", "validation_net_pnl", "development_net_sharpe", "max_holding_bars", "stop_loss_pct"],
        ascending=[False, False, False, True, True], kind="mergesort",
    ).reset_index(drop=True)
    final.to_csv(OUT / "validation_finalists.csv", index=False, float_format="%.12f")
    selected = final.iloc[0]
    selected_hold, selected_stop = int(selected.max_holding_bars), float(selected.stop_loss_pct)

    # Selection is complete.  From this point onward holdout may be opened once
    # for each frozen diagnostic/selected definition, never fed back upstream.
    for name in ("validation", "holdout", "full"):
        base_cases[name] = run_case(frame.loc[masks[name]], cfg)

    raw_hold = int(fitted["winner_duration_quantiles_bars"]["0.95"])
    raw_stop = float(fitted["winner_mae_pct_quantiles"]["0.95"])
    raw_results: dict[str, Any] = {}; selected_results: dict[str, Any] = {}
    holdout_opened = False
    for name in SPLITS:
        if name == "holdout": holdout_opened = True
        raw_case = run_case(frame.loc[masks[name]], cfg, raw_hold, raw_stop)
        selected_case = run_case(frame.loc[masks[name]], cfg, selected_hold, selected_stop)
        raw_results[name] = export_case(f"raw_q95_q95_{name}", raw_case, cfg, raw_hold, raw_stop)
        selected_results[name] = export_case(f"selected_{name}", selected_case, cfg, selected_hold, selected_stop)

    raw_dev = run_case(frame.loc[masks["development"]], cfg, raw_hold, raw_stop)
    raw_survival = survival(winners, raw_dev["bt"]["trades_df"])
    raw_summary = {
        "variant": "raw_q95_q95_diagnostic", "max_holding_bars": raw_hold,
        "stop_loss_pct": raw_stop, "development_winner_survival": raw_survival,
        "eligible": bool(raw_survival["matched_entry_pct"] == 100.0 and
                         raw_survival["still_net_profitable_pct"] >= MIN_WINNER_SURVIVAL_PCT),
        "results": raw_results,
        "selection_role": "diagnostic only; never used as holdout information",
    }
    write_json(OUT / "raw_q95_q95_summary.json", raw_summary)

    selected_survival_row = grid[(grid.max_holding_bars == selected_hold) &
                                 np.isclose(grid.stop_loss_pct, selected_stop, rtol=0, atol=1e-15)].iloc[0]
    only_noop = (len(grid[grid.eligible]) == 1 and
                 selected_hold == max(int(x["value"]) for x in duration_axis) and
                 math.isclose(selected_stop, max(float(x["value"]) for x in stop_axis), rel_tol=0, abs_tol=1e-15))
    summary = {
        "schema_version": 1, "study": "combined q95 time-stop + stop-loss on frozen classic QQQ to NVDA",
        "frozen_parameters": provenance["parameters"], "data": provenance["data"],
        "candidate_fit": {**fitted, "method": "development net-profitable winners only; observed higher quantiles",
                          "development_base_winners": len(winners), "candidate_pairs": len(grid),
                          "eligibility": "100% exact entry matching and >=95% still net-profitable development winners"},
        "selection": {
            "selected_max_holding_bars": selected_hold, "selected_stop_loss_pct": selected_stop,
            "development_net_sharpe": float(selected.development_net_sharpe),
            "validation_net_sharpe": float(selected.validation_net_sharpe),
            "validation_net_pnl": float(selected.validation_net_pnl), "robust_score": float(selected.robust_score),
            "development_winner_survival_pct": float(selected_survival_row.still_net_profitable_pct),
            "eligible_candidate_count": int(grid.eligible.sum()), "top_development_sent_to_validation": len(final),
            "method": "top development daily net Sharpe/net PnL; maximize min(dev,val Sharpe), tie validation PnL",
            "holdout_opened_once_after_selection": holdout_opened,
            "only_eligible_candidate_is_noop_max_max": only_noop,
        },
        "raw_q95_q95_diagnostic": raw_summary,
        "base_results": {name: case["metrics"] for name, case in base_cases.items()},
        "selected_results": selected_results,
        "execution": {
            "signal": "frozen close-generated classic signal; fill next raw open", "traded": "NVDA", "lead": "QQQ",
            "starting_capital_usd": cfg.backtest.initial_capital, "position_notional_usd": cfg.backtest.position_size_usd,
            "commission_usd_per_share_per_side": cfg.backtest.commission_per_share,
            "slippage_fraction_per_execution": cfg.backtest.slippage_pct,
            "overlay_order": "stop-loss first, then time-stop", "stop_gap": "adverse raw open",
            "forced_eod": "unchanged frozen engine final raw RTH close fallback",
        },
        "holdout_warning": "Holdout is one untouched historical diagnostic and was not used for fitting or selection",
    }
    write_json(OUT / "selected_summary.json", summary)
    write_json(OUT / "summary.json", summary)

    checks: dict[str, bool] = {
        "exact_raw_bars": provenance["data"]["raw_bars"] == 194_490,
        "exact_sessions": provenance["data"]["sessions"] == 501,
        "development_validation_holdout_sessions": [provenance["data"]["splits"][x]["sessions"] for x in SPLITS[:3]] == [250, 125, 126],
        "development_winners_frozen": len(winners) == 132,
        "raw_q95_diagnostic_published": (OUT / "raw_q95_q95_summary.json").is_file(),
        "selected_is_eligible": bool(selected_survival_row.eligible),
        "selected_winner_entry_match_100pct": float(selected_survival_row.matched_entry_pct) == 100.0,
        "selected_winner_survival_at_least_95pct": float(selected_survival_row.still_net_profitable_pct) >= 95.0,
        "holdout_opened_after_selection": holdout_opened,
        "selected_split_trade_additivity": sum(selected_results[x]["trades"] for x in SPLITS[:3]) == selected_results["full"]["trades"],
        "selected_split_net_additivity": abs(sum(selected_results[x]["net_pnl"] for x in SPLITS[:3]) - selected_results["full"]["net_pnl"]) <= 1e-8,
        "selected_gross_cost_net": abs(selected_results["full"]["gross_pnl"] - selected_results["full"]["costs"] - selected_results["full"]["net_pnl"]) <= 1e-8,
        "selected_final_equity": abs(selected_results["full"]["final_equity"] - (cfg.backtest.initial_capital + selected_results["full"]["net_pnl"])) <= 1e-8,
        "all_exact_trade_equity_files": all((OUT / f"{variant}_{split}_{kind}.csv").is_file()
                                              for variant in ("raw_q95_q95", "selected")
                                              for split in SPLITS for kind in ("trades", "equity")),
    }
    for variant in ("raw_q95_q95", "selected"):
        results = raw_results if variant == "raw_q95_q95" else selected_results
        for split in SPLITS:
            trades_csv = pd.read_csv(OUT / f"{variant}_{split}_trades.csv")
            equity_csv = pd.read_csv(OUT / f"{variant}_{split}_equity.csv")
            checks[f"{variant}_{split}_trade_rows"] = len(trades_csv) == results[split]["trades"]
            checks[f"{variant}_{split}_equity_rows"] = len(equity_csv) == results[split]["raw_bars"]
            checks[f"{variant}_{split}_no_overnight"] = bool(
                trades_csv.empty or (pd.to_datetime(trades_csv.entry_time, utc=True).dt.date ==
                                     pd.to_datetime(trades_csv.exit_time, utc=True).dt.date).all()
            )
    audit = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
             "selection_inputs": ["development", "validation"], "holdout_used_in_selection": False}
    write_json(OUT / "audit.json", audit)
    if audit["status"] != "PASS": raise AssertionError(audit)
    manifest = {
        "schema_version": 1, "status": "COMPLETE", "module": "rtharb.research.duration_stoploss_combined",
        "outputs": {"summary": "summary.json", "audit": "audit.json", "development_grid": "development_grid.csv",
                    "validation_finalists": "validation_finalists.csv", "raw_q95_diagnostic": "raw_q95_q95_summary.json"},
        "variants": ["raw_q95_q95", "selected"], "splits": list(SPLITS),
        "audit": {"status": "PASS", "file": "audit.json"},
    }
    write_json(OUT / "manifest.json", manifest)
    print(json.dumps({"status": "COMPLETE", "selected": {"max_holding_bars": selected_hold,
          "stop_loss_pct": selected_stop}, "holdout": selected_results["holdout"]},
          ensure_ascii=False, indent=2, default=_default), flush=True)


if __name__ == "__main__":
    main()
