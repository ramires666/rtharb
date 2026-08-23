"""Publish fixed-stop VWAP-Z risk/reward variants for the interactive report.

This is deliberately a reporting slice of ``research_risk_reward`` rather
than a second simulator.  It reuses the frozen causal VWAP-Z entry cohort and
the audited raw-OHLC bracket evaluator, changing only the reward multiple.
The original convergence exit is excluded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from rtharb.config import AppConfig
from rtharb.data.loader import DataLoader
from rtharb.research.risk_reward import CAPITAL, COMMISSION, SIZE, SLIP, _raw_target, evaluate


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_output" / "risk_reward"
RATIOS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def _ratio_slug(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _with_equity_metrics(metrics: dict, trades: pd.DataFrame) -> dict:
    result = dict(metrics)
    if trades.empty:
        result.update(final_equity=CAPITAL, max_drawdown=0.0, max_drawdown_pct=0.0)
        return result
    equity = CAPITAL + trades["net_pnl"].astype(float).cumsum()
    curve = pd.concat([pd.Series([CAPITAL]), equity.reset_index(drop=True)], ignore_index=True)
    peaks = curve.cummax()
    drawdown = peaks - curve
    drawdown_pct = drawdown / peaks.replace(0.0, float("nan")) * 100.0
    result.update(
        final_equity=float(curve.iloc[-1]),
        max_drawdown=float(drawdown.max()),
        max_drawdown_pct=float(drawdown_pct.max()),
    )
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base_summary_path = OUT / "summary.json"
    cohort_path = OUT / "vwap_z_entry_cohort.csv"
    base_summary = json.loads(base_summary_path.read_text(encoding="utf-8"))
    vwap_meta = base_summary["vwap_z"]
    stop_pct = float(vwap_meta["selected"]["stop_pct"])
    if stop_pct <= 0:
        raise ValueError(f"Invalid fixed VWAP-Z stop: {stop_pct}")

    cohort = pd.read_csv(cohort_path)
    # The frozen CSV spans EDT and EST, so pandas needs an explicit common
    # timezone before conversion back to the raw archive's New York timezone.
    for column in ("entry_time", "exit_time"):
        cohort[column] = pd.to_datetime(cohort[column], format="mixed", utc=True).dt.tz_convert("America/New_York")
    expected_candidates = int(vwap_meta["reconciliation"]["candidate_entries"])
    if len(cohort) != expected_candidates:
        raise AssertionError(f"VWAP-Z cohort changed: {len(cohort)} != {expected_candidates}")

    cfg = AppConfig.load(str(ROOT / "configs" / "default_config.yaml"))
    lead, target = DataLoader(cfg.cache_dir, "alpaca", "sip").get_synchronized_pair("QQQ", "NVDA")
    raw = _raw_target(lead, target)
    split = base_summary["splits"]
    d1 = pd.Timestamp(split["development_end"]).date()
    d2 = pd.Timestamp(split["validation_end"]).date()
    entry_dates = cohort["entry_time"].dt.date
    cohorts = {
        "development": cohort[entry_dates < d1],
        "validation": cohort[(entry_dates >= d1) & (entry_dates < d2)],
        "holdout": cohort[entry_dates >= d2],
        "full": cohort,
    }
    expected_period_counts = vwap_meta["reconciliation"]["period_candidate_counts"]
    actual_period_counts = {name: len(rows) for name, rows in cohorts.items()}
    for period in ("development", "validation", "holdout", "full"):
        if actual_period_counts[period] != int(expected_period_counts[period]):
            raise AssertionError(
                f"VWAP-Z split boundary mismatch for {period}: "
                f"{actual_period_counts[period]} != {expected_period_counts[period]}"
            )

    variants: dict[str, dict] = {}
    for rr in RATIOS:
        period_results: dict[str, dict] = {}
        period_trades: dict[str, pd.DataFrame] = {}
        for period, period_cohort in cohorts.items():
            metrics, trades = evaluate(period_cohort, raw, stop_pct, rr)
            metrics = _with_equity_metrics(metrics, trades)
            if metrics["trades"] + metrics["skipped_overlaps"] != len(period_cohort):
                raise AssertionError(f"Unreconciled cohort for RR {rr:g}, {period}")
            if abs(float(trades["net_pnl"].sum()) - float(metrics["net_pnl"])) > 1e-9:
                raise AssertionError(f"Net P&L mismatch for RR {rr:g}, {period}")
            if abs(float(trades["costs"].sum()) - float(metrics["costs"])) > 1e-9:
                raise AssertionError(f"Cost mismatch for RR {rr:g}, {period}")
            period_results[period] = metrics
            period_trades[period] = trades

        full_name = f"vwap_z_stop_{stop_pct * 100:g}pct_rr_{_ratio_slug(rr)}_full_trades.csv"
        full_path = OUT / full_name
        period_trades["full"].to_csv(full_path, index=False)

        split_rows = sum(len(period_trades[p]) for p in ("development", "validation", "holdout"))
        full_rows = len(period_trades["full"])
        split_net = sum(float(period_results[p]["net_pnl"]) for p in ("development", "validation", "holdout"))
        full_net = float(period_results["full"]["net_pnl"])
        if split_rows != full_rows or abs(split_net - full_net) > 1e-8:
            raise AssertionError(f"Split/full reconciliation failed for RR {rr:g}")

        variants[f"rr_{_ratio_slug(rr)}"] = {
            "rr": rr,
            "full_trades_csv": full_name,
            "results": period_results,
            "reconciliation": {
                "candidate_entries": expected_candidates,
                "full_trades": full_rows,
                "full_skipped_overlaps": int(period_results["full"]["skipped_overlaps"]),
                "candidate_entries_accounted": full_rows + int(period_results["full"]["skipped_overlaps"]) == expected_candidates,
                "split_trade_rows_equal_full": split_rows == full_rows,
                "split_net_pnl_equal_full": abs(split_net - full_net) <= 1e-8,
                "csv_net_pnl_equal_metrics": abs(float(pd.read_csv(full_path)["net_pnl"].sum()) - full_net) <= 1e-8,
            },
        }

    payload = {
        "study": "VWAP-Z entries with independent stop/take-profit exits",
        "entry_source": "frozen research_output/risk_reward/vwap_z_entry_cohort.csv",
        "entry_cohort_sha256": hashlib.sha256(cohort_path.read_bytes()).hexdigest(),
        "vwap_z_frozen_parameters": vwap_meta["frozen_parameters"],
        "fixed_stop_pct": stop_pct,
        "rr_ratios": list(RATIOS),
        "execution": {
            "entry": "existing frozen next-open entry",
            "exit": "raw NVDA 1m OHLC stop/target, stop-first on same bar, adverse open for gap-through stop, otherwise forced RTH EOD close",
            "convergence_exit": False,
            "notional_usd": SIZE,
            "commission_usd_per_share_per_side": COMMISSION,
            "slippage_fraction_per_execution": SLIP,
            "starting_capital_usd": CAPITAL,
        },
        "splits": split,
        "variants": variants,
    }
    output_path = OUT / "vwap_rr_ratios_summary.json"
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
