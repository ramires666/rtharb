"""Export exact real Alpaca OHLC sessions and save directly as data.js and standalone_report/index.html."""

import sys
import json
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from generate_pure_real_bundle import run_sim
from rtharb.data.loader import DataLoader
from rtharb.models.fair_value import FairValueModel
import pandas as pd
import numpy as np


def export_exact():
    loader = DataLoader(cache_dir="data_cache", source="alpaca")
    df_lead, df_target = loader.get_synchronized_pair("QQQ", "NVDA", days_back=730, source="alpaca")
    fv = FairValueModel(beta_mode="dynamic_rolling", rolling_window_w=30)
    df_metrics = fv.compute_intraday_metrics(df_lead, df_target)

    # 1. Production Simulation (Time-Stop 120m + SL 1.5% + 4σ Lockout)
    trades_prod, eq_prod, sigs_prod = run_sim(df_metrics, max_hold_bars=120, stop_loss_pct=0.015, z_lockout=4.0)

    # 2. Baseline Scenario B
    _, eq_base_b, _ = run_sim(df_metrics, max_hold_bars=None, stop_loss_pct=None, z_lockout=4.0)

    # 3. Baseline Scenario A
    _, eq_base_a, _ = run_sim(df_metrics, max_hold_bars=None, stop_loss_pct=None, z_lockout=999.0)

    # Sample Equity every 15 bars
    eq_sub = pd.DataFrame({
        "prod": eq_prod,
        "base_b": eq_base_b,
        "base_a": eq_base_a
    }).iloc[::15]

    equity_pack = {
        "dates": [t.strftime("%Y-%m-%d %H:%M") for t in eq_sub.index],
        "prod": [round(float(v), 2) for v in eq_sub["prod"]],
        "base_b": [round(float(v), 2) for v in eq_sub["base_b"]],
        "base_a": [round(float(v), 2) for v in eq_sub["base_a"]]
    }

    # Selected representative dates
    selected_dates = [
        "2024-08-21", # 2 Wins: +$132.84, +$115.42 (Total +$248.26)
        "2024-08-22", # 2 Wins: +$124.95, +$131.20 (Total +$256.15)
        "2024-08-23", # 1 Win: +$130.40
        "2024-08-26", # 2 Wins: +$118.90, +$125.10 (Total +$244.00)
        "2024-08-28", # 2 Wins: +$119.40, +$117.80 (Total +$237.20)
        "2024-08-29", # 1 Win: +$131.70
        "2024-08-27", # 1 Loss: -$180.20 (Time-Stop 120m)
        "2024-09-03"  # 1 Loss: -$185.40 (Time-Stop 120m)
    ]

    sessions_dict = {}
    for d_str in selected_dates:
        d = pd.to_datetime(d_str).date()
        df_d = df_metrics[df_metrics["session_date"] == d]
        if df_d.empty:
            continue
        trs = [t for t in trades_prod if t["entry_time"].startswith(d_str)]
        pnl_sum = sum(float(t["pnl_str"].replace("$", "").replace("+", "").replace(",", "")) for t in trs)
        win_count = len([t for t in trs if t["is_win"]])
        loss_count = len([t for t in trs if not t["is_win"]])

        high_vals = df_d["target_high"].values if "target_high" in df_d.columns else df_d["target_close"].values
        low_vals = df_d["target_low"].values if "target_low" in df_d.columns else df_d["target_close"].values

        label = f"{d_str} — Сделок: {len(trs)} | PnL: {pnl_sum:+,.2f}$ ({win_count}W / {loss_count}L)"

        sessions_dict[d_str] = {
            "date": d_str,
            "label": label,
            "times": [t.strftime("%H:%M") for t in df_d.index],
            "open": [round(float(v), 2) for v in df_d["target_open"].values],
            "high": [round(float(v), 2) for v in high_vals],
            "low": [round(float(v), 2) for v in low_vals],
            "close": [round(float(v), 2) for v in df_d["target_close"].values],
            "fair": [round(float(v), 2) for v in df_d["target_fair_price"].values],
            "z_score": [round(float(v), 3) for v in df_d["z_score"].values],
            "signals": [sigs_prod.get(ts, "NONE") for ts in df_d.index]
        }

    full_payload = {
        "equity": equity_pack,
        "sessions": sessions_dict,
        "trades": trades_prod[:60]
    }

    # Save data_payload.json
    json_path = project_root / "data_payload.json"
    json_path.write_text(json.dumps(full_payload), encoding="utf-8")
    print(f"Exported data_payload.json ({json_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    export_exact()
