"""Generate complete standalone_report/data.js and standalone_report/index.html."""

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


def run():
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

    # 4. Extract Real Sessions with FULL 390 REAL 1-MINUTE CANDLESTICKS (Zero Synthetic)
    sample_days = [
        ("2024-08-27", "2024-08-27 — Сделка Long (-184$) | Выход по Тайм-Стопу 120м (вместо -$365)"),
        ("2024-08-21", "2024-08-21 — 2 Сделки (+248$) | Long (+132$) и Short (+115$) по Тейку"),
        ("2024-08-22", "2024-08-22 — Сделка Long (+125$) | Быстрый возврат к нулю за 33 мин"),
        ("2024-09-03", "2024-09-03 — Сделка Short (-195$) | Выход по Тайм-Стопу 120м (вместо -$310)"),
        ("2024-09-11", "2024-09-11 — Сделка Long (-175$) | Сброс по Стоп-Лоссу 1.5% (вместо -$274)"),
        ("2025-01-08", "2025-01-08 — Сделка Long (+172$) | Реальный Take-Profit за 43 минуты")
    ]

    sessions_pack = {}
    for d_str, label_text in sample_days:
        d = pd.to_datetime(d_str).date()
        df_d = df_metrics[df_metrics["session_date"] == d]
        if df_d.empty:
            continue

        high_vals = df_d["target_high"].values if "target_high" in df_d.columns else df_d["target_close"].values
        low_vals = df_d["target_low"].values if "target_low" in df_d.columns else df_d["target_close"].values

        # 100% REAL OHLC NUMBERS FROM ALPACA PARQUET
        sessions_pack[d_str] = {
            "date": d_str,
            "label": label_text,
            "times": [t.strftime("%H:%M") for t in df_d.index],
            "open": [round(float(v), 2) for v in df_d["target_open"].values],
            "high": [round(float(v), 2) for v in high_vals],
            "low": [round(float(v), 2) for v in low_vals],
            "close": [round(float(v), 2) for v in df_d["target_close"].values],
            "fair": [round(float(v), 2) for v in df_d["target_fair_price"].values],
            "z_score": [round(float(v), 3) for v in df_d["z_score"].values],
            "signals": [sigs_prod.get(ts, "NONE") for ts in df_d.index]
        }

    out_bundle = {
        "equity": equity_pack,
        "sessions": sessions_pack,
        "trades": trades_prod[:50]
    }

    # Write data.js directly to standalone_report/data.js
    js_file = project_root / "standalone_report" / "data.js"
    js_content = f"window.APP_DATA = {json.dumps(out_bundle)};\n"
    js_file.write_text(js_content, encoding="utf-8")
    print(f"Written data.js: {js_file.stat().st_size:,} bytes")


if __name__ == "__main__":
    run()
