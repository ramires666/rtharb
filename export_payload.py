import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from rtharb.data.loader import DataLoader
from rtharb.models.fair_value import FairValueModel
from rtharb.analysis.matrix_comparator import MatrixComparator
from rtharb.config import AppConfig


def run():
    loader = DataLoader(cache_dir="data_cache", source="alpaca")
    df_lead, df_target = loader.get_synchronized_pair("QQQ", "NVDA", days_back=730, source="alpaca")
    fv = FairValueModel(beta_mode="dynamic_rolling", rolling_window_w=30)
    df_metrics = fv.compute_intraday_metrics(df_lead, df_target)
    cfg = AppConfig()
    comparator = MatrixComparator(cfg)
    matrix_res = comparator.run_all_scenarios(df_metrics)
    rec_res = matrix_res["results"]["B: Entry Lockout Only (Recommended)"]
    trades_b = rec_res["trades_df"]
    df_b_signals = rec_res["df_results"]

    # 1. Real Daily Equity Curves for 4 scenarios
    eq_df = matrix_res["equity_curves"]
    # Sample every 15 mins (15 bars)
    eq_sampled = eq_df.iloc[::15].copy()
    if eq_df.index[-1] != eq_sampled.index[-1]:
        eq_sampled = pd.concat([eq_sampled, eq_df.iloc[[-1]]])

    eq_pack = {
        "dates": [t.strftime("%Y-%m-%d %H:%M") for t in eq_sampled.index],
        "eqB": [round(float(v), 2) for v in eq_sampled["B: Entry Lockout Only (Recommended)"].values],
        "eqA": [round(float(v), 2) for v in eq_sampled["A: Pure Reversion (No 4σ caps)"].values],
        "eqD": [round(float(v), 2) for v in eq_sampled["D: Conservative (Lockout + Exit)"].values],
        "eqC": [round(float(v), 2) for v in eq_sampled["C: Emergency Exit Only"].values]
    }

    # 2. Pick 8 diverse real trading sessions (including profitable mean reversion and losing breakout days)
    days_to_pack = ["2024-08-21", "2024-08-22", "2024-08-27", "2024-09-03", "2024-09-11", "2024-10-04", "2025-01-08", "2025-03-14"]
    daily_data = {}
    for d_str in days_to_pack:
        d = pd.to_datetime(d_str).date()
        df_d = df_b_signals[df_b_signals["session_date"] == d]
        if df_d.empty:
            continue
        high_vals = df_d["target_high"].values if "target_high" in df_d else df_d["target_close"].values
        low_vals = df_d["target_low"].values if "target_low" in df_d else df_d["target_close"].values
        daily_data[d_str] = {
            "times": [t.strftime("%H:%M") for t in df_d.index],
            "open": [round(float(v), 2) for v in df_d["target_open"].values],
            "high": [round(float(v), 2) for v in high_vals],
            "low": [round(float(v), 2) for v in low_vals],
            "close": [round(float(v), 2) for v in df_d["target_close"].values],
            "fair": [round(float(v), 2) for v in df_d["target_fair_price"].values],
            "z_score": [round(float(v), 3) for v in df_d["z_score"].values],
            "signals": [str(s) for s in df_d["signal"].values]
        }

    # 3. Real Trades (Sample 35 trades including wins, losses, EOD liquidation)
    trades_pack = []
    for _, tr in trades_b.head(40).iterrows():
        pnl = float(tr["net_pnl"])
        sign = "+" if pnl >= 0 else ""
        trades_pack.append({
            "id": int(tr["trade_id"]),
            "dir": "🟢 LONG" if tr["direction"] == 1 else "🔴 SHORT",
            "entry_t": tr["entry_time"].strftime("%Y-%m-%d %H:%M"),
            "entry_p": f"${tr['entry_price']:.2f}",
            "exit_t": tr["exit_time"].strftime("%Y-%m-%d %H:%M"),
            "exit_p": f"${tr['exit_price']:.2f}",
            "pnl": f"{sign}${pnl:,.2f}",
            "ret": f"{tr['return_pct']*100:+.2f}%",
            "dur": f"{int(tr['duration_bars'])} мин",
            "rsn": tr["exit_reason"],
            "z_in": f"{tr['entry_z_score']:.2f}",
            "z_out": f"{tr['exit_z_score']:.2f}"
        })

    payload = {
        "equity": eq_pack,
        "daily": daily_data,
        "trades": trades_pack,
        "days": [d for d in days_to_pack if d in daily_data]
    }

    # Write JS file directly for index.html
    js_content = f"window.REAL_DATA = {json.dumps(payload)};\n"
    out_js = project_root / "standalone_report" / "real_data.js"
    with open(out_js, "w", encoding="utf-8") as f:
        f.write(js_content)
    print("Exported real_data.js:", out_js.stat().st_size)


if __name__ == "__main__":
    run()
