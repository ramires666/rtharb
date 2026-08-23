import sys
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

    days_to_pack = ["2024-08-21", "2024-08-27", "2024-09-03", "2024-09-11", "2025-01-08"]
    
    print("=== TRADES ===")
    for d_str in days_to_pack:
        d = pd.to_datetime(d_str).date()
        trs = trades_b[trades_b["entry_time"].dt.date == d]
        for _, tr in trs.iterrows():
            print(f"Trade #{int(tr['trade_id'])} | {'LONG' if tr['direction']==1 else 'SHORT'} | {tr['entry_time']} @ ${tr['entry_price']:.2f} -> {tr['exit_time']} @ ${tr['exit_price']:.2f} | Net: ${tr['net_pnl']:.2f} ({tr['return_pct']*100:+.2f}%) | Reason: {tr['exit_reason']} | Z_in: {tr['entry_z_score']:.2f} | Z_out: {tr['exit_z_score']:.2f}")

    print("\n=== SESSIONS SUMMARY ===")
    for d_str in days_to_pack:
        d = pd.to_datetime(d_str).date()
        df_d = df_b_signals[df_b_signals["session_date"] == d]
        print(f"Day {d_str}: {len(df_d)} bars, NVDA Open: ${df_d['target_open'].iloc[0]:.2f}, High: ${df_d['target_close'].max():.2f}, Low: ${df_d['target_close'].min():.2f}, Close: ${df_d['target_close'].iloc[-1]:.2f}")


if __name__ == "__main__":
    run()
