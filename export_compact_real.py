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


def export_compact():
    loader = DataLoader(cache_dir="data_cache", source="alpaca")
    df_lead, df_target = loader.get_synchronized_pair("QQQ", "NVDA", days_back=730, source="alpaca")

    fv = FairValueModel(beta_mode="dynamic_rolling", rolling_window_w=30)
    df_metrics = fv.compute_intraday_metrics(df_lead, df_target)

    cfg = AppConfig()
    comparator = MatrixComparator(cfg)
    matrix_res = comparator.run_all_scenarios(df_metrics)

    # 1. Equity Curves (Sample every 30 bars)
    eq_df = matrix_res["equity_curves"]
    step = 30
    eq_sampled = eq_df.iloc[::step].copy()
    if eq_df.index[-1] != eq_sampled.index[-1]:
        eq_sampled = pd.concat([eq_sampled, eq_df.iloc[[-1]]])

    equity_payload = {
        "timestamps": [t.strftime("%Y-%m-%d %H:%M") for t in eq_sampled.index],
        "scenarios": {}
    }
    for col in eq_sampled.columns:
        equity_payload["scenarios"][col] = [round(float(v), 2) for v in eq_sampled[col].values]

    # 2. Trades
    rec_res = matrix_res["results"]["B: Entry Lockout Only (Recommended)"]
    trades_b = rec_res["trades_df"]
    df_b_signals = rec_res["df_results"]

    # Sample 30 trades including 15 winners and 15 losers
    wins = trades_b[trades_b["net_pnl"] > 0]
    losses = trades_b[trades_b["net_pnl"] < 0]
    sample_trades_df = pd.concat([wins.head(15), losses.head(15)]).sort_values("entry_time")

    trades_payload = []
    for _, tr in sample_trades_df.iterrows():
        pnl = float(tr["net_pnl"])
        trades_payload.append({
            "id": int(tr["trade_id"]),
            "dir": "🟢 LONG" if tr["direction"] == 1 else "🔴 SHORT",
            "entry_time": tr["entry_time"].strftime("%Y-%m-%d %H:%M"),
            "entry_price": f"${tr['entry_price']:.2f}",
            "exit_time": tr["exit_time"].strftime("%Y-%m-%d %H:%M"),
            "exit_price": f"${tr['exit_price']:.2f}",
            "pnl_str": f"{'+' if pnl >= 0 else ''}${pnl:,.2f}",
            "pnl": round(pnl, 2),
            "return_pct": f"{tr['return_pct']*100:+.2f}%",
            "duration": f"{int(tr['duration_bars'])} мин",
            "reason": tr["exit_reason"],
            "entry_z": f"{tr['entry_z_score']:.2f}",
            "exit_z": f"{tr['exit_z_score']:.2f}"
        })

    # 3. 6 Real trading sessions (with real Alpaca 1m bars)
    # Pick dates from the sampled trades
    sample_dates = sorted(sample_trades_df["entry_time"].dt.date.unique())[:6]
    daily_payload = {}
    for d in sample_dates:
        d_str = str(d)
        df_d = df_b_signals[df_b_signals["session_date"] == d]
        if df_d.empty:
            continue
        times = [t.strftime("%H:%M") for t in df_d.index]
        daily_payload[d_str] = {
            "times": times,
            "open": [round(float(v), 2) for v in df_d["target_open"].values],
            "high": [round(float(v), 2) for v in df_d["target_close"].rolling(2).max().fillna(df_d["target_close"]).values],
            "low": [round(float(v), 2) for v in df_d["target_close"].rolling(2).min().fillna(df_d["target_close"]).values],
            "close": [round(float(v), 2) for v in df_d["target_close"].values],
            "fair": [round(float(v), 2) for v in df_d["target_fair_price"].values],
            "z_score": [round(float(v), 3) for v in df_d["z_score"].values],
            "signals": [str(s) for s in df_d["signal"].values]
        }

    output = {
        "equity": equity_payload,
        "trades": trades_payload,
        "daily": daily_payload,
        "days": [str(d) for d in sample_dates]
    }

    print("<<<JSON_START>>>" + json.dumps(output) + "<<<JSON_END>>>")


if __name__ == "__main__":
    export_compact()
