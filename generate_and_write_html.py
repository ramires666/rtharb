"""Generate standalone_report/index.html with 100% embedded real Alpaca data."""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from rtharb.data.loader import DataLoader
from rtharb.models.fair_value import FairValueModel


def run_sim(df_metrics, max_hold_bars=120, stop_loss_pct=0.015, z_lockout=4.0):
    z_enter = 1.5
    z_exit = 0.0
    delta_hook = 0.15
    comm_per_share = 0.0035
    slippage_bps = 0.0002
    capital = 100000.0
    pos_size = 20000.0

    trades = []
    equity_curve = []
    current_balance = capital

    in_position = False
    direction = 0
    entry_price = 0.0
    entry_time = None
    entry_z = 0.0
    shares = 0
    bars_held = 0

    armed = False
    armed_dir = 0
    extreme_z = 0.0

    sessions = df_metrics.groupby("session_date")
    signals_dict = {}

    for s_date, s_df in sessions:
        in_position = False
        direction = 0
        shares = 0
        bars_held = 0
        armed = False
        armed_dir = 0
        extreme_z = 0.0

        for i, (ts, row) in enumerate(s_df.iterrows()):
            z = row["z_score"]
            price = row["target_close"]
            is_eod = (i == len(s_df) - 1) or (ts.time() >= pd.to_datetime("15:55").time())
            current_sig = "NONE"

            if in_position:
                bars_held += 1
                exit_reason = None
                exit_price = price

                if stop_loss_pct is not None:
                    ret_unreal = (price - entry_price) / entry_price if direction == 1 else (entry_price - price) / entry_price
                    if ret_unreal <= -stop_loss_pct:
                        exit_reason = "STOP_LOSS_1.5%"
                        current_sig = "EXIT_STOP_LOSS"

                if exit_reason is None and max_hold_bars is not None:
                    if bars_held >= max_hold_bars:
                        exit_reason = f"TIME_STOP_{max_hold_bars}m"
                        current_sig = "EXIT_TIME_STOP"

                if exit_reason is None:
                    if direction == 1 and z >= -z_exit:
                        exit_reason = "TAKE_PROFIT"
                        current_sig = "EXIT_TAKE_PROFIT"
                    elif direction == -1 and z <= z_exit:
                        exit_reason = "TAKE_PROFIT"
                        current_sig = "EXIT_TAKE_PROFIT"

                if exit_reason is None and is_eod:
                    exit_reason = "FORCED_EOD"
                    current_sig = "EXIT_FORCED_EOD"

                if exit_reason is not None:
                    slip_cost = exit_price * slippage_bps
                    exec_exit_price = exit_price - slip_cost if direction == 1 else exit_price + slip_cost
                    gross_pnl = (exec_exit_price - entry_price) * shares if direction == 1 else (entry_price - exec_exit_price) * shares
                    comm = shares * comm_per_share * 2
                    net_pnl = gross_pnl - comm
                    ret_pct = net_pnl / (shares * entry_price)

                    current_balance += net_pnl
                    trades.append({
                        "id": len(trades) + 1,
                        "dir": "🟢 LONG" if direction == 1 else "🔴 SHORT",
                        "entry_time": entry_time.strftime("%Y-%m-%d %H:%M"),
                        "entry_price": f"${entry_price:.2f}",
                        "exit_time": ts.strftime("%Y-%m-%d %H:%M"),
                        "exit_price": f"${exec_exit_price:.2f}",
                        "pnl_str": f"{'+' if net_pnl>=0 else ''}${net_pnl:,.2f}",
                        "is_win": net_pnl >= 0,
                        "return_pct": f"{ret_pct*100:+.2f}%",
                        "duration": f"{bars_held} мин",
                        "reason": exit_reason,
                        "entry_z": f"{entry_z:.2f}",
                        "exit_z": f"{z:.2f}"
                    })

                    in_position = False
                    direction = 0
                    shares = 0
                    bars_held = 0
                    armed = False
                    armed_dir = 0
                    extreme_z = 0.0

            else:
                if not is_eod:
                    if abs(z) >= z_lockout:
                        armed = False
                        armed_dir = 0
                        extreme_z = 0.0
                    else:
                        if not armed:
                            if z <= -z_enter:
                                armed = True
                                armed_dir = 1
                                extreme_z = z
                            elif z >= z_enter:
                                armed = True
                                armed_dir = -1
                                extreme_z = z
                        else:
                            if armed_dir == 1:
                                if z < extreme_z:
                                    extreme_z = z
                                elif (z - extreme_z) >= delta_hook:
                                    slip_cost = price * slippage_bps
                                    exec_entry = price + slip_cost
                                    shares = int(pos_size / exec_entry)
                                    if shares > 0:
                                        in_position = True
                                        direction = 1
                                        entry_price = exec_entry
                                        entry_time = ts
                                        entry_z = z
                                        bars_held = 0
                                        armed = False
                                        current_sig = "BUY_LONG"
                            elif armed_dir == -1:
                                if z > extreme_z:
                                    extreme_z = z
                                elif (extreme_z - z) >= delta_hook:
                                    slip_cost = price * slippage_bps
                                    exec_entry = price - slip_cost
                                    shares = int(pos_size / exec_entry)
                                    if shares > 0:
                                        in_position = True
                                        direction = -1
                                        entry_price = exec_entry
                                        entry_time = ts
                                        entry_z = z
                                        bars_held = 0
                                        armed = False
                                        current_sig = "SELL_SHORT"

            equity_curve.append(current_balance)
            signals_dict[ts] = current_sig

    return trades, pd.Series(equity_curve, index=df_metrics.index), signals_dict


def main():
    loader = DataLoader(cache_dir="data_cache", source="alpaca")
    df_lead, df_target = loader.get_synchronized_pair("QQQ", "NVDA", days_back=730, source="alpaca")
    fv = FairValueModel(beta_mode="dynamic_rolling", rolling_window_w=30)
    df_metrics = fv.compute_intraday_metrics(df_lead, df_target)

    # 1. Production Simulation (Time-Stop 120m + SL 1.5% + 4σ Lockout)
    trades_prod, eq_prod, sigs_prod = run_sim(df_metrics, max_hold_bars=120, stop_loss_pct=0.015, z_lockout=4.0)

    # 2. Baseline Scenario B (Lockout only, no time-stop, no SL)
    _, eq_base_b, _ = run_sim(df_metrics, max_hold_bars=None, stop_loss_pct=None, z_lockout=4.0)

    # 3. Baseline Scenario A (Pure mean-reversion, no caps)
    _, eq_base_a, _ = run_sim(df_metrics, max_hold_bars=None, stop_loss_pct=None, z_lockout=999.0)

    # Equity points (sample every 15 bars)
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

    # Diverse 1-minute real sessions
    sample_days = [
        "2024-08-21", "2024-08-22", "2024-08-27", "2024-09-03", "2024-09-11",
        "2024-09-24", "2024-10-04", "2024-11-06", "2024-12-12", "2025-01-08",
        "2025-02-14", "2025-03-14", "2025-05-22", "2025-08-15", "2026-02-18"
    ]

    sessions_pack = {}
    for d_str in sample_days:
        d = pd.to_datetime(d_str).date()
        df_d = df_metrics[df_metrics["session_date"] == d]
        if df_d.empty:
            continue

        day_trades = [tr for tr in trades_prod if tr["entry_time"].startswith(d_str)]
        day_pnl = sum(float(tr["pnl_str"].replace("$", "").replace("+", "").replace(",", "")) for tr in day_trades)
        day_summary = f"Сделок: {len(day_trades)} | PnL: {'+' if day_pnl>=0 else ''}${day_pnl:,.2f}"

        high_vals = df_d["target_high"].values if "target_high" in df_d.columns else df_d["target_close"].values
        low_vals = df_d["target_low"].values if "target_low" in df_d.columns else df_d["target_close"].values

        sessions_pack[d_str] = {
            "date": d_str,
            "label": f"{d_str} — {day_summary}",
            "times": [t.strftime("%H:%M") for t in df_d.index],
            "open": [round(float(v), 2) for v in df_d["target_open"].values],
            "high": [round(float(v), 2) for v in high_vals],
            "low": [round(float(v), 2) for v in low_vals],
            "close": [round(float(v), 2) for v in df_d["target_close"].values],
            "fair": [round(float(v), 2) for v in df_d["target_fair_price"].values],
            "z_score": [round(float(v), 3) for v in df_d["z_score"].values],
            "signals": [sigs_prod.get(ts, "NONE") for ts in df_d.index]
        }

    # Print out JSON string for direct inclusion
    bundle = {
        "equity": equity_pack,
        "sessions": sessions_pack,
        "trades": trades_prod[:60]
    }
    print("=== BUNDLE_DATA_START ===")
    print(json.dumps(bundle))
    print("=== BUNDLE_DATA_END ===")


if __name__ == "__main__":
    main()
