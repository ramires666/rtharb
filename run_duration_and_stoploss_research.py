"""Comprehensive Quantitative Research:

1. Duration Analysis & Time-Stop Cutoffs (30m, 45m, 60m, 75m, 90m, 120m, 180m, EOD)
2. Stop-Loss Optimization (% Stop Loss & Z-Score Divergence Stop)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rtharb.data.loader import DataLoader
from rtharb.models.fair_value import FairValueModel
from rtharb.config import AppConfig


def simulate_with_rules(df_metrics, max_hold_bars=None, stop_loss_pct=None, z_stop=None, capital=100000.0, pos_size=20000.0):
    """Simulate single-leg stat-arb with custom time-stop, % stop-loss, and z-stop."""
    # Signal thresholds
    z_enter = 1.5
    z_exit = 0.0
    z_lockout = 4.0
    delta_hook = 0.15

    # Fees
    comm_per_share = 0.0035
    slippage_bps = 0.0002

    trades = []
    equity_curve = []
    current_balance = capital

    in_position = False
    direction = 0  # +1 Long, -1 Short
    entry_price = 0.0
    entry_time = None
    entry_idx = 0
    entry_z = 0.0
    shares = 0
    bars_held = 0

    armed = False
    armed_dir = 0
    extreme_z = 0.0

    # Group by session
    sessions = df_metrics.groupby("session_date")

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

            if in_position:
                bars_held += 1
                exit_reason = None
                exit_price = price

                # 1. Check Stop Loss %
                if stop_loss_pct is not None:
                    ret_unreal = (price - entry_price) / entry_price if direction == 1 else (entry_price - price) / entry_price
                    if ret_unreal <= -stop_loss_pct:
                        exit_reason = f"STOP_LOSS_{stop_loss_pct*100:.1f}%"

                # 2. Check Z-Score Divergence Stop
                if exit_reason is None and z_stop is not None:
                    if direction == 1 and z <= -z_stop:
                        exit_reason = f"Z_STOP_{z_stop}sigma"
                    elif direction == -1 and z >= z_stop:
                        exit_reason = f"Z_STOP_{z_stop}sigma"

                # 3. Check Time Stop (Max holding duration)
                if exit_reason is None and max_hold_bars is not None:
                    if bars_held >= max_hold_bars:
                        exit_reason = f"TIME_STOP_{max_hold_bars}m"

                # 4. Check Take Profit (Mean Reversion to 0)
                if exit_reason is None:
                    if direction == 1 and z >= -z_exit:
                        exit_reason = "TAKE_PROFIT"
                    elif direction == -1 and z <= z_exit:
                        exit_reason = "TAKE_PROFIT"

                # 5. Check Forced EOD (15:55 ET)
                if exit_reason is None and is_eod:
                    exit_reason = "FORCED_EOD"

                if exit_reason is not None:
                    # Execute Exit
                    slip_cost = exit_price * slippage_bps
                    exec_exit_price = exit_price - slip_cost if direction == 1 else exit_price + slip_cost
                    gross_pnl = (exec_exit_price - entry_price) * shares if direction == 1 else (entry_price - exec_exit_price) * shares
                    comm = shares * comm_per_share * 2
                    net_pnl = gross_pnl - comm
                    ret_pct = net_pnl / (shares * entry_price)

                    current_balance += net_pnl
                    trades.append({
                        "trade_id": len(trades) + 1,
                        "session_date": s_date,
                        "direction": direction,
                        "entry_time": entry_time,
                        "entry_price": entry_price,
                        "exit_time": ts,
                        "exit_price": exec_exit_price,
                        "duration_bars": bars_held,
                        "exit_reason": exit_reason,
                        "gross_pnl": gross_pnl,
                        "commissions": comm,
                        "net_pnl": net_pnl,
                        "return_pct": ret_pct,
                        "entry_z": entry_z,
                        "exit_z": z
                    })

                    in_position = False
                    direction = 0
                    shares = 0
                    bars_held = 0
                    armed = False
                    armed_dir = 0
                    extreme_z = 0.0

            else:
                # Not in position -> Check entry logic
                if not is_eod:
                    # Check 4-sigma lockout
                    if abs(z) >= z_lockout:
                        armed = False
                        armed_dir = 0
                        extreme_z = 0.0
                    else:
                        # Phase 1: Arming
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
                            # Update extreme
                            if armed_dir == 1:
                                if z < extreme_z:
                                    extreme_z = z
                                # Phase 2: Hook trigger
                                elif (z - extreme_z) >= delta_hook:
                                    # Enter Long
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
                            elif armed_dir == -1:
                                if z > extreme_z:
                                    extreme_z = z
                                # Phase 2: Hook trigger
                                elif (extreme_z - z) >= delta_hook:
                                    # Enter Short
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

            equity_curve.append(current_balance)

    trades_df = pd.DataFrame(trades)
    eq_series = pd.Series(equity_curve, index=df_metrics.index)

    # Compute metrics
    if trades_df.empty:
        return {"total_pnl": 0, "trades": 0, "sharpe": 0, "max_dd_pct": 0, "win_rate": 0, "trades_df": trades_df}

    wins = trades_df[trades_df["net_pnl"] > 0]
    losses = trades_df[trades_df["net_pnl"] <= 0]
    win_rate = len(wins) / len(trades_df) * 100

    tot_pnl = trades_df["net_pnl"].sum()
    gross_win = wins["net_pnl"].sum() if not wins.empty else 0
    gross_loss = abs(losses["net_pnl"].sum()) if not losses.empty else 1e-6
    profit_factor = gross_win / gross_loss if gross_loss > 0 else 999.0

    # Drawdown
    cummax = eq_series.cummax()
    dd = (eq_series - cummax) / cummax
    max_dd_pct = abs(dd.min()) * 100
    max_dd_usd = abs((eq_series - cummax).min())

    # Daily returns for Sharpe
    daily_eq = eq_series.resample("D").last().dropna()
    daily_ret = daily_eq.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0
    downside_ret = daily_ret[daily_ret < 0]
    sortino = (daily_ret.mean() / downside_ret.std()) * np.sqrt(252) if not downside_ret.empty and downside_ret.std() > 0 else 0

    return {
        "total_pnl": tot_pnl,
        "total_return_pct": (tot_pnl / capital) * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd_pct": max_dd_pct,
        "max_dd_usd": max_dd_usd,
        "total_trades": len(trades_df),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_win": wins["net_pnl"].mean() if not wins.empty else 0,
        "avg_loss": losses["net_pnl"].mean() if not losses.empty else 0,
        "max_win": trades_df["net_pnl"].max(),
        "max_loss": trades_df["net_pnl"].min(),
        "total_commissions": trades_df["commissions"].sum(),
        "trades_df": trades_df,
        "equity_series": eq_series
    }


def run_full_research():
    print("⏳ Loading 2-year 1m data from Alpaca Parquet cache...")
    loader = DataLoader(cache_dir="data_cache", source="alpaca")
    df_lead, df_target = loader.get_synchronized_pair("QQQ", "NVDA", days_back=730, source="alpaca")

    fv = FairValueModel(beta_mode="dynamic_rolling", rolling_window_w=30)
    df_metrics = fv.compute_intraday_metrics(df_lead, df_target)

    # -------------------------------------------------------------
    # 1. BASELINE ANALYSIS & DURATION DISTRIBUTION
    # -------------------------------------------------------------
    print("\n==================================================")
    print("📊 1. BASELINE STRATEGY (Scenario B, No Cutoffs)")
    print("==================================================")
    base_res = simulate_with_rules(df_metrics)
    b_df = base_res["trades_df"]

    print(f"Total Trades: {len(b_df)}")
    print(f"Net PnL: ${base_res['total_pnl']:,.2f} ({base_res['total_return_pct']:+.2f}%)")
    print(f"Sharpe: {base_res['sharpe']:.2f} | Max DD: {base_res['max_dd_pct']:.2f}% (${base_res['max_dd_usd']:,.2f})")
    print(f"Win Rate: {base_res['win_rate']:.1f}% ({len(b_df[b_df['net_pnl']>0])} wins / {len(b_df[b_df['net_pnl']<=0])} losses)")
    print(f"Avg Win: ${base_res['avg_win']:,.2f} | Avg Loss: ${base_res['avg_loss']:,.2f} | Max Loss: ${base_res['max_loss']:,.2f}")

    # Duration percentiles for WINS
    wins_df = b_df[b_df["net_pnl"] > 0]
    losses_df = b_df[b_df["net_pnl"] <= 0]

    print("\n⏱️ DURATION PERCENTILES FOR WINNING TRADES (Mean Reversion):")
    for p in [50, 75, 80, 85, 90, 95, 97, 99]:
        val = np.percentile(wins_df["duration_bars"], p)
        print(f"  {p}% of winning trades close within: {val:.0f} minutes")

    print("\n⏱️ DURATION PERCENTILES FOR LOSING TRADES:")
    for p in [50, 75, 80, 85, 90, 95, 99]:
        val = np.percentile(losses_df["duration_bars"], p)
        print(f"  {p}% of losing trades last up to: {val:.0f} minutes")

    print(f"\nAverage Duration Wins: {wins_df['duration_bars'].mean():.1f} min | Losses: {losses_df['duration_bars'].mean():.1f} min")
    print(f"Median Duration Wins: {wins_df['duration_bars'].median():.0f} min | Losses: {losses_df['duration_bars'].median():.0f} min")

    # -------------------------------------------------------------
    # 2. TIME-STOP (MAX HOLDING DURATION) RESEARCH
    # -------------------------------------------------------------
    print("\n==================================================")
    print("⏳ 2. TIME-STOP RESEARCH (Max Holding Duration Cutoff)")
    print("==================================================")
    time_cutoffs = [20, 30, 45, 60, 75, 90, 120, 150, 180, 240, None]
    time_results = []

    for tc in time_cutoffs:
        res = simulate_with_rules(df_metrics, max_hold_bars=tc)
        tc_name = f"{tc} min" if tc is not None else "No Time Stop (EOD)"
        time_results.append({
            "Time Stop": tc_name,
            "PnL ($)": f"${res['total_pnl']:,.2f}",
            "Return (%)": f"{res['total_return_pct']:+.2f}%",
            "Sharpe": round(res["sharpe"], 2),
            "Sortino": round(res["sortino"], 2),
            "Max DD (%)": f"{res['max_dd_pct']:.2f}%",
            "Max DD ($)": f"${res['max_dd_usd']:,.2f}",
            "Win Rate": f"{res['win_rate']:.1f}%",
            "Profit Factor": round(res["profit_factor"], 2),
            "Avg Loss": f"${res['avg_loss']:,.2f}",
            "Max Loss": f"${res['max_loss']:,.2f}",
            "Trades": res["total_trades"]
        })

    df_time_res = pd.DataFrame(time_results)
    print(df_time_res.to_string(index=False))

    # -------------------------------------------------------------
    # 3. STOP-LOSS % RESEARCH (Preserving Profit Priority)
    # -------------------------------------------------------------
    print("\n==================================================")
    print("🛡️ 3. STOP-LOSS % RESEARCH (Fixed Price Stop Loss)")
    print("==================================================")
    sl_grid = [0.003, 0.005, 0.0075, 0.010, 0.0125, 0.015, 0.0175, 0.020, 0.025, 0.030, None]
    sl_results = []

    for sl in sl_grid:
        res = simulate_with_rules(df_metrics, stop_loss_pct=sl)
        sl_name = f"{sl*100:.2f}%" if sl is not None else "No Stop Loss"
        sl_results.append({
            "Stop Loss": sl_name,
            "PnL ($)": f"${res['total_pnl']:,.2f}",
            "Return (%)": f"{res['total_return_pct']:+.2f}%",
            "Sharpe": round(res["sharpe"], 2),
            "Sortino": round(res["sortino"], 2),
            "Max DD (%)": f"{res['max_dd_pct']:.2f}%",
            "Max DD ($)": f"${res['max_dd_usd']:,.2f}",
            "Win Rate": f"{res['win_rate']:.1f}%",
            "Profit Factor": round(res["profit_factor"], 2),
            "Avg Loss": f"${res['avg_loss']:,.2f}",
            "Max Loss": f"${res['max_loss']:,.2f}",
            "Trades": res["total_trades"]
        })

    df_sl_res = pd.DataFrame(sl_results)
    print(df_sl_res.to_string(index=False))

    # -------------------------------------------------------------
    # 4. Z-SCORE DIVERGENCE STOP RESEARCH
    # -------------------------------------------------------------
    print("\n==================================================")
    print("📐 4. Z-SCORE DIVERGENCE STOP RESEARCH")
    print("==================================================")
    z_stops = [2.0, 2.5, 3.0, 3.5, 4.0, None]
    z_results = []

    for zs in z_stops:
        res = simulate_with_rules(df_metrics, z_stop=zs)
        zs_name = f"|Z| >= {zs:.1f}σ" if zs is not None else "No Z Stop"
        z_results.append({
            "Z Stop": zs_name,
            "PnL ($)": f"${res['total_pnl']:,.2f}",
            "Return (%)": f"{res['total_return_pct']:+.2f}%",
            "Sharpe": round(res["sharpe"], 2),
            "Sortino": round(res["sortino"], 2),
            "Max DD (%)": f"{res['max_dd_pct']:.2f}%",
            "Max DD ($)": f"${res['max_dd_usd']:,.2f}",
            "Win Rate": f"{res['win_rate']:.1f}%",
            "Profit Factor": round(res["profit_factor"], 2),
            "Avg Loss": f"${res['avg_loss']:,.2f}",
            "Max Loss": f"${res['max_loss']:,.2f}",
            "Trades": res["total_trades"]
        })

    df_z_res = pd.DataFrame(z_results)
    print(df_z_res.to_string(index=False))

    # -------------------------------------------------------------
    # 5. COMBINED OPTIMAL CONFIGURATIONS
    # -------------------------------------------------------------
    print("\n==================================================")
    print("🏆 5. COMBINED DUAL-PROTECTION MATRIX (Time-Stop + Stop-Loss)")
    print("==================================================")
    combos = [
        {"name": "Baseline (EOD + No SL)", "hold": None, "sl": None},
        {"name": "Time-Stop 60m (No SL)", "hold": 60, "sl": None},
        {"name": "Time-Stop 75m (No SL)", "hold": 75, "sl": None},
        {"name": "Time-Stop 90m (No SL)", "hold": 90, "sl": None},
        {"name": "Time-Stop 120m (No SL)", "hold": 120, "sl": None},
        {"name": "SL 1.0% (No Time Stop)", "hold": None, "sl": 0.010},
        {"name": "SL 1.5% (No Time Stop)", "hold": None, "sl": 0.015},
        {"name": "SL 2.0% (No Time Stop)", "hold": None, "sl": 0.020},
        {"name": "Combo: Time 75m + SL 1.5%", "hold": 75, "sl": 0.015},
        {"name": "Combo: Time 90m + SL 1.5%", "hold": 90, "sl": 0.015},
        {"name": "Combo: Time 120m + SL 1.5%", "hold": 120, "sl": 0.015},
        {"name": "Combo: Time 90m + SL 2.0%", "hold": 90, "sl": 0.020},
    ]

    combo_res = []
    for c in combos:
        res = simulate_with_rules(df_metrics, max_hold_bars=c["hold"], stop_loss_pct=c["sl"])
        combo_res.append({
            "Strategy Config": c["name"],
            "PnL ($)": f"${res['total_pnl']:,.2f}",
            "Return (%)": f"{res['total_return_pct']:+.2f}%",
            "Sharpe": round(res["sharpe"], 2),
            "Sortino": round(res["sortino"], 2),
            "Max DD (%)": f"{res['max_dd_pct']:.2f}%",
            "Max DD ($)": f"${res['max_dd_usd']:,.2f}",
            "Win Rate": f"{res['win_rate']:.1f}%",
            "Profit Factor": round(res["profit_factor"], 2),
            "Avg Loss": f"${res['avg_loss']:,.2f}",
            "Max Loss": f"${res['max_loss']:,.2f}",
            "Trades": res["total_trades"]
        })

    df_combos = pd.DataFrame(combo_res)
    print(df_combos.to_string(index=False))


if __name__ == "__main__":
    run_full_research()
