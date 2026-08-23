"""Frozen-pipeline, winner-preserving duration and MAE research.

Thresholds are fitted only on profitable development trades.  Each overlay is
then run independently with the unchanged frozen signals, causal next-open
engine, commissions/slippage and forced EOD.  No time-stop + stop-loss combo is
tested here.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from rtharb.config import AppConfig
from rtharb.data.loader import DataLoader
from rtharb.models.fair_value import FairValueModel
from rtharb.models.signals import SignalGenerator
from rtharb.backtest.engine import BacktestEngine
from rtharb.backtest.metrics import calculate_performance_metrics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "research_output" / "duration_stoploss_verified"
Q = [float(x) for x in os.environ.get("DURATION_QS", "0.90,0.95,0.975,0.99").split(",")]

def run(frame, cfg, hold=None, stop=None):
    eng = BacktestEngine(initial_capital=cfg.backtest.initial_capital, position_size_usd=cfg.backtest.position_size_usd,
                         commission_per_share=cfg.backtest.commission_per_share, slippage_pct=cfg.backtest.slippage_pct,
                         allow_short=cfg.backtest.allow_short, max_holding_bars=hold, stop_loss_pct=stop)
    bt = eng.run(frame, cfg.strategy.ticker_target)
    return bt, calculate_performance_metrics(bt["df_results"], bt["trades_df"], cfg.backtest.initial_capital)

def row(perf, trades):
    gross = float(trades["gross_pnl"].sum()) if not trades.empty else 0.0
    d = {"trades": perf.total_trades, "gross_pnl": gross, "net_pnl": perf.total_pnl, "return_pct": perf.total_return_pct,
         "sharpe": perf.sharpe_ratio, "sortino": perf.sortino_ratio, "max_dd_pct": perf.max_drawdown_pct,
         "win_rate_pct": perf.win_rate_pct, "profit_factor": perf.profit_factor, "avg_duration": perf.avg_duration_mins,
         "commissions": perf.total_commissions, "slippage": perf.total_slippage}
    d["reconciliation_error"] = d["gross_pnl"] - d["commissions"] - d["slippage"] - d["net_pnl"]
    return d

def add_mae(trades, metrics, slip):
    t = trades.copy()
    maes = []
    for _, tr in t.iterrows():
        entry = float(tr["entry_price"])/(1.0 + slip) if int(tr["direction"]) == 1 else float(tr["entry_price"])/(1.0 - slip)
        entry_ts, exit_ts = pd.Timestamp(tr["entry_time"]), pd.Timestamp(tr["exit_time"])
        # Entry bar is observable at next-open execution; exit bar's OHLC is
        # not observable before that open and must not enter MAE.
        start = int(metrics.index.searchsorted(entry_ts, side="left"))
        end = int(metrics.index.searchsorted(exit_ts, side="left"))
        path = metrics.iloc[start:end]
        if path.empty:
            maes.append(0.0); continue
        if int(tr["direction"]) == 1:
            mae = max(0.0, (entry - float(path["target_low"].min())) / entry)
        else:
            mae = max(0.0, (float(path["target_high"].max()) - entry) / entry)
        maes.append(mae)
    t["mae_pct"] = maes; t["mae_usd"] = t["mae_pct"] * t["entry_price"]
    return t

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = AppConfig.load(str(ROOT / "configs" / "default_config.yaml"))
    frozen = json.loads((ROOT / "research_output" / "base_strategy_summary.json").read_text(encoding="utf-8"))["selected_parameters"]
    cfg.strategy.beta_mode, cfg.strategy.beta_rolling_days = frozen["beta_mode"], int(frozen["beta_days"])
    cfg.strategy.rolling_window_w, cfg.strategy.z_entry = int(frozen["window"]), float(frozen["z_entry"])
    cfg.strategy.reversal_delta, cfg.strategy.reversal_timeout_bars = float(frozen["hook_delta"]), int(frozen["hook_timeout"])
    cfg.strategy.z_exit, cfg.strategy.z_max_allowed = float(frozen["exit_band"]), float(frozen["z_lockout"])
    lead, target = DataLoader(cfg.cache_dir, "alpaca", "sip").get_synchronized_pair("QQQ", "NVDA")
    model = FairValueModel(cfg.strategy.beta_mode, cfg.strategy.beta_rolling_days, cfg.strategy.rolling_window_w,
                           cfg.strategy.min_session_warmup_bars, cfg.strategy.min_sigma_history_days)
    metrics = model.compute_intraday_metrics(lead, target)
    frame = SignalGenerator(z_entry=cfg.strategy.z_entry, reversal_delta=cfg.strategy.reversal_delta,
                            reversal_timeout_bars=cfg.strategy.reversal_timeout_bars, enable_extreme_entry_lockout=True,
                            enable_extreme_emergency_exit=False, z_max_allowed=cfg.strategy.z_max_allowed,
                            lockout_mode="day_lockout", z_exit=cfg.strategy.z_exit,
                            forced_close_time=cfg.strategy.forced_close_time,
                            min_session_warmup_bars=cfg.strategy.min_session_warmup_bars).generate_signals(metrics)
    dates = sorted(frame.session_date.unique()); n=len(dates); dev=n//2; val=n*3//4
    masks={"development":frame.session_date<dates[dev],"validation":(frame.session_date>=dates[dev])&(frame.session_date<dates[val]),
           "holdout":frame.session_date>=dates[val],"full":frame.session_date>=dates[0]}
    base_trades={}; base_rows={}; all_mae=[]
    for name, mask in masks.items():
        bt, perf=run(frame.loc[mask],cfg); tr=add_mae(bt["trades_df"],metrics,cfg.backtest.slippage_pct)
        base_trades[name]=tr; base_rows[name]=row(perf,tr); tr.to_csv(OUT/f"base_{name}_trades.csv",index=False)
        if name=="development": all_mae=tr[tr.net_pnl>0].copy()
    # method='higher' selects an observed order statistic and therefore gives
    # empirical coverage >= the requested quantile, rather than interpolation.
    duration_q={str(q):int(all_mae.duration_bars.quantile(q, interpolation="higher")) for q in Q}
    mae_pct_q={str(q):float(all_mae.mae_pct.quantile(q, interpolation="higher")) for q in Q}
    mae_usd_q={str(q):float(all_mae.mae_usd.quantile(q, interpolation="higher")) for q in Q}
    time_results={}; stop_results={}
    for q in Q:
        hold=duration_q[str(q)]; stop=mae_pct_q[str(q)]
        time_results[str(q)]={}; stop_results[str(q)]={}
        for name,mask in masks.items():
            tb,tp=run(frame.loc[mask],cfg,hold=hold); sb,sp=run(frame.loc[mask],cfg,stop=stop)
            time_results[str(q)][name]={"threshold_bars":hold,**row(tp,tb["trades_df"])}
            stop_results[str(q)][name]={"threshold_pct":stop,"threshold_usd_quantile":mae_usd_q[str(q)],**row(sp,sb["trades_df"])}
            tb["trades_df"].to_csv(OUT/f"time_q{int(q*1000)}_{name}_trades.csv",index=False)
            sb["trades_df"].to_csv(OUT/f"stop_q{int(q*1000)}_{name}_trades.csv",index=False)
    def survival(overlay_kind, hold=None, stop=None):
        bt, _ = run(frame.loc[masks["development"]], cfg, hold=hold, stop=stop)
        ov = bt["trades_df"].copy(); base = base_trades["development"][base_trades["development"].net_pnl > 0]
        by_entry = {str(x): r for x, r in ov.set_index("entry_time").iterrows()}
        matched = not_early = profitable = 0
        for _, r in base.iterrows():
            x = by_entry.get(str(r.entry_time))
            if x is not None:
                matched += 1
                not_early += int(pd.Timestamp(x.exit_time) >= pd.Timestamp(r.exit_time))
                profitable += int(float(x.net_pnl) > 0)
        total = len(base)
        return {"baseline_dev_winner_count": total, "matched_entry_events": matched,
                "matched_survival_pct": 100*matched/total if total else 0.0,
                "not_prematurely_closed_pct": 100*not_early/total if total else 0.0,
                "still_net_profitable_pct": 100*profitable/total if total else 0.0}
    raw_stop_survival = survival("stop", stop=mae_pct_q["0.95"])
    selected_stop = mae_pct_q["0.95"]
    selected_stop_survival = raw_stop_survival
    if selected_stop_survival["still_net_profitable_pct"] < 95.0:
        for candidate in sorted(float(x) for x in all_mae.mae_pct.unique() if float(x) > selected_stop):
            candidate_survival = survival("stop", stop=candidate)
            if candidate_survival["still_net_profitable_pct"] >= 95.0:
                selected_stop = candidate
                selected_stop_survival = candidate_survival
                break
    selected_stop_results = {}
    for name, mask in masks.items():
        sb, sp = run(frame.loc[mask], cfg, stop=selected_stop)
        selected_stop_results[name] = {"threshold_pct": selected_stop, **row(sp, sb["trades_df"])}
        sb["trades_df"].to_csv(OUT / f"stop_selected_q95_{name}_trades.csv", index=False)
    survival_q95 = {
        "time_stop": survival("time", hold=duration_q["0.95"]),
        "stop_loss": selected_stop_survival,
    }
    summary={"frozen_parameters":frozen,"data":{"sessions":n,"development":dev,"validation":val-dev,"holdout":n-val},
             "winner_duration_quantiles_bars":duration_q,"winner_mae_pct_quantiles":mae_pct_q,"winner_mae_usd_quantiles":mae_usd_q,
             "base":base_rows,"time_stop_only":time_results,"stop_loss_only":stop_results,
             "selected_q95_overlays":{"time_stop":{"threshold_bars":duration_q["0.95"],"results":time_results["0.95"]},
                                      "stop_loss":{"raw_q95_threshold_pct":mae_pct_q["0.95"],
                                                   "selected_threshold_pct":selected_stop,
                                                   "raw_q95_survival":raw_stop_survival,"results":selected_stop_results}},
             "development_q95_survival":survival_q95,
             "threshold_fit_note":"Raw quantiles use profitable development trades only. The published stop is the minimum observed MAE threshold whose exact replay keeps at least 95% of original development winners net profitable. Overlays are independent."}
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    (OUT/"README.md").write_text("# Verified duration / stop-loss research\n\nExact frozen entries are generated from `base_strategy_summary.json` parameters and the standard SignalGenerator. Development profitable trades only determine 90/95/97.5/99% duration and intrabar MAE thresholds. Time-stop-only and stop-loss-only overlays are evaluated separately on development, validation, holdout and full periods using next-open fills, raw 1m high/low stop detection, conservative gap handling from BacktestEngine, costs and forced EOD. No combined overlay is tested.\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2,default=str))

if __name__=="__main__": main()
