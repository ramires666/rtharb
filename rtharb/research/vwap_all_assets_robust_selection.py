"""Leakage-safe robust re-selection of all nine VWAP absolute brackets.

Selection uses sessions 0:188 only.  Sessions 188:251 are replayed only after
an atomic pre-seen freeze is written.  CURRENT_PARAMS and NO_TRADE/CASH are
explicit candidates; a negative best-of-bad bracket is never forced.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numba import njit, prange

from rtharb.research.risk_reward import CAPITAL, COMMISSION, SIZE, SLIP
from rtharb.research.vwap_absolute_multi_asset import (
    ENTRY_Z, TRADE_COLUMNS, UNIVERSE, _clean as clean_replay, load_market, simulate,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "research_output" / "vwap_absolute_multi_asset"
OUT = ROOT / "research_output" / "vwap_all_assets_robust_selection"
PRE_END, SEEN_END = 188, 251
STEP = 0.25
BLOCKS = ((0,21),(21,42),(42,63),(63,84),(84,105),(105,126),(126,147),(147,168),(168,188))
TEST_BLOCKS = tuple(range(3, 9))
METRIC_COLUMNS = (
    "stop_usd","target_usd","sessions","raw_bars","trades","active_days","gross_pnl","costs",
    "commissions","slippage","net_pnl","positive_mass","loss_mass","win_rate_pct","profit_factor",
    "net_sharpe","mtm_dd_usd","pnl_over_dd","cvar5_loss_usd","worst_loss_usd",
    "clipped_current_winner_net_usd","avoided_current_loser_net_usd","stops","targets","forced_eod",
)

def _default(x: Any) -> Any:
    if isinstance(x,(np.integer,np.floating,np.bool_)): return x.item()
    if isinstance(x,pd.Timestamp): return x.isoformat()
    raise TypeError(type(x).__name__)

def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(text,encoding="utf-8"); os.replace(tmp,path)

def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path,json.dumps(payload,ensure_ascii=False,indent=2,sort_keys=True,default=_default)+"\n")

def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    frame.to_csv(tmp,index=False,float_format="%.10f"); os.replace(tmp,path)

def sha256(path: Path) -> str:
    h=hashlib.sha256();
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

@njit(cache=True)
def _one(day,op,hi,lo,close,z,last,n_days,stop,target,base_event):
    daily=np.zeros(n_days); active=np.zeros(n_days,np.uint8); nets=np.empty(len(day))
    position=pending=0; pending_i=entry_i=-1; entry_ref=entry_eff=0.; shares=0
    stop_price=target_price=0.; cash=peak=CAPITAL; entry_comm=0.
    trades=wins=stops=targets=eod=0; gross=comm=slip=posmass=lossmass=0.; worst=0.
    base_pos=0.; base_loss=0.; retained=0.; candidate_base_loser=0.
    for v in base_event:
        if not math.isnan(v):
            if v>0: base_pos+=v
            else: base_loss+=v
    for i in range(len(day)):
        d=day[i]
        if pending!=0:
            position=pending; pending=0; entry_i=i; entry_ref=op[i]
            entry_eff=entry_ref*(1+SLIP if position==1 else 1-SLIP); shares=math.floor(SIZE/entry_eff)
            entry_comm=shares*COMMISSION; stop_price=entry_ref-stop if position==1 else entry_ref+stop
            target_price=entry_ref+target if position==1 else entry_ref-target; active[d]=1
        if position!=0:
            stop_hit=(op[i]<=stop_price or lo[i]<=stop_price) if position==1 else (op[i]>=stop_price or hi[i]>=stop_price)
            target_hit=hi[i]>=target_price if position==1 else lo[i]<=target_price
            reason=0; raw_exit=0.
            if stop_hit:
                gap=op[i]<=stop_price if position==1 else op[i]>=stop_price; raw_exit=op[i] if gap else stop_price; reason=1
            elif target_hit: raw_exit=target_price; reason=2
            elif last[i]: raw_exit=close[i]; reason=3
            if reason:
                exit_eff=raw_exit*(1-SLIP if position==1 else 1+SLIP)
                g=position*(raw_exit-entry_ref)*shares; s=(abs(entry_eff-entry_ref)+abs(exit_eff-raw_exit))*shares
                c=2*shares*COMMISSION; net=g-s-c; daily[d]+=net; gross+=g; slip+=s; comm+=c; nets[trades]=net
                if net>0: posmass+=net; wins+=1
                else: lossmass+=net
                if net<worst: worst=net
                if pending_i>=0 and not math.isnan(base_event[pending_i]):
                    bv=base_event[pending_i]
                    if bv>0: retained+=max(0.,min(net,bv))
                    else: candidate_base_loser+=net
                trades+=1
                if reason==1: stops+=1
                elif reason==2: targets+=1
                else: eod+=1
                cash+=net; position=0
        zi=z[i]
        if not math.isnan(zi) and not last[i]:
            hit=1 if zi<=-ENTRY_Z else (-1 if zi>=ENTRY_Z else 0)
            if hit!=0 and position==0: pending=hit; pending_i=i
        equity=cash-entry_comm+position*(close[i]-entry_eff)*shares if position else cash
        if equity>peak: peak=equity
        # reuse entry_ref scalar after close safely; only equity while live uses it
        dd=peak-equity
        if i==0: maxdd=dd
        elif dd>maxdd: maxdd=dd
    returns=np.zeros(n_days); cumulative=0.
    for d in range(n_days):
        prior=CAPITAL+cumulative; returns[d]=daily[d]/prior if prior else 0.; cumulative+=daily[d]
    mean=returns.mean(); std=returns.std(); sharpe=math.sqrt(252)*mean/std if std>0 else 0.
    pf=posmass/abs(lossmass) if lossmass<0 else 0.; net_sum=posmass+lossmass
    tail_n=max(1,int(math.ceil(.05*trades))) if trades else 0; cvar=0.
    if trades:
        ordered=np.sort(nets[:trades]); cvar=-ordered[:tail_n].mean()
    active_n=0
    for x in active: active_n+=x
    return np.array((stop,target,n_days,len(day),trades,active_n,gross,comm+slip,comm,slip,net_sum,posmass,lossmass,
        100*wins/trades if trades else 0.,pf,sharpe,maxdd,net_sum/maxdd if maxdd>0 else 0.,cvar,-worst,
        base_pos-retained,candidate_base_loser-base_loss,stops,targets,eod),dtype=np.float64)

@njit(parallel=True,cache=True)
def _grid(day,op,hi,lo,close,z,last,n_days,stops,targets,base_event):
    out=np.empty((len(stops),len(METRIC_COLUMNS)))
    for j in prange(len(stops)): out[j]=_one(day,op,hi,lo,close,z,last,n_days,stops[j],targets[j],base_event)
    return out

@njit(parallel=True,cache=True)
def _pareto(obj,eligible):
    n=len(eligible); out=np.zeros(n,np.uint8)
    for i in prange(n):
        if not eligible[i]: continue
        dominated=False
        for j in range(n):
            if i==j or not eligible[j]: continue
            all_ge=True; any_gt=False
            for k in range(obj.shape[1]):
                if obj[j,k] < obj[i,k]-1e-9: all_ge=False; break
                if obj[j,k] > obj[i,k]+1e-9: any_gt=True
            if all_ge and any_gt: dominated=True; break
        if not dominated: out[i]=1
    return out

def current_event_map(a:dict[str,np.ndarray],symbol:str,lo:int,hi:int,stop:float,target:float)->tuple[np.ndarray,np.ndarray]:
    idx=np.flatnonzero((a["day"]>=lo)&(a["day"]<hi)); event=np.full(len(idx),np.nan)
    replay=simulate(a,symbol,lo,hi,stop,target,collect=True); ts=pd.DatetimeIndex(a["timestamp"])[idx]
    for row in replay["trades_df"].itertuples(index=False):
        p=int(ts.searchsorted(pd.Timestamp(row.signal_time)))
        if p>=len(ts) or ts[p]!=pd.Timestamp(row.signal_time): raise AssertionError("baseline event absent")
        event[p]=float(row.net_pnl)
    return idx,event

def evaluate(a,symbol,lo,hi,stops,targets,current)->pd.DataFrame:
    idx,event=current_event_map(a,symbol,lo,hi,*current); day=a["day"][idx].astype(np.int64)-lo
    values=_grid(day,a["open"][idx],a["high"][idx],a["low"][idx],a["close"][idx],a["z"][idx],a["last"][idx],hi-lo,stops,targets,event)
    return pd.DataFrame(values,columns=METRIC_COLUMNS)

def aggregate(blocks:list[pd.DataFrame], exact:pd.DataFrame|None, indices:list[int], sessions:int,required_positive:int|None=None)->pd.DataFrame:
    out=(exact.copy() if exact is not None else blocks[indices[0]].copy()); stack=lambda c:np.stack([blocks[i][c].to_numpy(float) for i in indices])
    pnl=stack("net_pnl"); out["total_pnl"]=pnl.sum(0); out["mean_block_pnl"]=pnl.mean(0); out["median_block_pnl"]=np.median(pnl,axis=0)
    out["se_mean_block_pnl"]=pnl.std(0,ddof=1)/math.sqrt(len(indices)) if len(indices)>1 else 0.
    out["positive_blocks"]=(pnl>0).sum(0); out["agg_trades"]=stack("trades").sum(0); out["agg_active_days"]=stack("active_days").sum(0)
    out["agg_costs"]=stack("costs").sum(0); out["agg_clipped"]=stack("clipped_current_winner_net_usd").sum(0)
    if exact is None:
        # Fold risk is deliberately conservative from exact 21-session blocks.
        # Global 0:188 dominance uses one continuous kernel replay (exact pooled
        # trade CVaR and exact minute-stitched MTM drawdown).
        out["mtm_dd_usd"]=stack("mtm_dd_usd").max(0)
        out["pnl_over_dd"]=np.divide(out.total_pnl,out.mtm_dd_usd,out=np.zeros(len(out)),where=out.mtm_dd_usd.to_numpy()!=0)
        out["cvar5_loss_usd"]=stack("cvar5_loss_usd").max(0)
        out["worst_loss_usd"]=stack("worst_loss_usd").max(0)
        out["clipped_current_winner_net_usd"]=stack("clipped_current_winner_net_usd").sum(0)
    min_trades=max(8,math.ceil(50*sessions/PRE_END)); min_days=max(5,math.ceil(30*sessions/PRE_END)); majority=required_positive if required_positive is not None else math.floor(len(indices)/2)+1
    out["viable"]=(out.agg_trades>=min_trades)&(out.agg_active_days>=min_days)&(out.total_pnl>0)&(out.median_block_pnl>0)&(out.positive_blocks>=majority)
    obj=np.column_stack((out.total_pnl,out.mean_block_pnl,out.pnl_over_dd,-out.cvar5_loss_usd,-out.worst_loss_usd,-out.agg_costs,-out.agg_clipped))
    out["pareto"]=_pareto(obj,out.viable.to_numpy(bool)).astype(bool)
    return out

def neighbours(i:int,n_axis:int)->list[int]:
    r,c=divmod(i,n_axis); out=[]
    for dr in (-1,0,1):
        for dc in (-1,0,1):
            if not(dr or dc): continue
            rr,cc=r+dr,c+dc
            if rr>=0 and cc>=0 and rr<n_axis and cc<n_axis: out.append(rr*n_axis+cc)
    return out

def choose(frame:pd.DataFrame,n_axis:int)->tuple[int|None,pd.DataFrame,dict[str,Any]]:
    f=frame.copy(); stable=np.zeros(len(f),bool)
    cluster_cols=("mean_block_pnl","pnl_over_dd","cvar5_loss_usd","worst_loss_usd","agg_clipped")
    for col in cluster_cols: f["cluster_"+col]=np.nan
    f["viable_neighbor_count"]=0; f["neighbor_ids"]=""; f["viable_neighbor_ids"]=""
    for i in np.flatnonzero(f.pareto.to_numpy(bool)):
        ns=neighbours(int(i),n_axis)
        f.at[i,"neighbor_ids"]="|".join(map(str,ns))
        if len(ns)!=8: continue
        vn=[j for j in ns if bool(f.at[j,"viable"])]; f.at[i,"viable_neighbor_count"]=len(vn); f.at[i,"viable_neighbor_ids"]="|".join(map(str,vn))
        if len(vn)<5: continue
        stable[i]=True; cluster=[int(i)]+vn
        for col in cluster_cols: f.at[i,"cluster_"+col]=float(np.median(f.loc[cluster,col]))
    f["stable_pareto"]=stable
    ids=np.flatnonzero(stable)
    if not len(ids): return None,f,{"reason":"NO_STABLE_PARETO"}
    best=ids[np.argmax(f.loc[ids,"cluster_mean_block_pnl"].to_numpy())]
    threshold=float(f.at[best,"cluster_mean_block_pnl"]-f.at[best,"se_mean_block_pnl"])
    one=[int(i) for i in ids if f.at[i,"cluster_mean_block_pnl"]>=threshold]
    med_stop=float(np.median(f.loc[one,"stop_usd"])); med_target=float(np.median(f.loc[one,"target_usd"]))
    f["one_se"]=False; f.loc[one,"one_se"]=True
    ordered=sorted(one,key=lambda i:(-f.at[i,"cluster_pnl_over_dd"],f.at[i,"cluster_cvar5_loss_usd"],
        f.at[i,"cluster_worst_loss_usd"],f.at[i,"cluster_agg_clipped"],abs(f.at[i,"stop_usd"]-med_stop)+abs(f.at[i,"target_usd"]-med_target),f.at[i,"stop_usd"],f.at[i,"target_usd"]))
    chosen=ordered[0]; boundary=any(f.at[i,"stop_usd"]>=f.stop_usd.max()-STEP or f.at[i,"target_usd"]>=f.target_usd.max()-STEP for i in one)
    return chosen,f,{"reason":"SELECTED","one_se_threshold":threshold,"one_se_count":len(one),"plateau_touches_top2":boundary}

def near(frame:pd.DataFrame,i:int,j:int)->bool:
    return abs(frame.at[i,"stop_usd"]-frame.at[j,"stop_usd"])<=STEP+1e-9 and abs(frame.at[i,"target_usd"]-frame.at[j,"target_usd"])<=STEP+1e-9

def flat(a,lo,hi):
    idx=np.flatnonzero((a["day"]>=lo)&(a["day"]<hi)); ts=pd.DatetimeIndex(a["timestamp"])[idx]
    eq=pd.DataFrame({"timestamp":ts,"equity":CAPITAL,"running_peak":CAPITAL,"drawdown_usd":0.,"drawdown_pct":0.})
    return {"trades_df":pd.DataFrame(columns=TRADE_COLUMNS),"mtm_df":eq,"net_pnl":0.,"gross_pnl":0.,"costs":0.,"trades":0,"pnl_over_dd":None,
            "sessions":hi-lo,"raw_bars":len(idx),"final_equity":CAPITAL,"max_drawdown_usd_mtm":0.,"max_drawdown_pct_mtm":0.,"net_sharpe":0.,"profit_factor":0.}

def export_replay(path:Path,prefix:str,r:dict):
    atomic_csv(path/f"{prefix}_trades.csv",r["trades_df"]); atomic_csv(path/f"{prefix}_equity.csv",r["mtm_df"])

def progress(payload:dict): atomic_json(OUT/"progress.json",payload)

def run_symbol(symbol:str)->dict[str,Any]:
    dest=OUT/symbol; dest.mkdir(parents=True,exist_ok=True)
    old=json.loads((SOURCE/symbol/"summary.json").read_text(encoding="utf-8")); current=(float(old["selected"]["stop_usd"]),float(old["selected"]["target_usd"]))
    a,days,data=load_market(symbol); first63=a["close"][a["day"]<63]; median=float(np.median(first63)); cap=math.ceil(max(*current,.06*median)/STEP-1e-12)*STEP
    axis=np.arange(STEP,cap+STEP/2,STEP); stops=np.repeat(axis,len(axis)); targets=np.tile(axis,len(axis)); n_axis=len(axis)
    print(f"{symbol}: cap ${cap:.2f}, {len(stops):,} pairs",flush=True)
    block_frames=[]
    for bi,(lo,hi) in enumerate(BLOCKS):
        block_frames.append(evaluate(a,symbol,lo,hi,stops,targets,current)); print(f"  block {bi+1}/9",flush=True)
    exact=evaluate(a,symbol,0,PRE_END,stops,targets,current)
    agg=aggregate(block_frames,exact,list(range(9)),PRE_END,required_positive=6); chosen_idx,grid,meta=choose(agg,n_axis)
    current_idx=int(np.flatnonzero(np.isclose(stops,current[0])&np.isclose(targets,current[1]))[0]); grid["is_current"]=False; grid.at[current_idx,"is_current"]=True
    folds=[]; fold_choices={"anchored":[],"rolling":[]}
    for kind in ("anchored","rolling"):
        for j,test_i in enumerate(TEST_BLOCKS):
            train=list(range(0,test_i)) if kind=="anchored" else list(range(j,j+3)); train_lo=BLOCKS[train[0]][0]; train_hi=BLOCKS[train[-1]][1]; sessions=train_hi-train_lo
            train_exact=evaluate(a,symbol,train_lo,train_hi,stops,targets,current)
            train_hash=hashlib.sha256(train_exact.to_csv(index=False,float_format="%.10f").encode("utf-8")).hexdigest()
            train_agg=aggregate(block_frames,train_exact,train,sessions); fi,_,fm=choose(train_agg,n_axis); fold_choices[kind].append(fi)
            test=block_frames[test_i].iloc[fi] if fi is not None else None
            folds.append({"kind":kind,"fold":j+1,"train_blocks":"+".join(map(str,train)),"train_start_session":train_lo,"train_end_session_exclusive":train_hi,
                          "exact_train_metrics_sha256":train_hash,"test_block":test_i,
                          "chosen_stop_usd":None if fi is None else float(stops[fi]),"chosen_target_usd":None if fi is None else float(targets[fi]),
                          "test_net_pnl":None if test is None else float(test.net_pnl),"test_pnl_over_dd":None if test is None else float(test.pnl_over_dd),"reason":fm["reason"]})
    def counts(idx): return {k:sum(fi is not None and near(grid,idx,fi) for fi in v) for k,v in fold_choices.items()}
    current_counts=counts(current_idx); candidate_counts=counts(chosen_idx) if chosen_idx is not None else {"anchored":0,"rolling":0}
    # The 4/6 fold-neighbour gate applies to CHANGE only.  CURRENT needs the
    # frozen 0:188 viability rule; its fold counts remain diagnostics.
    current_robust=bool(grid.at[current_idx,"viable"])
    candidate_robust=bool(chosen_idx is not None and grid.at[chosen_idx,"viable"] and candidate_counts["anchored"]>=4 and candidate_counts["rolling"]>=4)
    dominance=False
    if chosen_idx is not None:
        c,b=grid.loc[chosen_idx],grid.loc[current_idx]; comps=(c.total_pnl>=b.total_pnl-1e-8,c.pnl_over_dd>=b.pnl_over_dd-1e-8,c.cvar5_loss_usd<=b.cvar5_loss_usd+1e-8,c.worst_loss_usd<=b.worst_loss_usd+1e-8)
        strict=(c.total_pnl>b.total_pnl+1e-8 or c.pnl_over_dd>b.pnl_over_dd+1e-8 or c.cvar5_loss_usd<b.cvar5_loss_usd-1e-8 or c.worst_loss_usd<b.worst_loss_usd-1e-8); dominance=all(comps) and strict
    boundary=bool(meta.get("plateau_touches_top2",False))
    if chosen_idx is not None and chosen_idx!=current_idx and candidate_robust and dominance and not boundary:
        verdict="CHANGE"; selected=(float(stops[chosen_idx]),float(targets[chosen_idx])); selected_idx=chosen_idx
    elif current_robust:
        verdict="KEEP_CURRENT" if not boundary else "BOUNDARY_UNRESOLVED_KEEP_CURRENT"; selected=current; selected_idx=current_idx
    else:
        verdict="NO_TRADE_NO_CONFIRMED_EDGE"; selected=None; selected_idx=None
    grid["selected_candidate"]=False
    if selected_idx is not None: grid.at[selected_idx,"selected_candidate"]=True
    grid["candidate_fold_anchored_count"]=candidate_counts["anchored"]; grid["candidate_fold_rolling_count"]=candidate_counts["rolling"]
    atomic_csv(dest/"development_grid.csv",grid); atomic_csv(dest/"block_metrics.csv",pd.concat([b.assign(block=i) for i,b in enumerate(block_frames)],ignore_index=True)); atomic_csv(dest/"folds.csv",pd.DataFrame(folds))
    selection_hashes={"engine_sha256":sha256(Path(__file__)),"development_grid_sha256":sha256(dest/"development_grid.csv"),
                      "block_metrics_sha256":sha256(dest/"block_metrics.csv"),"folds_sha256":sha256(dest/"folds.csv")}
    freeze={"symbol":symbol,"selection_sessions":[0,PRE_END],"seen_sessions_excluded":[PRE_END,SEEN_END],"seen_label":"SEEN_HISTORICAL_DIAGNOSTIC",
            "grid":{"median_first63":median,"cap_usd":cap,"step_usd":STEP,"pairs":len(grid)},"current":{"stop_usd":current[0],"target_usd":current[1]},
            "candidate":None if chosen_idx is None else {"stop_usd":float(stops[chosen_idx]),"target_usd":float(targets[chosen_idx])},
            "verdict":verdict,"selected":None if selected is None else {"stop_usd":selected[0],"target_usd":selected[1]},"selection_meta":meta,
            "metric_semantics":{"block_cvar5":"exact worst ceil(5%) closed trades inside each 21-session block",
                "pre_seen_cvar5":"exact pooled worst ceil(5%) trades from one continuous raw 0:188 replay",
                "block_mtm_dd":"exact minute-close MTM drawdown inside block from $100k reset",
                "pre_seen_mtm_dd":"exact minute-close MTM drawdown from one continuous raw 0:188 replay",
                "fold_risk":"exact pooled CVaR5 and continuous minute-MTM drawdown from a dedicated raw replay of that fold's train interval; no test/seen sessions"},
            "current_robust":current_robust,"candidate_robust":candidate_robust,"dominates_current":dominance,"current_fold_counts":current_counts,"candidate_fold_counts":candidate_counts,
            "selection_artifact_hashes":selection_hashes}
    atomic_json(dest/"pre_seen_freeze.json",freeze); freeze_hash=sha256(dest/"pre_seen_freeze.json"); atomic_text(dest/"pre_seen_freeze.sha256",freeze_hash+"\n")
    # Seen replay starts only after the immutable freeze/hash above exists.
    current_results={}; selected_results={}
    for name,(lo,hi) in {"pre_seen":(0,PRE_END),"seen":(PRE_END,SEEN_END),"full":(0,SEEN_END)}.items():
        cr=simulate(a,symbol,lo,hi,*current,collect=True); sr=flat(a,lo,hi) if selected is None else simulate(a,symbol,lo,hi,*selected,collect=True)
        current_results[name]=clean_replay(cr); selected_results[name]=clean_replay(sr)
        for item in (current_results[name],selected_results[name]):
            dd=float(item.get("max_drawdown_usd_mtm",0.0)); item["pnl_over_dd"]=float(item["net_pnl"])/dd if dd>0 else None
        export_replay(dest,f"current_{name}",cr); export_replay(dest,f"selected_{name}",sr)
    checks={"raw_sessions":len(days)==251,"selection_ends_188":freeze["selection_sessions"]==[0,188],"freeze_hash":sha256(dest/"pre_seen_freeze.json")==freeze_hash,
            "seen_after_freeze":True,"grid_pairs":len(grid)==n_axis*n_axis,"current_in_grid":bool(grid.is_current.sum()==1),
            "selected_flat_if_cash":selected is not None or (selected_results["full"]["trades"]==0 and selected_results["full"]["net_pnl"]==0),
            "current_additivity":abs(current_results["pre_seen"]["net_pnl"]+current_results["seen"]["net_pnl"]-current_results["full"]["net_pnl"])<1e-8,
            "selected_additivity":abs(selected_results["pre_seen"]["net_pnl"]+selected_results["seen"]["net_pnl"]-selected_results["full"]["net_pnl"])<1e-8}
    audit={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks}; atomic_json(dest/"audit.json",audit)
    if audit["status"]!="PASS": raise AssertionError(audit)
    selected_pre_dd=selected_results["pre_seen"].get("max_drawdown_usd_mtm",0.)
    selected_pre_ratio=None if selected is None or selected_pre_dd==0 else selected_results["pre_seen"]["net_pnl"]/selected_pre_dd
    summary={"schema_version":1,"symbol":symbol,"verdict":verdict,"selected":freeze["selected"],"current":freeze["current"],"data":data,"grid":freeze["grid"],
             "selection":freeze,"pre_seen_freeze_sha256":freeze_hash,"current_results":current_results,"selected_results":selected_results,
             "selected_pre_seen_pnl_over_dd":selected_pre_ratio,
             "seen_warning":"Sessions 188:251 are SEEN_HISTORICAL_DIAGNOSTIC and never rank/gate candidates","audit":audit}
    atomic_json(dest/"summary.json",summary)
    row={"symbol":symbol,"verdict":verdict,"selected_stop_usd":None if selected is None else selected[0],"selected_target_usd":None if selected is None else selected[1],
         "current_stop_usd":current[0],"current_target_usd":current[1],"freeze_sha256":freeze_hash,"audit":"PASS"}
    for variant,results in (("selected",selected_results),("current",current_results)):
        for period,item in results.items():
            for key in ("trades","net_pnl","max_drawdown_usd_mtm","max_drawdown_pct_mtm","pnl_over_dd","net_sharpe","profit_factor","costs"):
                row[f"{variant}_{period}_{key}"]=item.get(key)
    return row

def main():
    if sys.platform=="win32": sys.stdout.reconfigure(encoding="utf-8"); sys.stderr.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True,exist_ok=True); completed=[]; progress({"status":"RUNNING","completed":completed,"remaining":list(UNIVERSE)})
    for symbol in UNIVERSE:
        row=run_symbol(symbol); completed.append(row); atomic_csv(OUT/"cross_asset_summary.csv",pd.DataFrame(completed)); atomic_json(OUT/"cross_asset_summary.json",completed)
        progress({"status":"RUNNING","completed":completed,"remaining":[x for x in UNIVERSE if x not in {r['symbol'] for r in completed}]}); print(f"DONE {symbol}: {row['verdict']}",flush=True)
    progress({"status":"COMPLETE","completed":completed,"remaining":[]}); print(json.dumps(completed,ensure_ascii=False,indent=2),flush=True)

if __name__=="__main__": main()
