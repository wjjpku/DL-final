#!/usr/bin/env python3
"""Leave-one-out across all 9 schedules: does MPL predict a held-out SHARP-decay
curve when it HAS seen other sharp decays? Decides formula-limit vs info-limit."""
import sys; sys.path.insert(0,'repro')
import numpy as np
from scipy.signal import savgol_filter
from reproduce_cosine_to_wsd import load_curve, MPL_PRECOMPUTED_INIT
from validate_theory import fit_mpl, mpl_pred, F_MPL, metrics
ALL9=["cosine_24000.csv","cosine_72000.csv","constant_24000.csv","constant_72000.csv",
      "wsd_20000_24000.csv","wsdld_20000_24000.csv","wsdcon_3.csv","wsdcon_9.csv","wsdcon_18.csv"]
def nf(c):
    y=c.loss; w=min(31,(len(y)//2)*2-1)
    return float(np.mean(np.abs(y-savgol_filter(y,max(w,5),3)))) if len(y)>=7 else 0
def decay_mae(p,c):
    m=c.step>=20000
    return float(np.mean(np.abs(c.loss[m]-mpl_pred(p,c,fast=False)[m]))) if m.any() else np.nan
print("Leave-one-out: 训练其余8条, 预测留出1条 (per scale)")
print(f"{'scale':>5} {'held-out':>22} {'test MAE':>9} {'decay MAE':>9} {'noise':>8} {'结构?':>6}")
for s in ["100","400"]:
    for held in ["wsd_20000_24000.csv","wsdld_20000_24000.csv","cosine_72000.csv","wsdcon_3.csv"]:
        tr=[load_curve(s,n) for n in ALL9 if n!=held]
        p=fit_mpl(tr, np.array(MPL_PRECOMPUTED_INIT[s],float), F_MPL)
        c=load_curve(s,held); e=metrics(c.loss,mpl_pred(p,c,fast=False))["mae"]; d=decay_mae(p,c); n=nf(c)
        tag="" 
        if held.startswith(("wsd_","wsdld")):
            tag="结构化" if d/max(n,1e-9)>2 else "噪声底"
        print(f"{s+'M':>5} {held:>22} {e:>9.5f} {d if not np.isnan(d) else 0:>9.5f} {n:>8.5f} {tag:>6}")
