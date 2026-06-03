#!/usr/bin/env python3
"""Is MPL's residual structured (improvable) or at the data noise floor?
Compare MPL prediction error vs an intrinsic noise-floor estimate (savgol)."""
import sys; sys.path.insert(0,'repro')
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from reproduce_cosine_to_wsd import load_curve, huber_log_residual, metrics, MPL_PRECOMPUTED_INIT
from validate_theory import ld_matrix, _coarse, mpl_pred, fit_mpl, F_MPL
SCALES=["25","100","400"]; COS=["cosine_24000.csv","cosine_72000.csv"]
WSD=["wsd_20000_24000.csv","wsdld_20000_24000.csv","wsdcon_3.csv","wsdcon_9.csv","wsdcon_18.csv"]

def noise_floor(curve):
    y=curve.loss
    n=len(y); w=min(31, n if n%2 else n-1)
    if w<5: return 0.0
    w=w if w%2 else w-1
    sm=savgol_filter(y, w, 3)                     # smooth trend (any real structure)
    return float(np.mean(np.abs(y-sm)))          # residual = intrinsic jitter

print("拟合 MPL on cosine, 比较 [MPL预测误差] vs [数据噪声底]（WSD曲线）")
print(f"{'scale':>5} {'curve':>22} {'noise_floor':>11} {'MPL_err':>9} {'比值':>6} {'结构?':>6}")
for s in SCALES:
    cur=[load_curve(s,n) for n in COS]
    p=fit_mpl(cur, np.array(MPL_PRECOMPUTED_INIT[s],float), F_MPL)
    for n in WSD:
        c=load_curve(s,n)
        nf=noise_floor(c)
        err=metrics(c.loss, mpl_pred(p,c,fast=False))["mae"]
        ratio=err/max(nf,1e-9)
        tag="结构化" if ratio>2.0 else ("噪声底" if ratio<1.3 else "接近")
        print(f"{s+'M':>5} {n:>22} {nf:>11.5f} {err:>9.5f} {ratio:>6.1f} {tag:>6}")
