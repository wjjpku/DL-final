#!/usr/bin/env python3
"""Is the DERIVED kernel form better than a generic decaying correction?

Fair few-shot (train = cosine x2 + 1 WSD, both arms refit MPL, test on the other 4 WSD).
We add, on top of the refit MPL, one extra term of each form (each with its own fitted
amplitude kappa; shape constant fixed at its best value), and compare held-out MAE:

  derived   : S-time exp of DROPS      sum (eta_{t-1}-eta_t)_+ exp(-lam (S_t-S_t'))   [ours]
  steptime  : step-time exp of drops   (wrong 'time' variable)
  floor     : (eta_peak - eta_t)_+     (no memory; the naive instantaneous floor)
  level     : S-time exp of LEVEL      relu(EMA_S(eta)-eta)        (drives on level not drops)

If 'derived' wins, the functional form matters -- it is not an arbitrary decaying term.
"""
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import lfilter

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import load_curve, huber_log_residual, metrics, MPL_PRECOMPUTED_INIT, SCALES, PEAK_LR
from validate_theory import fit_mpl, mpl_pred, F_MPL

COS = ["cosine_24000.csv", "cosine_72000.csv"]
NONCOS = ["wsd_20000_24000.csv", "wsdld_20000_24000.csv",
          "wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"]
LAM = 10.0


def feat(curve, kind):
    eta = curve.lrs.astype(np.float64)
    drop = np.zeros_like(eta); drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
    if kind == "derived":            # S-time exp of drops
        s = np.empty_like(eta); acc = 0.0
        for t in range(len(eta)):
            acc = acc * np.exp(-LAM * eta[t]) + drop[t]; s[t] = acc
        f = s / PEAK_LR
    elif kind == "steptime":         # step-time exp of drops (tau=1200)
        a = 1.0 - 1.0 / 1200.0
        s, _ = lfilter([1.0], [1.0, -a], drop, zi=[0.0]); f = s / PEAK_LR
    elif kind == "floor":            # instantaneous floor, no memory
        f = np.clip(PEAK_LR - eta, 0.0, None) / PEAK_LR
    elif kind == "level":            # S-time EMA of LEVEL minus eta
        # tilde_eta relaxes in S-time toward eta; drive on (tilde-eta)_+
        te = np.empty_like(eta); acc = PEAK_LR
        for t in range(len(eta)):
            acc = acc + LAM * eta[t] * (eta[t] - acc); te[t] = acc
        f = np.clip(te - eta, 0.0, None) / PEAK_LR
    return f[curve.step]


def fit_kappa(p_mpl, train, kind):
    base = [mpl_pred(p_mpl, c, fast=True) for c in train]
    F = [feat(c, kind) for c in train]
    la = np.concatenate([c.loss for c in train])
    obj = lambda k: 1e18 if k < 0 else huber_log_residual(
        la, np.concatenate([b + k * f for b, f in zip(base, F)]))
    return float(minimize_scalar(obj, bounds=(0.0, 1e4), method="bounded", options={"xatol": 1e-4}).x)


def main():
    kinds = ["derived", "steptime", "floor", "level"]
    print("=" * 70)
    print("Fair few-shot: derived kernel vs generic decaying corrections (held-out MAE)")
    print("=" * 70)
    mpl_all = []
    res = {k: [] for k in kinds}
    for seed in NONCOS:
        trn = COS + [seed]; tst = [x for x in NONCOS if x != seed]
        for s in SCALES:
            train = [load_curve(s, x) for x in trn]
            p = fit_mpl(train, MPL_PRECOMPUTED_INIT[s], F_MPL)
            for hc in (load_curve(s, x) for x in tst):
                mpl_all.append(metrics(hc.loss, mpl_pred(p, hc, fast=False))["mae"])
            for k in kinds:
                kap = fit_kappa(p, train, k)
                for hc in (load_curve(s, x) for x in tst):
                    res[k].append(metrics(hc.loss, mpl_pred(p, hc, fast=False) + kap * feat(hc, k))["mae"])
    mpl_all = np.array(mpl_all)
    print(f"  baseline MPL (refit) held-out mean MAE = {mpl_all.mean():.5f}\n")
    print(f"  {'correction form':12s} {'MAE':>9s} {'vs MPL':>8s} {'wins':>7s}")
    print(f"  {'(none=MPL)':12s} {mpl_all.mean():9.5f} {'  0.0%':>8s}")
    for k in kinds:
        m = np.array(res[k]); d = (m.mean()/mpl_all.mean()-1)*100
        tag = " <<" if k == "derived" else ""
        print(f"  {k:12s} {m.mean():9.5f} {d:+7.1f}% {int((m<mpl_all).sum()):3d}/{len(m):<2d}{tag}")
    print("\n  derived = S-time exp of drops (lambda_slow=10). The 'time' variable (S not")
    print("  steps), the drive (drops not level), and the no-memory floor all matter.")


if __name__ == "__main__":
    main()
