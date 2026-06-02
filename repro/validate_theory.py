#!/usr/bin/env python3
"""Comprehensive validation of the SGD-spectrum theory (docs/core/derivation.md).

Runs the full battery of falsifiable checks on the local MPL data (25/100/400M x
9 schedules), offline:

  E-INV   exponent invariance (P2/P3): per-scale fits, CV of every parameter.
  E-GAMMA gamma=0 ablation (P1): clean kernel vs full, train & test.
  E-NOISE noise floor ∝ eta (P4): wsdcon_{3,9,18} final loss vs stage-2 LR.
  E-WIN   SC-MPL vs MPL/Tissue PER CURVE (not just averaged), both scales.
  E-SRC   robustness to the shared-shape source (25 vs 100 vs 25+100).
  E-SPLIT robustness to train/test split (official MPL split vs cosine-only).

Speed: the annealing term LD is vectorised (ld_matrix) with a coarse decrement
grid; validated against the exact repo implementation before use.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    Curve, load_curve, compute_s1, compute_ld, huber_log_residual, metrics,
    MPL_PRECOMPUTED_INIT, TRAIN_CURVES, TEST_CURVES,
)
from sc_mpl_transfer import tissue_fast, fit_tissue_fast  # noqa: E402

ROOT = REPO.parent
OUT = ROOT / "results" / "validate_theory"
SCALES = ["25", "100", "400"]
NVAL = {"25": 25.0, "100": 100.0, "400": 400.0}
ORDER = ["L0", "A", "alpha", "B", "C", "beta", "gamma"]
ALL_CURVES = TRAIN_CURVES + ["constant_24000.csv"] + TEST_CURVES
# Physically-motivated bounds: alpha,beta = O(1) spectral exponents, C=2*lambda_0=O(1),
# gamma = small dynamical exponent. Bounds bracket the published MPL fits and exclude
# the degenerate gamma->large / beta->0 basin.
FULL_B = [(0.0, 10.0), (1e-8, 100.0), (0.05, 1.5), (1e-8, 1e6),
          (0.3, 6.0), (0.1, 1.5), (0.1, 1.5)]
FIT_NK = 1000  # fine grid (<0.3% LD err) for fitting; metrics use exact LD
FIT_PTS = 90   # log-spaced output points used during fitting (speed; metrics use full curve)


def _coarse(curve, npts=FIT_PTS):
    n = len(curve.step)
    if n <= npts:
        return curve
    idx = np.unique(np.round(np.geomspace(1, n, npts)).astype(int) - 1)
    idx = idx[(idx >= 0) & (idx < n)]
    from reproduce_cosine_to_wsd import Curve as _C
    return _C(curve.name, curve.scale, curve.step[idx], curve.loss[idx], curve.lrs)


# ----------------------------- fast vectorised LD -----------------------------
def ld_matrix(curve: Curve, C, beta, gamma, nk=500):
    """Vectorised LD at curve.step. Coarse decrement grid (nk nodes) approximates
    the smooth decrement integral; nk large -> exact repo LD."""
    lrs = curve.lrs.astype(np.float64)
    T = len(lrs)
    S = np.cumsum(lrs)
    kj = np.unique(np.linspace(1, T - 1, min(nk, T - 1)).astype(int))
    a = lrs[kj - 1] - lrs[kj]            # NOTE repo uses lr_gap=diff; sum of Δη over a cell
    # aggregate decrement over each cell [prev_node, node]: total drop = lrs[prev]-lrs[node]
    prev = np.concatenate([[0], kj[:-1]])
    a = lrs[prev] - lrs[kj]
    b = C * np.power(lrs[kj], -gamma)
    ref = S[prev]                         # remaining-progress reference
    sv = curve.step
    Ss = S[sv]
    gap = Ss[:, None] - ref[None, :]
    mask = (kj[None, :] <= sv[:, None]) & (gap > 0)
    kernel = 1.0 - np.power(1.0 + b[None, :] * np.maximum(gap, 0.0), -beta)
    return np.sum(np.where(mask, -a[None, :] * kernel, 0.0), axis=1)


def mpl_pred(p, curve, fast=True, nk=500):
    L0, A, alpha, B, C, beta, gamma = p
    s1 = compute_s1(curve)
    ld = ld_matrix(curve, C, beta, gamma, nk) if fast else compute_ld(curve, C, beta, gamma)
    return L0 + A * np.power(s1, -alpha) + B * ld


# ------------------------------- validation cell ------------------------------
def _validate_ld():
    c = load_curve("100", "wsd_20000_24000.csv")
    p = np.array([2.65, 0.60, 0.45, 437.9, 2.13, 0.598, 0.655])
    exact = compute_ld(c, p[4], p[5], p[6])
    for nk in (250, 500, 1000):
        approx = ld_matrix(c, p[4], p[5], p[6], nk)
        rel = np.max(np.abs(approx - exact)) / (np.max(np.abs(exact)) + 1e-12)
        print(f"  ld_matrix nk={nk}: max rel err vs exact = {rel:.4f}")


# ------------------------------- fitting helpers ------------------------------
def _obj(p, curves, nk):
    pr, ys = [], []
    for c in curves:
        v = mpl_pred(p, c, fast=True, nk=nk)
        if not np.all(np.isfinite(v)) or np.any(v <= 0):
            return 1e18
        pr.append(v); ys.append(c.loss)
    return huber_log_residual(np.concatenate(ys), np.concatenate(pr))


def fit_mpl(curves, init7, free_idx, nk=FIT_NK):
    free = sorted(free_idx)
    init7 = np.array(init7, float)
    bounds = [FULL_B[i] for i in free]
    fc = [_coarse(c) for c in curves]

    def asm(pf):
        f = init7.copy(); f[free] = pf; return f

    b = init7[free]
    inits = [b, b * 0.7, b * 1.3, b * 0.85, b * 1.5]
    best, bf = None, np.inf
    for x0 in inits:
        r = minimize(lambda pf: _obj(asm(pf), fc, nk), x0, method="L-BFGS-B",
                     bounds=bounds, options={"maxiter": 200, "ftol": 1e-10})
        if r.fun < bf:
            bf, best = r.fun, r.x
    return asm(best)


def mae_on(p, scale, names):
    """Exact-LD test/train MAE per curve."""
    out = {}
    for n in names:
        c = load_curve(scale, n)
        out[n] = metrics(c.loss, mpl_pred(p, c, fast=False))["mae"]
    return out


def shared_shape(fitted, srcs):
    """Mean of fitted {alpha,C,beta,gamma} over source scales (the small-model statistics)."""
    arr = np.array([fitted[s][[2, 4, 5, 6]] for s in srcs])
    return arr.mean(axis=0)


def honest_init(fitted, target, srcs):
    near = max(srcs, key=lambda s: NVAL[s])
    base = fitted[near].copy()
    base[[2, 4, 5, 6]] = shared_shape(fitted, srcs)
    return base


F_MPL = [0, 1, 2, 3, 4, 5, 6]
F_SC3 = [0, 1, 3]            # share alpha,C,beta,gamma -> fit {L0,A,B}
F_SC4 = [0, 1, 2, 3]        # share C,beta,gamma       -> fit {L0,A,alpha,B}

# Official MPL public split (the params in MPL_PRECOMPUTED_INIT were fit on this).
OFF_TRAIN = ["cosine_24000.csv", "constant_24000.csv", "wsdcon_9.csv"]
OFF_TEST = ["constant_72000.csv", "cosine_72000.csv", "wsd_20000_24000.csv",
            "wsdld_20000_24000.csv", "wsdcon_3.csv", "wsdcon_18.csv"]


# =============================== experiments ==================================
def official_params():
    """Trustworthy per-scale MPL fits (published). Re-fitting the 7-param MPL is
    unstable in {beta,gamma,C} (degenerate objective), so we anchor on these and
    only ever re-fit the STABLE amplitudes {L0,A,B}."""
    return {s: np.array(MPL_PRECOMPUTED_INIT[s], float) for s in SCALES}


def e_inv(fitted):
    print("\n=== E-INV: 指数不变性 (P2/P3) ===")
    print(f"{'param':>6} " + " ".join(f"{s+'M':>9}" for s in SCALES) + f" {'CV%':>7}  role")
    rows = {}
    for i, name in enumerate(ORDER):
        v = [fitted[s][i] for s in SCALES]
        cv = 100 * np.std(v) / abs(np.mean(v))
        role = "exponent(shape)" if name in ("alpha", "C", "beta", "gamma") else "amplitude"
        print(f"{name:>6} " + " ".join(f"{x:9.4f}" for x in v) + f" {cv:7.1f}  {role}")
        rows[name] = dict(vals=v, cv=cv, role=role)
    return rows


def e_gamma(fitted):
    """P1: isolate gamma. Hold exponents at official; compare official-gamma vs gamma=0,
    each given its BEST amplitude {L0,A,B} fit (stable). Official split."""
    print("\n=== E-GAMMA: η^(−γ) 项是否必要 (P1) — 仅重拟振幅,隔离 γ ===")
    print(f"{'scale':>6} {'full_tr':>8} {'g0_tr':>8} | {'full_te':>8} {'g0_te':>8} | {'γ(official)':>11}")
    rows = []
    for s in SCALES:
        cv = [load_curve(s, n) for n in OFF_TRAIN]
        full = fit_mpl(cv, fitted[s], F_SC3)               # amplitudes given official exponents
        base0 = fitted[s].copy(); base0[6] = 1e-4
        g0 = fit_mpl(cv, base0, F_SC3)                     # amplitudes given gamma=0
        ftr = np.mean(list(mae_on(full, s, OFF_TRAIN).values()))
        gtr = np.mean(list(mae_on(g0, s, OFF_TRAIN).values()))
        fte = np.mean(list(mae_on(full, s, OFF_TEST).values()))
        gte = np.mean(list(mae_on(g0, s, OFF_TEST).values()))
        print(f"{s+'M':>6} {ftr:8.5f} {gtr:8.5f} | {fte:8.5f} {gte:8.5f} | {fitted[s][6]:11.3f}")
        rows.append(dict(scale=s, full_train=ftr, g0_train=gtr, full_test=fte, g0_test=gte))
    return rows


def e_noise(fitted):
    """P4: noise floor ∝ eta. wsdcon_{3,9,18} differ only in stage-2 LR (3,9,18)e-5."""
    print("\n=== E-NOISE: 噪声地板 ∝ η (P4) — wsdcon 末端 loss vs stage-2 LR ===")
    rows = []
    for s in SCALES:
        finals = {}
        for lr in (3, 9, 18):
            c = load_curve(s, f"wsdcon_{lr}.csv")
            finals[lr] = float(c.loss[-1])
        mono = finals[3] < finals[9] < finals[18]
        # slope of final loss vs eta (e-5 units); theory: positive, ~linear
        etas = np.array([3, 9, 18.0]); ys = np.array([finals[3], finals[9], finals[18]])
        slope = np.polyfit(etas, ys, 1)[0]
        # linearity R^2
        pred = np.polyval(np.polyfit(etas, ys, 1), etas)
        r2 = 1 - np.sum((ys - pred) ** 2) / np.sum((ys - ys.mean()) ** 2)
        print(f"{s+'M':>6} final loss: η3={finals[3]:.4f} η9={finals[9]:.4f} η18={finals[18]:.4f}"
              f" | 单调递增={mono} 线性R²={r2:.3f} slope>0={slope>0}")
        rows.append(dict(scale=s, finals=finals, monotonic=mono, linear_r2=r2, slope=slope))
    return rows


def e_win(fitted):
    """Per-curve SC vs official MPL vs Tissue on the official split.
    MPL baseline = published per-scale params (its best, stable optimum).
    SC = transfer exponents from SMALLER scales, fit only amplitudes (stable)."""
    print("\n=== E-WIN: 逐曲线对比(官方划分)— SC(振幅拟合) vs 官方MPL(7p) vs Tissue(5p) ===")
    rows = []
    for t in ["100", "400"]:
        srcs = [s for s in SCALES if NVAL[s] < NVAL[t]]
        cv = [load_curve(t, n) for n in OFF_TRAIN]
        mpl = fitted[t]                                    # official, no refit
        base = honest_init(fitted, t, srcs)
        sc3 = fit_mpl(cv, base, F_SC3)                     # fit {L0,A,B}, exponents transferred
        sc4 = fit_mpl(cv, base, F_SC4)                     # fit {L0,A,alpha,B}
        tip, _ = fit_tissue_fast(cv)
        m_mpl = mae_on(mpl, t, OFF_TEST)
        m_sc3 = mae_on(sc3, t, OFF_TEST)
        m_sc4 = mae_on(sc4, t, OFF_TEST)
        m_ti = {n: metrics(load_curve(t, n).loss, tissue_fast(tip, load_curve(t, n)))["mae"] for n in OFF_TEST}
        w3 = sum(m_sc3[n] <= m_mpl[n] for n in OFF_TEST)
        w4 = sum(m_sc4[n] <= m_mpl[n] for n in OFF_TEST)
        a = lambda d: np.mean(list(d.values()))
        print(f"  [{t}M] 均值 test MAE: Tissue={a(m_ti):.5f} MPL={a(m_mpl):.5f} "
              f"SC-3={a(m_sc3):.5f} SC-4={a(m_sc4):.5f}")
        print(f"        逐曲线胜 MPL: SC-3 {w3}/{len(OFF_TEST)}, SC-4 {w4}/{len(OFF_TEST)}")
        for n in OFF_TEST:
            mk = "✓" if m_sc4[n] <= m_mpl[n] else "✗"
            print(f"     {n:22} Tissue={m_ti[n]:.5f} MPL={m_mpl[n]:.5f} SC4={m_sc4[n]:.5f} {mk}")
        rows.append(dict(target=t, tissue=a(m_ti), mpl=a(m_mpl), sc3=a(m_sc3), sc4=a(m_sc4),
                         win3=f"{w3}/{len(OFF_TEST)}", win4=f"{w4}/{len(OFF_TEST)}"))
    return rows


def e_src(fitted):
    """Robustness to which small scales supply the shared shape (400M target)."""
    print("\n=== E-SRC: 共享形状来源稳健性 (400M, 官方划分) ===")
    cv = [load_curve("400", n) for n in OFF_TRAIN]
    mpl_ref = np.mean(list(mae_on(fitted["400"], "400", OFF_TEST).values()))
    rows = []
    for srcs in (["25"], ["100"], ["25", "100"]):
        base = honest_init(fitted, "400", srcs)
        sc4 = fit_mpl(cv, base, F_SC4)
        te = np.mean(list(mae_on(sc4, "400", OFF_TEST).values()))
        print(f"  形状来自 {'+'.join(srcs)+'M':>9}: SC-4 test MAE = {te:.5f}  (官方MPL={mpl_ref:.5f})")
        rows.append(dict(src=srcs, test=te))
    return rows


def e_split(fitted):
    """Robustness to the train set: refit SC amplitudes on the COSINE-ONLY train
    instead of the official 3-curve train; compare to official MPL on WSD test."""
    print("\n=== E-SPLIT: 换训练集稳健性 (仅 cosine 训练 -> WSD 测试) ===")
    rows = []
    for t in ["100", "400"]:
        srcs = [s for s in SCALES if NVAL[s] < NVAL[t]]
        cv = [load_curve(t, n) for n in TRAIN_CURVES]      # cosine-only train
        base = honest_init(fitted, t, srcs)
        sc4 = fit_mpl(cv, base, F_SC4)
        m = np.mean(list(mae_on(fitted[t], t, TEST_CURVES).values()))   # official MPL
        sc = np.mean(list(mae_on(sc4, t, TEST_CURVES).values()))
        print(f"  [{t}M] WSD test MAE: 官方MPL={m:.5f}  SC-4(仅cosine训练)={sc:.5f}  {'SC赢' if sc < m else 'MPL赢'}")
        rows.append(dict(target=t, mpl=m, sc4=sc))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== validate vectorised LD against exact ===")
    _validate_ld()
    fitted = official_params()
    out = {
        "E_INV": e_inv(fitted),
        "E_GAMMA": e_gamma(fitted),
        "E_NOISE": e_noise(fitted),
        "E_WIN": e_win(fitted),
        "E_SRC": e_src(fitted),
        "E_SPLIT": e_split(fitted),
        "fitted_params": {s: fitted[s].tolist() for s in SCALES},
    }
    (OUT / "validation.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nsaved -> {OUT/'validation.json'}")


if __name__ == "__main__":
    main()
