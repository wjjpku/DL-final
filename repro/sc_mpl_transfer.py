#!/usr/bin/env python3
"""SC-MPL: Scale-Conditioned Multi-Power Law — small->large transfer experiments.

Premise (see docs/core/scaling_law_theory.md): in MPL's 7 parameters, the
annealing-kernel *shape* (C, beta, gamma) is a scale-invariant constant of the
optimization dynamics (CV ~1-2% over 25M->400M), while only the amplitudes
(L0, A, B) and weakly alpha drift with model size N.

SC-MPL therefore *shares* {C, beta, gamma} across scales (supplied by cheap small
models = the auxiliary statistics) and fits only the amplitudes on the target.
This trades parameters for inductive bias — the right direction for a scaling law.

Three falsifiable experiments on the local MPL data (3 scales x 9 schedules), all
offline:
  1. same-sample, fewer params : SC-MPL(4p) vs MPL(7p) test MAE on unseen WSD.
  2. few-shot (short prefix)    : fit on first p% of the cosine curve; who degrades slower.
  3. zero-shot cross-scale      : predict 400M from {25M,100M} amplitude-scaling, no 400M data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402  (reuse tested formula code)
    Curve, load_curve, mpl_predict, tissue_predict, fit_tissue, huber_log_residual,
    metrics, fit_with_restarts, subsample_curve, MPL_PRECOMPUTED_INIT,
    TRAIN_CURVES, TEST_CURVES,
)

ROOT = REPO.parent
OUT = ROOT / "results" / "sc_mpl_transfer"
SCALES = ["25", "100", "400"]
NVAL = {"25": 25.0, "100": 100.0, "400": 400.0}      # proxy for params (in M); trend is in log N
FIT_STRIDE = 16                                       # subsample steps for fitting speed
SHAPE_IDX = (4, 5, 6)                                 # C, beta, gamma in the 7-vector
AMP_IDX = (0, 1, 2, 3)                                # L0, A, alpha, B

# ---- bounds ----
FULL_BOUNDS = [(0.0, 10.0), (1e-8, 100.0), (1e-4, 3.0), (1e-8, 1e6),
               (1e-4, 1e3), (1e-4, 3.0), (1e-4, 3.0)]
AMP_BOUNDS = [(0.0, 10.0), (1e-8, 100.0), (1e-4, 3.0), (1e-8, 1e6)]


def load(scale, names):
    return [load_curve(scale, n) for n in names]


# --- fast Tissue (vectorize S2's IIR recurrence with lfilter; same formula as repo) ---
def tissue_fast(params, curve):
    L0, A, alpha, C, lam = params
    eta = curve.lrs.astype(float)
    delta = np.zeros_like(eta)
    delta[1:] = eta[:-1] - eta[1:]              # eta_{t-1} - eta_t
    anneal = lfilter([1.0], [1.0, -lam], delta)  # anneal[t] = lam*anneal[t-1] + delta[t]
    s2 = np.cumsum(anneal)[curve.step]
    s1 = np.cumsum(eta)[curve.step]
    return L0 + A * np.power(s1, -alpha) - C * s2


def fit_tissue_fast(curves, stride=FIT_STRIDE):
    fc = [subsample_curve(c, stride) for c in curves]
    min_loss = min(float(c.loss.min()) for c in curves)
    inits = []
    for lam in (0.99, 0.995, 0.997, 0.999):     # repo's 12-init grid for a fair Tissue fit
        inits.append(np.array([min_loss - 0.05, 0.5, 0.5, 100.0, lam]))
        inits.append(np.array([min_loss - 0.1, 1.0, 0.4, 10.0, lam]))
        inits.append(np.array([min_loss, 0.2, 0.7, 300.0, lam]))
    bounds = [(0.0, 10.0), (1e-8, 100.0), (1e-4, 3.0), (1e-8, 1e5), (0.9, 0.9999)]

    def obj(p):
        preds, ys = [], []
        for c in fc:
            pr = tissue_fast(p, c)
            if not np.all(np.isfinite(pr)) or np.any(pr <= 0):
                return 1e18
            preds.append(pr); ys.append(c.loss)
        return huber_log_residual(np.concatenate(ys), np.concatenate(preds))

    return fit_with_restarts(obj, inits, bounds)


def prefix(curve: Curve, frac: float) -> Curve:
    """Keep only the first `frac` fraction of observed steps (short-prefix few-shot)."""
    n = max(5, int(len(curve.step) * frac))
    return Curve(curve.name, curve.scale, curve.step[:n], curve.loss[:n], curve.lrs)


def _objective(params7, fit_curves):
    preds, ys = [], []
    for c in fit_curves:
        pred = mpl_predict(params7, c)
        if not np.all(np.isfinite(pred)) or np.any(pred <= 0):
            return 1e18
        preds.append(pred)
        ys.append(c.loss)
    return huber_log_residual(np.concatenate(ys), np.concatenate(preds))


def fit_partial(curves, init7, free_idx, stride=FIT_STRIDE):
    """Fit only parameters in `free_idx`; freeze the rest at init7.

    free_idx = all 7        -> plain MPL baseline
    free_idx = {0,1,2,3}    -> SC-MPL-4 (freeze shape C,beta,gamma from small models)
    free_idx = {0,1,3}      -> SC-MPL-3 (also freeze alpha)
    free_idx = {0}          -> 1-param calibration (only L0; everything else from small)
    """
    init7 = np.asarray(init7, float)
    free_idx = sorted(free_idx)
    fc = [subsample_curve(c, stride) for c in curves]
    bounds = [FULL_BOUNDS[i] for i in free_idx]

    def assemble(pf):
        full = init7.copy()
        full[free_idx] = pf
        return full

    base = init7[free_idx]
    # many diverse restarts so the comparison reflects the FORMULA, not optimizer luck
    inits = [base]
    for f in (0.7, 0.85, 1.15, 1.3):
        inits.append(base * f)
    for _ in range(6):
        inits.append(base * np.exp(0.4 * (np.arange(len(base)) % 3 - 1) * (1 + 0.3 * len(inits))))
    rng_scales = [0.5, 0.75, 1.25, 1.5, 2.0]
    for rs in rng_scales:
        jitter = base.copy(); jitter[::2] = jitter[::2] * rs
        inits.append(jitter)
    best, fun = fit_with_restarts(lambda pf: _objective(assemble(pf), fc), inits, bounds)
    return assemble(best), fun


def fit_mpl(curves, init7, stride=FIT_STRIDE):
    return fit_partial(curves, init7, [0, 1, 2, 3, 4, 5, 6], stride)


def test_mae(params7, scale):
    maes = []
    for n in TEST_CURVES:
        c = load_curve(scale, n)
        maes.append(metrics(c.loss, mpl_predict(params7, c))["mae"])
    return float(np.mean(maes)), maes


def train_mae(params7, scale):
    maes = [metrics(c.loss, mpl_predict(params7, c))["mae"] for c in load(scale, TRAIN_CURVES)]
    return float(np.mean(maes))


def frozen_init7(target, src, free_idx):
    """init7 whose FROZEN indices = small-scale mean, FREE indices = target's own init."""
    init7 = np.array(MPL_PRECOMPUTED_INIT[target], float).copy()
    small = np.array([MPL_PRECOMPUTED_INIT[s] for s in src]).mean(axis=0)
    frozen = [i for i in range(7) if i not in free_idx]
    init7[frozen] = small[frozen]
    return init7


def honest_init7(target, src):
    """HONEST init using only SMALLER models (no target oracle): amplitudes from the
    nearest smaller scale, exponents {alpha,C,beta,gamma} from the small-scale mean."""
    near = max(src, key=lambda s: NVAL[s])
    base = np.array(MPL_PRECOMPUTED_INIT[near], float).copy()
    small = np.array([MPL_PRECOMPUTED_INIT[s] for s in src]).mean(axis=0)
    base[[2, 4, 5, 6]] = small[[2, 4, 5, 6]]   # alpha, C, beta, gamma
    return base


# free-index presets
F_MPL = [0, 1, 2, 3, 4, 5, 6]   # MPL: all 7
F_SC4 = [0, 1, 2, 3]            # SC-MPL-4: freeze {C,beta,gamma}
F_SC3 = [0, 1, 3]              # SC-MPL-3: also freeze alpha
F_CAL1 = [0]                   # 1-param calibration: only L0


# ============================ experiments ============================

def exp1_same_sample():
    """Same training data, fewer params: MPL(7) vs SC-MPL-4 vs SC-MPL-3, test MAE on unseen WSD."""
    print("\n=== Exp1: 同样本、更少参数 (train=cosine, test=WSD) ===")
    print(f"{'target':>7} {'method':>9} {'#p':>3} {'train':>9} {'test':>9}  shape_src")
    print("(诚实初值:所有方法的初值只来自更小的模型,无目标尺度泄漏)")
    rows = []
    for t in ["100", "400"]:
        src = [s for s in SCALES if NVAL[s] < NVAL[t]]
        curves = load(t, TRAIN_CURVES)
        base = honest_init7(t, src)                 # honest init shared by all methods
        # Tissue baseline (5 params), fast vectorized
        ti_p, _ = fit_tissue_fast(curves)
        ti_tr = float(np.mean([metrics(c.loss, tissue_fast(ti_p, c))["mae"] for c in curves]))
        ti_te = float(np.mean([metrics(load_curve(t, n).loss, tissue_fast(ti_p, load_curve(t, n)))["mae"]
                               for n in TEST_CURVES]))
        print(f"{t+'M':>7} {'Tissue':>9} {5:>3} {ti_tr:9.5f} {ti_te:9.5f}  -")
        rows.append(dict(target=t, method="Tissue", nparam=5, train=ti_tr, test=ti_te))
        for name, free in [("MPL", F_MPL), ("SC-MPL-4", F_SC4), ("SC-MPL-3", F_SC3)]:
            p, _ = fit_partial(curves, base, free)   # same honest init for all
            tr, (te, _) = train_mae(p, t), test_mae(p, t)
            tag = "-" if name == "MPL" else "+".join(src) + "M"
            print(f"{t+'M':>7} {name:>9} {len(free):>3} {tr:9.5f} {te:9.5f}  {tag}")
            rows.append(dict(target=t, method=name, nparam=len(free), train=tr, test=te))
    return rows


def exp2_few_shot(target="400", src=("25", "100"), fracs=(0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 1.0)):
    """Short-prefix few-shot on one cosine curve, HONEST init (small-model prior, no target oracle):
    MPL(7) vs SC-MPL-3(3); who degrades slower."""
    print(f"\n=== Exp2: 短前缀少样本 (target={target}M, 诚实初值=小模型外推, 无目标泄漏) ===")
    print(f"{'p%':>5} {'MPL(7)':>9} {'SC-3(3)':>9}  winner")
    src = list(src)
    base = extrapolate7(target, src)                       # honest init from small models only
    cos = load_curve(target, "cosine_24000.csv")
    rows = []
    for f in fracs:
        pc = [prefix(cos, f)]
        mpl_p, _ = fit_partial(pc, base, F_MPL)            # MPL fits all 7 from honest init
        sc_p, _ = fit_partial(pc, base, F_SC3)             # SC-MPL-3 fits 3 amplitudes
        m_te, s_te = test_mae(mpl_p, target)[0], test_mae(sc_p, target)[0]
        print(f"{int(f*100):>5} {m_te:9.5f} {s_te:9.5f}  {'SC-MPL-3' if s_te < m_te else 'MPL'}")
        rows.append(dict(frac=f, mpl_test=m_te, sc_test=s_te))
    return rows


def exp3_zero_and_one_shot(target="400", src=("25", "100")):
    """Cross-scale: (a) zero-shot amplitude extrapolation; (b) 1-param L0 calibration."""
    print(f"\n=== Exp3: 跨尺度 ({'+'.join(src)}M -> {target}M) ===")
    src = list(src)
    # shape+alpha shared from small; fit amplitudes {L0,A,B} per small scale
    amp_idx = [0, 1, 3]
    amps = {s: fit_partial(load(s, TRAIN_CURVES), frozen_init7(s, [x for x in src if x != s] or src, F_SC3), F_SC3)[0]
            for s in src}
    logN = np.array([np.log(NVAL[s]) for s in src])
    shared = np.array([MPL_PRECOMPUTED_INIT[s] for s in src]).mean(axis=0)  # alpha,C,beta,gamma
    pred7 = shared.copy()
    for j in amp_idx:
        y = np.array([amps[s][j] for s in src])
        slope = (y[1] - y[0]) / (logN[1] - logN[0])
        pred7[j] = y[0] + slope * (np.log(NVAL[target]) - logN[0])
    z_te, _ = test_mae(pred7, target)
    # (b) 1-param calibration: freeze everything at the extrapolated pred7, fit only L0 on target cosine
    cal_p, _ = fit_partial([load_curve(target, "cosine_24000.csv")], pred7, F_CAL1)
    c_te, _ = test_mae(cal_p, target)
    # baseline MPL on full target train
    mpl_p, _ = fit_mpl(load(target, TRAIN_CURVES), MPL_PRECOMPUTED_INIT[target])
    m_te, _ = test_mae(mpl_p, target)
    print(f"  (a) 零样本外推       test_MAE = {z_te:.5f}  ({target}M 训练曲线: 0)")
    print(f"  (b) 1参数L0校准      test_MAE = {c_te:.5f}  ({target}M 训练曲线: 1, 只拟合 L0)")
    print(f"  --- MPL 全量拟合      test_MAE = {m_te:.5f}  ({target}M 训练曲线: {len(TRAIN_CURVES)}, 拟合 7 参数)")
    return dict(zero_shot=z_te, one_param_cal=c_te, mpl_full=m_te, pred7=pred7.tolist())


def extrapolate7(target, src):
    """Prior center: shared exponents from small mean; amplitudes log-linear-extrapolated in log N."""
    shared = np.array([MPL_PRECOMPUTED_INIT[s] for s in src]).mean(axis=0)
    amps = {s: fit_partial(load(s, TRAIN_CURVES), frozen_init7(s, [x for x in src if x != s] or src, F_SC3), F_SC3)[0]
            for s in src}
    logN = np.array([np.log(NVAL[s]) for s in src])
    base = shared.copy()
    for j in (0, 1, 3):
        y = np.array([amps[s][j] for s in src])
        slope = (y[1] - y[0]) / (logN[1] - logN[0])
        base[j] = y[0] + slope * (np.log(NVAL[target]) - logN[0])
    return base


def fit_amp_map(curves, base7, tau, stride=FIT_STRIDE):
    """MAP: fit amplitudes {L0,A,B} with relative-deviation prior centered on base7."""
    amp_idx = [0, 1, 3]
    center = np.asarray(base7, float)
    c0 = center[amp_idx]
    fc = [subsample_curve(c, stride) for c in curves]
    bounds = [FULL_BOUNDS[i] for i in amp_idx]

    def assemble(pa):
        full = center.copy(); full[amp_idx] = pa; return full

    def obj(pa):
        like = _objective(assemble(pa), fc)
        if like >= 1e17:
            return like
        return like + tau * float(np.sum(((pa - c0) / np.abs(c0)) ** 2))

    best, _ = fit_with_restarts(obj, [c0, c0 * 1.05, c0 * 0.95], bounds)
    return assemble(best)


def exp4_hierarchical(target="400", src=("25", "100"), taus=(0.0, 1e-3, 1e-2, 1e-1, 1.0, 1e3)):
    """Unified MAP across data regimes: one tau should be good everywhere.

    Regimes (target training data): full(2 curves), prefix 25%, prefix 10%, zero(0).
    Baselines per regime: MPL(7p, no prior) fit on the same data.
    """
    print(f"\n=== Exp4: 分层MAP统一方法 ({'+'.join(src)}M 先验 -> {target}M) ===")
    base = extrapolate7(target, src)
    cos = load_curve(target, "cosine_24000.csv")
    regimes = {"full(2)": load(target, TRAIN_CURVES),
               "prefix25%": [prefix(cos, 0.25)],
               "prefix10%": [prefix(cos, 0.10)],
               "zero(0)": []}
    # MPL baseline per regime, HONEST init (small-model prior, no target oracle)
    mpl_base = {}
    for name, cv in regimes.items():
        mpl_base[name] = test_mae(fit_mpl(cv, base)[0], target)[0] if cv else float("nan")
    hdr = "  ".join(f"{t:>8g}" for t in taus)
    print(f"{'regime':>10} {'MPL':>8}  | tau: {hdr}")
    rows = []
    for name, cv in regimes.items():
        line = []
        for tau in taus:
            p = fit_amp_map(cv, base, tau) if cv else base  # zero-data: pure prior
            line.append(test_mae(p, target)[0])
        best_tau = taus[int(np.argmin(line))]
        print(f"{name:>10} {mpl_base[name]:>8.5f}  | " + "  ".join(f"{v:>8.5f}" for v in line)
              + f"   best_tau={best_tau:g}")
        rows.append(dict(regime=name, mpl=mpl_base[name], taus=list(taus), test=line))
    return rows


from reproduce_cosine_to_wsd import compute_s1, compute_ld  # noqa: E402


def joint_predict(gp, N, curve):
    """Joint (N, schedule) law: 11 global params cover all scales."""
    E, a, p, A0, qA, B0, qB, alpha, C, beta, gamma = gp
    L0 = E + a * N ** (-p)
    A = A0 * N ** qA
    B = B0 * N ** qB
    s1 = compute_s1(curve)
    ld = compute_ld(curve, C, beta, gamma)
    return L0 + A * np.power(s1, -alpha) + B * ld


def coarse_log(curve, npts=40):
    """Log-spaced point subset: cheap fit grid (LD cost is dominated by high-step points)."""
    n = len(curve.step)
    if n <= npts:
        return curve
    idx = np.unique(np.round(np.geomspace(1, n, npts)).astype(int) - 1)
    idx = idx[(idx >= 0) & (idx < n)]
    return Curve(curve.name, curve.scale, curve.step[idx], curve.loss[idx], curve.lrs)


def fit_joint(scales, stride=FIT_STRIDE):
    """One global 11-param fit across all given scales' cosine train curves."""
    data = [(NVAL[s], coarse_log(load_curve(s, n), 40)) for s in scales for n in TRAIN_CURVES]
    init = np.array([2.0, 3.3, 0.37, 0.40, 0.08, 242.0, 0.13, 0.46, 2.1, 0.59, 0.65])
    bounds = [(0, 5), (0, 30), (0.01, 2), (1e-4, 10), (-1, 1), (1, 1e4), (-1, 2),
              (0.05, 2), (0.01, 100), (0.05, 3), (0.05, 3)]

    def obj(gp):
        preds, ys = [], []
        for N, c in data:
            pr = joint_predict(gp, N, c)
            if not np.all(np.isfinite(pr)) or np.any(pr <= 0):
                return 1e18
            preds.append(pr); ys.append(c.loss)
        return huber_log_residual(np.concatenate(ys), np.concatenate(preds))

    best, _ = fit_with_restarts(obj, [init], bounds)   # single good init for speed
    return best


def exp5_joint():
    """Single global 11-param joint law vs per-scale MPL (7x3=21 params)."""
    print("\n=== Exp5: 联合 (N,schedule) 律 [11 全局参数] vs 逐尺度 MPL [21 参数] ===")
    gp = fit_joint(SCALES)
    print(f"  全局参数: E={gp[0]:.3f} a={gp[1]:.3f} p={gp[2]:.3f} | "
          f"A0={gp[3]:.3f} qA={gp[4]:.3f} | B0={gp[5]:.1f} qB={gp[6]:.3f} | "
          f"alpha={gp[7]:.3f} C={gp[8]:.3f} beta={gp[9]:.3f} gamma={gp[10]:.3f}")
    print(f"{'scale':>7} {'JOINT(11)':>10} {'MPL(7/scale)':>13}")
    rows = []
    for s in SCALES:
        N = NVAL[s]
        j_te = float(np.mean([metrics(load_curve(s, n).loss, joint_predict(gp, N, load_curve(s, n)))["mae"]
                              for n in TEST_CURVES]))
        mpl_p, _ = fit_mpl(load(s, TRAIN_CURVES), MPL_PRECOMPUTED_INIT[s])
        m_te = test_mae(mpl_p, s)[0]
        print(f"{s+'M':>7} {j_te:>10.5f} {m_te:>13.5f}")
        rows.append(dict(scale=s, joint_test=j_te, mpl_test=m_te))
    tot_j, tot_m = sum(r["joint_test"] for r in rows), sum(r["mpl_test"] for r in rows)
    print(f"  合计 test MAE: JOINT(11p)={tot_j:.5f}  vs  MPL(21p)={tot_m:.5f}")
    return dict(global_params=gp.tolist(), per_scale=rows)


def exp6_gamma_test():
    """Test the leading-order theory prediction: the clean kernel 1-(1+C*dS)^-beta
    (gamma=0, no eta^-gamma modulation) should fit nearly as well as full MPL."""
    print("\n=== Exp6: 检验推导预言 gamma=0 (干净核 vs 完整 MPL) ===")
    print(f"{'scale':>7} {'MPL train':>10} {'g0 train':>10} | {'MPL test':>10} {'g0 test':>10} | {'gamma*':>7}")
    rows = []
    for s in SCALES:
        curves = load(s, TRAIN_CURVES)
        full, _ = fit_partial(curves, np.array(MPL_PRECOMPUTED_INIT[s], float), F_MPL)
        base = np.array(MPL_PRECOMPUTED_INIT[s], float); base[6] = 0.0
        g0, _ = fit_partial(curves, base, [0, 1, 2, 3, 4, 5])     # gamma frozen at 0
        ftr, (fte, _) = train_mae(full, s), test_mae(full, s)
        gtr, (gte, _) = train_mae(g0, s), test_mae(g0, s)
        print(f"{s+'M':>7} {ftr:>10.5f} {gtr:>10.5f} | {fte:>10.5f} {gte:>10.5f} | {full[6]:>7.3f}")
        rows.append(dict(scale=s, mpl_train=ftr, g0_train=gtr, mpl_test=fte, g0_test=gte, gamma=full[6]))
    return rows


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # Exp1: grouped bars of test MAE
    methods = ["Tissue", "MPL", "SC-MPL-4", "SC-MPL-3"]
    npar = {"Tissue": 5, "MPL": 7, "SC-MPL-4": 4, "SC-MPL-3": 3}
    colors = {"Tissue": "#999", "MPL": "#4C72B0", "SC-MPL-4": "#DD8452", "SC-MPL-3": "#C44E52"}
    scales = ["100", "400"]
    x = np.arange(len(scales)); w = 0.2
    for i, m in enumerate(methods):
        vals = [next(r["test"] for r in out["exp1_same_sample"] if r["target"] == s and r["method"] == m)
                for s in scales]
        axes[0].bar(x + (i - 1.5) * w, vals, w, label=f"{m} ({npar[m]}p)", color=colors[m])
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"{s}M" for s in scales])
    axes[0].set_ylabel("test MAE (unseen WSD)"); axes[0].set_title("Exp1: same-sample, fewer params")
    axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.3)
    # Exp2: MAE vs prefix
    e2 = out["exp2_few_shot"]
    fr = [r["frac"] * 100 for r in e2]
    axes[1].plot(fr, [r["mpl_test"] for r in e2], "o-", label="MPL (7p)", color="#4C72B0")
    axes[1].plot(fr, [r["sc_test"] for r in e2], "s-", label="SC-MPL-3 (3p)", color="#C44E52")
    axes[1].set_xlabel("prefix of cosine_24000 (%)"); axes[1].set_ylabel("test MAE (400M)")
    axes[1].set_title("Exp2: few-shot (short prefix)"); axes[1].set_yscale("log")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "sc_mpl_compare.png", dpi=130)
    print(f"saved -> {OUT/'sc_mpl_compare.png'}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    out["exp1_same_sample"] = exp1_same_sample()
    out["exp2_few_shot"] = exp2_few_shot()
    out["exp3_cross_scale"] = exp3_zero_and_one_shot()
    out["exp4_hierarchical"] = exp4_hierarchical()
    out["exp5_joint"] = exp5_joint()
    (OUT / "results.json").write_text(json.dumps(out, indent=2))
    plot(out)
    print(f"saved -> {OUT/'results.json'}")


if __name__ == "__main__":
    main()
