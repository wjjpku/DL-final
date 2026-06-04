"""
G2 -- CROSS-SHAPE transfer of the non-adiabatic lag kernel.

The paper (Wu, "Learning-Rate Schedules Are Not Adiabatic") validates the
DropRelaxS correction only across SCALES of a single schedule family. Its own
Limitations (sec 6) flag that (lambda_slow, c) are *measured*, not transferred.

Here we test a stronger claim in the controlled NQM where we KNOW {lambda_i,s_i}:

    Calibrate (lambda_slow, kappa = c * eta_peak * dLeq/deta) ONCE, on a single
    schedule SHAPE (WSD), then PREDICT the non-adiabatic lag on entirely NOVEL
    schedule shapes WITHOUT refitting either parameter.

    lag_hat(t) = kappa * droprelaxS(etas_shape, lambda_slow)
    r(t)       = L_true(t) - L_eq(t)            (measured residual)

Novel shapes (built at the SAME peak/end/warmup as calibration):
    - triangular        (warmup up, then linear down to end)
    - two-step staircase (peak -> mid -> low)
    - exponential decay  (geometric decay from peak to end)
    - cosine             (gradual smooth decay)
    - reverse-wsd        (decay then re-warm: decay to low, hold, ramp back up)

Spectrum: noise-dominated, lambdas = geomspace(0.3,3,10), sigmas = ones(10).
Theory anchor: lambda_slow_pred = 2*mean(lambda/sigma) (preconditioned curvature);
              kappa_pred ~ c * eta_peak * dLeq/deta.

HEADLINE: the SAME (lambda_slow, kappa) predicts the lag across UNSEEN schedule
shapes. SUCCESS if mean R2 > 0.6 over >= 3 novel shapes.

Run:
  cd c:/Users/21100/Desktop/represent && python repro/G2_cross_shape.py
"""
import sys, os, json
sys.path.insert(0, r'c:/Users/21100/Desktop/represent/repro')
import numpy as np
from engine import (adamw_nqm, nqm_linear_Leq, droprelaxS, drops, cumS,
                    wsd_lrs, cosine_lrs, const_lrs)

RESULTS = r'c:/Users/21100/Desktop/represent/results/G2.json'
FIG = r'c:/Users/21100/Desktop/represent/figs/G2.png'


# ---------------------------------------------------------------------------
# regression / metric helpers
# ---------------------------------------------------------------------------
def regress_through_origin(r, K):
    """kappa = argmin ||r - kappa K||^2 = sum(r*K)/sum(K*K); return (kappa, R2)."""
    denom = float(np.sum(K * K))
    if denom <= 0:
        return 0.0, 0.0
    kappa = float(np.sum(r * K) / denom)
    pred = kappa * K
    ss_res = float(np.sum((r - pred) ** 2))
    ss_tot = float(np.sum((r - r.mean()) ** 2)) + 1e-30
    return kappa, 1.0 - ss_res / ss_tot


def r2_of(r, pred):
    """R2 of a FIXED prediction (no free scale): 1 - SS_res/SS_tot."""
    r = np.asarray(r, float); pred = np.asarray(pred, float)
    ss_res = float(np.sum((r - pred) ** 2))
    ss_tot = float(np.sum((r - r.mean()) ** 2)) + 1e-30
    return 1.0 - ss_res / ss_tot


def rel_mae(r, pred):
    """Relative MAE: mean|r-pred| / mean|r|."""
    r = np.asarray(r, float); pred = np.asarray(pred, float)
    num = float(np.mean(np.abs(r - pred)))
    den = float(np.mean(np.abs(r))) + 1e-30
    return num / den


def best_lambda_slow(r, etas, lam_pred, t_fit=0, grid_factors=None):
    """Grid-search lambda_slow around lam_pred (through-origin kappa).
    Returns best (lam, kappa, R2) and full grid rows. Window [t_fit:]."""
    if grid_factors is None:
        grid_factors = np.geomspace(0.1, 10.0, 81)
    grid = lam_pred * grid_factors
    best = (lam_pred, 0.0, -np.inf)
    rows = []
    rw = r[t_fit:]
    for lam in grid:
        K = droprelaxS(etas, lam)[t_fit:]
        kappa, R2 = regress_through_origin(rw, K)
        rows.append((float(lam), float(kappa), float(R2)))
        if R2 > best[2]:
            best = (float(lam), float(kappa), float(R2))
    return best, rows


def dLeq_deta(lambdas, sigmas, eta, rel=1e-4):
    de = eta * rel
    return (nqm_linear_Leq(lambdas, sigmas, eta + de)
            - nqm_linear_Leq(lambdas, sigmas, eta - de)) / (2 * de)


# ---------------------------------------------------------------------------
# schedule builders (all share peak/end/warmup with the WSD calibrator)
# ---------------------------------------------------------------------------
def warmup_ramp(eta, n_warm, peak):
    """In-place linear warmup ramp 0->peak over n_warm steps (used by custom shapes)."""
    if n_warm > 0:
        eta[:n_warm] = peak * (np.arange(1, n_warm + 1) / n_warm)
    return eta


def triangular_lrs(total, peak, end, n_warm):
    """Warmup up to peak, then linear down to end over the rest."""
    eta = np.empty(total, float)
    warmup_ramp(eta, n_warm, peak)
    dec = np.arange(n_warm, total)
    frac = (dec - n_warm) / max(total - n_warm, 1)
    eta[n_warm:] = peak * (1 - frac) + end * frac
    return eta


def staircase_lrs(total, peak, mid, low, n_warm, s1, s2):
    """Two-step staircase: warmup -> peak (until s1) -> mid (until s2) -> low."""
    eta = np.empty(total, float)
    warmup_ramp(eta, n_warm, peak)
    eta[n_warm:s1] = peak
    eta[s1:s2] = mid
    eta[s2:] = low
    return eta


def expdecay_lrs(total, peak, end, n_warm):
    """Warmup, then geometric (exponential) decay from peak to end."""
    eta = np.empty(total, float)
    warmup_ramp(eta, n_warm, peak)
    n = total - n_warm
    frac = np.arange(n) / max(n - 1, 1)
    eta[n_warm:] = peak * (end / peak) ** frac
    return eta


def reverse_wsd_lrs(total, peak, low, n_warm, decay_start, rewarm_start):
    """Reverse-WSD: warmup -> peak (stable) -> linear DECAY to low (decay_start..rewarm_start)
    -> hold low briefly -> linear RE-WARM back up to peak at the end.
    Exercises the kernel's response to a drop FOLLOWED by a rise (no new drops
    during the re-warm, so K must relax through the rise)."""
    eta = np.empty(total, float)
    warmup_ramp(eta, n_warm, peak)
    eta[n_warm:decay_start] = peak
    # decay peak -> low
    dec = np.arange(decay_start, rewarm_start)
    fd = (dec - decay_start) / max(rewarm_start - decay_start, 1)
    eta[decay_start:rewarm_start] = peak * (1 - fd) + low * fd
    # re-warm low -> peak
    rw = np.arange(rewarm_start, total)
    fr = (rw - rewarm_start) / max(total - rewarm_start, 1)
    eta[rewarm_start:] = low * (1 - fr) + peak * fr
    return eta


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    np.set_printoptions(precision=4, suppress=True)

    # ---- spectrum (noise-dominated) ----
    d = 10
    lambdas = np.geomspace(0.3, 3.0, d)
    sigmas = np.ones(d)

    # ---- shared schedule params ----
    T = 4000
    eta_pk = 2e-2
    eta_end = 2e-3
    n_warm = 200
    n_rep = 4000
    seed = 0

    # eta_eff*lam_max sanity (eta_eff = eta/sigma = eta here)
    edge = float(eta_pk * lambdas.max() / sigmas.min())
    print(f"# eta_eff*lam_max = {edge:.4f}  (want << 1, far from edge of stability)")

    # ---- theory anchors ----
    lambda_slow_pred = float(2.0 * np.mean(lambdas / sigmas))
    dLeq = float(dLeq_deta(lambdas, sigmas, eta_pk))
    print(f"# lambda_slow_pred = 2*mean(lam/sig) = {lambda_slow_pred:.4f}")
    print(f"# dLeq/deta @ eta_pk = {dLeq:.6f}  ((1/4)sum s = {0.25*np.sum(sigmas):.4f})")

    # skip warmup ramp-up transient when fitting/scoring the decay-driven lag
    t_fit = n_warm + 100

    # =====================================================================
    # STEP 1 -- CALIBRATE on WSD ONLY
    # =====================================================================
    print("\n=== CALIBRATION schedule: WSD ===")
    etas_cal = np.asarray(
        wsd_lrs(T, decay_start=2800, peak=eta_pk, end=eta_end, n_warm=n_warm), float)
    L_true_cal = adamw_nqm(lambdas, sigmas, etas_cal, n_rep=n_rep, seed=seed)
    L_eq_cal = np.array([nqm_linear_Leq(lambdas, sigmas, e) for e in etas_cal])
    r_cal = L_true_cal - L_eq_cal

    (lam_cal, kappa_cal, R2_cal), grid_rows = best_lambda_slow(
        r_cal, etas_cal, lambda_slow_pred, t_fit=t_fit)
    # also record the through-origin kappa AT the theory-predicted lambda_slow
    K_pred_cal = droprelaxS(etas_cal, lambda_slow_pred)
    kappa_at_pred, R2_at_pred = regress_through_origin(r_cal[t_fit:], K_pred_cal[t_fit:])
    c_cal = kappa_cal / dLeq if dLeq != 0 else np.nan

    print(f"  calibrated lambda_slow = {lam_cal:.4f}  "
          f"(pred {lambda_slow_pred:.4f}, ratio {lam_cal/lambda_slow_pred:.2f})")
    print(f"  calibrated kappa       = {kappa_cal:.4f}  (in-sample R2={R2_cal:.3f})")
    print(f"  implied c = kappa/(dLeq/deta) = {c_cal:.3f}  (paper c~0.5, want O(1))")
    print(f"  @theory lambda_slow_pred: kappa={kappa_at_pred:.4f} R2={R2_at_pred:.3f}")

    if not np.isfinite(kappa_cal) or kappa_cal == 0:
        raise RuntimeError("calibration failed (kappa non-finite/zero)")

    # =====================================================================
    # STEP 2 -- PREDICT lag on NOVEL shapes with FROZEN (lam_cal, kappa_cal)
    # =====================================================================
    novel = {
        "triangular": triangular_lrs(T, eta_pk, eta_end, n_warm),
        "staircase": staircase_lrs(T, peak=eta_pk, mid=9e-3, low=2e-3,
                                   n_warm=n_warm, s1=1800, s2=3000),
        "expdecay": expdecay_lrs(T, eta_pk, eta_end, n_warm),
        "cosine": cosine_lrs(T, peak=eta_pk, end=eta_end, n_warm=n_warm),
        "reverse_wsd": reverse_wsd_lrs(T, peak=eta_pk, low=eta_end, n_warm=n_warm,
                                       decay_start=1500, rewarm_start=2800),
    }

    out = {
        "spectrum": {"lambdas": lambdas.tolist(), "sigmas": sigmas.tolist()},
        "params": {"T": T, "eta_pk": eta_pk, "eta_end": eta_end,
                   "n_warm": n_warm, "n_rep": n_rep, "seed": seed,
                   "t_fit": t_fit, "edge_eta_eff_lam_max": edge},
        "theory": {"lambda_slow_pred": lambda_slow_pred, "dLeq_deta": dLeq},
        "calibration": {
            "schedule": "wsd", "decay_start": 2800,
            "lambda_slow_cal": lam_cal,
            "lambda_slow_ratio_to_pred": float(lam_cal / lambda_slow_pred),
            "kappa_cal": kappa_cal,
            "c_implied": float(c_cal),
            "R2_in_sample": R2_cal,
            "kappa_at_pred_lambda_slow": kappa_at_pred,
            "R2_at_pred_lambda_slow": R2_at_pred,
            "r_last": float(r_cal[-1]),
            "maxabs_r": float(np.max(np.abs(r_cal[t_fit:]))),
        },
        "novel_shapes": {},
    }
    traces = {"wsd_cal": dict(etas=etas_cal, L_true=L_true_cal, L_eq=L_eq_cal,
                              r=r_cal, K=droprelaxS(etas_cal, lam_cal),
                              kappa=kappa_cal, lam=lam_cal)}

    print("\n=== TRANSFER to novel shapes (frozen lambda_slow & kappa) ===")
    r2_list = []
    for name, etas in novel.items():
        etas = np.asarray(etas, float)
        L_true = adamw_nqm(lambdas, sigmas, etas, n_rep=n_rep, seed=seed)
        L_eq = np.array([nqm_linear_Leq(lambdas, sigmas, e) for e in etas])
        r = L_true - L_eq

        # FROZEN prediction: same lambda_slow & same kappa, no refit
        K = droprelaxS(etas, lam_cal)
        lag_hat = kappa_cal * K

        rw = r[t_fit:]
        ph = lag_hat[t_fit:]
        R2_frozen = r2_of(rw, ph)
        rmae_frozen = rel_mae(rw, ph)

        # SHAPE-ONLY transfer: Pearson corr(r, K). Isolates whether the kernel's
        # TIME-SHAPE transfers, independent of amplitude/sign of kappa. The
        # frozen-R2 additionally demands the calibrated AMPLITUDE be right.
        Kw = K[t_fit:]
        if np.std(Kw) > 0 and np.std(rw) > 0:
            pearson = float(np.corrcoef(rw, Kw)[0, 1])
        else:
            pearson = 0.0

        # diagnostics: how much better could a per-shape refit do?
        kappa_refit, R2_refit_kappa = regress_through_origin(rw, K[t_fit:])
        (lam_refit, kappa_refit_full, R2_refit_full), _ = best_lambda_slow(
            r, etas, lambda_slow_pred, t_fit=t_fit)

        maxabs = float(np.max(np.abs(rw)))
        r2_list.append(R2_frozen)

        print(f"\n  -- {name} --")
        print(f"     max|r|(decay)      = {maxabs:.5e}   r[-1]={r[-1]:.5e}")
        print(f"     FROZEN R2          = {R2_frozen:.3f}   rel_MAE={rmae_frozen:.3f}"
              f"   pearson(r,K)={pearson:.3f}")
        print(f"     (refit kappa only) R2={R2_refit_kappa:.3f} kappa={kappa_refit:.3f}")
        print(f"     (refit lam+kappa)  R2={R2_refit_full:.3f} "
              f"lam={lam_refit:.3f} kappa={kappa_refit_full:.3f}")

        out["novel_shapes"][name] = {
            "frozen": {"R2": float(R2_frozen), "rel_MAE": float(rmae_frozen),
                       "pearson_r_K": pearson,
                       "lambda_slow": float(lam_cal), "kappa": float(kappa_cal)},
            "refit_kappa_only": {"R2": float(R2_refit_kappa),
                                 "kappa": float(kappa_refit)},
            "refit_lam_kappa": {"R2": float(R2_refit_full),
                                "lambda_slow": float(lam_refit),
                                "kappa": float(kappa_refit_full)},
            "maxabs_r_decay": maxabs,
            "r_last": float(r[-1]),
        }
        traces[name] = dict(etas=etas, L_true=L_true, L_eq=L_eq, r=r,
                            lag_hat=lag_hat)

    # =====================================================================
    # STEP 3 -- HEADLINE
    # =====================================================================
    r2_arr = np.array(r2_list, float)
    pear_arr = np.array([out["novel_shapes"][k]["frozen"]["pearson_r_K"]
                         for k in novel], float)
    n_pass = int(np.sum(r2_arr > 0.6))
    mean_r2 = float(np.mean(r2_arr))
    median_r2 = float(np.median(r2_arr))
    mean_pear = float(np.mean(pear_arr))
    median_pear = float(np.median(pear_arr))
    # shapes whose kernel SHAPE transfers (pearson>0.6); separates shape from amp
    n_shape_ok = int(np.sum(pear_arr > 0.6))
    # primary task success: mean frozen R2 > 0.6 over >= 3 novel shapes
    success = bool(mean_r2 > 0.6 and len(r2_arr) >= 3)
    # secondary (shape-transfer) success: >=3 shapes with frozen R2>0.6
    success_majority = bool(n_pass >= 3)

    out["summary"] = {
        "n_novel_shapes": int(len(r2_arr)),
        "frozen_R2_per_shape": {k: out["novel_shapes"][k]["frozen"]["R2"]
                                for k in novel},
        "frozen_relMAE_per_shape": {k: out["novel_shapes"][k]["frozen"]["rel_MAE"]
                                    for k in novel},
        "frozen_pearson_per_shape": {k: out["novel_shapes"][k]["frozen"]["pearson_r_K"]
                                     for k in novel},
        "mean_frozen_R2": mean_r2,
        "median_frozen_R2": median_r2,
        "mean_pearson_r_K": mean_pear,
        "median_pearson_r_K": median_pear,
        "n_shapes_R2_gt_0.6": n_pass,
        "n_shapes_pearson_gt_0.6": n_shape_ok,
        "calibrated_on": "wsd",
        "lambda_slow_cal": lam_cal,
        "kappa_cal": kappa_cal,
        "success": success,
        "success_majority_R2_gt_0.6": success_majority,
        "failing_shapes": [k for k in novel
                           if out["novel_shapes"][k]["frozen"]["R2"] <= 0.6],
    }

    print("\n=== SUMMARY (CROSS-SHAPE TRANSFER) ===")
    print(f"  calibrated on WSD: lambda_slow={lam_cal:.3f}, kappa={kappa_cal:.3f}")
    for k in novel:
        s = out["novel_shapes"][k]["frozen"]
        print(f"    {k:12s}: frozen R2={s['R2']:+.3f}  relMAE={s['rel_MAE']:.3f}"
              f"  pearson={s['pearson_r_K']:+.3f}")
    print(f"  mean frozen R2     = {mean_r2:.3f}   (task success wants > 0.6)")
    print(f"  median frozen R2   = {median_r2:.3f}")
    print(f"  mean pearson(r,K)  = {mean_pear:.3f}   (shape-only transfer)")
    print(f"  shapes R2>0.6      = {n_pass}/{len(r2_arr)}")
    print(f"  shapes pearson>0.6 = {n_shape_ok}/{len(r2_arr)}")
    print(f"  failing shapes     = {out['summary']['failing_shapes']}")
    print(f"  SUCCESS (mean R2>0.6) = {success}")
    print(f"  SUCCESS (majority R2>0.6) = {success_majority}")

    # ---- sanity: no NaNs anywhere ----
    def _check(o, path=""):
        bad = []
        if isinstance(o, dict):
            for k, v in o.items():
                bad += _check(v, f"{path}.{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                bad += _check(v, f"{path}[{i}]")
        elif isinstance(o, float):
            if not np.isfinite(o):
                bad.append(path)
        return bad
    nans = _check(out)
    if nans:
        print(f"!! WARNING non-finite values at: {nans}")
    else:
        print("  (no NaNs/inf in output)")
    out["summary"]["no_nans"] = (len(nans) == 0)

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {RESULTS}")

    # ---- figure (best-effort) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(FIG), exist_ok=True)
        names = list(novel.keys())
        ncol = len(names)
        fig, axes = plt.subplots(2, ncol, figsize=(3.2 * ncol, 6.5), squeeze=False)
        for j, name in enumerate(names):
            tr = traces[name]
            ax = axes[0, j]
            ax.plot(tr["etas"], color="gray", lw=1.0)
            ax.set_title(f"{name}\nLR schedule", fontsize=9)
            ax.set_xlabel("step"); ax.set_ylabel("eta")
            ax = axes[1, j]
            ax.plot(tr["r"], color="C3", lw=1.2, label="measured r")
            ax.plot(tr["lag_hat"], color="C0", ls="--", lw=1.2,
                    label="frozen kappa*K")
            ax.axvline(t_fit, color="gray", ls=":", lw=0.8, alpha=0.6)
            s = out["novel_shapes"][name]["frozen"]
            ax.set_title(f"R2={s['R2']:.2f} relMAE={s['rel_MAE']:.2f}", fontsize=9)
            ax.set_xlabel("step"); ax.set_ylabel("residual")
            ax.legend(fontsize=7)
        fig.suptitle(
            f"G2 cross-shape transfer (calibrated on WSD: "
            f"lambda_slow={lam_cal:.2f}, kappa={kappa_cal:.2f})  "
            f"mean R2={mean_r2:.2f}", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(FIG, dpi=110)
        print(f"saved {FIG}")
    except Exception as e:
        print(f"(figure skipped: {e})")

    return out


if __name__ == "__main__":
    main()
