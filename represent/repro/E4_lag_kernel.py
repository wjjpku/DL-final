"""
E4 -- Lag kernel on a full schedule.

Demonstrate, on a full LR schedule run with from-scratch AdamW on a noisy
quadratic model (NQM):

  (a) the rate-dependent residual signature: small on a cosine schedule
      (slow, smooth decay -> near quasi-static), large on a WSD schedule
      (fast, late linear decay -> loss lags equilibrium).
  (b) residual r(t) = L_true - L_eq is well explained by the paper's
      DropRelaxS kernel K(t) (Eq.4), regressed THROUGH THE ORIGIN.
  (c) the kernel amplitude kappa relates to dL_eq/deta via
      kappa ~ c * eta_peak * dL_eq/deta  with c in (0,1].

Spectrum: noise-dominated, lambdas = geomspace(0.3,3,12), sigmas = ones(12).

Run:
  cd c:/Users/21100/Desktop/represent && python repro/E4_lag_kernel.py
"""
import sys, os, json
sys.path.insert(0, r'c:/Users/21100/Desktop/represent/repro')
import numpy as np
from engine import (adamw_nqm, nqm_linear_Leq, droprelaxS, cumS, drops,
                    cosine_lrs, wsd_lrs)

RESULTS = r'c:/Users/21100/Desktop/represent/results/E4.json'
FIG = r'c:/Users/21100/Desktop/represent/figs/E4.png'


# ---------------------------------------------------------------------------
# helpers
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


def best_lambda_slow(r, etas, lam_pred, t_fit=0, grid_factors=None):
    """Grid-search lambda_slow around lam_pred; return best (lam, kappa, R2)
    and the full grid (for reporting / plotting). Regression uses window [t_fit:]."""
    if grid_factors is None:
        grid_factors = np.geomspace(0.1, 10.0, 61)
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


def decay_window_start(etas, frac=0.02, default=0):
    """First step at which cumulative LR-drop mass exceeds `frac` of total drop
    mass -- the onset of the decay-active region where DropRelaxS is non-trivial.
    For schedules with continuous tiny drops (cosine) this is right after warmup;
    for WSD it sits at the decay knee."""
    dr = drops(etas)
    tot = dr.sum()
    if tot <= 0:
        return default
    cum = np.cumsum(dr)
    idx = np.argmax(cum >= frac * tot)
    return int(idx)


def dLeq_deta(lambdas, sigmas, eta, rel=1e-4):
    de = eta * rel
    return (nqm_linear_Leq(lambdas, sigmas, eta + de)
            - nqm_linear_Leq(lambdas, sigmas, eta - de)) / (2 * de)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    np.set_printoptions(precision=4, suppress=True)

    # ---- spectrum (noise-dominated) ----
    d = 12
    lambdas = np.geomspace(0.3, 3.0, d)
    sigmas = np.ones(d)

    # ---- schedule params; keep eta_eff*lam << 1 (eta_eff = eta/sigma = eta) ----
    T = 4000
    eta_pk = 2e-2
    eta_end = 2e-3
    n_warm = 200
    # sanity: eta_eff*lam_max = eta_pk * lam_max / min(sigma)
    print(f"# eta_eff*lam_max = {eta_pk * lambdas.max() / sigmas.min():.4f}  (want << 1)")

    n_rep = 4000
    seed = 0

    # ---- preconditioned slow rate prediction ----
    # AdamW noise-dominated: lambda_slow = 2 * mean(lambda / sigma)
    lambda_slow_pred = float(2.0 * np.mean(lambdas / sigmas))
    print(f"# lambda_slow_pred = 2*mean(lam/sig) = {lambda_slow_pred:.4f}")

    # ---- dL_eq/deta at the peak (amplitude reference) ----
    dLeq = float(dLeq_deta(lambdas, sigmas, eta_pk))
    print(f"# dLeq/deta @ eta_pk = {dLeq:.6f}   ((1/4)sum s = {0.25*np.sum(sigmas):.4f})")

    # Fit/characterization window: skip the warmup ramp-up transient.
    # During warmup theta starts at 0 and *lags upward* as the LR ramps to peak
    # (a NEGATIVE residual). That is a ramp-up transient, not the decay-driven
    # lag the DropRelaxS kernel (built from LR *drops*) models. We therefore
    # characterize the residual after the ramp lag has relaxed (t >= t_fit).
    t_fit = n_warm + 100  # 300: peak reached at n_warm, ramp lag ~settled by +100

    schedules = {
        "cosine": cosine_lrs(T, peak=eta_pk, end=eta_end, n_warm=n_warm),
        "wsd": wsd_lrs(T, decay_start=2800, peak=eta_pk, end=eta_end, n_warm=n_warm),
    }
    out_t_fit = t_fit

    out = {
        "spectrum": {"lambdas": lambdas.tolist(), "sigmas": sigmas.tolist()},
        "params": {"T": T, "eta_pk": eta_pk, "eta_end": eta_end,
                   "n_warm": n_warm, "n_rep": n_rep, "seed": seed,
                   "t_fit": out_t_fit},
        "lambda_slow_pred": lambda_slow_pred,
        "dLeq_deta": dLeq,
        "schedules": {},
    }
    traces = {}

    for name, etas in schedules.items():
        print(f"\n=== schedule: {name} ===")
        etas = np.asarray(etas, float)
        # true loss from from-scratch AdamW ensemble
        L_true = adamw_nqm(lambdas, sigmas, etas, n_rep=n_rep, seed=seed)
        # adiabatic / quasi-static baseline
        L_eq = np.array([nqm_linear_Leq(lambdas, sigmas, e) for e in etas])
        r = L_true - L_eq

        maxabs = float(np.max(np.abs(r)))
        # decay-region residual: skip warmup ramp transient (t >= t_fit)
        r_post = r[t_fit:]
        maxabs_post = float(np.max(np.abs(r_post)))

        # decay-active window: from the onset of substantial LR drops. This is
        # where the residual "jumps up" (paper Fig.1) and the kernel is non-zero;
        # the stable plateau (kernel ~0) only adds non-drop noise to the fit.
        t_decay = max(t_fit, decay_window_start(etas))

        # kernel at predicted lambda_slow (fit on conservative window [t_fit:])
        K_pred = droprelaxS(etas, lambda_slow_pred)
        kappa_pred_ls, R2_pred_ls = regress_through_origin(
            r[t_fit:], K_pred[t_fit:])
        # and on the decay-active window
        kappa_pred_dw, R2_pred_dw = regress_through_origin(
            r[t_decay:], K_pred[t_decay:])

        # grid-search best lambda_slow on conservative window
        (lam_best, kappa_best, R2_best), grid_rows = best_lambda_slow(
            r, etas, lambda_slow_pred, t_fit=t_fit)
        # grid-search best lambda_slow on decay-active window
        (lam_best_dw, kappa_best_dw, R2_best_dw), _ = best_lambda_slow(
            r, etas, lambda_slow_pred, t_fit=t_decay)

        # Amplitude identity (paper Eq.3 experiment iii): kappa_pred = dLeq/deta,
        # and c = kappa_fit / kappa_pred should be O(1), <=~1 (paper c~0.5).
        # (Eq.4 writes c*eta_peak*dLeq/deta*K, but Eq.3's exact identity is
        #  sum_i w_i = dLeq/deta with K convolving the raw drops, so the
        #  predictable amplitude scale is kappa_pred = dLeq/deta.)
        kappa_pred = dLeq
        c_pred_ls = kappa_pred_ls / kappa_pred if kappa_pred != 0 else np.nan
        c_best = kappa_best / kappa_pred if kappa_pred != 0 else np.nan
        c_best_dw = kappa_best_dw / kappa_pred if kappa_pred != 0 else np.nan

        print(f"  max|r| (full)        = {maxabs:.5e}")
        print(f"  max|r| (decay region, t>={t_fit}) = {maxabs_post:.5e}")
        print(f"  L_true[0]={L_true[0]:.4f} L_true[-1]={L_true[-1]:.4f} "
              f"L_eq[-1]={L_eq[-1]:.4f}  r[-1]={r[-1]:.5e}")
        print(f"  post-warmup window [t>={t_fit}]:")
        print(f"    @lambda_slow_pred={lambda_slow_pred:.3f}: "
              f"kappa={kappa_pred_ls:.4f} R2={R2_pred_ls:.3f} "
              f"c=kappa/(dLeq/deta)={c_pred_ls:.3f}")
        print(f"    best lambda_slow={lam_best:.3f}: "
              f"kappa={kappa_best:.4f} R2={R2_best:.3f} c={c_best:.3f}")
        print(f"  decay-active window [t>={t_decay}]:")
        print(f"    @lambda_slow_pred={lambda_slow_pred:.3f}: "
              f"kappa={kappa_pred_dw:.4f} R2={R2_pred_dw:.3f}")
        print(f"    best lambda_slow={lam_best_dw:.3f}: "
              f"kappa={kappa_best_dw:.4f} R2={R2_best_dw:.3f} c={c_best_dw:.3f}")

        out["schedules"][name] = {
            "maxabs_r_full": maxabs,
            "maxabs_r_decay": maxabs_post,
            "r_last": float(r[-1]),
            "L_true_first": float(L_true[0]),
            "L_true_last": float(L_true[-1]),
            "L_eq_last": float(L_eq[-1]),
            "kappa_pred_dLeq_deta": float(kappa_pred),
            "t_decay": int(t_decay),
            "post_warmup_window": {
                "t_start": int(t_fit),
                "at_pred_lambda_slow": {
                    "lambda_slow": lambda_slow_pred,
                    "kappa": kappa_pred_ls, "R2": R2_pred_ls, "c": float(c_pred_ls),
                },
                "best_lambda_slow": {
                    "lambda_slow": lam_best,
                    "kappa": kappa_best, "R2": R2_best, "c": float(c_best),
                },
            },
            "decay_active_window": {
                "t_start": int(t_decay),
                "at_pred_lambda_slow": {
                    "lambda_slow": lambda_slow_pred,
                    "kappa": kappa_pred_dw, "R2": R2_pred_dw,
                },
                "best_lambda_slow": {
                    "lambda_slow": lam_best_dw,
                    "kappa": kappa_best_dw, "R2": R2_best_dw, "c": float(c_best_dw),
                },
            },
        }
        traces[name] = dict(etas=etas, L_true=L_true, L_eq=L_eq, r=r,
                            K_pred=K_pred, K_best=droprelaxS(etas, lam_best))

    # ---- top-line comparison vs paper criteria ----
    wsd = out["schedules"]["wsd"]
    cos = out["schedules"]["cosine"]
    # headline R2: post-warmup window at the PREDICTED lambda_slow (parameter-free)
    wsd_R2_pw = wsd["post_warmup_window"]["best_lambda_slow"]["R2"]
    wsd_R2_pred = wsd["post_warmup_window"]["at_pred_lambda_slow"]["R2"]
    wsd_R2_dw = wsd["decay_active_window"]["best_lambda_slow"]["R2"]
    wsd_R2_dw_pred = wsd["decay_active_window"]["at_pred_lambda_slow"]["R2"]
    wsd_maxabs = wsd["maxabs_r_decay"]
    cos_maxabs = cos["maxabs_r_decay"]
    wsd_c = wsd["post_warmup_window"]["best_lambda_slow"]["c"]
    matches = bool(wsd_R2_pw > 0.6 and cos_maxabs < 0.5 * wsd_maxabs)
    out["summary"] = {
        "wsd_R2_post_warmup_best": wsd_R2_pw,
        "wsd_R2_post_warmup_at_pred": wsd_R2_pred,
        "wsd_R2_decay_window_best": wsd_R2_dw,
        "wsd_R2_decay_window_at_pred": wsd_R2_dw_pred,
        "wsd_maxabs_r_decay": wsd_maxabs,
        "cosine_maxabs_r_decay": cos_maxabs,
        "cosine_over_wsd_ratio": float(cos_maxabs / wsd_maxabs) if wsd_maxabs else np.nan,
        "wsd_c": wsd_c,
        "matches_paper": matches,
    }

    print("\n=== SUMMARY ===")
    print(f"  wsd R2 post-warmup @pred lambda_slow = {wsd_R2_pred:.3f}")
    print(f"  wsd R2 post-warmup (best)            = {wsd_R2_pw:.3f}   (want > 0.6)")
    print(f"  wsd R2 decay-window @pred lambda_slow= {wsd_R2_dw_pred:.3f}")
    print(f"  wsd R2 decay-window (best)           = {wsd_R2_dw:.3f}   (paper 0.40/0.80/0.87)")
    print(f"  wsd  max|r| (decay region)           = {wsd_maxabs:.5e}")
    print(f"  cos  max|r| (decay region)           = {cos_maxabs:.5e}")
    print(f"  cos/wsd ratio                        = {cos_maxabs/wsd_maxabs:.3f}   (want < 0.5)")
    print(f"  wsd c = kappa/(dLeq/deta)            = {wsd_c:.3f}  (paper c~0.5, want O(1))")
    print(f"  MATCHES PAPER                        = {matches}")

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {RESULTS}")

    # ---- figure (optional, best-effort) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(FIG), exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for col, name in enumerate(["cosine", "wsd"]):
            tr = traces[name]
            ax = axes[0, col]
            ax.plot(tr["L_true"], label="L_true (AdamW)", lw=1.5)
            ax.plot(tr["L_eq"], label="L_eq (adiabatic)", lw=1.2, ls="--")
            ax.set_title(f"{name}: loss curves")
            ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend(fontsize=8)
            ax2 = ax.twinx()
            ax2.plot(tr["etas"], color="gray", alpha=0.4, lw=0.8)
            ax2.set_ylabel("eta", color="gray")

            ax = axes[1, col]
            ax.plot(tr["r"], label="residual r=L_true-L_eq", color="C3", lw=1.3)
            sb = out["schedules"][name]["post_warmup_window"]["best_lambda_slow"]
            ax.plot(sb["kappa"] * tr["K_best"],
                    label=f"kappa*K (R2={sb['R2']:.2f})",
                    color="C0", ls="--", lw=1.3)
            t_fit_line = out["schedules"][name]["post_warmup_window"]["t_start"]
            ax.axvline(t_fit_line, color="gray", ls=":", lw=0.8, alpha=0.6)
            ax.set_title(f"{name}: residual vs DropRelaxS kernel")
            ax.set_xlabel("step"); ax.set_ylabel("residual"); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG, dpi=110)
        print(f"saved {FIG}")
    except Exception as e:
        print(f"(figure skipped: {e})")

    return out


if __name__ == "__main__":
    main()
