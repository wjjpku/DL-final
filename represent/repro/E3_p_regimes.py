"""
E3 -- THE KEY GENERALIZATION the paper PREDICTS but DID NOT TEST.

Paper claim (Hessian eigenbasis, AdamW preconditioned second-moment dynamics):
  * NOISE-dominated modes  (Adam preconditioner v* ~ s^2):   tau ~ eta^{-1}   (p = 1)
  * SIGNAL-dominated modes (v* ~ lam^2 V, self-consistent):  tau ~ eta^{-2/3} (p = 2/3)

This script measures the relaxation time tau(eta) of a real AdamW Noisy-Quadratic-Model
in BOTH regimes and fits tau ~ eta^{-p}.

Protocols
---------
(A) NOISE-dominated.  Large per-mode gradient noise (sigma = 1) so the Adam second
    moment v is dominated by noise s^2 (preconditioner ~ const).  Equilibrate at a high
    constant eta_hi, step DOWN to eta_lo, record the loss transient, fit an exponential
    relaxation tau.  Sweep eta_lo and fit a power law.  Expect p ~ 1.

(B) SIGNAL-dominated.  Here the preconditioner must track the SIGNAL g^2 ~ (lam*theta)^2,
    not the noise.  The 2/3 exponent is a *self-consistent* statement about the relaxation
    AROUND a (weakly) noisy equilibrium:
        rate r ~ eta * lam / sqrt(v),   v ~ lam^2 V,   V ~ (eta s / lam)^{2/3}
        =>  r ~ eta^{2/3} (lam/s)^{1/3}   =>  tau ~ eta^{-2/3}.
    For v to be signal-dominated (lam^2 V >> s^2) AND for the relaxation to be slow
    enough to fit a clean exponential (tau >> 1 step) we must tune lam/s: the two
    constraints are only marginally compatible, both pointing to lam/s ~ 30-40 with a
    small gradient noise s.  We equilibrate at eta_hi = 3*eta_lo (a modest drop so v
    stays near its eta_lo value during relaxation), step down, fit tau, sweep eta_lo.
    We run THREE (lam, s) variants to bracket p_signal robustly.

(B') CONTRAST: the *noise-free large-displacement* sign-descent limit.
    Starting far from the optimum with TINY noise, the AdamW update collapses to pure
    sign descent (m/sqrt(v) -> sign(g)), theta decays LINEARLY, and tau ~ eta^{-1}
    (p = 1) regardless of beta2.  We report this to be candid that the 2/3 does NOT
    come from the naive "tiny-noise large-displacement" picture -- it is specifically
    the self-consistent noisy-equilibrium relaxation that bends the exponent below 1.

matches_paper := (|p_noise - 1| < 0.2)  AND  (p_signal clearly < p_noise, ideally ~2/3).
"""
import sys, os, json
sys.path.insert(0, r'c:/Users/21100/Desktop/represent/repro')
import numpy as np
from engine import (equilibrate, adamw_nqm_from_state, adamw_nqm,
                    measure_tau, fit_powerlaw)

OUT = r'c:/Users/21100/Desktop/represent/results/E3.json'
D = 8
N_REP = 3000
SEED = 0


def sweep_stepdown(lambdas, sigmas, etas_lo, eta_hi=None, ratio=None,
                   n_eq=8000, n_relax=8000, n_rep=N_REP, r2_thresh=0.85):
    """Equilibrate (at eta_hi if given, else ratio*eta_lo), step down to each eta_lo,
    fit an exponential relaxation tau, then fit tau ~ eta_lo^{-p}.

    Returns dict with per-eta tau/r2/signal-dominance and the power-law fit on the
    points whose exponential fit is clean (r2 > r2_thresh)."""
    taus, fitr2, sigdom = [], [], []
    for e in etas_lo:
        eh = eta_hi if eta_hi is not None else ratio * e
        state = equilibrate(lambdas, sigmas, eta=eh, n_steps=n_eq,
                            n_rep=n_rep, seed=SEED)
        theta, m, v = state
        Vth = np.mean(theta ** 2, axis=0)
        # signal-dominance of the Adam second moment: lam^2 V vs s^2 (slow-mode avg)
        sd = float(np.mean(lambdas ** 2 * Vth) / np.mean(np.asarray(sigmas) ** 2))
        loss = adamw_nqm_from_state(lambdas, sigmas, np.full(n_relax, e),
                                    state, seed=SEED + 1)
        r = measure_tau(loss, t0=0)
        taus.append(r["tau"]); fitr2.append(r["r2"]); sigdom.append(sd)
    taus = np.array(taus); fitr2 = np.array(fitr2)
    mask = fitr2 > r2_thresh
    if mask.sum() >= 3:
        p, c, plr2 = fit_powerlaw(np.array(etas_lo)[mask], taus[mask])
    else:  # fall back to all points
        p, c, plr2 = fit_powerlaw(etas_lo, taus)
        mask = np.ones_like(fitr2, dtype=bool)
    return dict(etas=list(map(float, etas_lo)),
                taus=[float(x) for x in taus],
                exp_fit_r2=[float(x) for x in fitr2],
                signal_dominance=[float(x) for x in sigdom],
                n_clean=int(mask.sum()),
                p=float(p), c=float(c), powerlaw_r2=float(plr2))


def sweep_largedisp(lambdas, sigmas, etas, theta0=1.0, n_steps=12000,
                    n_rep=300, frac=1.0 / np.e):
    """Noise-free-ish large-displacement decay: start at theta0 (far from optimum),
    measure time for the loss to fall to frac*L0, fit time ~ eta^{-p}."""
    ts = []
    for e in etas:
        loss = adamw_nqm(lambdas, sigmas, np.full(n_steps, e),
                         theta0=np.full(len(lambdas), theta0),
                         n_rep=n_rep, seed=SEED)
        target = loss[0] * frac
        idx = np.where(loss <= target)[0]
        ts.append(int(idx[0]) if len(idx) else n_steps)
    p, c, r2 = fit_powerlaw(etas, ts)
    return dict(etas=list(map(float, etas)), tchar=[int(x) for x in ts],
                p=float(p), c=float(c), powerlaw_r2=float(r2))


def main():
    np.set_printoptions(precision=4)
    results = {}

    # ----------------------------------------------------------------- (A) NOISE
    print("=" * 70)
    print("(A) NOISE-dominated: sigma=1.0, equilibrate at fixed eta_hi, step down")
    print("=" * 70)
    lam_N = np.geomspace(0.5, 5.0, D)
    sig_N = np.ones(D) * 1.0
    etas_N = [3e-3, 6e-3, 1.2e-2, 2.4e-2]
    noise = sweep_stepdown(lam_N, sig_N, etas_N, eta_hi=6e-2,
                           n_eq=3000, n_relax=5000)
    noise["lambdas"] = lam_N.tolist(); noise["sigma"] = 1.0
    results["noise"] = noise
    for e, t, r in zip(noise["etas"], noise["taus"], noise["exp_fit_r2"]):
        print(f"   eta={e:.4f}  tau={t:7.1f}  exp_r2={r:.3f}")
    print(f"   --> p_noise = {noise['p']:.3f}  (predict 1.0)  "
          f"powerlaw_r2={noise['powerlaw_r2']:.3f}  n_clean={noise['n_clean']}")

    # ----------------------------------------------------------------- (B) SIGNAL
    print()
    print("=" * 70)
    print("(B) SIGNAL-dominated: tuned lam/s, equilibrate at 3*eta_lo, step down")
    print("    (self-consistent noisy-equilibrium relaxation -> predict 2/3)")
    print("=" * 70)
    # three variants bracketing lam/s ~ 30-40, each with small gradient noise
    signal_variants = [
        dict(name="v1_lam1.0_s0.027",
             lam=np.geomspace(0.7, 1.4, D), s=0.027,
             etas=[2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3]),
        dict(name="v2_lam2.0_s0.05",
             lam=np.geomspace(1.4, 2.8, D), s=0.05,
             etas=[1.25e-4, 2.5e-4, 5e-4, 1e-3, 2e-3]),
        dict(name="v3_lam0.5_s0.014",
             lam=np.geomspace(0.35, 0.7, D), s=0.014,
             etas=[5e-4, 1e-3, 2e-3, 4e-3, 8e-3]),
    ]
    sig_results = []
    for var in signal_variants:
        res = sweep_stepdown(var["lam"], np.full(D, var["s"]), var["etas"],
                             ratio=3.0, n_eq=8000, n_relax=8000)
        res["name"] = var["name"]
        res["lambdas"] = var["lam"].tolist(); res["sigma"] = var["s"]
        sig_results.append(res)
        print(f"  [{var['name']}]")
        for e, t, r in zip(res["etas"], res["taus"], res["exp_fit_r2"]):
            print(f"     eta={e:.5f}  tau={t:7.1f}  exp_r2={r:.3f}")
        print(f"     --> p = {res['p']:.3f}  powerlaw_r2={res['powerlaw_r2']:.3f}"
              f"  n_clean={res['n_clean']}")
    p_signal_list = [r["p"] for r in sig_results]
    p_signal = float(np.median(p_signal_list))
    results["signal"] = dict(variants=sig_results,
                             p_median=p_signal,
                             p_min=float(min(p_signal_list)),
                             p_max=float(max(p_signal_list)))
    print(f"   ==> p_signal (median of variants) = {p_signal:.3f}  "
          f"range [{min(p_signal_list):.3f}, {max(p_signal_list):.3f}]  (predict 0.667)")

    # ----------------------------------------------------- (B') CONTRAST sign-descent
    print()
    print("=" * 70)
    print("(B') CONTRAST: noise-free large-displacement (pure sign descent)")
    print("=" * 70)
    lam_LD = np.geomspace(0.5, 5.0, D)
    sig_LD = np.full(D, 1e-3)
    etas_LD = [1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2]
    ld = sweep_largedisp(lam_LD, sig_LD, etas_LD)
    ld["lambdas"] = lam_LD.tolist(); ld["sigma"] = 1e-3
    results["large_displacement_signdescent"] = ld
    for e, t in zip(ld["etas"], ld["tchar"]):
        print(f"   eta={e:.4f}  t_char(1/e)={t:6d}")
    print(f"   --> p = {ld['p']:.3f}  (pure sign descent => p=1, NOT 2/3)  "
          f"powerlaw_r2={ld['powerlaw_r2']:.3f}")

    # ----------------------------------------------------------------- VERDICT
    p_noise = noise["p"]
    cond_noise = abs(p_noise - 1.0) < 0.2
    cond_signal_below = (p_signal < p_noise - 0.1)
    cond_signal_near23 = abs(p_signal - 2.0 / 3.0) < 0.15
    matches = bool(cond_noise and cond_signal_below)
    results["verdict"] = dict(
        p_noise=float(p_noise), p_signal=float(p_signal),
        predict_p_noise=1.0, predict_p_signal=2.0 / 3.0,
        cond_noise_near1=bool(cond_noise),
        cond_signal_below_noise=bool(cond_signal_below),
        cond_signal_near_2_3=bool(cond_signal_near23),
        matches_paper=matches)

    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  p_noise  = {p_noise:.3f}  (predict 1.00)   |p-1|<0.2 ? {cond_noise}")
    print(f"  p_signal = {p_signal:.3f}  (predict 0.667)  < p_noise ? {cond_signal_below}"
          f"   near 2/3 ? {cond_signal_near23}")
    print(f"  matches_paper = {matches}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {OUT}")
    return results


if __name__ == "__main__":
    main()
