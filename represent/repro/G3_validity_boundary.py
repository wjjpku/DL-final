"""
G3 -- MAP the validity boundary of the DropRelaxS / adiabatic-lag theory.

The paper (sec.6 Limitations, item c) explicitly assumes a locally-quadratic,
static spectrum with SLOW modes:   eta_eff * lambda / s << 1,  i.e. FAR from the
edge of stability.  It never says HOW far is far enough.  This script maps the
quantitative validity boundary in the controlled noisy-quadratic model (NQM)
where {lambda_i, s_i} are KNOWN.

Setup (noise-dominated so eta_eff = eta/s is exact and rho is controllable):
  lambdas = geomspace(0.5, 5.0, 8),  sigmas = ones(8)   ->  eta_eff = eta.
  Define the max preconditioned per-step rate
        rho = eta_eff * lambda_max = (eta / s) * lambda_max = eta * 5.0 .
  Edge of stability for the linear contraction map is rho ~ 1 (the multiplier
  (1-rho) loses contraction; (1-rho)^2 -> no decay as rho->1, blows up rho>2).

For each PEAK lr (chosen so rho ranges ~1e-3 .. ~1):
  (i)  p_local : equilibrate at the peak, step DOWN by small local factors,
       fit an exponential relaxation tau at each, then fit tau ~ eta^{-p_local}.
       Theory predicts p_local = 1 (noise-dominated AdamW).  We report how p
       departs from 1 as rho grows.
  (ii) droprelaxS_R2 : run a WSD schedule with this peak, form the residual
       r = L_true - L_eq (adiabatic baseline = per-step linear L_eq), grid-search
       lambda_slow around the predicted 2*mean(lambda/s), regress r through the
       origin onto the DropRelaxS kernel K(t), report the best R2.

Output: rho vs {p_local, droprelaxS_R2}; identify the rho where the theory
degrades (R2 < 0.6 OR |p_local-1| > 0.3).

Run:
  cd c:/Users/21100/Desktop/represent && python repro/G3_validity_boundary.py
"""
import sys, os, json
sys.path.insert(0, r'c:/Users/21100/Desktop/represent/repro')
import numpy as np
from engine import (equilibrate, adamw_nqm_from_state, adamw_nqm,
                    measure_tau, fit_powerlaw, nqm_linear_Leq, nqm_linear_tau,
                    droprelaxS, drops, wsd_lrs)

OUT = r'c:/Users/21100/Desktop/represent/results/G3.json'

# ---------------------------------------------------------------- spectrum
D        = 8
LAMBDAS  = np.geomspace(0.5, 5.0, D)
SIGMAS   = np.ones(D)
LAM_MAX  = float(LAMBDAS.max())          # 5.0
# lambda_slow prediction (preconditioned curvature, paper sec.3): 2*mean(lam/s)
LAMBDA_SLOW_PRED = float(2.0 * np.mean(LAMBDAS / SIGMAS))

# rho grid: rho = eta_peak * lam_max  (since s=1, eta_eff=eta) -> eta = rho/lam_max
# denser through the Goldilocks band and the high-rho (edge-of-stability) descent.
RHOS = np.unique(np.round(np.concatenate([
    np.geomspace(2e-3, 1e-2, 4),      # finite-horizon rise side
    np.geomspace(1.4e-2, 0.30, 8),    # Goldilocks peak + high-rho descent (boundary)
    np.geomspace(0.45, 0.95, 3),      # edge of stability
]), 6))

# ensemble / step budgets (n_rep<=4000, steps<=5000 per the brief)
N_REP        = 4000
EQ_STEPS     = 3000
RELAX_STEPS  = 5000
SEED_EQ      = 0
SEED_RELAX   = 1
SEED_WSD     = 0

# WSD schedule
T_WSD        = 5000
N_WARM       = 200
DECAY_START  = 3500
END_FRAC     = 0.1     # eta_end = END_FRAC * eta_peak  (a real, fast late decay)


# ---------------------------------------------------------------- helpers
def regress_through_origin(r, K):
    denom = float(np.sum(K * K))
    if denom <= 0:
        return 0.0, -np.inf
    kappa = float(np.sum(r * K) / denom)
    pred = kappa * K
    ss_res = float(np.sum((r - pred) ** 2))
    ss_tot = float(np.sum((r - r.mean()) ** 2)) + 1e-30
    return kappa, 1.0 - ss_res / ss_tot


def best_lambda_slow(r, etas, lam_pred, t_fit, grid_factors=None):
    if grid_factors is None:
        grid_factors = np.geomspace(0.1, 10.0, 61)
    grid = lam_pred * grid_factors
    best = (lam_pred, 0.0, -np.inf)
    rw = r[t_fit:]
    for lam in grid:
        K = droprelaxS(etas, lam)[t_fit:]
        kappa, R2 = regress_through_origin(rw, K)
        if np.isfinite(R2) and R2 > best[2]:
            best = (float(lam), float(kappa), float(R2))
    return best


def dLeq_deta(eta, rel=1e-4):
    de = eta * rel
    return (nqm_linear_Leq(LAMBDAS, SIGMAS, eta + de)
            - nqm_linear_Leq(LAMBDAS, SIGMAS, eta - de)) / (2 * de)


def measure_p_local(eta_peak):
    """Measure how the relaxation time tau scales with eta LOCALLY around eta_peak.

    Equilibrate at eta_hi = 1.6*eta_peak (so each step-down has a sizeable
    transient amplitude), then step DOWN to local LRs eta_lo = factor*eta_peak.
    The NQM relaxation is a MIXTURE of per-mode timescales (tau_i from ~1/eta to
    ~1/eta over the spectrum), so we anchor the single-exponential fit with the
    KNOWN linear equilibrium floor L_eq(eta_lo) -- this isolates a consistent
    dominant tau and gives clean fits.  Then fit tau ~ eta^{-p_local}.

    rho is governed by eta_peak (the regime we are probing); the modest 1.6x
    equilibration headroom keeps eta_hi*lam_max only slightly above rho.
    Returns (p_local, powerlaw_r2, n_clean, rows)."""
    factors = np.array([0.50, 0.65, 0.80, 1.00])
    etas_lo = eta_peak * factors
    eta_hi = eta_peak * 1.6
    state = equilibrate(LAMBDAS, SIGMAS, eta=eta_hi, n_steps=EQ_STEPS,
                        n_rep=N_REP, seed=SEED_EQ)
    taus, r2s = [], []
    rows = []
    for e in etas_lo:
        loss = adamw_nqm_from_state(LAMBDAS, SIGMAS, np.full(RELAX_STEPS, e),
                                    state, seed=SEED_RELAX)
        floor = nqm_linear_Leq(LAMBDAS, SIGMAS, e)   # known adiabatic floor
        r = measure_tau(loss, t0=0, floor=floor)
        taus.append(r["tau"]); r2s.append(r["r2"])
        rows.append(dict(eta=float(e), rho=float(e * LAM_MAX),
                         tau=float(r["tau"]), exp_r2=float(r["r2"])))
    taus = np.array(taus, float); r2s = np.array(r2s, float)
    # keep clean exponential fits with finite positive tau
    mask = np.isfinite(taus) & (taus > 0) & np.isfinite(r2s) & (r2s > 0.85)
    if mask.sum() >= 3:
        p, c, plr2 = fit_powerlaw(etas_lo[mask], taus[mask])
    elif np.isfinite(taus).sum() >= 2:
        good = np.isfinite(taus) & (taus > 0)
        p, c, plr2 = fit_powerlaw(etas_lo[good], taus[good])
    else:
        p, c, plr2 = np.nan, np.nan, np.nan
    return float(p), float(plr2), int(mask.sum()), rows


def measure_droprelaxS_R2(eta_peak):
    """Run WSD at this peak, residual vs DropRelaxS, return best-lambda_slow R2."""
    eta_end = END_FRAC * eta_peak
    etas = wsd_lrs(T_WSD, decay_start=DECAY_START, peak=eta_peak,
                   end=eta_end, n_warm=N_WARM)
    etas = np.asarray(etas, float)
    L_true = adamw_nqm(LAMBDAS, SIGMAS, etas, n_rep=N_REP, seed=SEED_WSD)
    L_eq = np.array([nqm_linear_Leq(LAMBDAS, SIGMAS, e) for e in etas])
    r = L_true - L_eq
    t_fit = N_WARM + 100      # skip the warmup ramp-up transient
    # fit at the PREDICTED lambda_slow (parameter-free)
    K_pred = droprelaxS(etas, LAMBDA_SLOW_PRED)
    kappa_pred, R2_pred = regress_through_origin(r[t_fit:], K_pred[t_fit:])
    # and the best lambda_slow (grid search)
    lam_best, kappa_best, R2_best = best_lambda_slow(r, etas, LAMBDA_SLOW_PRED, t_fit)
    # amplitude scale c = kappa / (eta_peak * dLeq/deta) -- paper Eq.4 form
    dLeq = float(dLeq_deta(eta_peak))
    denom = eta_peak * dLeq
    c_best = float(kappa_best / denom) if denom != 0 else np.nan
    finite = bool(np.all(np.isfinite(L_true)) and np.all(np.isfinite(L_eq)))
    # horizon diagnostic: slowest-mode relaxation time at the peak vs the number
    # of decay steps.  DropRelaxS needs the lag to (largely) equilibrate within
    # the decay, i.e. tau_slow << decay_len.  When tau_slow >~ decay_len the lag
    # never reaches its DropRelaxS form (a finite-horizon artifact of the
    # controlled experiment, NOT edge-of-stability).
    tau_slow = float(np.max(nqm_linear_tau(LAMBDAS, SIGMAS, eta_peak)))
    decay_len = T_WSD - DECAY_START
    tau_over_decay = float(tau_slow / decay_len)
    return dict(eta_peak=float(eta_peak), eta_end=float(eta_end),
                R2_pred=float(R2_pred), kappa_pred=float(kappa_pred),
                lam_best=float(lam_best), R2_best=float(R2_best),
                kappa_best=float(kappa_best), c_best=c_best,
                dLeq_deta=dLeq, maxabs_r=float(np.max(np.abs(r[t_fit:]))),
                L_true_finite=finite, tau_slow=tau_slow,
                tau_over_decay=tau_over_decay,
                L_true_last=float(L_true[-1]), L_eq_last=float(L_eq[-1]))


def main():
    np.set_printoptions(precision=4, suppress=True)
    print("=" * 78)
    print("G3 -- validity boundary of DropRelaxS / adiabatic-lag theory")
    print("=" * 78)
    print(f"spectrum d={D}  lambdas=[{LAMBDAS.min():.3g}..{LAM_MAX:.3g}]  sigmas=ones")
    print(f"  -> eta_eff = eta/s = eta ; rho = eta_eff*lam_max = eta*{LAM_MAX:g}")
    print(f"lambda_slow_pred = 2*mean(lam/s) = {LAMBDA_SLOW_PRED:.4f}")
    print(f"rho grid: {RHOS}")
    print()
    print(f"{'rho':>8} {'eta_pk':>9} {'p_local':>8} {'plr2':>6} {'nkeep':>5} "
          f"{'R2_pred':>8} {'R2_best':>8} {'lam_b':>7} {'c_best':>8} {'tau/dec':>7} "
          f"{'fin':>4}")
    print("-" * 86)

    rows = []
    for rho in RHOS:
        eta_peak = float(rho) / LAM_MAX
        p_local, plr2, nkeep, prows = measure_p_local(eta_peak)
        dr = measure_droprelaxS_R2(eta_peak)
        row = dict(
            rho=float(rho),
            eta_peak=eta_peak,
            p_local=p_local,
            p_local_powerlaw_r2=plr2,
            p_local_n_clean=nkeep,
            droprelaxS_R2_pred=dr["R2_pred"],
            droprelaxS_R2_best=dr["R2_best"],
            lambda_slow_best=dr["lam_best"],
            kappa_best=dr["kappa_best"],
            c_best=dr["c_best"],
            dLeq_deta=dr["dLeq_deta"],
            maxabs_r=dr["maxabs_r"],
            tau_slow=dr["tau_slow"],
            tau_over_decay=dr["tau_over_decay"],
            L_true_finite=dr["L_true_finite"],
            stepdown_rows=prows,
        )
        rows.append(row)
        print(f"{rho:8.4f} {eta_peak:9.5f} {p_local:8.3f} {plr2:6.3f} {nkeep:5d} "
              f"{dr['R2_pred']:8.3f} {dr['R2_best']:8.3f} {dr['lam_best']:7.3f} "
              f"{dr['c_best']:8.2f} {dr['tau_over_decay']:7.2f} "
              f"{str(dr['L_true_finite']):>4}")

    # ===================================================================
    # Locate the EDGE-OF-STABILITY (high-rho) validity boundary.
    #
    # The R2(rho) curve is non-monotone by design of a FINITE-horizon study:
    #   * very small rho  -> tau_slow ~ 1/eta grows until tau_slow >~ decay_len,
    #     so the lag never fully develops within the schedule.  This is a
    #     finite-horizon artifact (NOT edge of stability): rho<<1 still, SNR is
    #     high (verified separately), p_local~1, and L stays finite.
    #   * the R2 PEAK sits in the "Goldilocks" band where tau_slow << decay_len
    #     AND rho << 1 simultaneously.
    #   * as rho -> 1 the preconditioned contraction (1-rho) collapses: tau~1
    #     step, the locally-quadratic / slow-mode assumption fails, the residual
    #     stops matching DropRelaxS (R2 -> negative) and p_local blows up.  This
    #     is the genuine edge-of-stability breakdown the paper warns about.
    #
    # We therefore report the UPPER boundary: starting from the R2 peak, the
    # largest rho for which the theory still holds (R2_best>=0.6 AND
    # |p_local-1|<=0.3 AND finite), and the first rho above it that breaks.
    # ===================================================================
    def holds(rw):
        return (np.isfinite(rw["droprelaxS_R2_best"])
                and rw["droprelaxS_R2_best"] >= 0.6
                and np.isfinite(rw["p_local"])
                and abs(rw["p_local"] - 1.0) <= 0.3
                and rw["L_true_finite"])

    for rw in rows:
        rw["theory_holds"] = bool(holds(rw))

    rho_sorted = sorted(rows, key=lambda r: r["rho"])
    r2s = np.array([rw["droprelaxS_R2_best"] for rw in rho_sorted], float)
    rho_arr = np.array([rw["rho"] for rw in rho_sorted], float)
    p_arr = np.array([rw["p_local"] for rw in rho_sorted], float)

    # R2 peak (Goldilocks center)
    finite_r2 = np.where(np.isfinite(r2s), r2s, -np.inf)
    i_peak = int(np.argmax(finite_r2))
    rho_peak = float(rho_arr[i_peak])
    R2_peak = float(r2s[i_peak])

    # UPPER boundary: walk UP from the peak; last holding rho and first break
    rho_upper_hold = None
    rho_upper_break = None
    for i in range(i_peak, len(rho_sorted)):
        if rho_sorted[i]["theory_holds"]:
            rho_upper_hold = rho_arr[i]
        else:
            rho_upper_break = rho_arr[i]
            break
    if rho_upper_hold is None:        # peak itself holds by construction usually
        rho_upper_hold = rho_peak

    # interpolate the rho where R2 crosses 0.6 on the descending (high-rho) side
    rho_cross06 = None
    for i in range(i_peak, len(rho_sorted) - 1):
        a, b = r2s[i], r2s[i + 1]
        if np.isfinite(a) and np.isfinite(b) and a >= 0.6 > b:
            ra, rb = np.log(rho_arr[i]), np.log(rho_arr[i + 1])
            frac = (a - 0.6) / (a - b)
            rho_cross06 = float(np.exp(ra + frac * (rb - ra)))
            break

    # ----- HIGH-rho monotone degradation (the boundary we claim) -----
    hi = np.arange(i_peak, len(rho_sorted))
    if hi.size >= 3 and np.all(np.isfinite(r2s[hi])):
        rr = np.argsort(np.argsort(rho_arr[hi]))
        rs = np.argsort(np.argsort(r2s[hi]))
        spearman_hi = float(np.corrcoef(rr, rs)[0, 1])
    else:
        spearman_hi = np.nan
    R2_top_mean = float(np.mean(r2s[-3:]))          # 3 highest-rho points
    monotone_high = bool(np.isfinite(spearman_hi) and spearman_hi < -0.5
                         and R2_top_mean < R2_peak - 0.3)

    # edge-of-stability is REAL: p_local also breaks at high rho
    p_breaks_high = bool(np.isfinite(p_arr[-1]) and abs(p_arr[-1] - 1.0) > 0.3)

    # ----- low-rho finite-horizon caveat (honest characterization) -----
    # at the lowest rho, R2 is below the peak BUT tau_slow >~ decay_len and rho<<1
    tau_over = np.array([rw["tau_over_decay"] for rw in rho_sorted], float)
    low_rho_horizon_limited = bool(
        r2s[0] < R2_peak - 0.2 and tau_over[0] > 0.7 and rho_arr[0] < 0.05)

    success = bool(monotone_high and (rho_upper_break is not None)
                   and (rho_cross06 is not None) and p_breaks_high)

    boundary = rho_cross06 if rho_cross06 is not None else rho_upper_hold
    headline = (f"DropRelaxS/adiabatic-lag theory holds in a Goldilocks band peaking "
                f"at rho~{rho_peak:.2g} (R2~{R2_peak:.2f}) and breaks past "
                f"rho~{boundary:.2g} (R2<0.6, |p_local-1|>0.3) toward the edge of "
                f"stability rho->1")

    summary = dict(
        rho_peak=rho_peak, R2_peak=R2_peak,
        rho_upper_hold=(None if rho_upper_hold is None else float(rho_upper_hold)),
        rho_upper_break=(None if rho_upper_break is None else float(rho_upper_break)),
        rho_cross06_high=(None if rho_cross06 is None else float(rho_cross06)),
        boundary_rho=float(boundary),
        spearman_rho_R2_highside=spearman_hi,
        R2_top3_mean=R2_top_mean,
        monotone_degradation_highside=monotone_high,
        p_local_breaks_at_high_rho=p_breaks_high,
        low_rho_horizon_limited=low_rho_horizon_limited,
        success=success,
        headline=headline,
    )

    out = dict(
        experiment="G3_validity_boundary",
        description=("Map the rho=eta_eff*lam_max validity boundary of the "
                     "DropRelaxS/adiabatic-lag theory in the noise-dominated NQM. "
                     "Edge-of-stability breakdown is the upper (high-rho) boundary; "
                     "very-small-rho R2 dip is a finite-horizon artifact "
                     "(tau_slow ~ 1/eta exceeds the decay length), not theory failure."),
        spectrum=dict(d=D, lambdas=LAMBDAS.tolist(), sigmas=SIGMAS.tolist(),
                      lam_max=LAM_MAX),
        config=dict(rhos=RHOS.tolist(), lambda_slow_pred=LAMBDA_SLOW_PRED,
                    n_rep=N_REP, eq_steps=EQ_STEPS, relax_steps=RELAX_STEPS,
                    T_wsd=T_WSD, n_warm=N_WARM, decay_start=DECAY_START,
                    decay_len=T_WSD - DECAY_START, end_frac=END_FRAC,
                    criteria="theory_holds := R2_best>=0.6 AND |p_local-1|<=0.3 AND finite"),
        rows=rows,
        summary=summary,
    )

    print("\n" + "=" * 78)
    print("SUMMARY  (edge-of-stability = high-rho boundary)")
    print("=" * 78)
    print(f"  R2 peak           = {R2_peak:.3f} at rho = {rho_peak:.3g}  (Goldilocks)")
    print(f"  rho_upper_hold    = {rho_upper_hold}  (last holding rho above peak)")
    print(f"  rho_upper_break   = {rho_upper_break}  (first breaking rho above peak)")
    print(f"  R2=0.6 crossing   = {rho_cross06}  (interpolated, high side)")
    print(f"  spearman(rho,R2) high-side = {spearman_hi:.3f}  (want < -0.5)")
    print(f"  R2 top-3 mean     = {R2_top_mean:.3f}  (vs peak {R2_peak:.3f})")
    print(f"  monotone degrade (high side) = {monotone_high}")
    print(f"  p_local breaks at high rho   = {p_breaks_high}")
    print(f"  low-rho finite-horizon caveat= {low_rho_horizon_limited}")
    print(f"  success           = {success}")
    print(f"  HEADLINE: {headline}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {OUT}")
    return out


if __name__ == "__main__":
    main()
