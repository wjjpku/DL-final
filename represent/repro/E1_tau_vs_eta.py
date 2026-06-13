"""
E1 -- reproduce tau ~ 1/eta for from-scratch AdamW on a noisy quadratic (NQM).

Theory (paper, AdamW noise-dominated regime):
  preconditioner v* ~ s^2  => effective step eta_eff = eta/s
  per-step relaxation rate  2 eta lam / s  =>  tau ~ 1/eta   (p = 1).

Protocol:
  - Noise-dominated spectrum: d=10, lambdas = geomspace(0.5, 5.0, 10), sigmas = ones(10).
  - Equilibrate at a HIGH LR eta_hi=3e-2 (large n_steps, large n_rep), capture (theta,m,v).
  - For each eta_lo in geomspace(2e-3, 3.2e-2, 8): continue from the SAME equilibrium
    state at constant eta_lo and record the relaxation transient; measure_tau(loss, t0=0).
  - Discard fits with r2<0.9 or near the edge of stability (eta_lo*max(lam)/min(sig) not <<1).
  - Fit tau vs eta with fit_powerlaw -> report p, c, r2.
  - Cross-check smallest-eta tau against nqm_linear_tau (slowest mode), agree within ~30%.

PAPER: p ~ 1.01 (sim); real-data pooled p = 0.84 +/- 0.17 (deep_tau_pooled.py).  matches_paper if |p_sim-1| < 0.2.
"""
import sys, os, json
sys.path.insert(0, r'c:/Users/21100/Desktop/represent/repro')
import numpy as np
from engine import (equilibrate, adamw_nqm_from_state, measure_tau,
                    fit_powerlaw, nqm_linear_tau)

# ---------------------------------------------------------------- config
D          = 10
LAMBDAS    = np.geomspace(0.5, 5.0, D)
SIGMAS     = np.ones(D)
ETA_HI     = 3e-2          # high LR to equilibrate at
ETA_LOS    = np.geomspace(2e-3, 3.2e-2, 8)
EQ_STEPS   = 3500
RELAX_STEPS= 4000
N_REP      = 3500
SEED_EQ    = 0
SEED_RELAX = 1
R2_MIN     = 0.9
STAB_MAX   = 0.5          # require eta_lo*max(lam)/min(sig) < this ("<<1")

RESULTS_PATH = r'c:/Users/21100/Desktop/represent/results/E1.json'


def main():
    print("=" * 70)
    print("E1 -- tau ~ 1/eta : from-scratch AdamW on noise-dominated NQM")
    print("=" * 70)
    print(f"spectrum d={D}  lambdas=[{LAMBDAS.min():.3g}..{LAMBDAS.max():.3g}]  "
          f"sigmas=ones  ->  noise-dominated (v* ~ s^2)")
    print(f"equilibrate at eta_hi={ETA_HI:.3g}  (n_steps={EQ_STEPS}, n_rep={N_REP})")
    print(f"relax {RELAX_STEPS} steps at each eta_lo  (seed={SEED_RELAX})\n")

    # equilibrate ONCE at the high LR; reuse the same state for every eta_lo
    state = equilibrate(LAMBDAS, SIGMAS, eta=ETA_HI, n_steps=EQ_STEPS,
                        n_rep=N_REP, seed=SEED_EQ)
    L_hi = 0.5 * (np.mean(state[0] ** 2, axis=0) @ LAMBDAS)
    print(f"equilibrium loss at eta_hi: L_hi = {L_hi:.5f}\n")

    rows = []
    print(f"{'eta_lo':>10} {'stab':>8} {'tau':>10} {'amp':>10} "
          f"{'floor':>10} {'r2':>7} {'keep':>5}")
    print("-" * 70)
    for eta_lo in ETA_LOS:
        stab = eta_lo * LAMBDAS.max() / SIGMAS.min()   # ~ eta_eff*lam, want <<1
        etas = np.full(RELAX_STEPS, eta_lo)
        loss = adamw_nqm_from_state(LAMBDAS, SIGMAS, etas, state, seed=SEED_RELAX)
        r = measure_tau(loss, t0=0)
        keep = (np.isfinite(r["tau"]) and np.isfinite(r["r2"])
                and r["r2"] >= R2_MIN and stab < STAB_MAX and r["tau"] > 0)
        rows.append(dict(eta=float(eta_lo), stab=float(stab),
                         tau=float(r["tau"]), amp=float(r["amp"]),
                         floor=float(r["floor"]), r2=float(r["r2"]),
                         keep=bool(keep)))
        print(f"{eta_lo:10.4g} {stab:8.3f} {r['tau']:10.2f} {r['amp']:10.4e} "
              f"{r['floor']:10.4e} {r['r2']:7.3f} {str(keep):>5}")

    kept = [row for row in rows if row["keep"]]
    print(f"\nkept {len(kept)}/{len(rows)} fits "
          f"(r2>={R2_MIN}, stab<{STAB_MAX})")
    if len(kept) < 3:
        print("WARNING: fewer than 3 usable fits -- power law unreliable")

    x = np.array([row["eta"] for row in kept])
    y = np.array([row["tau"] for row in kept])
    p, c, r2 = fit_powerlaw(x, y)
    print(f"\npower-law fit  tau = {c:.3g} * eta^(-{p:.3f})   r2={r2:.4f}")
    print(f"  (paper: p ~ 1.01 sim; real-data pooled p = 0.84 +/- 0.17, deep_tau_pooled.py)")

    # cross-check: smallest-eta tau vs linear-theory slowest mode
    eta_min = x.min()
    tau_meas_min = float(y[np.argmin(x)])
    tau_lin = nqm_linear_tau(LAMBDAS, SIGMAS, eta_min)
    tau_lin_slow = float(np.max(tau_lin))      # slowest mode dominates
    ratio = tau_meas_min / tau_lin_slow
    agree = abs(ratio - 1.0) <= 0.30
    print(f"\ncross-check at eta_min={eta_min:.4g}:")
    print(f"  measured tau      = {tau_meas_min:.2f}")
    print(f"  linear slowest tau= {tau_lin_slow:.2f}")
    print(f"  ratio meas/lin    = {ratio:.3f}  -> agree(<=30%)={agree}")

    matches_paper = abs(p - 1.0) < 0.2

    out = dict(
        experiment="E1_tau_vs_eta",
        description="from-scratch AdamW on noise-dominated NQM: tau ~ eta^-p",
        spectrum=dict(d=D, lambdas=LAMBDAS.tolist(), sigmas=SIGMAS.tolist()),
        config=dict(eta_hi=ETA_HI, eta_los=ETA_LOS.tolist(),
                    eq_steps=EQ_STEPS, relax_steps=RELAX_STEPS, n_rep=N_REP,
                    r2_min=R2_MIN, stab_max=STAB_MAX),
        rows=rows,
        n_kept=len(kept),
        powerlaw=dict(p=p, c=c, r2=r2),
        crosscheck=dict(eta_min=eta_min, tau_measured=tau_meas_min,
                        tau_linear_slowest=tau_lin_slow, ratio=ratio,
                        agree_within_30pct=bool(agree)),
        paper=dict(p_sim=1.01, p_real_pooled=0.84, p_real_err=0.17),
        matches_paper=bool(matches_paper),
    )
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved -> {RESULTS_PATH}")
    print(f"\nRESULT: p={p:.3f}  |p-1|={abs(p-1):.3f}  "
          f"matches_paper={matches_paper}")
    return out


if __name__ == "__main__":
    main()
