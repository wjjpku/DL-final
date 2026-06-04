"""
EXPERIMENT E2 -- tau is INDEPENDENT of beta2 in the noise-dominated regime.

Paper claim: in the noise-dominated regime the AdamW preconditioner at equilibrium
is v* ~ s^2 (the gradient-noise second moment), which is set by the noise variance,
NOT by beta2.  beta2 only controls the *averaging window* of the EMA; once equilibrated
the EMA has converged to the same expectation E[g^2] ~ s^2 for any beta2 in (0,1).
Hence the effective step eta_eff = eta / sqrt(v*) ~ eta / s and the per-step relaxation
rate 2*eta*lam/s are beta2-independent  => tau is FLAT vs beta2.

Protocol (same noise-dominated spectrum as E1):
  lambdas = geomspace(0.5, 5.0, 10), sigmas = ones(10).
  eta_hi = 3e-2 (equilibrate), eta_lo = 6e-3 (relaxation).
  For each beta2 in [0.9, 0.95, 0.99, 0.999, 0.9999]:
    equilibrate at eta_hi WITH THAT beta2 -> state
    adamw_nqm_from_state(eta_lo) WITH THE SAME beta2 -> relaxation transient
    measure_tau -> tau
  Report taus, mean, std, CV = std/mean.  matches_paper if CV < 0.10.
"""
import sys, os, json
sys.path.insert(0, r'c:/Users/21100/Desktop/represent/repro')
import numpy as np
from engine import (equilibrate, adamw_nqm_from_state, measure_tau,
                    nqm_linear_tau)

RESULTS = r'c:/Users/21100/Desktop/represent/results'
os.makedirs(RESULTS, exist_ok=True)


def main():
    # --- noise-dominated spectrum (identical to E1) ---
    d = 10
    lambdas = np.geomspace(0.5, 5.0, d)
    sigmas = np.ones(d)

    eta_hi = 3e-2     # equilibrate here
    eta_lo = 6e-3     # step down to here, record transient

    # numerics
    n_rep = 4000
    eps = 1e-8
    # The Adam v-EMA has an effective averaging window ~ 1/(1-beta2).  To make the
    # preconditioner converge to its equilibrium v* ~ s^2 even at beta2=0.9999
    # (window ~ 1e4), scale equilibration/relaxation lengths with the largest window.
    # This keeps every fit clean (r2>0.9) without changing the physics.
    # beta2-dependent equilibration: pre-roll for several EMA windows (~5/(1-beta2)),
    # floored at 4000, so v* genuinely converges to s^2 even at beta2=0.9999
    # (window ~ 1e4). This isolates the equilibrium physics from EMA warm-up.
    def n_eq_for(b2):
        return int(np.clip(5.0 / (1.0 - b2), 4000, 60000))
    n_relax = 5000    # relaxation steps recorded

    beta2_list = [0.9, 0.95, 0.99, 0.999, 0.9999]

    # sanity: stay well inside stability  (eta_eff * lam = eta/sigma * lam << 1)
    edge = eta_lo / sigmas.min() * lambdas.max()
    print(f"stability check: eta_lo/sigma_min * lam_max = {edge:.4f}  (want << 1)")

    # linear cross-check (beta2-independent by construction): slowest-mode tau
    tau_lin = nqm_linear_tau(lambdas, sigmas, eta_lo)
    tau_lin_slow = float(np.max(tau_lin))
    print(f"linear-approx slowest-mode tau (cross-check) = {tau_lin_slow:.1f} steps\n")

    rows = []
    taus = []
    print(f"{'beta2':>8} | {'tau (steps)':>12} | {'amp':>9} | {'floor':>9} | {'r2':>6}")
    print("-" * 56)
    for b2 in beta2_list:
        # equilibrate at high LR with THIS beta2
        n_eq = n_eq_for(b2)
        state = equilibrate(lambdas, sigmas, eta=eta_hi,
                            n_steps=n_eq, n_rep=n_rep, seed=0,
                            beta1=0.9, beta2=b2, eps=eps)
        # step down to eta_lo with the SAME beta2; record transient
        etas = np.full(n_relax, eta_lo)
        loss = adamw_nqm_from_state(lambdas, sigmas, etas, state,
                                    beta1=0.9, beta2=b2, eps=eps, seed=1)
        # Fit over a window that captures the transient (~the whole relaxation, many tau)
        # but is not so long it is dominated by the noisy floor.  Window = ~1500 steps
        # (>~ 20 * tau for tau~66) gives a clean exponential fit for every beta2.
        r = measure_tau(loss, t0=0, fit_len=1500)
        rows.append(dict(beta2=b2, tau=r['tau'], amp=r['amp'],
                         floor=r['floor'], r2=r['r2'], n_eq=n_eq,
                         loss0=float(loss[0]), loss_end=float(loss[-1])))
        taus.append(r['tau'])
        print(f"{b2:>8.4f} | {r['tau']:>12.2f} | {r['amp']:>9.4f} | "
              f"{r['floor']:>9.4f} | {r['r2']:>6.3f}")

    taus = np.array(taus, float)
    mean_tau = float(np.mean(taus))
    std_tau = float(np.std(taus, ddof=0))
    cv = float(std_tau / mean_tau) if mean_tau != 0 else float('nan')
    spread = float((taus.max() - taus.min()) / mean_tau)

    print("-" * 56)
    print(f"mean(tau)  = {mean_tau:.2f} steps")
    print(f"std(tau)   = {std_tau:.2f} steps")
    print(f"CV = std/mean = {cv:.4f}  ({100*cv:.2f}%)")
    print(f"min-max spread / mean = {spread:.4f}  ({100*spread:.2f}%)")
    print(f"\nPAPER: tau FLAT, CV ~ 2%.")
    matches = (cv < 0.10) and np.all(np.isfinite(taus)) and \
              np.all(np.array([row['r2'] for row in rows]) > 0.9)
    print(f"matches_paper (CV<0.10 & all finite & all r2>0.9): {matches}")

    out = dict(
        experiment="E2_beta2_ablation",
        description="tau independent of beta2 in noise-dominated regime",
        spectrum=dict(d=d, lambdas=lambdas.tolist(), sigmas=sigmas.tolist(),
                      regime="noise-dominated (sigmas O(1))"),
        eta_hi=eta_hi, eta_lo=eta_lo,
        n_rep=n_rep, n_eq_per_beta2=[n_eq_for(b) for b in beta2_list],
        n_relax=n_relax, fit_len=1500, eps=eps,
        beta2_list=beta2_list,
        per_beta2=rows,
        taus=taus.tolist(),
        mean_tau=mean_tau, std_tau=std_tau, cv=cv, minmax_spread_over_mean=spread,
        tau_linear_slowmode_crosscheck=tau_lin_slow,
        paper_value="tau flat, CV ~ 2%",
        our_value=f"CV = {100*cv:.2f}% (mean tau = {mean_tau:.1f} steps over beta2 in [0.9,0.9999])",
        matches_paper=bool(matches),
        why=("At equilibrium the Adam preconditioner v* equals E[g^2] ~ s^2 (the "
             "gradient-noise variance) for ANY beta2 in (0,1): beta2 only sets the EMA "
             "averaging window, not its converged mean. So eta_eff = eta/sqrt(v*) ~ eta/s "
             "and the per-step relaxation rate 2*eta*lam/s are beta2-independent, giving a "
             "flat tau. (Smaller beta2 only adds more EMA jitter -> slightly noisier fits.)"),
    )
    path = os.path.join(RESULTS, "E2.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {path}")
    return out


if __name__ == "__main__":
    main()
