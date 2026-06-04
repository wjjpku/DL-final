"""
E6 -- Scale-invariance of the DropRelaxS relaxation rate lambda_slow.

Claim (paper):  lambda_slow = 2 * (lambda/sigma)_eff is a *preconditioned curvature*.
Because the AdamW preconditioner divides out the noise scale, lambda_slow should be
(approximately) INVARIANT across "model scales" as long as the (lambda/sigma)
distribution is preserved -- even when the overall loss magnitude, the number of
modes, or the noise amplitude changes.

We build 4 NQM configs that mimic different "model sizes" but all share the SAME
lambda/sigma ratio distribution:

  cfgA: lambdas = geomspace(0.5,5,8),  sigmas = 1.0
  cfgB: lambdas = lambdas_A * 2,       sigmas = 2.0     (amplitude up 2x, ratio kept)
  cfgC: lambdas = geomspace(0.5,5,16), sigmas = 1.0     (more modes d=16, same ratio range)
  cfgD: lambdas = lambdas_A * 0.5,     sigmas = 0.5     (curvature & noise 0.5x, ratio kept)

For each config we:
  1. Equilibrate AdamW at the peak LR (true equilibrium state, no startup transient).
  2. Run a wsdcon-style schedule: constant peak -> SHARP DROP to a lower constant LR.
     A sharp drop followed by a plateau makes the lag relax exponentially in S-time,
     which cleanly identifies lambda_slow (a continuous decay would be degenerate).
  3. Build the adiabatic baseline = the true (measured) AdamW equilibrium loss in each
     phase, and form the residual = actual_loss - adiabatic_baseline.
  4. Grid-fit the paper's DropRelaxS kernel K(t; lambda_slow) (Eq.4) to the residual,
     pick lambda_slow that maximizes R2.

We report lambda_slow per config, the CV across configs, and compare to
2*mean(lambda/sigma) and 2*harmonic_mean(lambda/sigma).

EXPECT: CV small (< ~0.25) -> lambda_slow is ~scale-invariant.  matches_paper if CV < 0.3.

Run:  cd c:/Users/21100/Desktop/represent && python repro/E6_scale_inv.py
"""
import sys, json, os
sys.path.insert(0, r'c:/Users/21100/Desktop/represent/repro')
import numpy as np
from engine import (equilibrate, adamw_nqm_from_state, droprelaxS, drops,
                    two_stage_lrs, measure_tau)

# ----------------------------------------------------------------------------
# experiment knobs (kept fast: n_rep<=4000, steps<=5000)
# ----------------------------------------------------------------------------
PEAK   = 2e-2      # peak LR (eta_eff*lam_max = 0.1 for all configs -> stable, slow modes)
LR_B   = 4e-3      # post-drop constant LR
TOTAL  = 4000      # total steps in the run
STEP   = 1000      # step at which the sharp drop happens
N_REP  = 4000      # replicas for ensemble loss
N_EQ   = 3000      # equilibration steps
SEED   = 0


def make_configs():
    lamA = np.geomspace(0.5, 5.0, 8)
    cfgs = {
        "cfgA": dict(lambdas=lamA,                    sigmas=np.full(8, 1.0),
                     note="base: d=8, sigma=1"),
        "cfgB": dict(lambdas=lamA * 2.0,              sigmas=np.full(8, 2.0),
                     note="amplitude 2x (lam,sig both x2), ratio kept"),
        "cfgC": dict(lambdas=np.geomspace(0.5, 5, 16), sigmas=np.full(16, 1.0),
                     note="more modes d=16, same ratio range"),
        "cfgD": dict(lambdas=lamA * 0.5,              sigmas=np.full(8, 0.5),
                     note="curvature & noise 0.5x, ratio kept"),
    }
    return cfgs


def measure_Leq(lambdas, sigmas, eta, n_eq=2500, n_avg=800, seed=0):
    """True AdamW equilibrium loss at constant `eta` (equilibrate then average)."""
    st = equilibrate(lambdas, sigmas, eta, n_steps=n_eq, n_rep=N_REP, seed=seed)
    loss = adamw_nqm_from_state(lambdas, sigmas, np.full(n_avg, eta), st, seed=seed + 7)
    return float(loss.mean())


def run_config(name, cfg):
    lambdas = np.asarray(cfg["lambdas"], float)
    sigmas = np.asarray(cfg["sigmas"], float)
    ratio = lambdas / sigmas
    pred_mean = 2.0 * ratio.mean()
    pred_hmean = 2.0 / np.mean(1.0 / ratio)

    # --- schedule: constant peak -> sharp drop -> constant lr_b ---
    etas = two_stage_lrs(TOTAL, peak=PEAK, lr_b=LR_B, step=STEP, n_warm=1)
    etas[:STEP] = PEAK   # pure stable plateau (no warmup transient)

    # --- run from the true peak-equilibrium state (no startup transient) ---
    st = equilibrate(lambdas, sigmas, PEAK, n_steps=N_EQ, n_rep=N_REP, seed=SEED)
    loss = adamw_nqm_from_state(lambdas, sigmas, etas, st, seed=SEED + 1)

    # --- adiabatic baseline: measured equilibrium loss in each phase ---
    Leq_peak = float(loss[:STEP].mean())                 # peak equilib (from the run itself)
    Leq_b = measure_Leq(lambdas, sigmas, LR_B, seed=SEED)
    base = np.where(np.arange(TOTAL) < STEP, Leq_peak, Leq_b)
    resid = loss - base

    # --- DropRelaxS grid fit over the post-drop region ---
    region = slice(STEP, TOTAL)
    yr = resid[region]
    grid = np.geomspace(0.2, 40.0, 80)

    def fit_ls(ls):
        K = droprelaxS(etas, ls)
        x = K[region]
        A = np.vstack([x, np.ones_like(x)]).T          # fit y = c*K + b
        coef, *_ = np.linalg.lstsq(A, yr, rcond=None)
        pred = A @ coef
        r2 = 1.0 - np.sum((yr - pred) ** 2) / (np.sum((yr - yr.mean()) ** 2) + 1e-30)
        return float(coef[0]), float(coef[1]), float(r2)

    fits = [(ls,) + fit_ls(ls) for ls in grid]
    best = max(fits, key=lambda r: r[3])
    ls_best, c_best, b_best, r2_best = best

    # --- cross-check: single-exponential tau in step-time -> lambda_slow ---
    mt = measure_tau(loss, t0=STEP + 1, floor=None)
    ls_tau = float(1.0 / (mt["tau"] * LR_B)) if np.isfinite(mt["tau"]) else np.nan

    out = dict(
        name=name, note=cfg["note"], d=int(len(lambdas)),
        mean_ratio=float(ratio.mean()), pred_2mean=float(pred_mean),
        pred_2hmean=float(pred_hmean),
        Leq_peak=Leq_peak, Leq_b=Leq_b,
        resid_drop_jump=float(resid[STEP]),          # immediate lag right after the drop
        resid_end=float(resid[-1]),
        lambda_slow=float(ls_best), kernel_c=float(c_best), kernel_r2=float(r2_best),
        tau_steps=float(mt["tau"]), tau_r2=float(mt["r2"]),
        lambda_slow_from_tau=ls_tau,
    )
    return out


def cv(vals):
    v = np.asarray(vals, float)
    return float(np.std(v) / (np.mean(v) + 1e-30))


def main():
    cfgs = make_configs()
    print("=" * 72)
    print("E6 -- scale-invariance of lambda_slow (preconditioned curvature)")
    print("=" * 72)
    results = {}
    for name, cfg in cfgs.items():
        r = run_config(name, cfg)
        results[name] = r
        print(f"\n[{name}] {r['note']}  (d={r['d']})")
        print(f"   mean(lam/sig)={r['mean_ratio']:.3f}  "
              f"pred 2*mean={r['pred_2mean']:.3f}  2*hmean={r['pred_2hmean']:.3f}")
        print(f"   Leq_peak={r['Leq_peak']:.5f}  Leq_b={r['Leq_b']:.5f}  "
              f"resid jump(after drop)={r['resid_drop_jump']:.5f}")
        print(f"   DropRelaxS fit:  lambda_slow={r['lambda_slow']:.3f}  "
              f"R2={r['kernel_r2']:.3f}  (c={r['kernel_c']:.3g})")
        print(f"   tau cross-check: tau={r['tau_steps']:.1f} steps  R2={r['tau_r2']:.3f}  "
              f"-> lambda_slow_from_tau={r['lambda_slow_from_tau']:.3f}")

    ls_vals = [results[n]["lambda_slow"] for n in cfgs]
    ls_tau_vals = [results[n]["lambda_slow_from_tau"] for n in cfgs]
    pred_mean_vals = [results[n]["pred_2mean"] for n in cfgs]
    pred_hmean_vals = [results[n]["pred_2hmean"] for n in cfgs]

    cv_ls = cv(ls_vals)
    cv_tau = cv(ls_tau_vals)
    matches = bool(cv_ls < 0.3)

    summary = dict(
        configs=results,
        lambda_slow_per_config={n: results[n]["lambda_slow"] for n in cfgs},
        lambda_slow_mean=float(np.mean(ls_vals)),
        lambda_slow_std=float(np.std(ls_vals)),
        CV_lambda_slow=cv_ls,
        CV_lambda_slow_from_tau=cv_tau,
        pred_2mean_mean=float(np.mean(pred_mean_vals)),
        pred_2hmean_mean=float(np.mean(pred_hmean_vals)),
        lambda_slow_over_2mean=float(np.mean(ls_vals) / np.mean(pred_mean_vals)),
        lambda_slow_over_2hmean=float(np.mean(ls_vals) / np.mean(pred_hmean_vals)),
        matches_paper=matches,
        criterion="CV(lambda_slow) < 0.3 across configs with preserved lambda/sigma distribution",
        knobs=dict(PEAK=PEAK, LR_B=LR_B, TOTAL=TOTAL, STEP=STEP, N_REP=N_REP, N_EQ=N_EQ),
    )

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print("  lambda_slow per config (DropRelaxS grid fit):")
    for n in cfgs:
        print(f"     {n}: {results[n]['lambda_slow']:.3f}  (R2={results[n]['kernel_r2']:.3f})")
    print(f"  mean lambda_slow = {summary['lambda_slow_mean']:.3f} "
          f"+/- {summary['lambda_slow_std']:.3f}")
    print(f"  CV(lambda_slow)            = {cv_ls:.3f}   "
          f"(EXPECT < 0.25; matches_paper if < 0.3)")
    print(f"  CV(lambda_slow_from_tau)   = {cv_tau:.3f}")
    print(f"  lambda_slow / 2*mean(lam/sig)  = {summary['lambda_slow_over_2mean']:.3f}")
    print(f"  lambda_slow / 2*hmean(lam/sig) = {summary['lambda_slow_over_2hmean']:.3f}")
    print(f"  matches_paper (CV<0.3): {matches}")

    os.makedirs(r'c:/Users/21100/Desktop/represent/results', exist_ok=True)
    out_path = r'c:/Users/21100/Desktop/represent/results/E6.json'
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsaved -> {out_path}")
    return summary


if __name__ == "__main__":
    main()
