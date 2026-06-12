"""Phase-0 prototype: variational optimal-schedule solve under the lag-corrected law
(MPL + DropRelaxS, second-order closure) on the 10.7M bed calibration.

- Backbone: MPL fit on [constant, cosine, wsdcon_20] (same protocol as analyze_gen.py).
- Lag arms: (d=0, kappa=0.0027611), (d=0.5, kappa=0.0132498), (d=0.75, kappa=0.0223383),
  lam*=1.0 (probe-selected, = measured flat tau ~ 850 steps) -- from GEN_REPORT.json.
- Variational problem: fixed budget T=6000 (warmup 400 fixed), monotone non-increasing
  eta in [5e-5, 1.5e-3], knots every 80 steps, minimize predicted final loss
  J = L_MPL(T) + kappa * lagfeat(T)   vs adiabatic  J0 = L_MPL(T).
- Outputs: optimal shapes, decay-start shift, predicted A1-vs-A2 gap under the full law,
  A6 (late-sharp, MPL-matched) construction, seed-noise floor estimate from existing
  multi-seed curves.
"""
import os, sys, json, glob
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "represent", "repro"))
import analyze_curves as AC
from engine import cumS, mpl_loss_at

PEAK, WARM, TOTAL, ETA_MIN = 1.5e-3, 400, 6000, 5e-5
KNOT_EVERY = 80
ARMS = [("d=0", 0.0, 0.0027610792134732053),
        ("d=0.5", 0.5, 0.013249845363695585),
        ("d=0.75", 0.75, 0.02233834262339137)]
LAM = 1.0


def lagfeat_final(etas, delta, lam=LAM):
    """Weighted DropRelaxS at the final step, drops normalized by PEAK (analyze_gen conv)."""
    eta = np.asarray(etas, float)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
    if delta > 0:
        drop = drop * np.power(np.maximum(eta / PEAK, 1e-12), delta)
    dec = np.exp(-lam * eta)
    acc = 0.0
    for t in range(len(eta)):
        acc = acc * dec[t] + drop[t]
    return acc / PEAK


def build_sched(u):
    """u: unconstrained knot params -> monotone non-increasing eta after warmup."""
    # cumulative softplus in log space; eta_knots[0] = PEAK
    sp = np.log1p(np.exp(-np.abs(u))) + np.maximum(u, 0)   # softplus, stable
    logeta = np.log(PEAK) - np.concatenate([[0.0], np.cumsum(sp)])
    eta_k = np.clip(np.exp(logeta), ETA_MIN, PEAK)
    knots = np.arange(WARM, TOTAL + 1, KNOT_EVERY)
    if knots[-1] != TOTAL:
        knots = np.append(knots, TOTAL)
    # interpolate to per-step schedule
    e = np.empty(TOTAL)
    e[:WARM] = PEAK * (np.arange(1, WARM + 1) / WARM)
    e[WARM:] = np.interp(np.arange(WARM, TOTAL), knots, eta_k)
    return e


N_KNOTS = len(np.arange(WARM, TOTAL + 1, KNOT_EVERY)) - 1  # number of u params


def final_mpl(etas, params):
    return float(mpl_loss_at(etas, np.array([TOTAL - 1]), *params)[0])


def make_obj(params, delta, kappa):
    def obj(u):
        e = build_sched(u)
        J = final_mpl(e, params)
        if kappa > 0:
            J += kappa * lagfeat_final(e, delta)
        return J
    return obj


def solve(params, delta, kappa, n_restarts=4, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for r in range(n_restarts):
        u0 = rng.uniform(0.0, 0.08, N_KNOTS) - 2.0 - r * 0.8  # mostly-flat inits, varied
        res = minimize(make_obj(params, delta, kappa), u0, method="Nelder-Mead",
                       options=dict(maxiter=20000, maxfev=40000, xatol=1e-4, fatol=1e-7))
        if best is None or res.fun < best.fun:
            best = res
    return build_sched(best.x), best.fun


def decay_start(e, frac=0.99):
    """First step (post warmup) where eta < frac*PEAK."""
    idx = np.where(e[WARM:] < frac * PEAK)[0]
    return WARM + (int(idx[0]) if len(idx) else TOTAL)


def main():
    cv = AC.load_scale("m")
    train = ["constant", "cosine", "wsdcon_20"]
    params, fobj = AC.fit_mpl(cv, train, n_starts=8)
    print(f"MPL fit on {train}: obj={fobj:.5f}")
    print("params L0,A,alpha,B,C,beta,gamma =", np.round(params, 4))

    scheds = json.load(open(os.path.join(REPO, "represent", "results", "curves",
                                         "schedules.json")))
    out = {"mpl_params": [float(p) for p in params]}

    for tag, delta, kappa in ARMS:
        # adiabatic optimum (kappa=0) and lag-aware optimum, same machinery
        eA, JA = solve(params, delta, 0.0)
        eL, JL = solve(params, delta, kappa)
        # evaluate both under the FULL law
        full = lambda e: final_mpl(e, params) + kappa * lagfeat_final(e, delta)
        gap = full(eA) - full(eL)
        # also under full law: existing baselines
        base = {}
        for s in ["wsd", "wsdld", "cosine"]:
            e = np.asarray(scheds[s], float)
            base[s] = full(e)
        print(f"\n== arm {tag} (kappa={kappa:.5f}, lam={LAM}) ==")
        print(f"  adiabatic-opt: J_mpl={final_mpl(eA, params):.5f} "
              f"full={full(eA):.5f} decay_start~{decay_start(eA)} "
              f"eta_T={eA[-1]:.2e} S_T={cumS(eA)[-1]:.2f}")
        print(f"  lag-opt:       J_mpl={final_mpl(eL, params):.5f} "
              f"full={full(eL):.5f} decay_start~{decay_start(eL)} "
              f"eta_T={eL[-1]:.2e} S_T={cumS(eL)[-1]:.2f}")
        print(f"  predicted gap (adia-opt - lag-opt under full law) = {gap:+.5f}")
        print(f"  baselines under full law: " +
              " ".join(f"{k}={v:.5f}" for k, v in base.items()))
        # lag features
        print(f"  lagfeat: adia-opt {lagfeat_final(eA, delta):.4f}  "
              f"lag-opt {lagfeat_final(eL, delta):.4f}")
        # coarse shape summary (eta at selected steps)
        qs = [3000, 4000, 4500, 5000, 5400, 5700, 5900, 5999]
        print("  shape eta(t)/peak:")
        print("    t      : " + " ".join(f"{q:6d}" for q in qs))
        print("    adia   : " + " ".join(f"{eA[q]/PEAK:6.3f}" for q in qs))
        print("    lag    : " + " ".join(f"{eL[q]/PEAK:6.3f}" for q in qs))
        out[tag] = dict(gap=float(gap),
                        adia=dict(J=float(full(eA)), ds=int(decay_start(eA)),
                                  etaT=float(eA[-1]), S=float(cumS(eA)[-1])),
                        lag=dict(J=float(full(eL)), ds=int(decay_start(eL)),
                                 etaT=float(eL[-1]), S=float(cumS(eL)[-1])),
                        sched_adia=[float(x) for x in eA],
                        sched_lag=[float(x) for x in eL],
                        baselines={k: float(v) for k, v in base.items()})

        # ---- A6: late-sharp, MPL-matched to the adiabatic optimum ----
        targetJ = final_mpl(eA, params)
        bestA6 = None
        for w in [100, 150, 200, 300]:
            for eta_end_f in np.linspace(0.033, 0.6, 40):
                e6 = np.full(TOTAL, PEAK)
                e6[:WARM] = PEAK * (np.arange(1, WARM + 1) / WARM)
                tail = np.arange(TOTAL - w, TOTAL)
                fr = (tail - (TOTAL - w)) / w
                e6[TOTAL - w:] = PEAK * (1 - fr) + eta_end_f * PEAK * fr
                dJ = abs(final_mpl(e6, params) - targetJ)
                if bestA6 is None or dJ < bestA6[0]:
                    bestA6 = (dJ, w, eta_end_f, e6)
        dJ, w, eef, e6 = bestA6
        g6 = full(e6) - full(eA)
        print(f"  A6 late-sharp MPL-matched: w={w} eta_end={eef:.3f}*peak "
              f"|dJ_mpl|={dJ:.5f}; predicted pure-lag gap vs adia-opt = {g6:+.5f}")
        out[tag]["A6"] = dict(w=int(w), eta_end_frac=float(eef), dJ=float(dJ),
                              gap=float(g6), sched=[float(x) for x in e6])

    # ---- seed-noise floor from existing multi-seed curves ----
    print("\n== seed-noise floor (final-window mean of raw eval loss) ==")
    GEN = os.path.join(REPO, "represent", "results", "curves_gen")
    sch_gen = json.load(open(os.path.join(GEN, "schedules.json")))
    for fam, names in [("constant", ["constant", "constant_s1338", "constant_s1339"]),
                       ("twodrop", ["twodrop", "twodrop_s1338", "twodrop_s1339"]),
                       ("onedrop", ["onedrop", "onedrop_s1338", "onedrop_s1339"])]:
        vals = []
        for n in names:
            f = os.path.join(GEN, n + ".csv")
            if n == "constant":
                f = os.path.join(REPO, "represent", "results", "curves", "m_constant.csv")
            if not os.path.exists(f):
                continue
            rows = np.genfromtxt(f, delimiter=",", names=True)
            st = np.atleast_1d(rows["step"]).astype(int)
            ls = np.atleast_1d(rows["eval_loss"]).astype(float)
            m = st >= TOTAL - 200
            vals.append(float(ls[m].mean()))
        if len(vals) >= 2:
            print(f"  {fam}: finals {['%.4f' % v for v in vals]}  "
                  f"SD={np.std(vals, ddof=1):.4f}")
            out.setdefault("seed_noise", {})[fam] = dict(
                finals=vals, sd=float(np.std(vals, ddof=1)))
    # paired diff SD: (twodrop - onedrop) per seed
    try:
        pairs = []
        for sfx in ["", "_s1338", "_s1339"]:
            a = np.genfromtxt(os.path.join(GEN, f"twodrop{sfx}.csv"),
                              delimiter=",", names=True)
            b = np.genfromtxt(os.path.join(GEN, f"onedrop{sfx}.csv"),
                              delimiter=",", names=True)
            ma = np.atleast_1d(a["step"]) >= TOTAL - 200
            mb = np.atleast_1d(b["step"]) >= TOTAL - 200
            pairs.append(float(np.atleast_1d(a["eval_loss"])[ma].mean()
                               - np.atleast_1d(b["eval_loss"])[mb].mean()))
        print(f"  paired (twodrop-onedrop): {['%.4f' % p for p in pairs]}  "
              f"SD={np.std(pairs, ddof=1):.4f}")
        out["seed_noise"]["paired_two_minus_one"] = dict(
            diffs=pairs, sd=float(np.std(pairs, ddof=1)))
    except Exception as ex:
        print(f"  paired diff failed: {ex}")

    json.dump(out, open(os.path.join(HERE, "opt_schedule_proto.json"), "w"))
    print("\nsaved", os.path.join(HERE, "opt_schedule_proto.json"))


if __name__ == "__main__":
    main()
