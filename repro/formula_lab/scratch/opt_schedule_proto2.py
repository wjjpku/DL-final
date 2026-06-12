"""Phase-0 part 2: (a) stronger variational solve (warm starts from wsd/wsdld + cross
warm-starting + Powell polish); (b) MPL-matched discrimination pairs: gradual reference
vs late-sharp arm with equal L_MPL(T), so any realized loss gap is pure non-adiabatic
and the law's predicted gap kappa*(dfeat) is a committed quantitative prediction.
"""
import os, sys, json
import numpy as np
from scipy.optimize import minimize, brentq

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "represent", "repro"))
import analyze_curves as AC
from engine import cumS, mpl_loss_at

PEAK, WARM, TOTAL, ETA_MIN = 1.5e-3, 400, 6000, 5e-5
KNOT_EVERY = 80
KNOTS = np.arange(WARM, TOTAL + 1, KNOT_EVERY)
N_KNOTS = len(KNOTS) - 1
ARMS = [("d=0", 0.0, 0.0027610792134732053),
        ("d=0.5", 0.5, 0.013249845363695585),
        ("d=0.75", 0.75, 0.02233834262339137)]
LAM = 1.0


def lagfeat_final(etas, delta, lam=LAM):
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
    sp = np.log1p(np.exp(-np.abs(u))) + np.maximum(u, 0)
    logeta = np.log(PEAK) - np.concatenate([[0.0], np.cumsum(sp)])
    eta_k = np.clip(np.exp(logeta), ETA_MIN, PEAK)
    e = np.empty(TOTAL)
    e[:WARM] = PEAK * (np.arange(1, WARM + 1) / WARM)
    e[WARM:] = np.interp(np.arange(WARM, TOTAL), KNOTS, eta_k)
    return e


def u_from_sched(e):
    """Inverse of build_sched at the knots (approx)."""
    e = np.asarray(e, float)
    ee = np.append(e, e[-1]) if len(e) == TOTAL else e[:TOTAL + 1]
    ek = np.clip(np.interp(KNOTS, np.arange(len(ee)), ee), ETA_MIN, PEAK)
    d = np.diff(-np.log(ek / PEAK))
    d = np.maximum(d, 1e-9)
    # invert softplus
    return np.where(d > 30, d, np.log(np.expm1(d)))


def final_mpl(etas, params):
    return float(mpl_loss_at(etas, np.array([TOTAL - 1]), *params)[0])


def solve(params, delta, kappa, inits):
    def obj(u):
        e = build_sched(u)
        J = final_mpl(e, params)
        if kappa > 0:
            J += kappa * lagfeat_final(e, delta)
        return J
    best = None
    for u0 in inits:
        res = minimize(obj, u0, method="Nelder-Mead",
                       options=dict(maxiter=30000, maxfev=60000,
                                    xatol=1e-5, fatol=1e-8))
        res2 = minimize(obj, res.x, method="Powell",
                        options=dict(maxiter=8000, xtol=1e-6, ftol=1e-9))
        cand = res2 if res2.fun < res.fun else res
        if best is None or cand.fun < best.fun:
            best = cand
    return build_sched(best.x), best.fun, best.x


def lin_sched(ds, ef, w=None):
    """constant peak to ds, linear decay to ef*peak by step ds+w (default: to TOTAL)."""
    e = np.full(TOTAL, PEAK)
    e[:WARM] = PEAK * (np.arange(1, WARM + 1) / WARM)
    end = TOTAL if w is None else min(ds + w, TOTAL)
    t = np.arange(ds, end)
    fr = (t - ds) / max(end - ds, 1)
    e[ds:end] = PEAK * (1 - fr) + ef * PEAK * fr
    e[end:] = ef * PEAK
    return e


def main():
    cv = AC.load_scale("m")
    train = ["constant", "cosine", "wsdcon_20"]
    params, fobj = AC.fit_mpl(cv, train, n_starts=8)
    print(f"MPL fit obj={fobj:.5f}; params:", np.round(params, 4))
    scheds = json.load(open(os.path.join(REPO, "represent", "results", "curves",
                                         "schedules.json")))
    rng = np.random.default_rng(1)
    base_inits = [u_from_sched(np.asarray(scheds["wsd"], float)),
                  u_from_sched(np.asarray(scheds["wsdld"], float)),
                  u_from_sched(lin_sched(1200, 0.034)),
                  u_from_sched(lin_sched(3000, 0.034)),
                  rng.uniform(-2.2, -1.8, N_KNOTS)]

    out = {"mpl_params": [float(p) for p in params]}
    for tag, delta, kappa in ARMS:
        eA, JA, uA = solve(params, delta, 0.0, base_inits)
        # lag solve warm-started from the adiabatic solution too
        eL, JL, _ = solve(params, delta, kappa, base_inits + [uA])
        full = lambda e: final_mpl(e, params) + kappa * lagfeat_final(e, delta)
        gap = full(eA) - full(eL)
        wsdld = np.asarray(scheds["wsdld"], float)
        print(f"\n== {tag} ==")
        print(f"  adia-opt J_mpl={final_mpl(eA, params):.5f} full={full(eA):.5f}")
        print(f"  lag-opt  J_mpl={final_mpl(eL, params):.5f} full={full(eL):.5f}")
        print(f"  gap(adia-opt - lag-opt under full law) = {gap:+.6f}")
        print(f"  wsdld under full law = {full(wsdld):.5f}  (opt should beat this)")
        qs = [1000, 2000, 3000, 4000, 4500, 5000, 5400, 5700, 5900, 5999]
        print("    t    : " + " ".join(f"{q:6d}" for q in qs))
        print("    adia : " + " ".join(f"{eA[q]/PEAK:6.3f}" for q in qs))
        print("    lag  : " + " ".join(f"{eL[q]/PEAK:6.3f}" for q in qs))
        out[tag] = dict(gap=float(gap), J_adia=float(full(eA)),
                        J_lag=float(full(eL)), J_wsdld=float(full(wsdld)),
                        sched_adia=[float(x) for x in eA],
                        sched_lag=[float(x) for x in eL])

    # ---------- MPL-matched discrimination pairs ----------
    # reference arms: gradual linear cooldowns; matched arms: late-sharp linear drops
    # tuned so L_MPL(T) is EQUAL -> realized gap is pure lag, predicted = kappa*dfeat.
    print("\n== MPL-matched pairs (gradual ref vs late-sharp, equal L_MPL(T)) ==")
    pairs = []
    for ds_ref, ds_sharp in [(3000, 5700), (3500, 5500), (4000, 5800), (2500, 5900)]:
        ref = lin_sched(ds_ref, 0.1)
        Jref = final_mpl(ref, params)

        def mismatch(ef):
            return final_mpl(lin_sched(ds_sharp, ef), params) - Jref
        lo, hi = 0.034, 0.95
        try:
            if mismatch(lo) * mismatch(hi) > 0:
                print(f"  ds_ref={ds_ref} ds_sharp={ds_sharp}: no match in range "
                      f"(m(lo)={mismatch(lo):+.4f}, m(hi)={mismatch(hi):+.4f})")
                continue
            ef = brentq(mismatch, lo, hi, xtol=1e-5)
        except Exception as ex:
            print(f"  pair failed: {ex}")
            continue
        sharp = lin_sched(ds_sharp, ef)
        row = dict(ds_ref=ds_ref, ds_sharp=ds_sharp, ef_sharp=float(ef),
                   J_mpl=float(Jref))
        msg = (f"  ref(ds={ds_ref},end=0.10) vs sharp(ds={ds_sharp},"
               f"end={ef:.3f}) | J_mpl={Jref:.5f} | predicted pure-lag gap: ")
        for tag, delta, kappa in ARMS:
            g = kappa * (lagfeat_final(sharp, delta) - lagfeat_final(ref, delta))
            row[f"gap_{tag}"] = float(g)
            msg += f"{tag}:{g:+.5f} "
        print(msg)
        row["sched_ref"] = [float(x) for x in ref]
        row["sched_sharp"] = [float(x) for x in sharp]
        pairs.append(row)
    out["pairs"] = pairs

    json.dump(out, open(os.path.join(HERE, "opt_schedule_proto2.json"), "w"))
    print("\nsaved", os.path.join(HERE, "opt_schedule_proto2.json"))


if __name__ == "__main__":
    main()
