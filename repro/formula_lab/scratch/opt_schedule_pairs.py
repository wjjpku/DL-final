"""Phase-0 quick part: MPL-matched discrimination pairs only (no variational solve).
Gradual reference vs late-sharp arm with equal L_MPL(T); predicted pure-lag gap
kappa*dfeat per closure arm. Also: lagfeat for existing baselines."""
import os, sys, json
import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "represent", "repro"))
import analyze_curves as AC
from engine import mpl_loss_at

PEAK, WARM, TOTAL = 1.5e-3, 400, 6000
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


def final_mpl(etas, params):
    return float(mpl_loss_at(etas, np.array([TOTAL - 1]), *params)[0])


def lin_sched(ds, ef):
    e = np.full(TOTAL, PEAK)
    e[:WARM] = PEAK * (np.arange(1, WARM + 1) / WARM)
    t = np.arange(ds, TOTAL)
    fr = (t - ds) / max(TOTAL - ds, 1)
    e[ds:] = PEAK * (1 - fr) + ef * PEAK * fr
    return e


def main():
    cv = AC.load_scale("m")
    params, fobj = AC.fit_mpl(cv, ["constant", "cosine", "wsdcon_20"], n_starts=8)
    print(f"MPL fit obj={fobj:.5f}")
    out = {"mpl_params": [float(p) for p in params]}
    pairs = []
    for ds_ref, ds_sharp in [(2500, 5400), (3000, 5600), (3500, 5700),
                             (4000, 5800), (3000, 5000)]:
        ref = lin_sched(ds_ref, 0.1)
        Jref = final_mpl(ref, params)

        def mm(ef):
            return final_mpl(lin_sched(ds_sharp, ef), params) - Jref
        lo, hi = 0.034, 0.98
        if mm(lo) * mm(hi) > 0:
            print(f"ref ds={ds_ref} sharp ds={ds_sharp}: NO MATCH "
                  f"(m(lo)={mm(lo):+.4f} m(hi)={mm(hi):+.4f})")
            continue
        ef = brentq(mm, lo, hi, xtol=1e-5)
        sharp = lin_sched(ds_sharp, ef)
        row = dict(ds_ref=ds_ref, ds_sharp=ds_sharp, ef_sharp=float(ef),
                   J_mpl=float(Jref))
        msg = (f"ref(ds={ds_ref}->0.10peak) vs sharp(ds={ds_sharp}->"
               f"{ef:.3f}peak)  J_mpl={Jref:.5f}  pure-lag gap: ")
        for tag, delta, kappa in ARMS:
            g = kappa * (lagfeat_final(sharp, delta) - lagfeat_final(ref, delta))
            row[f"gap_{tag}"] = float(g)
            msg += f"{tag}:{g:+.5f} "
        print(msg)
        pairs.append(row)
    out["pairs"] = pairs
    # baselines
    scheds = json.load(open(os.path.join(REPO, "represent", "results", "curves",
                                         "schedules.json")))
    for s in ["wsd", "wsdld", "cosine"]:
        e = np.asarray(scheds[s], float)
        print(f"baseline {s}: J_mpl={final_mpl(e, params):.5f} " +
              " ".join(f"feat({t})={lagfeat_final(e, d):.4f}"
                       for t, d, _ in ARMS))
    json.dump(out, open(os.path.join(HERE, "opt_schedule_pairs.json"), "w"))
    print("saved", os.path.join(HERE, "opt_schedule_pairs.json"))


if __name__ == "__main__":
    main()
