#!/usr/bin/env python3
"""Attempt 2 schedule appendix, pre-launch CPU artifacts (G4: committed
before the Phase-B GPU derby at l).

Same derby design as 1F but on the 25M matched-recipe bed, with TWO
pre-registered kappa-transfer families:
  B_identity : kappa_l = kappa_m * B_l/B_m  (the shipped zero-measurement
               chain dL_eq/deta = B; per ensemble member, matched split/seed)
  naive      : kappa_l = kappa_m  (no rescaling)
Backbone ensemble: splits [constant,cosine,wsdcon_20] x [.._40], fit seeds
{0,1}, fitted independently on the m suite (results/curves) and the l suite
(represent/results/curves_suite_l).

Verdicts (3 seeds, tail [5800,6000), ref ds=3000):
  V1_l/V2_l : lag pricing at l -- g(5000)/g(5700) - (MPL-alone + spread)
              > 2*SE  (SE = paired SD / sqrt(3); SD prior 0.99e-3 from the
              m trios; contingency: if observed l paired SD > 1.5e-3, add
              seeds 1340/1341 on ds {3000,5700})
  V5_transfer: GLS-lite chi2 over the 3 informative gaps, per closure:
              B_identity beats naive with delta-chi2 >= 6 (or vice versa);
              else NOT SEPARATED.
Output: results/formula_lab/optsched_predictions_l.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "represent", "repro"))
sys.path.insert(0, HERE)
import analyze_curves as AC  # noqa: E402
from engine import mpl_loss_at, cumS  # noqa: E402
from opt_schedule import lagfeat_final, lin_sched, ARMS, DS, TOTAL  # noqa: E402

SUITE_L = os.path.join(REPO, "represent", "results", "curves_suite_l")
SPLITS = (["constant", "cosine", "wsdcon_20"],
          ["constant", "cosine", "wsdcon_40"])
FIT_SEEDS = (0, 1)
REF = 3000


def load_suite_l():
    """l suite -> the same dict format as AC.load_scale."""
    scheds = json.load(open(os.path.join(SUITE_L, "schedules.json")))
    out = {}
    for name, lr in scheds.items():
        f = os.path.join(SUITE_L, f"{name}.csv")
        if not os.path.exists(f):
            continue
        rows = np.genfromtxt(f, delimiter=",", names=True)
        step = np.atleast_1d(rows["step"]).astype(int)
        loss = np.atleast_1d(rows[AC.LOSS_COL]).astype(float)
        keep = step >= AC.T_MIN
        st, ls = step[keep], loss[keep]
        out[name] = dict(step=st, lr=np.asarray(lr, float),
                         loss=AC.smooth_by_step(st, ls), loss_raw=ls)
    return out


def member_fits(cv):
    out = []
    for split in SPLITS:
        for seed in FIT_SEEDS:
            params, fobj = AC.fit_mpl(cv, split, n_starts=8, seed=seed)
            out.append(dict(split=split, fit_seed=seed, obj=float(fobj),
                            params=[float(p) for p in params]))
    return out


def gaps_for(params, kappas):
    rows = {}
    for ds in DS:
        e = lin_sched(ds)
        J = float(mpl_loss_at(e, np.array([TOTAL - 1]), *params)[0])
        feats = {t: float(lagfeat_final(e, d)) for t, d, _ in ARMS}
        rows[ds] = (J, feats)
    out = {}
    for ds in DS:
        dmpl = rows[ds][0] - rows[REF][0]
        cell = {"dMPL": float(dmpl)}
        for t, d, _ in ARMS:
            cell[t] = float(dmpl + kappas[t]
                            * (rows[ds][1][t] - rows[REF][1][t]))
        out[str(ds)] = cell
    return out


def main():
    cv_m = AC.load_scale("m")
    cv_l = load_suite_l()
    print("l suite schedules:", sorted(cv_l))
    fits_m = member_fits(cv_m)
    fits_l = member_fits(cv_l)
    kap_m = {t: k for t, _, k in ARMS}
    ens = []
    for fm, fl in zip(fits_m, fits_l):
        B_m, B_l = fm["params"][3], fl["params"][3]
        ratio = B_l / B_m
        fam = {}
        fam["B_identity"] = gaps_for(fl["params"],
                                     {t: kap_m[t] * ratio for t in kap_m})
        fam["naive"] = gaps_for(fl["params"], kap_m)
        ens.append(dict(split=fl["split"], fit_seed=fl["fit_seed"],
                        obj_l=fl["obj"], obj_m=fm["obj"],
                        B_m=float(B_m), B_l=float(B_l),
                        kappa_ratio=float(ratio), gaps=fam))
        print(f"split={fl['split']} seed={fl['fit_seed']} "
              f"obj_l={fl['obj']:.5f} B_m={B_m:.2f} B_l={B_l:.2f} "
              f"ratio={ratio:.3f} "
              f"gBid(5700,d=.75)={fam['B_identity']['5700']['d=0.75']*1e3:+.2f}e-3 "
              f"gnaive={fam['naive']['5700']['d=0.75']*1e3:+.2f}e-3")

    def spread(fam, ds, arm):
        return float(np.ptp([e["gaps"][fam][str(ds)][arm] for e in ens]))

    out = dict(
        name="25M matched-recipe cooldown derby (Attempt 2 appendix)",
        committed_before_launch=True,
        design=("4 linear-cooldown starts ds in {1300,3000,5000,5700} -> "
                "0.1*peak hold, total 6000, seeds {1337,1338,1339}, bs=48, "
                "l recipe d=512 nh=8 nl=8 (train_optsched.py --scale l); "
                "metric mean raw eval loss [5800,6000); ref ds=3000"),
        ensemble=ens,
        ensemble_spread_g5000_dMPL=spread("B_identity", 5000, "dMPL"),
        ensemble_spread_g5700_dMPL=spread("B_identity", 5700, "dMPL"),
        paired_sd_prior=0.99e-3,
        verdict_rule=("V1_l/V2_l: lag pricing at l fires if "
                      "g_meas - (mean dMPL + spread_dMPL) > 2*SE, "
                      "SE=SD/sqrt(3) with SD prior 0.99e-3 replaced by the "
                      "observed l paired SD; contingency seeds 1340/1341 on "
                      "ds {3000,5700} if SD > 1.5e-3.  V5_transfer: per "
                      "closure, GLS-lite chi2 over gaps {1300,5000,5700}: "
                      "B_identity vs naive separated only at delta-chi2>=6."),
        dof_note=("zero new DOF: B_identity ratio uses only fitted backbone "
                  "B's; kappa_m frozen from the m bed (G5 safe)"))
    op = os.path.join(REPO, "results", "formula_lab",
                      "optsched_predictions_l.json")
    json.dump(out, open(op, "w"), indent=1)
    print("wrote", op)


if __name__ == "__main__":
    main()
