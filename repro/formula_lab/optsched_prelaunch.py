#!/usr/bin/env python3
"""Attempt 1F pre-launch CPU artifacts (committed before any GPU run):
1. Backbone-ensemble gap predictions: derby gaps under the primary split
   [constant, cosine, wsdcon_20] AND the alternate split
   [constant, cosine, wsdcon_40], multi-start; spread = ensemble uncertainty.
2. Paired-SD artifact: per-seed tail means over [5800, 6000) from the
   existing 3-seed trunk-shared trios (constant/onedrop/twodrop x
   {1337, 1338, 1339}); SD of the paired gaps pins the noise floor for the
   verdict rule (V1 fires only if gap - (prediction + ensemble spread)
   > 2*SE with SE = SD/sqrt(n_seeds)).
Output: results/formula_lab/optsched_predictions_m.json
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


def gaps_for_split(cv, split, seed):
    params, fobj = AC.fit_mpl(cv, split, n_starts=8, seed=seed)
    rows = {}
    for ds in DS:
        e = lin_sched(ds)
        J = float(mpl_loss_at(e, np.array([TOTAL - 1]), *params)[0])
        feats = {t: float(lagfeat_final(e, d)) for t, d, _ in ARMS}
        rows[ds] = (J, feats)
    ref = 3000
    out = {}
    for ds in DS:
        dmpl = rows[ds][0] - rows[ref][0]
        cell = {"dMPL": dmpl}
        for t, d, kap in ARMS:
            cell[t] = dmpl + kap * (rows[ds][1][t] - rows[ref][1][t])
        out[ds] = cell
    return out, fobj


def main():
    cv = AC.load_scale("m")
    ens = []
    for split in (["constant", "cosine", "wsdcon_20"],
                  ["constant", "cosine", "wsdcon_40"]):
        for seed in (0, 1):
            g, fobj = gaps_for_split(cv, split, seed)
            ens.append({"split": split, "fit_seed": seed, "obj": fobj,
                        "gaps": g})
            print(f"split={split} seed={seed} obj={fobj:.5f} "
                  f"g(5000)={g[5000]['d=0']*1e3:+.2f}e-3 "
                  f"g(5700)={g[5700]['d=0']*1e3:+.2f}e-3")

    # paired-SD from existing 3-seed trios
    cdir = os.path.join(REPO, "represent", "results", "curves_gen")
    mdir = os.path.join(REPO, "represent", "results", "curves")

    def tail_mean(path):
        rows = np.genfromtxt(path, delimiter=",", names=True)
        st = np.atleast_1d(rows["step"])
        ev = np.atleast_1d(rows["eval_loss"])
        m = (st >= 5800) & (st < 6000)
        return float(np.mean(ev[m]))

    def curve_path(name, seed):
        sfx = "" if seed == 1337 else f"_s{seed}"
        p1 = os.path.join(cdir, f"{name}{sfx}.csv")
        if os.path.exists(p1):
            return p1
        if name == "constant" and seed == 1337:
            return os.path.join(mdir, "m_constant.csv")
        return None

    gaps = {"onedrop-constant": [], "twodrop-onedrop": []}
    for seed in (1337, 1338, 1339):
        vals = {}
        for n in ("constant", "onedrop", "twodrop"):
            p = curve_path(n, seed)
            if p:
                vals[n] = tail_mean(p)
        if "onedrop" in vals and "constant" in vals:
            gaps["onedrop-constant"].append(vals["onedrop"] - vals["constant"])
        if "twodrop" in vals and "onedrop" in vals:
            gaps["twodrop-onedrop"].append(vals["twodrop"] - vals["onedrop"])
    paired_sd = {k: float(np.std(v, ddof=1)) if len(v) >= 2 else None
                 for k, v in gaps.items()}
    print("paired tail-gap values:", {k: np.round(v, 5).tolist()
                                      for k, v in gaps.items()})
    print("paired SD:", paired_sd)

    out = {
        "ensemble": ens,
        "ensemble_spread_g5000_d0": float(np.ptp([e["gaps"][5000]["d=0"] for e in ens])),
        "ensemble_spread_g5700_d0": float(np.ptp([e["gaps"][5700]["d=0"] for e in ens])),
        "paired_tail_gaps": gaps,
        "paired_sd": paired_sd,
        "verdict_rule": ("V1: g(5000-3000) - (pred + ensemble spread) > 2*SE; "
                         "V2 same for 5700; SE = SD/sqrt(3); contingency: if "
                         "observed paired SD > 1.2e-3 add seeds 1340/1341 on "
                         "ds in {3000,5000}"),
    }
    op = os.path.join(REPO, "results", "formula_lab", "optsched_predictions_m.json")
    json.dump(out, open(op, "w"), indent=1)
    print("wrote", op)


if __name__ == "__main__":
    main()
