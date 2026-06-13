"""Attempt 2 appendix verdicts -- executes optsched_predictions_l.json
verbatim (committed 8af32fa before the Phase-B launch).

V1_l/V2_l: lag pricing at 25M -- measured paired gap minus (ensemble-mean
dMPL + ensemble spread) > 2*SE, SE = paired SD / sqrt(3).
V5_transfer: per closure, GLS-lite chi2 over gaps {1300,5000,5700} for the
B_identity vs naive kappa families; separated only at |delta chi2| >= 6.
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CDIR = os.path.join(ROOT, "results", "curves_optsched_l")
PRED = os.path.join(ROOT, "..", "results", "formula_lab",
                    "optsched_predictions_l.json")
SEEDS = [1337, 1338, 1339]
DS = [1300, 3000, 5000, 5700]
CLOSURES = ["d=0", "d=0.5", "d=0.75"]


def tail_mean(tag):
    rows = np.genfromtxt(os.path.join(CDIR, tag + ".csv"), delimiter=",",
                         names=True)
    st = np.atleast_1d(rows["step"])
    ev = np.atleast_1d(rows["eval_loss"])
    m = (st >= 5800) & (st < 6000)
    return float(np.mean(ev[m]))


def main():
    pred = json.load(open(os.path.normpath(PRED)))
    ens = pred["ensemble"]
    L = {(ds, s): tail_mean(f"ds{ds}_s{s}") for ds in DS for s in SEEDS}
    print("== per-arm tail means ==")
    for ds in DS:
        print(f"  ds={ds}: " + " ".join(f"s{s}={L[(ds,s)]:.4f}"
                                        for s in SEEDS))
    stats = {}
    print("\n== paired gaps vs ds=3000 ==")
    for ds in [1300, 5000, 5700]:
        g = np.array([L[(ds, s)] - L[(3000, s)] for s in SEEDS])
        sd = float(np.std(g, ddof=1))
        stats[ds] = (float(np.mean(g)), sd / np.sqrt(len(g)), sd)
        print(f"  g({ds}) = {np.mean(g)*1e3:+.2f}e-3 +/- "
              f"{sd/np.sqrt(3)*1e3:.2f}e-3  (SD {sd*1e3:.2f}e-3, per-seed "
              + " ".join(f"{x*1e3:+.2f}" for x in g) + ")")
    sd_max = max(s[2] for s in stats.values())
    if sd_max > 1.5e-3:
        print(f"  CONTINGENCY: max paired SD {sd_max*1e3:.2f}e-3 > 1.5e-3 "
              f"-> prereg asks seeds 1340/1341 on ds {{3000,5700}}")

    def emean(fam, ds, arm):
        return float(np.mean([e["gaps"][fam][str(ds)][arm] for e in ens]))

    print("\n== verdicts ==")
    for vname, ds, spread_key in [("V1_l", 5000, "ensemble_spread_g5000_dMPL"),
                                  ("V2_l", 5700, "ensemble_spread_g5700_dMPL")]:
        gm, se, _ = stats[ds]
        base = emean("B_identity", ds, "dMPL") + pred[spread_key]
        fires = (gm - base) > 2 * se
        print(f"  {vname}: measured {gm*1e3:+.2f}e-3 vs MPL-alone "
              f"{emean('B_identity', ds, 'dMPL')*1e3:+.2f}e-3 + spread "
              f"{pred[spread_key]*1e3:.2f}e-3; 2SE={2*se*1e3:.2f}e-3 -> "
              f"{'FIRES' if fires else 'null-consistent'}")

    print("\n== V5_transfer: chi2 per closure (gaps 1300/5000/5700) ==")
    meas = np.array([stats[ds][0] for ds in [1300, 5000, 5700]])
    ses = np.array([max(stats[ds][1], 1e-4) for ds in [1300, 5000, 5700]])
    chi = {}
    for fam in ["B_identity", "naive"]:
        row = {}
        for arm in ["dMPL"] + CLOSURES:
            pv = np.array([emean(fam, ds, arm) for ds in [1300, 5000, 5700]])
            row[arm] = float(np.sum(((meas - pv) / ses) ** 2))
        chi[fam] = row
        print(f"  {fam:11s}: " + " ".join(f"{a}={row[a]:9.2f}"
                                          for a in ["dMPL"] + CLOSURES))
    print("  per-closure delta-chi2 (naive - B_identity; >=6 separates):")
    for arm in CLOSURES:
        d = chi["naive"][arm] - chi["B_identity"][arm]
        tag = ("B_identity wins" if d >= 6 else
               "naive wins" if d <= -6 else "NOT SEPARATED")
        print(f"    {arm:7s}: {d:+8.2f}  -> {tag}")
    json.dump({"tail_means": {f"{k[0]}_{k[1]}": v for k, v in L.items()},
               "gaps": {str(k): v for k, v in stats.items()},
               "chi2": chi},
              open(os.path.join(ROOT, "results", "OPTSCHED_L_REPORT.json"),
                   "w"), indent=1)
    print("\nsaved results/OPTSCHED_L_REPORT.json")


if __name__ == "__main__":
    main()
