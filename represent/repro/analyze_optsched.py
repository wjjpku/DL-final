"""Attempt 1F derby verdicts -- executes optsched_predictions_m.json verbatim.

Metric: mean raw eval loss over [5800, 6000) per arm; per-seed paired gaps
vs ds=3000.  V1: g(5000) - (pred_d0 + ensemble spread) > 2*SE.
V2: same for g(5700).  V4 sanity: g(1300) within +/-3e-3 of +9.8e-3.
Closure separation: GLS-lite comparison of which closure's predicted gap
vector best matches the measured one (delta-chi2 >= 6 to claim).
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CDIR = os.path.join(ROOT, "results", "curves_optsched")
PRED = os.path.join(ROOT, "..", "results", "formula_lab",
                    "optsched_predictions_m.json")
SEEDS = [1337, 1338, 1339]
DS = [1300, 3000, 5000, 5700]


def tail_mean(tag):
    rows = np.genfromtxt(os.path.join(CDIR, tag + ".csv"), delimiter=",",
                         names=True)
    st = np.atleast_1d(rows["step"])
    ev = np.atleast_1d(rows["eval_loss"])
    m = (st >= 5800) & (st < 6000)
    return float(np.mean(ev[m]))


def main():
    pred = json.load(open(os.path.normpath(PRED)))
    L = {(ds, s): tail_mean(f"ds{ds}_s{s}") for ds in DS for s in SEEDS}
    print("== per-arm tail means ==")
    for ds in DS:
        print(f"  ds={ds}: " + " ".join(f"s{s}={L[(ds,s)]:.4f}" for s in SEEDS))

    gaps = {ds: [L[(ds, s)] - L[(3000, s)] for s in SEEDS] for ds in DS}
    print("\n== paired gaps vs ds=3000 (per seed) ==")
    stats = {}
    for ds in [1300, 5000, 5700]:
        g = np.array(gaps[ds])
        se = float(np.std(g, ddof=1) / np.sqrt(len(g)))
        stats[ds] = (float(np.mean(g)), se)
        print(f"  g({ds}) = {np.mean(g)*1e3:+.2f}e-3 +/- {se*1e3:.2f}e-3 "
              f"(per-seed: {[f'{x*1e3:+.2f}' for x in g]})")

    ens = pred["ensemble"]
    spread5000 = pred["ensemble_spread_g5000_d0"]
    spread5700 = pred["ensemble_spread_g5700_d0"]
    pred_d0_5000 = float(np.mean([e["gaps"]["5000"]["d=0"] for e in ens]))
    pred_d0_5700 = float(np.mean([e["gaps"]["5700"]["d=0"] for e in ens]))
    pred_mpl_5000 = float(np.mean([e["gaps"]["5000"]["dMPL"] for e in ens]))
    pred_mpl_5700 = float(np.mean([e["gaps"]["5700"]["dMPL"] for e in ens]))

    print("\n== verdicts ==")
    g5000, se5000 = stats[5000]
    v1 = (g5000 - (pred_mpl_5000 + spread5000)) > 2 * se5000
    print(f"  V1 (lag pricing at 5000): measured {g5000*1e3:+.2f}e-3 vs "
          f"MPL-alone {pred_mpl_5000*1e3:+.2f}e-3 + spread {spread5000*1e3:.2f}e-3; "
          f"2SE={2*se5000*1e3:.2f}e-3 -> {'FIRES' if v1 else 'null-consistent'}")
    g5700, se5700 = stats[5700]
    v2 = (g5700 - (pred_mpl_5700 + spread5700)) > 2 * se5700
    print(f"  V2 (lag pricing at 5700): measured {g5700*1e3:+.2f}e-3 vs "
          f"MPL-alone {pred_mpl_5700*1e3:+.2f}e-3 + spread {spread5700*1e3:.2f}e-3; "
          f"2SE={2*se5700*1e3:.2f}e-3 -> {'FIRES' if v2 else 'null-consistent'}")
    g1300, _ = stats[1300]
    v4 = abs(g1300 - 9.8e-3) <= 3e-3
    print(f"  V4 sanity (1300): measured {g1300*1e3:+.2f}e-3 vs ~+9.8e-3 "
          f"-> {'OK' if v4 else 'FLAG: pricing verdicts contingent'}")

    # closure selection (chi2 over the 3 informative gaps, per ensemble mean)
    print("\n== closure chi2 (lower better; need delta>=6 to claim) ==")
    meas = np.array([stats[1300][0], stats[5000][0], stats[5700][0]])
    ses = np.array([max(stats[ds][1], 1e-4) for ds in [1300, 5000, 5700]])
    for arm in ["dMPL", "d=0", "d=0.5", "d=0.75"]:
        pv = np.array([float(np.mean([e["gaps"][str(ds)][arm] for e in ens]))
                       for ds in [1300, 5000, 5700]])
        chi2 = float(np.sum(((meas - pv) / ses) ** 2))
        print(f"  {arm:7s}: chi2 = {chi2:8.2f}")

    # wsdld cross-check (3 seeds incl. existing 1337 m_wsdld)
    try:
        w = [tail_mean("wsdld_s1338"), tail_mean("wsdld_s1339")]
        rows = np.genfromtxt(os.path.join(ROOT, "results", "curves",
                                          "m_wsdld.csv"), delimiter=",",
                             names=True)
        st = np.atleast_1d(rows["step"]); ev = np.atleast_1d(rows["eval_loss"])
        m = (st >= 5800) & (st < 6000)
        w.append(float(np.mean(ev[m])))
        print(f"\n  wsdld tail means (s1338/s1339/s1337): "
              + " ".join(f"{x:.4f}" for x in w))
    except Exception as e:
        print("  wsdld check skipped:", e)


if __name__ == "__main__":
    main()
