#!/usr/bin/env python3
"""Audit remediation (mid-round adversarial audit, findings 1F-1/3/4 and the
l-bed inheritance): like-for-like derby re-analysis.

The committed predictions were ENDPOINT (step 5999) values while the verdict
metric is the mean over [5800,6000) of still-cooling curves.  Here both sides
use the same 21-step eval grid (5800..5990 step 10, plus 5999) -- zero new
freedom: identical splits/fit-seeds/kappas as the committed prereg, only the
evaluation grid is corrected.  Also executes:
  - m contingency restatement on 5 seeds when ds3000/5000 s1340/41 exist;
  - l restatement of V2_l on 5 seeds (contingency landed);
  - widened backbone ensemble (5 extra single-schedule splits) for an honest
    dMPL spread (sensitivity only, G5-safe);
  - closure chi2 like-for-like, labeled POST-HOC for the m bed (not prereg'd
    there; prereg'd only at l).
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
from engine import mpl_loss_at  # noqa: E402
from opt_schedule import lin_sched, ARMS, DS, TOTAL, PEAK, LAM  # noqa: E402
from optsched_prelaunch_l import load_suite_l  # noqa: E402

EVAL_STEPS = np.array(sorted(set(range(5800, 6000, 10)) | {5999}))
SPLITS_CORE = (["constant", "cosine", "wsdcon_20"],
               ["constant", "cosine", "wsdcon_40"])
SPLITS_WIDE = (["constant", "cosine", "wsd"],
               ["constant", "cosine", "wsdld"],
               ["constant", "cosine", "wsdcon_5"],
               ["constant", "cosine", "wsdcon_10"],
               ["constant", "cosine", "wsdcon_80"])
REF = 3000
KAP_M = {t: k for t, _, k in ARMS}


def lagfeat_steps(etas, delta, steps):
    eta = np.asarray(etas, float)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
    if delta > 0:
        drop = drop * np.power(np.maximum(eta / PEAK, 1e-12), delta)
    dec = np.exp(-LAM * eta)
    acc, out, want = 0.0, {}, set(int(s) for s in steps)
    for t in range(len(eta)):
        acc = acc * dec[t] + drop[t]
        if t in want:
            out[t] = acc / PEAK
    return float(np.mean([out[int(s)] for s in steps]))


def win_gaps(params, kappas):
    rows = {}
    for ds in DS:
        e = lin_sched(ds)
        J = float(np.mean(mpl_loss_at(e, EVAL_STEPS, *params)))
        feats = {t: lagfeat_steps(e, d, EVAL_STEPS) for t, d, _ in ARMS}
        rows[ds] = (J, feats)
    out = {}
    for ds in DS:
        dm = rows[ds][0] - rows[REF][0]
        cell = {"dMPL": float(dm)}
        for t, d, _ in ARMS:
            cell[t] = float(dm + kappas[t] * (rows[ds][1][t]
                                              - rows[REF][1][t]))
        out[ds] = cell
    return out


def tail_mean(cdir, tag):
    f = os.path.join(cdir, tag + ".csv")
    if not os.path.exists(f):
        return None
    rows = np.genfromtxt(f, delimiter=",", names=True)
    st = np.atleast_1d(rows["step"])
    ev = np.atleast_1d(rows["eval_loss"])
    m = (st >= 5800) & (st < 6000)
    return float(np.mean(ev[m]))


def measured(cdir, seeds_by_ds):
    L = {}
    for ds, seeds in seeds_by_ds.items():
        for s in seeds:
            v = tail_mean(cdir, f"ds{ds}_s{s}")
            if v is not None:
                L[(ds, s)] = v
    return L


def gap_stats(L, ds, ref_seeds):
    seeds = [s for s in ref_seeds if (ds, s) in L and (REF, s) in L]
    g = np.array([L[(ds, s)] - L[(REF, s)] for s in seeds])
    return float(np.mean(g)), float(np.std(g, ddof=1) / np.sqrt(len(g))), \
        len(seeds)


def run_bed(name, cv, kap_fams, cdir, seeds_by_ds, prereg_label):
    print(f"\n======== {name} bed (like-for-like, window-mean both sides) "
          f"========")
    ens, wide = [], []
    for split_set, sink in ((SPLITS_CORE, ens), (SPLITS_WIDE, wide)):
        for split in split_set:
            try:
                params, fobj = AC.fit_mpl(cv, split, n_starts=8, seed=0)
            except Exception as e:
                print(f"  split {split}: fit failed ({e})")
                continue
            fams = {fn: win_gaps(params, kf(params)) for fn, kf in kap_fams}
            sink.append(dict(split=split, obj=float(fobj), fams=fams))
    L = measured(cdir, seeds_by_ds)
    allseeds = sorted({s for (_, s) in L})
    res = {"bed": name, "gaps": {}, "verdicts": {}}
    print("  measured paired gaps (vs ds=3000, matched seeds):")
    for ds in [1300, 5000, 5700]:
        gm, se, n = gap_stats(L, ds, allseeds)
        res["gaps"][ds] = (gm, se, n)
        print(f"    g({ds}) = {gm*1e3:+.2f}e-3 +/- {se*1e3:.2f}e-3 (n={n})")
    fam0 = ens[0]["fams"].keys().__iter__().__next__()
    for vn, ds in (("V1", 5000), ("V2", 5700)):
        gm, se, n = res["gaps"][ds]
        preds = [e["fams"][fam0][ds]["dMPL"] for e in ens]
        pmean, spread = float(np.mean(preds)), float(np.ptp(preds))
        wpreds = preds + [e["fams"][fam0][ds]["dMPL"] for e in wide]
        wspread = float(np.ptp(wpreds))
        fires = (gm - (pmean + spread)) > 2 * se
        fires_w = (gm - (pmean + wspread)) > 2 * se
        res["verdicts"][vn] = dict(measured=gm, pred=pmean, spread=spread,
                                   wide_spread=wspread, se=se, n=n,
                                   fires=bool(fires),
                                   fires_wide=bool(fires_w))
        print(f"  {vn}({ds}): meas {gm*1e3:+.2f} vs dMPL {pmean*1e3:+.2f} "
              f"+ spread {spread*1e3:.2f} (wide {wspread*1e3:.2f}); "
              f"2SE={2*se*1e3:.2f} -> {'FIRES' if fires else 'null'}"
              f" / wide-ensemble {'FIRES' if fires_w else 'null'}")
    meas = np.array([res["gaps"][d][0] for d in [1300, 5000, 5700]])
    ses = np.array([max(res["gaps"][d][1], 1e-4) for d in [1300, 5000, 5700]])
    print(f"  closure chi2 over (1300,5000,5700) [{prereg_label}]:")
    chi = {}
    for fam in ens[0]["fams"]:
        row = {}
        for arm in ["dMPL"] + [t for t, _, _ in ARMS]:
            pv = np.array([float(np.mean([e["fams"][fam][d][arm]
                                          for e in ens]))
                           for d in [1300, 5000, 5700]])
            row[arm] = float(np.sum(((meas - pv) / ses) ** 2))
        chi[fam] = row
        print("    " + fam + ": " + " ".join(f"{a}={row[a]:8.2f}"
                                             for a in row))
    res["chi2"] = chi
    res["ensemble_objs"] = [e["obj"] for e in ens] + [e["obj"] for e in wide]
    return res


def main():
    out = {}
    cv_m = AC.load_scale("m")
    m_dirs = os.path.join(REPO, "represent", "results", "curves_optsched")
    m_seeds = {1300: [1337, 1338, 1339], 3000: [1337, 1338, 1339, 1340, 1341],
               5000: [1337, 1338, 1339, 1340, 1341],
               5700: [1337, 1338, 1339]}
    out["m"] = run_bed("m", cv_m,
                       [("fixed", lambda p: KAP_M)],
                       m_dirs, m_seeds, "POST-HOC at m (not prereg'd)")

    cv_l = load_suite_l()
    pl = json.load(open(os.path.join(REPO, "results", "formula_lab",
                                     "optsched_predictions_l.json")))
    rat = float(np.mean([e["kappa_ratio"] for e in pl["ensemble"]]))
    rats = sorted(set(round(e["kappa_ratio"], 3) for e in pl["ensemble"]))
    print(f"\n(kappa_ratio ensemble mean {rat:.3f}; split-instability "
          f"committed values {rats} -- reported per audit item 5)")
    l_dirs = os.path.join(REPO, "represent", "results", "curves_optsched_l")
    l_seeds = {1300: [1337, 1338, 1339], 3000: [1337, 1338, 1339, 1340, 1341],
               5000: [1337, 1338, 1339],
               5700: [1337, 1338, 1339, 1340, 1341]}

    def bid(params):
        return {t: KAP_M[t] * rat for t in KAP_M}

    out["l"] = run_bed("l", cv_l,
                       [("B_identity", bid), ("naive", lambda p: KAP_M)],
                       l_dirs, l_seeds, "prereg'd at l (8af32fa)")
    ch = out["l"]["chi2"]
    print("  V5_transfer like-for-like (naive - B_identity, >=6 separates):")
    for arm in [t for t, _, _ in ARMS]:
        d = ch["naive"][arm] - ch["B_identity"][arm]
        print(f"    {arm:7s}: {d:+8.2f} -> "
              + ("B_identity" if d >= 6 else
                 "naive" if d <= -6 else "NOT SEPARATED"))
    op = os.path.join(REPO, "results", "formula_lab",
                      "derby_likeforlike.json")
    json.dump(out, open(op, "w"), indent=1, default=float)
    print("\nwrote", op)


if __name__ == "__main__":
    main()
