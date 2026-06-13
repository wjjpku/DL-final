#!/usr/bin/env python3
"""ITEM A2: public floor-protocol extraction on OUR beds (zero-GPU).

Context (DECISION_TABLE): the public-bed superlinear floor exponents
p = 1.06/1.49/1.25 (25/100/400M) came from SETTLED WSDCON FLOORS -- each
probe holds stage-2 LR for many tau after an instant drop from peak, the
late-window loss is read off, and a power law in eta2 is fitted across the
probes.  Our AUDIT-C-corrected equal-S ladders instead give SUBLINEAR
p = 0.647 (m) / 0.641 (l).  Parameter count (scale ladder) and batch size
(bs192 ladder) are positively excluded as lone drivers; the attribution is
"source not localized (data/recipe/horizon/schedule family/floor protocol)".

This script gives the FLOOR-PROTOCOL hypothesis its direct test: run the
public settled-wsdcon protocol on our beds, from existing suite curves only.

  m bed (10.7M): represent/results/curves/m_wsdcon_{5,10,15,20,40,80}.csv
  l bed (25M)  : represent/results/curves_suite_l/wsdcon_{5,10,20,40,80}.csv

Suite wsdcon layout: warmup 400 -> hold peak 1.5e-3 until step 3000 ->
INSTANT drop to eta2 = tag*1e-5 -> hold to step 6000.  T2 = 3000 fixed for
every probe, so total S at measurement DIFFERS across probes (the public
protocol's backbone confound), unlike the equal-S ladder (T2 = 1.2/eta2).

PRIMARY (verdict-bearing, zero MPL refit, per G5):
  floor_i = mean smoothed eval loss over the design window
            step >= 3000 + 0.75*T2 = 5250  (same window rule as the
            AUDIT-C-corrected ladder -> the ONLY change vs the ladder
            number is the protocol itself);
  fit     = analyze_floor2.fit_p: floor = L + a*(eta2/peak)^p, free offset,
            residual bootstrap 90% CI (N=1000, seed 0).
  Report p_protocol(m), p_protocol(l).  Fire line: p_protocol > 1.

SENSITIVITIES (labeled, non-verdict-bearing):
  s1 m without wsdcon_15 (probe set matched to l's five);
  s2 floors = mean of last 5 RAW eval rows (the floor_powerlaw.py
     settled-tail mimic, no smoothing/window);
  s3 three-point subsets matching the public eta2/peak ratios
     (public {3,9,18}e-5 / 3e-4 = 0.1/0.3/0.6; m: wsdcon_{15,40,80} =
     0.1/0.267/0.533, l: wsdcon_{20,40,80} = 0.133/0.267/0.533) --
     3 params on 3 points = exact solve, point estimate only;
  s4 [G5: SENSITIVITY-ONLY MPL REFIT] the EXACT public-bed computation:
     subtract the fitted MPL backbone L0 + A*S(t)^-alpha (official split
     [constant, cosine, wsdcon_20], n_starts=8, seed=0) and fit the pure
     2-param log-log power law floor ~ eta^p on positive floors, as in
     repro/nonadiabatic_theory.estimate_dLeq_deta / floor_powerlaw.py.

Output: results/formula_lab/a2_protocol_extraction.json
Usage:  KMP_DUPLICATE_LIB_OK=TRUE python repro/formula_lab/a2_protocol_extraction.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "represent", "repro"))
import analyze_curves as AC          # noqa: E402
import analyze_floor2 as AF          # noqa: E402
from engine import cumS              # noqa: E402

SUITE_L = os.path.join(REPO, "represent", "results", "curves_suite_l")
OUT = os.path.join(REPO, "results", "formula_lab",
                   "a2_protocol_extraction.json")
DROP, T2 = 3000, 3000
CUT = DROP + int(0.75 * T2)          # 5250: design-window rule, fixed T2
PEAK = 1.5e-3                        # == analyze_floor2.PEAK == suite peak
THREE_POINT = {"m": ["wsdcon_15", "wsdcon_40", "wsdcon_80"],
               "l": ["wsdcon_20", "wsdcon_40", "wsdcon_80"]}
OFFICIAL_SPLIT = ["constant", "cosine", "wsdcon_20"]


def load_suite_l():
    """l suite -> the same dict format as AC.load_scale (cf.
    optsched_prelaunch_l.load_suite_l)."""
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


def probes(cv):
    """{eta2: name} for the wsdcon probes present in a suite."""
    out = {}
    for name in cv:
        if name.startswith("wsdcon_"):
            out[int(name.split("_")[1]) * 1e-5] = name
    return dict(sorted(out.items()))


def floor_window(c):
    m = c["step"] >= CUT
    assert m.sum() >= 4, f"only {m.sum()} eval rows in the design window"
    return float(np.mean(c["loss"][m])), int(m.sum())


def floor_last5_raw(c):
    return float(np.mean(c["loss_raw"][-5:]))


def fitp(etas, fls):
    p, lo, hi, L, a = AF.fit_p(np.asarray(etas, float),
                               np.asarray(fls, float))
    at_bound = ("lower(0.2)" if p < 0.2 + 1e-6 else
                "upper(3.5)" if p > 3.5 - 1e-6 else None)
    return dict(p=p, ci90=[lo, hi], L=L, a=a, p_at_bound=at_bound)


def loglog_p(etas, fls):
    """Public-bed 2-param fit: log floor ~ p log eta, positive floors only."""
    etas, fls = np.asarray(etas, float), np.asarray(fls, float)
    ok = fls > 0
    if ok.sum() < 2:
        return None, int(ok.sum())
    return float(np.polyfit(np.log(etas[ok]), np.log(fls[ok]), 1)[0]), \
        int(ok.sum())


def run_scale(scale, cv):
    pr = probes(cv)
    etas = np.array(list(pr))
    names = list(pr.values())
    print(f"\n== scale {scale}: {len(names)} settled wsdcon probes ==")
    fls_w, fls_r = [], []
    for e, n in pr.items():
        fw, nrows = floor_window(cv[n])
        fr = floor_last5_raw(cv[n])
        fls_w.append(fw)
        fls_r.append(fr)
        print(f"  {n:10s} eta2={e:.1e}  floor[win>={CUT}]={fw:.4f} "
              f"({nrows} rows)  floor[last5 raw]={fr:.4f}")
    fls_w, fls_r = np.array(fls_w), np.array(fls_r)

    res = {"probes": names, "eta2": etas.tolist(),
           "floors_window": fls_w.tolist(), "floors_last5_raw": fls_r.tolist(),
           "monotone_window": bool(np.all(np.diff(fls_w) > 0))}

    # PRIMARY
    prim = fitp(etas, fls_w)
    res["primary"] = prim
    print(f"  PRIMARY p_protocol({scale}) = {prim['p']:.3f} "
          f"90% CI [{prim['ci90'][0]:.3f}, {prim['ci90'][1]:.3f}]  "
          f"(L={prim['L']:.4f}, a={prim['a']:.4f}, "
          f"monotone={res['monotone_window']}, "
          f"at_bound={prim['p_at_bound']})")

    sens = {}
    # s1: m without wsdcon_15 (match l's probe set)
    if "wsdcon_15" in names:
        keep = [i for i, n in enumerate(names) if n != "wsdcon_15"]
        sens["s1_no_wsdcon15"] = fitp(etas[keep], fls_w[keep])
        print(f"  s1 (no wsdcon_15): p = {sens['s1_no_wsdcon15']['p']:.3f} "
              f"CI {sens['s1_no_wsdcon15']['ci90']}")
    # s2: last-5-raw floors
    sens["s2_last5_raw"] = fitp(etas, fls_r)
    print(f"  s2 (last5 raw):    p = {sens['s2_last5_raw']['p']:.3f} "
          f"CI {sens['s2_last5_raw']['ci90']}")
    # s3: matched-ratio 3-point exact fit (point estimate only)
    sel = [i for i, n in enumerate(names) if n in THREE_POINT[scale]]
    if len(sel) == 3:
        f3 = fitp(etas[sel], fls_w[sel])
        sens["s3_three_point"] = {"p": f3["p"], "probes": THREE_POINT[scale],
                                  "p_at_bound": f3["p_at_bound"],
                                  "note": "3-param/3-point; bootstrap CI "
                                          "not meaningful"}
        print(f"  s3 (3pt {','.join(THREE_POINT[scale])}): p = {f3['p']:.3f} "
              f"at_bound={f3['p_at_bound']}")
    # s5: monotone rising edge above the U minimum (removes the
    # non-monotonicity objection from the bound-pinned fits)
    imin = int(np.argmin(fls_w))
    if len(names) - imin >= 3:
        rng = list(range(imin, len(names)))
        f5 = fitp(etas[rng], fls_w[rng])
        sens["s5_rising_edge"] = {"p": f5["p"], "p_at_bound": f5["p_at_bound"],
                                  "probes": [names[i] for i in rng],
                                  "ci90": f5["ci90"]}
        print(f"  s5 (rising edge {','.join(names[i] for i in rng)}): "
              f"p = {f5['p']:.3f} at_bound={f5['p_at_bound']}")
    # s4: G5 SENSITIVITY-ONLY MPL refit -> backbone-subtracted log-log
    params, fobj = AC.fit_mpl(cv, OFFICIAL_SPLIT, n_starts=8, seed=0)
    L0, A, alpha = params[0], params[1], params[2]
    sub = []
    for n in names:
        c = cv[n]
        bb = L0 + A * np.power(cumS(c["lr"])[c["step"]], -alpha)
        m = c["step"] >= CUT
        sub.append(float(np.mean((c["loss"] - bb)[m])))
    p_ll, nin = loglog_p(etas, sub)
    s4 = {"label": "G5 SENSITIVITY-ONLY (MPL refit, official split, seed 0)",
          "mpl_obj": float(fobj), "L0_A_alpha": [float(L0), float(A),
                                                 float(alpha)],
          "floors_backbone_subtracted": sub,
          "p_loglog_all": p_ll, "n_positive": nin}
    p3 = loglog_p(etas[sel], np.array(sub)[sel])[0] if len(sel) == 3 else None
    s4["p_loglog_three_point"] = p3
    sens["s4_backbone_subtracted_loglog"] = s4
    print(f"  s4 (G5-labeled, backbone-subtracted log-log): "
          f"p_all = {p_ll if p_ll is None else round(p_ll, 3)} "
          f"({nin}/{len(names)} positive), 3pt = "
          f"{p3 if p3 is None else round(p3, 3)}")
    res["sensitivities"] = sens
    return res


def main():
    report = {"item": "A2 public floor-protocol extraction",
              "window": f"step >= {CUT} (design window, fixed T2={T2})",
              "fit": "analyze_floor2.fit_p (L + a*(eta2/1.5e-3)^p, "
                     "residual bootstrap 90% CI)",
              "ladder_reference": {"m": 0.647, "l": 0.641,
                                   "src": "AUDIT-C corrected equal-S ladder"},
              "public_reference": {"25M": 1.06, "100M": 1.49, "400M": 1.25}}
    report["m"] = run_scale("m", AC.load_scale("m"))
    report["l"] = run_scale("l", load_suite_l())

    pm, pl = report["m"]["primary"], report["l"]["primary"]
    fire = pm["p"] > 1 or pl["p"] > 1
    degenerate = (not report["m"]["monotone_window"]
                  or not report["l"]["monotone_window"]
                  or pm["p_at_bound"] or pl["p_at_bound"])
    report["verdict"] = {
        "fire_line": "p_protocol > 1 at either scale (all-probe fit_p)",
        "p_m": pm["p"], "p_l": pl["p"], "fires": bool(fire),
        "degenerate": bool(degenerate),
        "reading": ("protocol REPRODUCES superlinearity -> unlocalized-source "
                    "sentence localizes to floor protocol" if fire else
                    "all-probe protocol fit does NOT read superlinear on our "
                    "beds; see degeneracy flags and 3-point/rising-edge "
                    "sensitivities before striking the protocol")}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2, default=float)
    print(f"\nVERDICT: p_protocol(m)={pm['p']:.3f} {pm['ci90']}, "
          f"p_protocol(l)={pl['p']:.3f} {pl['ci90']} -> "
          f"{'FIRES (p>1)' if fire else 'no fire (sublinear)'}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
