#!/usr/bin/env python3
"""ITEM E1-CERT: backbone-null matched-S paired derby -- CPU certification
with self-bucketing kill criterion.  NO GPU launch here.

Design goal: schedule PAIRS over 6000 steps (peak 1.5e-3, warmup 400) that
share (i) total S = sum(eta), (ii) the same terminal LR (0.1*peak), and
(iii) the same end step, but differ in cooldown SHAPE.  Every design reaches
the terminal LR by step <= 5700 and holds, so S(t) is matched at EVERY step
of the eval window [5800,6000), not just at the end -- the MPL main term
A*(S+Sw)^-alpha cancels EXACTLY within a pair; only the loss-drop term's
placement (eta_k, S_k of the decrements) survives.  That residual, evaluated
under ALL 7 ensemble backbones (SPLITS_CORE + SPLITS_WIDE from
derby_likeforlike.py, m suite, identical fit settings), IS the null
uncertainty: dMPL across backbones (mean = systematic non-cancellation,
ptp spread = backbone uncertainty).

The lag law predicts a nonzero pair difference: kappa * (feat_1 - feat_2)
with lagfeat over the same 21-step eval grid (cloned verbatim from
derby_likeforlike.lagfeat_steps) and kappa FIXED from the m-bed ARMS values
(opt_schedule.ARMS; lam* = 1).  No MPL refit is verdict-bearing: the 7
backbone fits are the same ensemble machinery as the shipped derby
(fit on training splits, never on a target).

CERTIFICATION (per closure):
  |lag-predicted difference| > 3 * (7-backbone ptp spread of dMPL)
  AND |lag-predicted difference| > 2 * SE_paired_hi (1.4e-3, the
  conservative end of the known paired-SE band 0.7-1.4e-3 at 3-5 seeds).
Headline pair certification = BOTH d=0.5 (shipped frontier) AND d=0.75
closures pass; d=0 is reported separately (structurally near-null here:
for monotone same-endpoint cooldowns the unweighted deposit integral is
shape-invariant, so d=0 differences come from S-time decay timing only).
If NO pair certifies -> E1 self-buckets to B.

Outputs results/formula_lab/e1_designs.json (generation parameters for all
pairs; exact eta arrays for the certified/best pairs).

Run:  KMP_DUPLICATE_LIB_OK=TRUE python repro/formula_lab/e1_certify.py
"""
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "represent", "repro"))
sys.path.insert(0, HERE)
import analyze_curves as AC                       # noqa: E402
from engine import mpl_loss_at                    # noqa: E402
from opt_schedule import ARMS, LAM, PEAK, WARM, TOTAL  # noqa: E402

END_LR = 0.1 * PEAK            # terminal LR, identical for every arm
TE_LATEST = 5700               # all arms at END_LR from here -> S(t) matched
                               # at every eval step within a pair
EVAL_STEPS = np.array(sorted(set(range(5800, 6000, 10)) | {5999}))
SPLITS_CORE = (["constant", "cosine", "wsdcon_20"],
               ["constant", "cosine", "wsdcon_40"])
SPLITS_WIDE = (["constant", "cosine", "wsd"],
               ["constant", "cosine", "wsdld"],
               ["constant", "cosine", "wsdcon_5"],
               ["constant", "cosine", "wsdcon_10"],
               ["constant", "cosine", "wsdcon_80"])
SE_HI, SE_LO = 1.4e-3, 0.7e-3  # known paired-SE band at 3-5 seeds
SIG1_HI, SIG1_LO = 2.4e-3, 1.6e-3  # implied per-seed paired SD band
BACKBONE_CACHE = os.path.join(REPO, "results", "formula_lab",
                              "e1_backbones.json")
OUT_PATH = os.path.join(REPO, "results", "formula_lab", "e1_designs.json")


# ---------------- lag feature: cloned verbatim from derby_likeforlike ----
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


# ---------------- schedule generators (continuous params, integer steps) --
def sched_ramp(ts, te, q):
    """warmup 400 -> hold peak -> power ramp (1-u)^q from peak to END_LR over
    [ts, te] (ts may be float; eta sampled at integer steps) -> hold END_LR."""
    t = np.arange(TOTAL, dtype=float)
    u = np.clip((t - ts) / max(te - ts, 1e-9), 0.0, 1.0)
    e = END_LR + (PEAK - END_LR) * np.power(1.0 - u, q)
    e[:WARM] = PEAK * np.arange(1, WARM + 1) / WARM
    return e


def sched_steps(drop_list):
    """warmup 400 -> hold peak -> instantaneous drops [(t_d, level), ...]
    (levels strictly decreasing, last level == END_LR)."""
    e = np.full(TOTAL, PEAK)
    for td, lv in drop_list:
        e[int(td):] = lv
    e[:WARM] = PEAK * np.arange(1, WARM + 1) / WARM
    return e


def solve_partner(S_target, te, q):
    """Solve ramp start ts (float) so sum(sched_ramp(ts,te,q)) == S_target.
    sum is continuous & strictly increasing in ts.  Returns ts or None."""
    lo, hi = WARM + 50.0, te - 300.0   # keep a non-degenerate ramp
    f = lambda ts: float(np.sum(sched_ramp(ts, te, q)) - S_target)
    if f(lo) > 0 or f(hi) < 0:
        return None
    return float(brentq(f, lo, hi, xtol=1e-9))


# ---------------- design space (modest grid) -----------------------------
def design_pairs():
    """Each pair: anchor arm (integer params) + partner ramp with ts solved
    for exact S-match.  diff convention everywhere: arm1(anchor) - arm2."""
    specs = []
    # F1: single sharp drop (wsdcon-like) vs linear cooldown to 5700
    for td in (3600, 4200, 4800, 5200):
        specs.append(dict(
            name=f"sharp{td}_vs_lin5700", family="sharp_vs_linear",
            anchor=dict(kind="steps", drops=[[td, END_LR]]),
            partner=dict(te=TE_LATEST, q=1.0)))
    # F1b: sharp drop vs SHORTER linear (te=5500)
    specs.append(dict(
        name="sharp4800_vs_lin5500", family="sharp_vs_linear",
        anchor=dict(kind="steps", drops=[[4800, END_LR]]),
        partner=dict(te=5500, q=1.0)))
    # F2: two-phase step schedule vs linear cooldown
    for t1, t2, mid in ((3000, 5200, 0.5), (2500, 5400, 0.5),
                        (3500, 5000, 0.4), (2000, 4800, 0.6)):
        specs.append(dict(
            name=f"twophase{t1}-{t2}m{mid}_vs_lin5700",
            family="twophase_vs_linear",
            anchor=dict(kind="steps",
                        drops=[[t1, mid * PEAK], [t2, END_LR]]),
            partner=dict(te=TE_LATEST, q=1.0)))
    # F3: fast early linear cooldown (long END_LR tail) vs slow long ramp
    for a0 in (2500, 3000, 3500, 4000):
        specs.append(dict(
            name=f"fast{a0}-{a0+400}_vs_lin5700", family="fastearly_vs_slow",
            anchor=dict(kind="ramp", ts=float(a0), te=float(a0 + 400), q=1.0),
            partner=dict(te=TE_LATEST, q=1.0)))
    # F4: concave (holds high, plunges late) vs convex (drops early, flat)
    for qa, tsa, qb in ((0.4, 3600.0, 2.5), (0.4, 3000.0, 2.5),
                        (0.3, 4000.0, 3.0), (0.5, 3300.0, 2.0)):
        specs.append(dict(
            name=f"concq{qa}@{int(tsa)}_vs_convq{qb}",
            family="concave_vs_convex",
            anchor=dict(kind="ramp", ts=tsa, te=float(TE_LATEST), q=qa),
            partner=dict(te=TE_LATEST, q=qb)))
    # F5: sharp drop vs concave ramp (max deposit-weight contrast)
    for td in (4200, 4800):
        specs.append(dict(
            name=f"sharp{td}_vs_concq0.4", family="sharp_vs_concave",
            anchor=dict(kind="steps", drops=[[td, END_LR]]),
            partner=dict(te=TE_LATEST, q=0.4)))
    return specs


def build_arm(a):
    if a["kind"] == "steps":
        return sched_steps([(int(t), float(lv)) for t, lv in a["drops"]])
    return sched_ramp(a["ts"], a["te"], a["q"])


# ---------------- backbone ensemble (7 fits, cached) ----------------------
def get_backbones(cv):
    cache = {}
    if os.path.exists(BACKBONE_CACHE):
        cache = json.load(open(BACKBONE_CACHE))
    fits, dirty = {}, False
    for split in list(SPLITS_CORE) + list(SPLITS_WIDE):
        key = "+".join(split)
        if key in cache:
            fits[key] = cache[key]
            continue
        params, fobj = AC.fit_mpl(cv, split, n_starts=8, seed=0)
        fits[key] = dict(params=[float(p) for p in params], obj=float(fobj))
        cache[key] = fits[key]
        dirty = True
        print(f"  fitted backbone {key}: obj={fobj:.5f}")
    if dirty:
        json.dump(cache, open(BACKBONE_CACHE, "w"), indent=1)
    return fits


def j_window(eta, params):
    return float(np.mean(mpl_loss_at(eta, EVAL_STEPS, *params)))


# ---------------- evaluation ---------------------------------------------
def evaluate_pair(spec, fits):
    e1 = build_arm(spec["anchor"])
    S1 = float(np.sum(e1))
    ts = solve_partner(S1, spec["partner"]["te"], spec["partner"]["q"])
    if ts is None:
        return None, None, None
    e2 = sched_ramp(ts, spec["partner"]["te"], spec["partner"]["q"])
    res = dict(name=spec["name"], family=spec["family"],
               anchor=spec["anchor"],
               partner=dict(kind="ramp", ts=ts, **spec["partner"]),
               S_total=S1, dS=float(np.sum(e1) - np.sum(e2)),
               terminal_lr=[float(e1[-1]), float(e2[-1])])
    dJ = {}
    for key, f in fits.items():
        dJ[key] = j_window(e1, f["params"]) - j_window(e2, f["params"])
    vals = np.array(list(dJ.values()))
    core = np.array([dJ["+".join(s)] for s in SPLITS_CORE])
    res["dMPL"] = dict(per_backbone=dJ, mean=float(vals.mean()),
                       spread7=float(np.ptp(vals)),
                       spread_core=float(np.ptp(core)))
    sp7 = res["dMPL"]["spread7"]
    res["closures"] = {}
    for tname, delta, kappa in ARMS:
        f1 = lagfeat_steps(e1, delta, EVAL_STEPS)
        f2 = lagfeat_steps(e2, delta, EVAL_STEPS)
        lag = kappa * (f1 - f2)
        cl = dict(kappa=kappa, feat1=f1, feat2=f2, lag_diff=float(lag),
                  pass_3x_spread=bool(abs(lag) > 3 * sp7),
                  pass_2se_hi=bool(abs(lag) > 2 * SE_HI),
                  pass_2se_lo=bool(abs(lag) > 2 * SE_LO),
                  certified=bool(abs(lag) > 3 * sp7
                                 and abs(lag) > 2 * SE_HI))
        for tag, s1 in (("hi", SIG1_HI), ("lo", SIG1_LO)):
            cl[f"seeds_required_sig{tag}"] = (
                int(math.ceil((2 * s1 / abs(lag)) ** 2))
                if lag != 0 else None)
        res["closures"][tname] = cl
    c = res["closures"]
    res["certified_d05_d075"] = bool(c["d=0.5"]["certified"]
                                     and c["d=0.75"]["certified"])
    res["certified_d0"] = bool(c["d=0"]["certified"])
    # ranking margin: weakest of the two d>0 closures, vs the binding gate
    gate = max(3 * sp7, 2 * SE_HI)
    res["margin"] = float(min(abs(c["d=0.5"]["lag_diff"]),
                              abs(c["d=0.75"]["lag_diff"])) / gate)
    return res, e1, e2


def main():
    print("E1-CERT: backbone-null matched-S paired derby (CPU only)")
    cv = AC.load_scale("m")
    need = sorted({s for sp in list(SPLITS_CORE) + list(SPLITS_WIDE)
                   for s in sp})
    missing = [s for s in need if s not in cv]
    if missing:
        raise SystemExit(f"missing m-suite schedules: {missing}")
    fits = get_backbones(cv)
    print(f"backbone ensemble: {len(fits)} fits  "
          f"objs={[round(f['obj'], 4) for f in fits.values()]}")

    results, arrays = [], {}
    for spec in design_pairs():
        res, e1, e2 = evaluate_pair(spec, fits)
        if res is None:
            print(f"  {spec['name']}: S-match infeasible, skipped")
            continue
        results.append(res)
        arrays[res["name"]] = (e1, e2)
        c = res["closures"]
        print(f"  {res['name']:34s} S={res['S_total']:.4f} "
              f"dS={res['dS']:+.1e} dMPLmean={res['dMPL']['mean']*1e3:+.2f}e-3 "
              f"sp7={res['dMPL']['spread7']*1e3:.2f}e-3 | lag e-3: "
              f"d0={c['d=0']['lag_diff']*1e3:+.2f} "
              f"d05={c['d=0.5']['lag_diff']*1e3:+.2f} "
              f"d075={c['d=0.75']['lag_diff']*1e3:+.2f} | "
              f"cert={'YES' if res['certified_d05_d075'] else 'no'}"
              f" margin={res['margin']:.2f}")

    certified = [r for r in results if r["certified_d05_d075"]]
    certified.sort(key=lambda r: -r["margin"])
    best = certified[:3]
    print(f"\ncertified pairs (d=0.5 AND d=0.75 pass both gates): "
          f"{len(certified)}/{len(results)}")
    for r in best:
        c = r["closures"]
        print(f"  BEST {r['name']}: lag(d05)={c['d=0.5']['lag_diff']*1e3:+.2f}"
              f"e-3 lag(d075)={c['d=0.75']['lag_diff']*1e3:+.2f}e-3 "
              f"spread7={r['dMPL']['spread7']*1e3:.2f}e-3 "
              f"seeds(sig_hi)={max(c['d=0.5']['seeds_required_sighi'], c['d=0.75']['seeds_required_sighi'])} "
              f"d0: {'cert' if r['certified_d0'] else 'NULL (by design)'}")
    if not certified:
        print("\nNO PAIR CERTIFIES -> E1 self-buckets to B.")

    out = dict(
        item="E1-CERT",
        protocol=dict(
            peak=PEAK, warmup=WARM, total=TOTAL, terminal_lr=END_LR,
            lam_star=LAM, kappas={t: k for t, _, k in ARMS},
            eval_steps=[int(s) for s in EVAL_STEPS],
            backbone_splits=["+".join(s) for s in
                             list(SPLITS_CORE) + list(SPLITS_WIDE)],
            criteria="certified iff |kappa*dfeat| > 3*spread7(dMPL) AND "
                     "> 2*SE_hi per closure; headline = d=0.5 AND d=0.75",
            se_paired_band=[SE_LO, SE_HI],
            per_seed_sd_band=[SIG1_LO, SIG1_HI],
            s_matching="partner ramp start ts solved by brentq so "
                       "sum(eta) matches the anchor exactly (dS ~ 1e-9); "
                       "all arms hold END_LR from te<=5700 -> S(t) matched "
                       "at every eval step"),
        backbones=fits,
        pairs=results,
        certified=[r["name"] for r in certified],
        best=[r["name"] for r in best],
        eta_arrays={r["name"]: dict(arm1=list(map(float, arrays[r["name"]][0])),
                                    arm2=list(map(float, arrays[r["name"]][1])))
                    for r in best})
    json.dump(out, open(OUT_PATH, "w"), indent=1, default=float)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
