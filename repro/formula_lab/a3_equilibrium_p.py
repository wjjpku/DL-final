"""ITEM A3: equilibrium-p from fitted asymptotes F_inf (zero-GPU).

The shipped ladder exponent p (AUDIT-C corrected: m 0.647 [0.610, 0.683],
l 0.641 [0.613, 0.673]) uses design-window tail MEANS of still-relaxing
curves (slopes -0.002..-0.025 / 1k steps) -> p is a budget-indexed equal-S
exponent, not an equilibrium exponent.  Here each rung is instead fit on
its FULL stage-2 window [3000, 3000+T2) with the analyze_floor local
protocol

    r(t) = F_inf + b*(t-3000) + A*exp(-(t-3000)/tau),   |b| <= 1e-3,

and the floor is taken as
  (i)  the fitted asymptote F_inf                       -> p_Finf
  (ii) the b-extrapolated end value F_inf + b*T2
       (transient removed, secular drift kept)          -> p_end
p is then refit per scale exactly like analyze_floor2.fit_p (8 rungs x 2
seeds pooled, residual bootstrap, 90% CI; identical bounds/N_BOOT/rng).

Hard gate for E4: p_Finf sublinear with comparable CIs -> sublinearity is
not a truncation artifact (E4 tests horizon as a bed property);
p_Finf crossing toward/above 1 -> equal-S budget-indexing is load-bearing
and E4 is mandatory.

G5-safe: no MPL fit anywhere.  Labeled sensitivities: degenerate-rung
exclusion; step-spacing-weighted per-rung fits (the dense post-drop
sampling otherwise over-weights the transient ~12x per step).

Usage:  KMP_DUPLICATE_LIB_OK=TRUE python repro/formula_lab/a3_equilibrium_p.py
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "represent", "repro"))

import analyze_curves as AC            # noqa: E402  (smooth_by_step)
import analyze_floor2 as AF2           # noqa: E402  (fit_p, collect, RUNG_ETA, tail_floor)

OUT = os.path.join(REPO, "results", "formula_lab", "a3_equilibrium_p.json")
SHIPPED = {"m": (0.647, 0.610, 0.683), "l": (0.641, 0.613, 0.673)}
DROP = 3000


def mdl(t, a, b, amp, tau):
    return a + b * t + amp * np.exp(-t / tau)


def fit_rung(path, eta2, weighted=False):
    """analyze_floor local protocol on the FULL stage-2 window.

    Returns dict with F_inf (=a), end value a+b*T2, diagnostics and
    degeneracy flags."""
    rows = np.genfromtxt(path, delimiter=",", names=True)
    step = np.atleast_1d(rows["step"]).astype(int)
    loss = np.atleast_1d(rows["eval_loss"]).astype(float)
    sm = AC.smooth_by_step(step, loss)
    T2 = int(round(1.2 / eta2))
    m = (step >= DROP) & (step < DROP + T2)
    t = (step[m] - DROP).astype(float)
    y = sm[m]
    if len(t) < 10:
        return None
    sigma = None
    if weighted:
        # weight each sample by the step interval it represents (integral
        # weighting); curve_fit sigma ~ 1/sqrt(weight)
        d = np.diff(t)
        w = np.concatenate([[d[0]], (d[:-1] + d[1:]) / 2, [d[-1]]])
        sigma = 1.0 / np.sqrt(np.clip(w, 1e-6, None))
    lo = [0.0, -1e-3, 0.0, 10.0]
    hi = [5.0, 1e-3, 2.0, 6000.0]
    a0 = float(y[-1])
    A0 = max(float(y[0] - y[-1]), 1e-3)
    best, best_sse = None, np.inf
    for tau0 in (100.0, 300.0, 1000.0, 3000.0):
        for amp0 in (A0, 0.01):
            try:
                po, _ = curve_fit(mdl, t, y, p0=[a0, 0.0, amp0, tau0],
                                  sigma=sigma, maxfev=50000, bounds=(lo, hi))
            except Exception:
                continue
            sse = float(np.sum((y - mdl(t, *po)) ** 2))
            if sse < best_sse:
                best, best_sse = po, sse
    if best is None:
        return None
    a, b, amp, tau = (float(v) for v in best)
    pred = mdl(t, *best)
    r2 = 1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-30)
    exp_left = float(np.exp(-T2 / tau))      # transient remaining at window end
    flags = []
    if amp <= 2e-3:
        flags.append("amp~0")
    if tau >= 5700:
        flags.append("tau@hi")
    if tau <= 11:
        flags.append("tau@lo")
    if tau > T2 / 2:
        flags.append("tau>T2/2")             # exp not decayed in-window: F_inf extrapolated
    if abs(b) >= 0.95e-3:
        flags.append("b@bound")
    if r2 < 0.6:
        flags.append("r2<0.6")
    return dict(F_inf=a, b=b, amp=amp, tau=tau, r2=float(r2), T2=T2,
                end_value=a + b * T2, end_value_with_exp=a + b * T2 + amp * exp_left,
                exp_left=exp_left, b_per_1k=b * 1000, n=int(len(t)),
                degenerate=bool(flags), flags=flags)


def pool_and_fit(per_rung_vals):
    """per_rung_vals: {rung: {seed: floor}} -> fit_p on seed-pooled means."""
    rungs = sorted(per_rung_vals, key=lambda r: AF2.RUNG_ETA[r])
    etas = np.array([AF2.RUNG_ETA[r] for r in rungs])
    fls = np.array([np.mean(list(per_rung_vals[r].values())) for r in rungs])
    p, lo, hi, L0, a0 = AF2.fit_p(etas, fls)
    return dict(p=p, ci=[lo, hi], L=L0, a=a0, rungs=rungs,
                floors=[float(f) for f in fls],
                monotone=bool(np.all(np.diff(fls) > 0)))


def main():
    report = {}
    for scale in ("m", "l"):
        files = AF2.collect(scale)
        print(f"\n===== scale {scale} ({len(files)} rung files) =====")
        fits = {}
        shipped_tail, finf, endv, finf_w = {}, {}, {}, {}
        for (r, seed), f in sorted(files.items(),
                                   key=lambda kv: (AF2.RUNG_ETA[kv[0][0]], kv[0][1])):
            eta2 = AF2.RUNG_ETA[r]
            res = fit_rung(f, eta2)
            res_w = fit_rung(f, eta2, weighted=True)
            tail, _, _ = AF2.tail_floor(f, eta2)
            fits[f"{r}_s{seed}"] = dict(res, file=os.path.relpath(f, REPO),
                                        tail_mean=tail,
                                        F_inf_weighted=res_w["F_inf"] if res_w else None,
                                        weighted_flags=res_w["flags"] if res_w else None)
            shipped_tail.setdefault(r, {})[seed] = tail
            finf.setdefault(r, {})[seed] = res["F_inf"]
            endv.setdefault(r, {})[seed] = res["end_value"]
            if res_w:
                finf_w.setdefault(r, {})[seed] = res_w["F_inf"]
            print(f"  rung {r:>3s} s{seed}: tail={tail:.4f}  F_inf={res['F_inf']:.4f}"
                  f"  end={res['end_value']:.4f}  tau={res['tau']:6.0f}"
                  f"  A={res['amp']:.4f}  b/1k={res['b_per_1k']:+.4f}"
                  f"  r2={res['r2']:.3f}  {'DEGEN ' + ','.join(res['flags']) if res['flags'] else ''}")

        out = {"per_rung_fits": fits}
        out["p_tail_sanity"] = pool_and_fit(shipped_tail)     # must reproduce shipped
        out["p_Finf"] = pool_and_fit(finf)
        out["p_end"] = pool_and_fit(endv)
        out["p_Finf_weighted_SENSITIVITY"] = pool_and_fit(finf_w)

        # sensitivity: drop rungs whose fit degenerates in EITHER seed
        bad = sorted({k.split("_")[0] for k, v in fits.items() if v["degenerate"]},
                     key=lambda r: AF2.RUNG_ETA[r])
        keep_f = {r: v for r, v in finf.items() if r not in bad}
        keep_e = {r: v for r, v in endv.items() if r not in bad}
        out["degenerate_rungs"] = bad
        if len(keep_f) >= 6:
            out["p_Finf_exclDegen"] = pool_and_fit(keep_f)
            out["p_end_exclDegen"] = pool_and_fit(keep_e)
            keep_w = {r: v for r, v in finf_w.items() if r not in bad}
            if len(keep_w) >= 6:
                out["p_Finf_weighted_exclDegen_SENSITIVITY"] = pool_and_fit(keep_w)

        s = SHIPPED[scale]
        print(f"  -- p sanity (design-window tails): {out['p_tail_sanity']['p']:.3f} "
              f"{[round(c,3) for c in out['p_tail_sanity']['ci']]} "
              f"(shipped {s[0]} [{s[1]}, {s[2]}])")
        for k in ("p_Finf", "p_end", "p_Finf_weighted_SENSITIVITY",
                  "p_Finf_exclDegen", "p_end_exclDegen",
                  "p_Finf_weighted_exclDegen_SENSITIVITY"):
            if k in out:
                v = out[k]
                print(f"  -- {k:32s}: p={v['p']:.3f} 90%CI [{v['ci'][0]:.3f}, "
                      f"{v['ci'][1]:.3f}]  monotone={v['monotone']}")
        if bad:
            print(f"  degenerate rungs (either seed): {bad}")
        report[scale] = out

    # E4 gate verdict.  When rungs degenerate, the full-8-rung fit is
    # corrupted (p slams to the 0.2 bound); the verdict-bearing fit is the
    # clean-rung one.
    verdict = {}
    for scale in ("m", "l"):
        key = ("p_Finf_exclDegen" if "p_Finf_exclDegen" in report[scale]
               else "p_Finf")
        p = report[scale][key]["p"]
        lo, hi = report[scale][key]["ci"]
        band = ("SUBLINEAR (CI excludes 1)" if hi < 1 else
                ("SUPERLINEAR (CI excludes 1)" if lo > 1 else "STRADDLES 1"))
        verdict[scale] = dict(fit_used=key, p=p, ci=[lo, hi], band=band)
    report["E4_gate"] = verdict
    report["shipped_reference"] = {k: dict(p=v[0], ci=[v[1], v[2]])
                                   for k, v in SHIPPED.items()}
    print(f"\nE4 gate: {verdict}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2, default=float)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
