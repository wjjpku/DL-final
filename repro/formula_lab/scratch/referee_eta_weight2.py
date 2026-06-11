#!/usr/bin/env python3
"""Follow-up referee controls.

T1: ORACLE per-curve kappa (lam=10): does the eta_post-weighted feature SHAPE
    beat unweighted when amplitude is fit on the test curve itself?
T2: kappa-rescale control: delta=0 probe-calibrated kappa scaled by constant c.
    If c*kappa matches eta_post OOS gains, the weight is just a kappa inflator.
T3: exact (unbinned) features for the headline OOS comparison (binning check).
T4: cooldown half-split of OOS gains (from referee_eta_weight.json).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPRO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPRO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    load_curve, metrics, MPL_PRECOMPUTED_INIT, SCALES, PEAK_LR,
    compute_s1, compute_ld,
)
from formula_lab.kernels import conv_feature  # noqa: E402

CURVES = [
    "cosine_72000.csv",
    "wsd_20000_24000.csv",
    "wsdld_20000_24000.csv",
    "wsdcon_3.csv",
    "wsdcon_9.csv",
    "wsdcon_18.csv",
]
SHARP = ["wsd_20000_24000.csv", "wsdld_20000_24000.csv"]
PROBES = ["wsdcon_3.csv", "wsdcon_9.csv", "wsdcon_18.csv"]
LAM = 10.0


def exact_feat(curve, delta):
    return conv_feature(curve, lambda u: np.exp(-LAM * u), eta_weight_delta=delta)


def fit_kappa(feats, resids):
    x = np.concatenate(feats)
    y = np.concatenate(resids)
    xx = float(np.dot(x, x))
    return 0.0 if xx <= 1e-18 else max(0.0, float(np.dot(x, y) / xx))


def main():
    data = {}
    for scale in SCALES:
        curves = {c: load_curve(scale, c) for c in CURVES}
        p = MPL_PRECOMPUTED_INIT[scale]
        L0, A, alpha, B, C, beta, gamma = p
        base, resid = {}, {}
        for c in CURVES:
            s1 = compute_s1(curves[c])
            ld = compute_ld(curves[c], C, beta, gamma)
            base[c] = L0 + A * np.power(s1, -alpha) + B * ld
            resid[c] = curves[c].loss - base[c]
        feats = {}
        for d in [0.0, 0.5]:
            for c in CURVES:
                feats[(d, c)] = exact_feat(curves[c], d)
        data[scale] = dict(curves=curves, base=base, resid=resid, feats=feats)

    # ---- T1: oracle per-curve kappa (shape-only comparison), EXACT feats ----
    print("== T1 ORACLE: kappa fit per test curve itself (lam=10, exact features) ==")
    print("   -> isolates feature SHAPE from amplitude calibration")
    for scale in SCALES:
        D = data[scale]
        for c in SHARP + ["cosine_72000.csv"]:
            row = f"  {scale:>4s} {c.replace('.csv','').replace('_20000_24000',''):>12s}"
            for d in [0.0, 0.5]:
                f, r = D["feats"][(d, c)], D["resid"][c]
                k = fit_kappa([f], [r])
                sse = float(np.sum((r - k * f) ** 2))
                sst = float(np.sum((r - r.mean()) ** 2))
                r2 = 1.0 - sse / sst
                mae0 = float(np.mean(np.abs(r)))
                mae1 = float(np.mean(np.abs(r - k * f)))
                row += (f" | d={d:.1f}: k={k:.4f} R2={r2:+.3f} "
                        f"dMAE={100*(mae1/mae0-1):+6.1f}%")
            print(row)

    # ---- T2: kappa-rescale control on OOS protocol (exact feats) -----------
    print("\n== T2 kappa-rescale control: delta=0, kappa = c * k_probes(delta=0) ==")
    print("   eta_post d=0.5 OOS numbers shown for comparison")
    for scale in SCALES:
        D = data[scale]
        k0 = fit_kappa([D["feats"][(0.0, c)] for c in PROBES],
                       [D["resid"][c] for c in PROBES])
        k5 = fit_kappa([D["feats"][(0.5, c)] for c in PROBES],
                       [D["resid"][c] for c in PROBES])

        def sharp_dmae(d, kappa):
            out = []
            for c in SHARP:
                cur = D["curves"][c]
                mae0 = metrics(cur.loss, D["base"][c])["mae"]
                mae1 = metrics(cur.loss, D["base"][c] + kappa * D["feats"][(d, c)])["mae"]
                out.append(100.0 * (mae1 / mae0 - 1.0))
            return float(np.mean(out))

        line = f"  {scale:>4s} k_probes(d=0)={k0:.4f} k_probes(d=.5)={k5:.4f} infl={k5/k0:.2f}x"
        print(line)
        scan = {c: sharp_dmae(0.0, c * k0) for c in [1.0, 1.5, 2.0, 2.39, 2.5, 3.0, 3.5, 4.0]}
        print("    d=0 scaled:  " + "  ".join(f"c={c:.2f}:{v:+6.1f}%" for c, v in scan.items()))
        print(f"    eta_post d=0.5 (k={k5:.4f}): {sharp_dmae(0.5, k5):+6.1f}%  "
              f"| d=0 best over scan: {min(scan.values()):+6.1f}%")

    # ---- T3: exact-feature headline OOS table (binning check) --------------
    print("\n== T3 exact-feature OOS (kappa on probes, lam=10): per-curve dMAE% ==")
    for scale in SCALES:
        D = data[scale]
        for d in [0.0, 0.5]:
            k = fit_kappa([D["feats"][(d, c)] for c in PROBES],
                          [D["resid"][c] for c in PROBES])
            per = []
            for c in CURVES:
                cur = D["curves"][c]
                mae0 = metrics(cur.loss, D["base"][c])["mae"]
                mae1 = metrics(cur.loss, D["base"][c] + k * D["feats"][(d, c)])["mae"]
                per.append(100.0 * (mae1 / mae0 - 1.0))
            sharp = float(np.mean([per[CURVES.index(c)] for c in SHARP]))
            print(f"  {scale:>4s} d={d:.1f} k={k:.4f} sharp dMAE={sharp:+6.1f}% | "
                  + " ".join(f"{c.split('_')[0]}{c.split('_')[1][:2] if 'wsdcon' in c else ''}={v:+6.1f}"
                             for c, v in zip(CURVES, per)))

    # ---- T4: cooldown halves from first script's JSON -----------------------
    print("\n== T4 cooldown half-split (OOS probe-cal lam=10) from referee JSON ==")
    jpath = REPRO.parent / "results" / "formula_lab" / "referee_eta_weight.json"
    rows = json.loads(jpath.read_text(encoding="utf-8"))
    for r in rows:
        if r.get("protocol") == "oos_probe_lam10" and (r["mode"], r["delta"]) in [
                ("none", 0.0), ("eta_post", 0.5)]:
            h = r["cooldown_halves_dmae_pct"]
            print(f"  {r['scale']:>4s} {r['mode']:>8s} d={r['delta']:.2f} "
                  + " ".join(f"{k}={v:+6.1f}%" for k, v in h.items()))


if __name__ == "__main__":
    main()
