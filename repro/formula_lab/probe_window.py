#!/usr/bin/env python3
"""Is the probe-kappa deficit an amplitude effect or tail contamination?

Fit per-probe kappa using only the EARLY post-drop window (drop step 8000 to
8000+W) vs the full curve, for lr and pow d=0.5 features; compare to the
sharp-curve kappa.  If early-window kappa ~ sharp kappa, the deficit is a
kernel/tail problem, not amplitude.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from reproduce_cosine_to_wsd import (  # noqa: E402
    load_curve, mpl_predict, MPL_PRECOMPUTED_INIT, SCALES,
)
from formula_lab.lab import feature, fit_origin, DECAY, PROBES  # noqa: E402

DROP_STEP = 8000


def main():
    for spec, tag in [({"form": "lr", "lam": 10}, "lr@10"),
                      ({"form": "pow", "delta": 0.5, "lam": 10}, "pow d=0.5@10")]:
        print(f"== {tag} ==")
        for scale in SCALES:
            p = MPL_PRECOMPUTED_INIT[scale]
            # sharp reference
            xs, ys = [], []
            for n in DECAY:
                c = load_curve(scale, n)
                xs.append(feature(c, spec))
                ys.append(c.loss - mpl_predict(p, c))
            k_sharp = fit_origin(np.concatenate(xs), np.concatenate(ys))[0]
            line = [f"{scale:>4}M k_sharp={k_sharp:.4f}"]
            for n in PROBES:
                c = load_curve(scale, n)
                phi = feature(c, spec)
                r = c.loss - mpl_predict(p, c)
                k_full = max(0.0, fit_origin(phi, r)[0])
                for W in [2000]:
                    m = (c.step >= DROP_STEP) & (c.step < DROP_STEP + W)
                    k_w = max(0.0, fit_origin(phi[m], r[m])[0]) if m.sum() > 3 else float("nan")
                line.append(f"{n.split('.')[0]}: full={k_full/k_sharp:.2f} "
                            f"early={k_w/k_sharp:.2f}")
            print("   " + "  ".join(line))


if __name__ == "__main__":
    main()
