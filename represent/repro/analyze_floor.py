"""Analyze the equal-S constant-LR floor ladder (results/curves_floor/).

All rungs end at the same cumulative LR S*, so backbone terms cancel and the
final smoothed eval losses differ only by the eta-dependent floor (plus eval
noise).  Outputs:
  G4a: monotonicity of floors in eta2 (the old U-shape was the backbone
       confound fingerprint);
  G4b: power-law fit  floor(eta) = L_base + a*(eta/peak)^p, bootstrap CI on p;
  tau(eta2) from the dense post-drop window (backbone-free: equal-S applies
       only at the end, so use a local linear+exp fit).
"""
import os, sys, json, glob
import numpy as np
from scipy.optimize import least_squares, curve_fit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_curves as AC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CDIR = os.path.join(ROOT, "results", "curves_floor")
PEAK = 1.5e-3


def load():
    out = {}
    for f in sorted(glob.glob(os.path.join(CDIR, "floor_*.csv"))):
        name = os.path.basename(f)[:-4]
        if name.endswith("_s1338"):
            continue
        rows = np.genfromtxt(f, delimiter=",", names=True)
        step = np.atleast_1d(rows["step"]).astype(int)
        loss = np.atleast_1d(rows["eval_loss"]).astype(float)
        lr = np.atleast_1d(rows["lr"]).astype(float)
        out[name] = dict(step=step, loss=loss, lr=lr,
                         smooth=AC.smooth_by_step(step, loss))
    return out


def main():
    cv = load()
    print("rungs:", sorted(cv.keys()))
    floors = []
    for name, c in sorted(cv.items(), key=lambda kv: float(kv[0].split("_")[1])):
        eta2 = float(name.split("_")[1]) * 1e-5
        n = len(c["step"])
        tail = max(4, n // 4)
        fl = float(np.mean(c["smooth"][-tail:]))
        floors.append((eta2, fl))
        print(f"  {name:10s} eta2={eta2:.1e} floor={fl:.4f} (tail n={tail})")

    etas = np.array([e for e, _ in floors])
    fls = np.array([f for _, f in floors])
    order = np.argsort(etas)
    etas, fls = etas[order], fls[order]
    mono = bool(np.all(np.diff(fls) > 0))
    print(f"\nG4a monotone in eta2: {mono}  (diffs: "
          + " ".join(f"{d:+.4f}" for d in np.diff(fls)) + ")")

    # G4b: floor = L_base + a*(eta/peak)^p
    x = etas / PEAK

    def resid(th):
        L, loga, p = th
        return fls - (L + np.exp(loga) * x ** p)

    best = None
    for p0 in [0.8, 1.0, 1.3, 1.6, 2.0]:
        r = least_squares(resid, x0=[fls.min() - 0.02, np.log(0.02), p0],
                          bounds=([0.5, -10, 0.2], [fls.min(), 3, 3.5]))
        if best is None or r.cost < best.cost:
            best = r
    L, loga, p = best.x
    print(f"G4b fit: floor = {L:.4f} + {np.exp(loga):.4f} * (eta/peak)^{p:.3f}")

    # bootstrap on residuals
    res = resid(best.x)
    rng = np.random.default_rng(0)
    ps = []
    for _ in range(1000):
        fb = (L + np.exp(loga) * x ** p) + rng.choice(res, len(res), replace=True)
        try:
            r = least_squares(lambda th: fb - (th[0] + np.exp(th[1]) * x ** th[2]),
                              x0=best.x,
                              bounds=([0.5, -10, 0.2], [fb.min(), 3, 3.5]))
            ps.append(r.x[2])
        except Exception:
            pass
    lo, hi = np.percentile(ps, [5, 95])
    print(f"G4b p_real = {p:.3f}  (90% CI [{lo:.3f}, {hi:.3f}]) "
          f"-> superlinear: {lo > 1.0}")

    # tau(eta2) from dense post-drop window
    print("\ntau(eta2), local backbone+exp fit:")
    e_t, taus = [], []
    for name, c in sorted(cv.items(), key=lambda kv: float(kv[0].split("_")[1])):
        eta2 = float(name.split("_")[1]) * 1e-5
        if eta2 >= PEAK:
            continue
        drop = 3000
        m = (c["step"] >= drop) & (c["step"] <= drop + 2500)
        t = (c["step"][m] - drop).astype(float)
        y = c["smooth"][m]
        if len(t) < 10:
            continue

        def mdl(t, a, b, amp, tau):
            return a + b * t + amp * np.exp(-t / tau)
        try:
            p0 = [y[-1], 0.0, max(y[0] - y[-1], 1e-3), 300.0]
            po, _ = curve_fit(mdl, t, y, p0=p0, maxfev=50000,
                              bounds=([0, -1e-3, 0, 10], [5, 1e-3, 2, 6000]))
            pred = mdl(t, *po)
            r2 = 1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-30)
            print(f"  {name:10s} tau={po[3]:7.0f} amp={po[2]:+.4f} r2={r2:.3f}")
            if po[2] > 1e-3 and r2 > 0.6 and 10 < po[3] < 5900:
                e_t.append(eta2); taus.append(po[3])
        except Exception as ex:
            print(f"  {name}: fit failed {ex}")
    if len(e_t) >= 3:
        sl = np.polyfit(np.log(e_t), np.log(taus), 1)[0]
        print(f"  tau ~ eta^{sl:.2f}  (n={len(e_t)}; -1 = inverse-LR law)")

    json.dump({"floors": [[float(e), float(f)] for e, f in zip(etas, fls)],
               "monotone": mono, "p": float(p), "p_ci": [float(lo), float(hi)]},
              open(os.path.join(ROOT, "results", "FLOOR_REPORT.json"), "w"), indent=2)
    print("\nsaved results/FLOOR_REPORT.json")


if __name__ == "__main__":
    main()
