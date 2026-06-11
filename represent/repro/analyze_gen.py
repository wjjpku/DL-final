"""Evaluate correction-law variants on the REAL ~10.7M transformer:
existing m_* suite (results/curves/) + new out-of-family suite
(results/curves_gen/: twodrop, cyclic, invsqrt, sharp600).

Protocol (leakage-clean):
  MPL backbone : fit on [constant, cosine, wsdcon_20]   (AC conventions)
  calibration  : wsdcon_{5,10,40,80} probes -- pooled origin-LS kappa,
                 lam chosen on the probes per arm (grid)
  held-out     : wsd, wsdld, + the four new schedules
  arms         : lr (paper), pow d=0.5, affine rho=0.5
Extra diagnostics: twodrop superposition, cyclic sign test.
"""
import os, sys, json, glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_curves as AC
from engine import cumS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENDIR = os.path.join(ROOT, "results", "curves_gen")
PEAK = 1.5e-3

PROBES = ["wsdcon_5", "wsdcon_10", "wsdcon_40", "wsdcon_80"]
HELD = ["wsd", "wsdld", "twodrop", "cyclic", "invsqrt", "sharp600"]
LAM_GRID = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]

ARMS = [
    ("lr (paper)", {"form": "lr"}),
    ("pow d=0.5", {"form": "pow", "delta": 0.5}),
    ("pow d=0.75", {"form": "pow", "delta": 0.75}),
    ("affine r=0.5", {"form": "affine", "rho": 0.5}),
]


def load_gen():
    scheds = json.load(open(os.path.join(GENDIR, "schedules.json")))
    out = {}
    for f in glob.glob(os.path.join(GENDIR, "*.csv")):
        name = os.path.basename(f)[:-4]
        if name not in scheds:
            continue
        rows = np.genfromtxt(f, delimiter=",", names=True)
        step = np.atleast_1d(rows["step"]).astype(int)
        loss = np.atleast_1d(rows["eval_loss"]).astype(float)
        keep = step >= AC.T_MIN
        st, ls = step[keep], loss[keep]
        out[name] = dict(step=st, lr=np.asarray(scheds[name], float),
                         loss=AC.smooth_by_step(st, ls), loss_raw=ls)
    return out


def wfeature(lr, spec, lam):
    """Weighted DropRelaxS, dimensionless (drops normalized by PEAK)."""
    eta = np.asarray(lr, float)
    drop = np.zeros_like(eta)
    drop[1:] = np.maximum(eta[:-1] - eta[1:], 0.0)
    form = spec.get("form", "lr")
    if form == "pow":
        drop = drop * np.power(np.maximum(eta / PEAK, 1e-12), spec["delta"])
    elif form == "affine":
        rho = spec["rho"]
        drop = drop * ((1 - rho) + rho * eta / PEAK)
    dec = np.exp(-lam * eta)
    s = np.empty_like(eta)
    acc = 0.0
    for t in range(len(eta)):
        acc = acc * dec[t] + drop[t]
        s[t] = acc
    return s / PEAK


def fit_origin(x, y):
    xx = float(np.dot(x, x))
    k = 0.0 if xx <= 1e-18 else max(0.0, float(np.dot(x, y) / xx))
    resid = y - k * x
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss if ss > 0 else float("nan")
    return k, r2


def main():
    cv = AC.load_scale("m")
    gen = load_gen()
    for k, v in gen.items():
        cv[k] = v
    print("curves:", sorted(cv.keys()))

    train = ["constant", "cosine", "wsdcon_20"]
    params, fobj = AC.fit_mpl(cv, train, n_starts=8)
    print(f"MPL fit on {train}: obj={fobj:.5f}")
    print("params:", np.round(params, 4))

    resid = {}
    for s in cv:
        st = cv[s]["step"]
        pred = AC.mpl_pred_at(cv[s]["lr"], st, params)
        resid[s] = cv[s]["loss"] - pred

    results = {}
    for tag, spec in ARMS:
        # choose lam + kappa on probes (pooled)
        best = None
        for lam in LAM_GRID:
            xs = [wfeature(cv[s]["lr"], spec, lam)[cv[s]["step"]] for s in PROBES]
            ys = [resid[s] for s in PROBES]
            x, y = np.concatenate(xs), np.concatenate(ys)
            k, _ = fit_origin(x, y)
            sse = float(np.sum((y - k * x) ** 2))
            if best is None or sse < best[2]:
                best = (lam, k, sse)
        lam, kappa, _ = best

        rows = {}
        for s in HELD:
            if s not in cv:
                continue
            phi = wfeature(cv[s]["lr"], spec, lam)[cv[s]["step"]]
            y = cv[s]["loss"]
            base = y - resid[s]
            mae0 = float(np.mean(np.abs(resid[s])))
            mae1 = float(np.mean(np.abs(y - (base + kappa * phi))))
            _, r2 = fit_origin(phi, resid[s])
            rows[s] = {"mae_mpl": mae0, "mae_corr": mae1,
                       "delta_pct": 100.0 * (mae1 / mae0 - 1.0), "r2_resid": r2}
        clean = {s: r for s, r in rows.items() if s != "cyclic"}
        mean_delta = float(np.mean([r["delta_pct"] for r in clean.values()]))
        results[tag] = {"lam": lam, "kappa": kappa, "rows": rows,
                        "mean_delta": mean_delta}
        print(f"\n== {tag}: lam*={lam:g} kappa={kappa:.4f} ==")
        for s, r in rows.items():
            extra = "  [excluded from mean: MPL re-warm backbone failure]" \
                if s == "cyclic" else ""
            print(f"  {s:10s} MAE {r['mae_mpl']:.4f} -> {r['mae_corr']:.4f} "
                  f"({r['delta_pct']:+6.1f}%)  resid-R2={r['r2_resid']:+.3f}{extra}")
        print(f"  mean delta (excl. cyclic): {mean_delta:+.1f}%")

    # --- twodrop superposition test (G1): two-kernel decomposition ---
    if "twodrop" in cv:
        print("\n--- twodrop superposition (G1) ---")
        lr = cv["twodrop"]["lr"]
        st = cv["twodrop"]["step"]
        S = cumS(lr)
        lam = results["lr (paper)"]["lam"]
        d1, d2 = 2500, 4500
        K1 = np.where(st >= d1, np.exp(-lam * (S[st] - S[d1])), 0.0)
        K2 = np.where(st >= d2, np.exp(-lam * (S[st] - S[d2])), 0.0)
        r = resid["twodrop"]
        m = st >= AC.T_MIN
        X = np.column_stack([K1[m], K2[m]])
        y = r[m]

        def nnls2(X, y):
            from scipy.optimize import nnls
            return nnls(X, y)[0]

        A = nnls2(X, y)
        ratio = A[1] / A[0] if A[0] > 0 else float("nan")
        # block bootstrap (blocks of 12 points)
        rng = np.random.default_rng(0)
        n = len(y); bs = 12
        nblk = (n + bs - 1) // bs
        ratios = []
        for _ in range(500):
            idx = np.concatenate([
                np.arange(s, min(s + bs, n))
                for s in rng.integers(0, max(n - bs, 1), nblk)])
            try:
                Ab = nnls2(X[idx], y[idx])
                if Ab[0] > 1e-9:
                    ratios.append(Ab[1] / Ab[0])
            except Exception:
                pass
        lo, hi = (np.percentile(ratios, [5, 95]) if ratios else (np.nan, np.nan))
        pred = {"delta=0": 0.700, "p=1.25": 0.564, "delta=0.5": 0.383, "p=2": 0.303}
        print(f"  measured A2/A1 = {ratio:.3f}  (90% CI [{lo:.3f}, {hi:.3f}], "
              f"n_boot={len(ratios)})")
        print(f"  predictions: {pred}")
        fit = X @ A
        ss = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - float(np.sum((y - fit) ** 2)) / ss if ss > 0 else float("nan")
        print(f"  joint two-kernel R2 = {r2:.3f}")
        results["twodrop_G1"] = {"ratio": ratio, "ci": [float(lo), float(hi)],
                                 "r2": r2, "pred": pred}

    # --- G1-paired: twodrop vs onedrop vs constant (backbone-free) ---
    if "onedrop" in cv and "twodrop" in cv and "constant" in cv:
        print("\n--- G1-paired (difference curves, local-linear backbone) ---")
        from scipy.optimize import curve_fit

        def lag_amp(c_a, c_b, drop, window=1500):
            """Fit (L_a - L_b)(t) = c0 + c1*(t-drop) + A*exp(-(t-drop)/tau)
            on [drop, drop+window]; curves share state at t=drop."""
            m = (c_a["step"] >= drop) & (c_a["step"] <= drop + window)
            t = (c_a["step"][m] - drop).astype(float)
            ya = c_a["loss"][m]
            yb = np.interp(c_a["step"][m], c_b["step"], c_b["loss"])
            d = ya - yb

            def mdl(t, c0, c1, A, tau):
                return c0 + c1 * t + A * np.exp(-t / tau)
            p0 = [d[-1], 0.0, max(d[0] - d[-1], 1e-3), 300.0]
            po, _ = curve_fit(mdl, t, d, p0=p0, maxfev=60000,
                              bounds=([-1, -1e-2, 0, 20], [1, 1e-2, 1, 4000]))
            pred = mdl(t, *po)
            ss = float(np.sum((d - d.mean()) ** 2))
            r2 = 1 - float(np.sum((d - pred) ** 2)) / ss if ss > 0 else float("nan")
            return po[2], po[3], r2, d

        try:
            A1, tau1, r21, _ = lag_amp(cv["onedrop"], cv["constant"], 2500)
            A2, tau2, r22, _ = lag_amp(cv["twodrop"], cv["onedrop"], 4500)
            print(f"  drop1 (peak->0.5):  A1={A1:+.4f} tau={tau1:.0f} r2={r21:.3f}")
            print(f"  drop2 (0.5->0.15):  A2={A2:+.4f} tau={tau2:.0f} r2={r22:.3f}")
            ratio = A2 / A1 if A1 > 0 else float("nan")
            pred = {"delta=0": 0.700, "p=1.25": 0.564, "delta=0.5": 0.383,
                    "p=2": 0.303}
            print(f"  paired A2/A1 = {ratio:.3f}   predictions: {pred}")
            results["twodrop_G1_paired"] = {
                "A1": float(A1), "A2": float(A2), "tau1": float(tau1),
                "tau2": float(tau2), "r2_1": float(r21), "r2_2": float(r22),
                "ratio": float(ratio), "pred": pred}
        except Exception as ex:
            print(f"  paired fit failed: {ex}")

    # --- cyclic sign test ---
    if "cyclic" in cv:
        print("\n--- cyclic (re-warm) sign test ---")
        st = cv["cyclic"]["step"]
        r = resid["cyclic"]
        for lo, hi, label in [(2500, 3500, "after drop1"),
                              (3700, 4800, "after re-warm"),
                              (4800, 6000, "after drop2")]:
            m = (st >= lo) & (st < hi)
            if m.sum():
                print(f"  {label:15s} mean resid {float(np.mean(r[m])):+.4f}")

    json.dump(results, open(os.path.join(ROOT, "results", "GEN_REPORT.json"), "w"),
              indent=2, default=float)
    print("\nsaved results/GEN_REPORT.json")


if __name__ == "__main__":
    main()
