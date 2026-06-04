"""Analyze the showcase run (results/curves_show/) for the non-adiabatic lag:
  (1) rate-dependence signature: wsd_sharp (fast sweep) vs wsd_grad (slow sweep) vs cosine,
      all reaching the SAME final LR -> sharp should lag highest above a well-fit MPL.
  (2) residual = DropRelaxS kernel (R^2) and finite-lambda vs cumulative-drop fair baseline.
  (3) tau ~ 1/eta from the wsdcon two-stage probes.
Reuses engine + analyze_curves analysis functions."""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_curves as AC
from engine import droprelaxS, fit_powerlaw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CDIR = os.path.join(ROOT, "results", "curves_show")


def load():
    scheds = json.load(open(os.path.join(CDIR, "schedules.json")))
    cv = {}
    for f in glob.glob(os.path.join(CDIR, "*.csv")):
        name = os.path.basename(f)[:-4]
        if name not in scheds:
            continue
        rows = np.genfromtxt(f, delimiter=",", names=True)
        step = np.atleast_1d(rows["step"]).astype(int)
        loss = np.atleast_1d(rows["eval_loss"]).astype(float)
        keep = step >= AC.T_MIN
        st = step[keep]; ls = loss[keep]
        cv[name] = dict(step=st, lr=np.asarray(scheds[name], float),
                        loss=AC.smooth_by_step(st, ls), loss_raw=ls)
    return cv


def main():
    cv = load()
    print("schedules:", sorted(cv.keys()))
    train = [s for s in ["constant", "cosine", "wsdcon_20"] if s in cv]
    params, f = AC.fit_mpl(cv, train, n_starts=5)
    print(f"MPL fit on {train}: obj={f:.5f}")
    print("params L0,A,alpha,B,C,beta,gamma =", np.round(params, 4))

    print("\n--- (1) rate-dependence signature (residual above well-fit MPL) ---")
    sig = {}
    for s in ["cosine", "wsd_grad", "wsd_sharp"]:
        if s in cv:
            st, r, _ = AC.residual(cv, params, s)
            # end = mean of last 15% of decay
            n = max(1, len(r) // 7)
            sig[s] = dict(end=float(np.mean(r[-n:])), maxabs=float(np.max(np.abs(r))))
            print(f"  {s:10s}: end-lag={sig[s]['end']:+.4f}  max|r|={sig[s]['maxabs']:.4f}")
    if "wsd_sharp" in sig and "wsd_grad" in sig:
        print(f"  >>> sharp/grad end-lag ratio = {sig['wsd_sharp']['end']/ (sig['wsd_grad']['end'] if sig['wsd_grad']['end'] else 1e-9):.2f}  (expect >1: faster sweep, bigger lag)")

    print("\n--- (2) residual = DropRelaxS + fair baseline ---")
    for s in ["wsd_sharp", "wsd_grad"]:
        if s in cv:
            b, _, _ = AC.fit_droprelaxS(cv, params, s)
            print(f"  DropRelaxS {s}: R2={b['R2']:.3f} lam={b['lam']:.2f} kappa={b['kappa']:.3e}")
    fb = AC.fair_baseline_test(cv, params, scheds=("cosine", "wsd_grad", "wsd_sharp"))
    print(f"  fair baseline: finite-lambda R2={fb['finite']['R2']:.3f} (lam={fb['finite']['lam']:.2f}) "
          f"vs cumulative-drop R2={fb['cumdrop']['R2']:.3f} -> advantage {fb['advantage']:+.3f}")

    print("\n--- (3) tau ~ 1/eta (wsdcon probes, MPL-residual) ---")
    tv = AC.tau_vs_eta(cv, params)
    print(f"  p={tv['p']:.3f} (r2={tv['r2']:.3f}, n={tv['n']})")
    for pt in sorted(tv["points"], key=lambda z: z["eta"]):
        print(f"    eta={pt['eta']:.1e} tau={pt.get('tau', float('nan')):.0f} r2={pt.get('r2', float('nan')):.3f} used={pt['used']}")

    # also a model-free tau (linear backbone + exp) as a cross-check
    print("  [model-free cross-check: a+b*t+c*exp(-t/tau)]")
    from scipy.optimize import curve_fit
    def mdl(t, a, b, c, tau): return a + b * t + c * np.exp(-t / tau)
    e2, ta = [], []
    for name in sorted(cv):
        if not name.startswith("wsdcon_"): continue
        s2 = float(cv[name]["lr"][-1]); dstep = AC._drop_step(cv[name]["lr"])
        m = cv[name]["step"] >= dstep
        t = (cv[name]["step"][m] - dstep).astype(float); y = cv[name]["loss"][m]
        if len(t) < 8: continue
        try:
            p0 = [y[-1], (y[-1]-y[0])/max(t[-1],1), y[0]-y[-1], 200.0]
            po, _ = curve_fit(mdl, t, y, p0=p0, maxfev=40000, bounds=([0,-1,-1,5],[5,1,5,8000]))
            if po[2] > 0 and 5 < po[3] < 8000:
                e2.append(s2); ta.append(po[3])
                print(f"    {name}: tau_mf={po[3]:.0f} amp={po[2]:+.4f}")
        except Exception:
            pass
    if len(e2) >= 3:
        p, c, r2 = fit_powerlaw(np.array(e2), np.array(ta))
        print(f"  >>> model-free tau ~ eta^-{p:.3f} (r2={r2:.3f}, n={len(e2)})")

    # ---- (3b) BACKBONE-AWARE tau: fix MPL backbone (L0,A,alpha), fit floor-relaxation exp on top ----
    print("\n--- (3b) backbone-aware tau (L0+A*(S_drop+eta*t')^-alpha + amp*exp(-t'/tau)) ---")
    from engine import cumS
    L0, A, alpha = params[0], params[1], params[2]
    e3, ta3 = [], []
    tau3_pts = []
    for name in sorted(cv):
        if not name.startswith("wsdcon_"):
            continue
        c = cv[name]; s2 = float(c["lr"][-1]); dstep = AC._drop_step(c["lr"])
        S = cumS(c["lr"]); S_drop = S[dstep]
        m = c["step"] >= dstep
        tp = (c["step"][m] - dstep).astype(float); y = c["loss"][m]
        if len(tp) < 8:
            continue
        def bbexp(t, amp, tau, db):
            return (L0 + db) + A * (S_drop + s2 * t) ** (-alpha) + amp * np.exp(-t / tau)
        try:
            p0 = [max(y[0] - y[-1], 1e-3), 200.0, 0.0]
            po, _ = curve_fit(bbexp, tp, y, p0=p0, maxfev=60000,
                              bounds=([0, 5, -0.5], [5, 8000, 0.5]))
            pred = bbexp(tp, *po); r2 = 1 - np.sum((y - pred) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-30)
            used = po[0] > 1e-3 and 5 < po[1] < 8000 and r2 > 0.8
            tau3_pts.append(dict(eta=s2, tau=float(po[1]), amp=float(po[0]), r2=float(r2), used=bool(used)))
            print(f"    {name}: tau={po[1]:.0f} amp={po[0]:+.4f} r2={r2:.3f} used={used}")
            if used:
                e3.append(s2); ta3.append(po[1])
        except Exception as ex:
            print(f"    {name}: fit fail {ex}")
    tau_bb = dict(points=tau3_pts)
    if len(e3) >= 3:
        p, c_, r2 = fit_powerlaw(np.array(e3), np.array(ta3))
        tau_bb.update(p=float(p), c=float(c_), r2=float(r2), n=len(e3))
        print(f"  >>> backbone-aware tau ~ eta^-{p:.3f} (r2={r2:.3f}, n={len(e3)})  [expect ~1]")

    json.dump(dict(params=list(map(float, params)), fit_obj=float(f), signature=sig,
                   fair_baseline=fb, tau_vs_eta=tv, tau_backbone=tau_bb), open(os.path.join(ROOT, "results", "SHOWCASE_REPORT.json"), "w"), indent=2)
    print("\nsaved results/SHOWCASE_REPORT.json")

    # ---- figures ----
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        figs = os.path.join(ROOT, "figs"); os.makedirs(figs, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for s, col in [("cosine", "tab:green"), ("wsd_grad", "tab:orange"), ("wsd_sharp", "tab:red")]:
            if s in cv:
                st, r, _ = AC.residual(cv, params, s)
                ax[0].plot(st, r, color=col, lw=1.5, label=f"{s}")
        if "wsd_sharp" in cv:
            best, _, _ = AC.fit_droprelaxS(cv, params, "wsd_sharp")
            st = cv["wsd_sharp"]["step"]; K = droprelaxS(cv["wsd_sharp"]["lr"], best["lam"])[st]
            ax[0].plot(st, best["kappa"] * K, "k--", lw=1.6, label=f"DropRelaxS (R²={best['R2']:.2f})")
        ax[0].axhline(0, color="gray", lw=0.6); ax[0].set_xlabel("step"); ax[0].set_ylabel("L_true - L_MPL")
        ax[0].set_title("Non-adiabatic residual: fast vs slow decay (showcase, ~10M)"); ax[0].legend(fontsize=8)
        pts = [p for p in tv["points"] if p.get("used")]
        if len(pts) >= 2:
            e = np.array([p["eta"] for p in pts]); t = np.array([p["tau"] for p in pts])
            ax[1].loglog(e, t, "o", ms=8, label=f"measured (p={tv['p']:.2f})")
            xx = np.geomspace(e.min(), e.max(), 30); ax[1].loglog(xx, tv["c"] * xx ** (-tv["p"]), "-")
            ax[1].loglog(xx, t[0] * (e[0] / xx), "k--", lw=1, label="slope -1 (τ∝1/η)")
            ax[1].set_xlabel("stage-2 LR η"); ax[1].set_ylabel("τ (steps)"); ax[1].set_title("τ ∝ 1/η (wsdcon)"); ax[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(figs, "showcase.png"), dpi=130)
        print("wrote figs/showcase.png")
    except Exception as e:
        print("fig error:", e)


if __name__ == "__main__":
    main()
