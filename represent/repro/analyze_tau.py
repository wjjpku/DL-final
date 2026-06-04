"""Clean tau~1/eta from the long-window wsdcon run (results/curves_tau/).
Fit MPL backbone (L0,A,alpha) on 'constant', then for each wsdcon fit the post-drop curve
  loss(t') = (L0+db) + A*(S_drop + eta*t')^(-alpha) + amp*exp(-t'/tau)
(backbone continuation + exponential floor-relaxation). Then tau ~ eta^{-p}."""
import os, sys, json, glob
import numpy as np
from scipy.optimize import curve_fit
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_curves as AC
from engine import cumS, fit_powerlaw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CDIR = os.path.join(ROOT, "results", "curves_tau")


def load():
    scheds = json.load(open(os.path.join(CDIR, "schedules.json")))
    cv = {}
    for f in glob.glob(os.path.join(CDIR, "*.csv")):
        name = os.path.basename(f)[:-4]
        if name not in scheds:
            continue
        rows = np.genfromtxt(f, delimiter=",", names=True)
        step = np.atleast_1d(rows["step"]).astype(int); loss = np.atleast_1d(rows["eval_loss"]).astype(float)
        keep = step >= AC.T_MIN
        cv[name] = dict(step=step[keep], lr=np.asarray(scheds[name], float),
                        loss=AC.smooth_by_step(step[keep], loss[keep]))
    return cv


def main():
    cv = load()
    print("schedules:", sorted(cv.keys()))
    # MPL backbone from 'constant'
    params, f = AC.fit_mpl(cv, ["constant"], n_starts=5)
    L0, A, alpha = params[0], params[1], params[2]
    print(f"backbone fit (constant): L0={L0:.4f} A={A:.4f} alpha={alpha:.4f} (obj={f:.5f})")

    rows = []; es, ts = [], []
    for name in sorted(cv):
        if not name.startswith("wsdcon_"):
            continue
        c = cv[name]; s2 = float(c["lr"][-1]); dstep = AC._drop_step(c["lr"])
        S = cumS(c["lr"]); S_drop = float(S[dstep])
        m = c["step"] >= dstep
        tp = (c["step"][m] - dstep).astype(float); y = c["loss"][m]
        if len(tp) < 10:
            continue
        def bbexp(t, amp, tau, db):
            return (L0 + db) + A * (S_drop + s2 * t) ** (-alpha) + amp * np.exp(-t / tau)
        try:
            po, _ = curve_fit(bbexp, tp, y, p0=[max(y[0]-y[-1],1e-3), 300.0, 0.0],
                              maxfev=80000, bounds=([0, 5, -0.5], [5, 20000, 0.5]))
            pred = bbexp(tp, *po); r2 = 1 - np.sum((y-pred)**2)/(np.sum((y-y.mean())**2)+1e-30)
            used = po[0] > 1e-3 and 5 < po[1] < 19000 and r2 > 0.85
            rows.append(dict(eta=s2, tau=float(po[1]), amp=float(po[0]), r2=float(r2),
                             used=bool(used), window=int(tp[-1])))
            print(f"  {name}: eta={s2:.1e} tau={po[1]:.0f} amp={po[0]:+.4f} r2={r2:.3f} window={int(tp[-1])} used={used}")
            if used:
                es.append(s2); ts.append(po[1])
        except Exception as ex:
            print(f"  {name}: fail {ex}")
    out = dict(backbone=[float(L0), float(A), float(alpha)], points=rows)
    if len(es) >= 3:
        p, c_, r2 = fit_powerlaw(np.array(es), np.array(ts))
        out.update(p=float(p), c=float(c_), r2=float(r2), n=len(es))
        print(f"\n>>> tau ~ eta^-{p:.3f}  (log-log r2={r2:.3f}, n={len(es)})  [paper: p=1.00+/-0.18]")
        # predicted lambda_slow = 1/(tau*eta)
        lam = [1.0/(t*e) for t, e in zip(ts, es)]
        print(f">>> implied lambda_slow=1/(tau*eta) = {np.round(lam,2)} (should be ~const if tau~1/eta)")
        out["lambda_slow_implied"] = list(map(float, lam))
    json.dump(out, open(os.path.join(ROOT, "results", "TAU_REPORT.json"), "w"), indent=2)
    print("saved results/TAU_REPORT.json")

    # figure
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        u = [r for r in rows if r["used"]]
        if len(u) >= 2:
            e = np.array([r["eta"] for r in u]); t = np.array([r["tau"] for r in u])
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.loglog(e, t, "o", ms=9, label=f"measured (p={out.get('p',float('nan')):.2f})")
            xx = np.geomspace(e.min(), e.max(), 30)
            if "p" in out: ax.loglog(xx, out["c"]*xx**(-out["p"]), "-")
            ax.loglog(xx, t[0]*(e[0]/xx), "k--", lw=1.2, label="slope -1 (τ∝1/η)")
            ax.set_xlabel("stage-2 LR η"); ax.set_ylabel("relaxation τ (steps)")
            ax.set_title("Real transformer (~10M): τ vs η, long-window wsdcon")
            ax.legend(); fig.tight_layout()
            fig.savefig(os.path.join(ROOT, "figs", "tau_real.png"), dpi=130); print("wrote figs/tau_real.png")
    except Exception as ex:
        print("fig err", ex)


if __name__ == "__main__":
    main()
