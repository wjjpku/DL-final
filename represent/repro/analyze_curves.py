"""
analyze_curves.py -- Real-curve reproduction of the paper's experiments on loss curves
we generate ourselves (results/curves/<scale>_<sched>.csv + schedules.json).

Reproduces, on a real transformer:
  (Fig 1)  the non-adiabatic residual signature: well-fit MPL residual ~0 on stable/cosine,
           jumps positive on fast wsd/wsdld decays.
  (sec ii) residual = DropRelaxS kernel  (R^2 of residual ~ kappa * K(t)).
  (Fig 2)  tau ~ 1/eta from the wsdcon two-stage relaxation transients (pooled p).
  (Table1) parameter-free cross-scale prediction: MPL vs MPL+correction MAE on held-out wsd/wsdld.

Pure numpy/scipy; imports engine.py.  Usage:  python repro/analyze_curves.py
"""
import os, sys, json, glob
import numpy as np
from scipy.optimize import minimize
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import (mpl_loss, mpl_loss_at, droprelaxS, droprelaxS_twoexp, cumS, drops,
                    measure_tau, fit_powerlaw)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURVEDIR = os.path.join(ROOT, "results", "curves")
OUT = os.path.join(ROOT, "results")

LOSS_COL = "eval_loss"   # clean curve for law fitting
T_MIN = 700              # exclude warmup + early transient (warmup=400); MPL models post-warmup descent
                         # (S(t)=cumsum of FULL lr schedule is still used; only recorded points are filtered)


def huber(r, d=1e-3):
    a = np.abs(r)
    return np.where(a <= d, 0.5 * r * r, d * (a - 0.5 * d))


SMOOTH_STEPS = 80   # step-window for denoising the fast SGD-noise-floor fluctuation in eval loss


def smooth_by_step(step, loss, win=SMOOTH_STEPS):
    """Moving average over a +/- win/2 STEP window (sampling density varies, so window by step
    not by sample index). Suppresses the fast per-step SGD-noise floor jitter (~1e-2) while
    preserving the slow lag/relaxation structure (~1e2 steps)."""
    step = np.asarray(step, float); loss = np.asarray(loss, float)
    out = np.empty_like(loss)
    h = win / 2.0
    for i in range(len(step)):
        m = np.abs(step - step[i]) <= h
        out[i] = loss[m].mean()
    return out


def load_scale(scale):
    """Return dict sched -> {step, lr(full schedule), loss}."""
    scheds = json.load(open(os.path.join(CURVEDIR, "schedules.json")))
    out = {}
    for f in glob.glob(os.path.join(CURVEDIR, f"{scale}_*.csv")):
        name = os.path.basename(f)[len(scale) + 1:-4]
        rows = np.genfromtxt(f, delimiter=",", names=True)
        step = np.atleast_1d(rows["step"]).astype(int)
        loss = np.atleast_1d(rows[LOSS_COL]).astype(float)
        if name not in scheds:
            continue
        keep = step >= T_MIN                     # drop warmup/early transient from the analysis
        st = step[keep]; ls = loss[keep]
        out[name] = dict(step=st, lr=np.asarray(scheds[name], float),
                         loss=smooth_by_step(st, ls), loss_raw=ls)
    return out


# ----------------------- MPL fit (numpy) -----------------------
def mpl_pred_at(etas, step, params):
    return mpl_loss_at(etas, step, *params)


def fit_mpl(curves, train_scheds, n_starts=4, seed=0):
    """Fit 7 MPL params on the training schedules (log-space Huber)."""
    rng = np.random.default_rng(seed)
    # init from a power-law fit of cumulative-LR vs loss across train curves
    minloss = min(curves[s]["loss"].min() for s in train_scheds)
    lx, ly = [], []
    for s in train_scheds:
        c = curves[s]
        S = cumS(c["lr"])[c["step"]]
        lx.append(np.log(S)); ly.append(np.log(np.clip(c["loss"] - minloss + 0.01, 1e-6, None)))
    lx = np.concatenate(lx); ly = np.concatenate(ly)
    A_ = np.vstack([lx, np.ones_like(lx)]).T
    slope, inter = np.linalg.lstsq(A_, ly, rcond=None)[0]
    alpha0 = max(-slope, 0.1); A0 = max(np.exp(inter), 0.05); L0_0 = max(minloss - 0.1, 0.1)

    def obj(p):
        L0, A, alpha, B, C, beta, gamma = p
        if min(A, alpha, B, C, beta, gamma) <= 0 or L0 < 0:
            return 1e9
        tot = 0.0
        for s in train_scheds:
            c = curves[s]
            try:
                pred = mpl_pred_at(c["lr"], c["step"], p)
            except Exception:
                return 1e9
            if np.any(pred <= 0) or np.any(~np.isfinite(pred)):
                return 1e9
            r = np.log(c["loss"]) - np.log(pred)
            tot += huber(r).sum()
        return tot

    best = None; bestf = np.inf
    base = [L0_0, A0, alpha0, 300.0, 2.0, 0.6, 0.6]
    starts = [base]
    for _ in range(n_starts - 1):
        starts.append([L0_0 * rng.uniform(0.8, 1.1), A0 * rng.uniform(0.6, 1.6),
                       alpha0 * rng.uniform(0.7, 1.4), rng.uniform(100, 600),
                       rng.uniform(1, 3), rng.uniform(0.4, 0.8), rng.uniform(0.4, 0.9)])
    bnds = [(0, 10), (1e-3, 5), (0.05, 1.5), (1, 8000), (0.05, 300), (0.02, 3), (0.02, 3)]
    for s0 in starts:
        try:
            res = minimize(obj, s0, method="L-BFGS-B", bounds=bnds,
                           options=dict(maxiter=4000, ftol=1e-11, gtol=1e-8))
            if res.fun < bestf:
                bestf = res.fun; best = res.x
        except Exception:
            pass
    return best, bestf


# ----------------------- residual + DropRelaxS -----------------------
def residual(curves, params, sched):
    c = curves[sched]
    pred = mpl_pred_at(c["lr"], c["step"], params)
    return c["step"], c["loss"] - pred, pred


def fair_baseline_test(curves, params, scheds=("cosine", "wsd", "wsdld")):
    """The verifier's key diagnostic: does the FINITE-lambda_slow relaxation kernel beat the
    adiabatic 'cumulative-drop' baseline (lambda_slow->0, i.e. lag prop to total LR dropped)?
    We fit ONE amplitude across all schedules jointly (through origin) for each kernel and compare
    joint R^2. The relaxation kernel's unique content is rate-dependence: cosine and wsd reach the
    SAME final LR (same total drop) yet only the fast wsd lags -> cumulative-drop CANNOT explain the
    contrast, a finite lambda_slow can. Also report the end-of-curve lag ratio wsd/cosine."""
    avail = [s for s in scheds if s in curves]
    R = {s: residual(curves, params, s) for s in avail}   # (step, r, pred)
    def joint_r2(kernel_fn):
        num = den = 0.0; ys = []; ks = []
        for s in avail:
            step, r, _ = R[s]
            K = kernel_fn(curves[s]["lr"])[step]
            ys.append(r); ks.append(K)
        y = np.concatenate(ys); k = np.concatenate(ks)
        kappa = np.sum(y * k) / (np.sum(k * k) + 1e-30)
        pred = kappa * k
        return 1 - np.sum((y - pred) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-30), float(kappa)
    # finite lambda_slow: scan, pick best joint R2
    best = dict(R2=-np.inf)
    for lam in np.geomspace(0.3, 500, 50):
        r2, kap = joint_r2(lambda e, L=lam: droprelaxS(e, L))
        if r2 > best["R2"]:
            best = dict(R2=float(r2), lam=float(lam), kappa=kap)
    # cumulative-drop baseline (lambda_slow -> 0): K = cumulative sum of drops so far
    cum_r2, cum_kap = joint_r2(lambda e: np.cumsum(drops(e)))
    # end-of-curve lag ratio wsd/cosine (the 'same destination, faster sweep, bigger lag' signature)
    ratio = None
    if "wsd" in R and "cosine" in R:
        rc = R["cosine"][1]; rw = R["wsd"][1]
        # compare over the last 20% of steps
        nc = max(1, len(rc) // 5); nw = max(1, len(rw) // 5)
        ec = np.mean(rc[-nc:]); ew = np.mean(rw[-nw:])
        ratio = dict(cosine_end=float(ec), wsd_end=float(ew),
                     wsd_over_cosine=float(ew / ec) if ec != 0 else float("inf"))
    return dict(finite=best, cumdrop=dict(R2=float(cum_r2), kappa=float(cum_kap)),
                advantage=float(best["R2"] - cum_r2), end_lag_ratio=ratio, scheds=avail)


def fit_droprelaxS(curves, params, sched, lam_grid=None):
    """Regress residual on K(t) through origin; scan lambda_slow for best R^2."""
    step, r, _ = residual(curves, params, sched)
    etas = curves[sched]["lr"]
    if lam_grid is None:
        lam_grid = np.geomspace(0.5, 500, 40)
    best = dict(R2=-np.inf)
    for lam in lam_grid:
        K = droprelaxS(etas, lam)[step]
        denom = np.sum(K * K) + 1e-30
        kappa = np.sum(r * K) / denom
        pred = kappa * K
        ss_res = np.sum((r - pred) ** 2)
        ss_tot = np.sum((r - r.mean()) ** 2) + 1e-30
        R2 = 1 - ss_res / ss_tot
        if R2 > best["R2"]:
            best = dict(R2=float(R2), lam=float(lam), kappa=float(kappa))
    return best, step, r


# ----------------------- tau ~ 1/eta from wsdcon -----------------------
def _drop_step(lr):
    """Index (in full-schedule units) of the LR drop in a wsdcon schedule."""
    d = np.diff(lr)
    k = int(np.argmin(d))            # most negative step (the drop)
    return k + 1


def tau_vs_eta(curves, params, r2_min=0.7):
    """Paper Fig.2: fit the RESIDUAL transient r(t)=L_true-L_MPL after the wsdcon drop to
    an exponential -> tau; then fit tau vs stage-2 LR as a power law tau ~ eta^{-p}.
    Using the residual (not raw loss) removes the MPL backbone + MPL's adiabatic drop response,
    isolating the non-adiabatic lag (exactly what the paper does)."""
    etas2, taus, info = [], [], []
    for name, c in sorted(curves.items()):
        if not name.startswith("wsdcon_"):
            continue
        eta2 = float(c["lr"][-1])                       # stage-2 LR
        dstep = _drop_step(c["lr"])
        step, r_full, _ = residual(curves, params, name)
        mask = step >= dstep
        if mask.sum() < 6:
            info.append(dict(eta=eta2, tau=np.nan, r2=np.nan, used=False, reason="few pts")); continue
        rr = r_full[mask]
        tt = step[mask].astype(float)           # actual step times (irregularly sampled)
        # the lag is positive right after the drop and decays toward 0
        res = measure_tau(rr, t0=0, floor=None, t=tt)
        used = np.isfinite(res["tau"]) and res["r2"] >= r2_min and res["tau"] > 1 and res["amp"] > 0
        info.append(dict(eta=eta2, tau=float(res["tau"]), amp=float(res["amp"]), floor=float(res["floor"]),
                         r2=float(res["r2"]), used=bool(used), drop=int(dstep), npts=int(mask.sum())))
        if used:
            etas2.append(eta2); taus.append(res["tau"])
    if len(etas2) >= 3:
        p, c_, r2 = fit_powerlaw(np.array(etas2), np.array(taus))
    else:
        p, c_, r2 = np.nan, np.nan, np.nan
    return dict(p=float(p), c=float(c_), r2=float(r2), n=len(etas2), points=info)


def dLeq_deta(curves, drop_step=4000):
    """Estimate dL_eq/deta from wsdcon final (equilibrated) losses vs stage-2 LR (noise-floor slope)."""
    xs, ys = [], []
    for name, c in curves.items():
        if not name.startswith("wsdcon_"):
            continue
        eta2 = c["lr"][-1]
        # use the last few samples as the equilibrated floor
        floor = np.mean(c["loss"][-5:])
        xs.append(eta2); ys.append(floor)
    xs = np.array(xs); ys = np.array(ys)
    if len(xs) < 2:
        return np.nan, xs, ys
    A = np.vstack([xs, np.ones_like(xs)]).T
    slope, inter = np.linalg.lstsq(A, ys, rcond=None)[0]
    return float(slope), xs, ys


# ----------------------- cross-scale prediction (Table 1) -----------------------
def correction_term(etas, lam_slow, c, eta_peak, dLeq):
    return c * eta_peak * dLeq * droprelaxS(etas, lam_slow)


def cross_scale(all_curves, targets=("wsd", "wsdld"), train_scheds=("cosine", "constant", "wsdcon_9")):
    scales = list(all_curves.keys())
    # fit a well-fit MPL per scale on the train schedules
    fits = {}
    perscale = {}
    for sc in scales:
        cv = all_curves[sc]
        ts = [s for s in train_scheds if s in cv]
        if "wsdcon_9" not in cv:  # pick a middle wsdcon as train
            wc = [s for s in cv if s.startswith("wsdcon_")]
            if wc:
                ts = [s for s in train_scheds if s in cv and not s.startswith("wsdcon_")] + [sorted(wc)[len(wc) // 2]]
        params, f = fit_mpl(cv, ts)
        fits[sc] = params
        # per-scale lambda_slow and c (kappa = c*eta_peak*dLeq) measured on its OWN wsd
        eta_peak = max(cv["cosine"]["lr"]) if "cosine" in cv else max(next(iter(cv.values()))["lr"])
        dL, _, _ = dLeq_deta(cv)
        best_wsd, _, _ = fit_droprelaxS(cv, params, "wsd") if "wsd" in cv else (dict(R2=np.nan, lam=np.nan, kappa=np.nan), None, None)
        lam = best_wsd["lam"]; kappa = best_wsd["kappa"]
        c_const = kappa / (eta_peak * dL) if (np.isfinite(dL) and dL != 0) else np.nan
        perscale[sc] = dict(eta_peak=eta_peak, dLeq=dL, lam_slow=lam, kappa=kappa, c=c_const, params=list(map(float, params)))
    # leave-one-scale-out parameter-free prediction
    rows = []
    for sc in scales:
        cv = all_curves[sc]; params = fits[sc]
        others = [s for s in scales if s != sc]
        lam_pred = np.nanmean([perscale[o]["lam_slow"] for o in others]) if others else perscale[sc]["lam_slow"]
        c_pred = np.nanmean([perscale[o]["c"] for o in others]) if others else perscale[sc]["c"]
        eta_peak = perscale[sc]["eta_peak"]; dL = perscale[sc]["dLeq"]  # target's OWN cheap probe
        for tgt in targets:
            if tgt not in cv:
                continue
            c_t = cv[tgt]
            pred_mpl = mpl_pred_at(c_t["lr"], c_t["step"], params)
            corr = correction_term(c_t["lr"], lam_pred, c_pred, eta_peak, dL)[c_t["step"]]
            pred_ours = pred_mpl + corr
            mae_mpl = np.mean(np.abs(c_t["loss"] - pred_mpl))
            mae_ours = np.mean(np.abs(c_t["loss"] - pred_ours))
            rows.append(dict(scale=sc, target=tgt, mae_mpl=float(mae_mpl), mae_ours=float(mae_ours),
                             delta=float((mae_ours - mae_mpl) / mae_mpl), lam_pred=float(lam_pred),
                             c_pred=float(c_pred), dLeq=float(dL)))
    return rows, perscale, fits


def main():
    scales = []
    for f in glob.glob(os.path.join(CURVEDIR, "*_cosine.csv")):
        scales.append(os.path.basename(f).split("_")[0])
    scales = sorted(set(scales))
    print("scales found:", scales)
    all_curves = {}
    # only analyze scales that are fully trained (avoid partial scales during live training)
    complete = []
    for sc in scales:
        cv = load_scale(sc)
        nwc = len([k for k in cv if k.startswith("wsdcon_")])
        if "cosine" in cv and "constant" in cv and "wsd" in cv and nwc >= 3:
            complete.append(sc); all_curves[sc] = cv
        else:
            print(f"  [skip incomplete scale {sc}: {sorted(cv.keys())}]")
    scales = complete
    report = dict(scales=scales, per_scale={})
    for sc in scales:
        cv = all_curves[sc]
        print(f"\n===== scale {sc}: schedules = {sorted(cv.keys())} =====")
        # MPL fit on train set
        train = [s for s in ("cosine", "constant") if s in cv]
        wc = sorted([s for s in cv if s.startswith("wsdcon_")])
        if wc:
            train.append(wc[len(wc) // 2])
        params, f = fit_mpl(cv, train)
        print(f"  MPL fit on {train}: obj={f:.4f}")
        print(f"  params L0,A,alpha,B,C,beta,gamma = {np.array(params)}")
        # residual signature
        sig = {}
        for s in ("cosine", "wsd", "wsdld"):
            if s in cv:
                _, r, _ = residual(cv, params, s)
                sig[s] = dict(max_abs=float(np.max(np.abs(r))), end=float(r[-1]), mean=float(r.mean()))
        print(f"  residual signature (max_abs / end): " +
              ", ".join(f"{k}:{v['max_abs']:.4f}/{v['end']:.4f}" for k, v in sig.items()))
        # droprelaxS fit on wsd
        drs = {}
        for s in ("wsd", "wsdld"):
            if s in cv:
                best, _, _ = fit_droprelaxS(cv, params, s)
                drs[s] = best
                print(f"  DropRelaxS fit {s}: R2={best['R2']:.3f} lam={best['lam']:.2f}")
        # fair-baseline diagnostic (finite lambda_slow vs cumulative-drop)
        fb = fair_baseline_test(cv, params)
        if fb.get("end_lag_ratio"):
            elr = fb["end_lag_ratio"]
            print(f"  non-adiabatic signature: end-lag wsd/cosine = {elr['wsd_over_cosine']:.2f} "
                  f"(wsd={elr['wsd_end']:.4f}, cosine={elr['cosine_end']:.4f})")
        print(f"  fair baseline: finite-lambda R2={fb['finite']['R2']:.3f} (lam={fb['finite']['lam']:.2f}) "
              f"vs cumdrop R2={fb['cumdrop']['R2']:.3f}  -> advantage {fb['advantage']:+.3f}")
        # tau vs eta
        tv = tau_vs_eta(cv, params)
        print(f"  tau~1/eta: p={tv['p']:.3f} (r2={tv['r2']:.3f}, n={tv['n']})")
        dL, xs, ys = dLeq_deta(cv)
        print(f"  dLeq/deta (noise-floor slope) = {dL:.3f}")
        report["per_scale"][sc] = dict(params=list(map(float, params)), fit_obj=float(f),
                                       signature=sig, droprelaxS=drs, fair_baseline=fb,
                                       tau_vs_eta=tv, dLeq_deta=float(dL))
    # cross-scale
    if len(scales) >= 2:
        rows, perscale, fits = cross_scale(all_curves)
        report["cross_scale"] = rows
        report["perscale_constants"] = {k: {kk: vv for kk, vv in v.items() if kk != "params"} for k, v in perscale.items()}
        if rows:
            avg = np.mean([r["delta"] for r in rows])
            wins = sum(1 for r in rows if r["mae_ours"] < r["mae_mpl"])
            print(f"\n===== CROSS-SCALE (Table 1 analog) =====")
            for r in rows:
                print(f"  {r['scale']} {r['target']}: MPL={r['mae_mpl']:.5f} +ours={r['mae_ours']:.5f} ({r['delta']*100:+.0f}%)")
            print(f"  overall avg delta = {avg*100:+.0f}%  wins {wins}/{len(rows)}")
            report["cross_scale_summary"] = dict(avg_delta=float(avg), wins=wins, n=len(rows))
    json.dump(report, open(os.path.join(OUT, "REAL_CURVE_REPORT.json"), "w"), indent=2)
    print(f"\nsaved {os.path.join(OUT, 'REAL_CURVE_REPORT.json')}")


if __name__ == "__main__":
    main()
