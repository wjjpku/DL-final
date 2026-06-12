"""Attempt 3 verdicts -- executes repin_prereg.json verbatim.

Per arm: tau_loss from the arm's own loss curve (analyze_floor exp protocol);
S_pre(dS) = top preconditioned-Hessian eigenvalue (probes with Ritz residual
> 0.05 dropped); R(dS) = (S_pre(dS)-S_pre(0)) / (38/eta2 - S_pre(0));
tau_spectral = 1-1/e crossing of the monotone-rectified R.

Verdicts: OFF_REGIME (control S_pre(0) outside [0.5,1.5]x38/eta1),
C_REPIN (R monotone rho>=0.8 AND tau ratio in [1/3,3]),
C_DECOUPLED (R<0.2 throughout 2.5*tau_loss OR ratio outside [0.1,10]),
else AMBIGUOUS.  B-axis secondary only if C_REPIN at m.
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_curves as AC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ETA1, B1 = 1.5e-3, 0.9
EDGE = lambda eta: (2 + 2 * B1) / ((1 - B1) * eta)   # 38/eta
ETA2 = {"spec_e10": 1e-4, "spec_e40": 4e-4, "spec_nodrop": 1.5e-3,
        "spec_e10_b12": 1e-4, "spec_e10_b192": 1e-4}
RES_GATE = 0.05


def tau_loss(path):
    rows = np.genfromtxt(path, delimiter=",", names=True)
    step = np.atleast_1d(rows["step"]).astype(int)
    loss = np.atleast_1d(rows["eval_loss"]).astype(float)
    sm = AC.smooth_by_step(step, loss)
    m = (step >= 3000) & (step <= 3000 + 5500)
    t = (step[m] - 3000).astype(float)
    y = sm[m]

    def mdl(t, a, b, amp, tau):
        return a + b * t + amp * np.exp(-t / tau)
    po, _ = curve_fit(mdl, t, y,
                      p0=[y[-1], 0.0, max(y[0] - y[-1], 1e-3), 300.0],
                      maxfev=50000,
                      bounds=([0, -1e-3, 0, 10], [5, 1e-3, 2, 9000]))
    pred = mdl(t, *po)
    r2 = 1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-30)
    return float(po[3]), float(po[2]), float(r2)


def tau_from_R(ds, R):
    """1-1/e crossing of the monotone-rectified R (linear interp)."""
    Rm = np.maximum.accumulate(R)
    thr = 1 - 1 / np.e
    if Rm[-1] < thr:
        return None
    i = int(np.argmax(Rm >= thr))
    if i == 0:
        return float(ds[0])
    return float(np.interp(thr, [Rm[i - 1], Rm[i]], [ds[i - 1], ds[i]]))


def analyze_arm(scale, tag):
    base = os.path.join(ROOT, "results", f"curves_spectrum_{scale}")
    spec = os.path.join(base, f"{tag}_spec.csv")
    curve = os.path.join(base, f"{tag}.csv")
    rows = np.genfromtxt(spec, delimiter=",", names=True)
    ds = np.atleast_1d(rows["dS"]).astype(float)
    sp = np.atleast_1d(rows["lp1"]).astype(float)
    res = np.atleast_1d(rows["res_p"]).astype(float)
    keep = res <= RES_GATE
    eta2 = ETA2[tag]
    tl, amp, r2 = tau_loss(curve)
    s0 = sp[ds == 0][0] if np.any(ds == 0) else sp[0]
    R = (sp - s0) / (EDGE(eta2) - s0)
    out = dict(tag=tag, eta2=eta2, tau_loss=tl, r2_loss=r2, S_pre0=float(s0),
               edge_target=float(EDGE(eta2)),
               dropped_probes=int((~keep).sum()),
               ds=ds[keep].tolist(), R=R[keep].tolist(),
               S_pre=sp[keep].tolist())
    win = keep & (ds <= 2.5 * tl) & (ds > 0)
    if win.sum() >= 4:
        rho = float(spearmanr(ds[win], R[win]).statistic)
        out["rho"] = rho
    ts = tau_from_R(ds[keep], R[keep])
    out["tau_spectral"] = ts
    if ts is not None:
        # bootstrap: leave-2-out over interior probes
        rng = np.random.default_rng(0)
        idx = np.where(keep)[0]
        tss = []
        for _ in range(400):
            sub = np.sort(rng.choice(idx, max(len(idx) - 2, 4), replace=False))
            t_ = tau_from_R(ds[sub], R[sub])
            if t_ is not None:
                tss.append(t_)
        if tss:
            out["tau_spectral_ci"] = [float(np.percentile(tss, 5)),
                                      float(np.percentile(tss, 95))]
    return out


def verdict(scale):
    base = os.path.join(ROOT, "results", f"curves_spectrum_{scale}")
    arms = {os.path.basename(f)[:-9]: None
            for f in glob.glob(os.path.join(base, "*_spec.csv"))}
    rep = {"scale": scale}
    for tag in sorted(arms):
        try:
            rep[tag] = analyze_arm(scale, tag)
        except Exception as e:
            rep[tag] = {"error": str(e)}
    ctrl = rep.get("spec_nodrop")
    if ctrl and "S_pre0" in ctrl:
        ratio = ctrl["S_pre0"] / EDGE(ETA1)
        rep["sanity_ratio"] = ratio
        if not (0.5 <= ratio <= 1.5):
            rep["verdict"] = f"OFF_REGIME (S_pre0/edge = {ratio:.2f})"
            return rep
    votes = {}
    for tag in ["spec_e10", "spec_e40"]:
        a = rep.get(tag)
        if not a or "error" in a:
            continue
        tl, ts, rho = a["tau_loss"], a.get("tau_spectral"), a.get("rho", -9)
        Rmax = max((a["R"][i] for i in range(len(a["ds"]))
                    if 0 < a["ds"][i] <= 2.5 * tl), default=0)
        if ts is not None and rho >= 0.8 and 1 / 3 <= ts / tl <= 3:
            votes[tag] = "C_REPIN"
        elif Rmax < 0.2 or (ts is not None and not 0.1 <= ts / tl <= 10):
            votes[tag] = "C_DECOUPLED"
        else:
            votes[tag] = "AMBIGUOUS"
    rep["votes"] = votes
    vs = set(votes.values())
    rep["verdict"] = votes and (vs.pop() if len(vs) == 1 else
                                "MIXED: " + json.dumps(votes)) or "NO DATA"
    return rep


def main():
    full = {}
    for scale in ["m", "ml", "l", "xl"]:
        if glob.glob(os.path.join(ROOT, "results", f"curves_spectrum_{scale}",
                                  "*_spec.csv")):
            rep = verdict(scale)
            full[scale] = rep
            print(f"\n== scale {scale} ==")
            for tag, a in rep.items():
                if isinstance(a, dict) and "tau_loss" in a:
                    print(f"  {tag:14s} tau_loss={a['tau_loss']:7.0f} "
                          f"tau_spec={a.get('tau_spectral')} "
                          f"rho={a.get('rho', float('nan')):.2f} "
                          f"S_pre0={a['S_pre0']:.0f}/{a['edge_target']:.0f} "
                          f"drop={a['dropped_probes']}")
            print(f"  sanity={rep.get('sanity_ratio')}")
            print(f"  VERDICT: {rep.get('verdict')}")
    # B-axis secondary
    m = full.get("m", {})
    if m.get("votes", {}).get("spec_e10") == "C_REPIN":
        b12 = m.get("spec_e10_b12", {})
        b192 = m.get("spec_e10_b192", {})
        t12, t192 = b12.get("tau_spectral"), b192.get("tau_spectral")
        if t12 and t192:
            spectral_slower_smallB = t12 > t192
            print(f"\nB-axis secondary: tau_spec(b12)={t12:.0f} vs "
                  f"tau_spec(b192)={t192:.0f} -> spectral channel is "
                  f"{'the B-DEPENDENT' if spectral_slower_smallB else 'the B-BLIND'}"
                  f" channel of the 1A mixture")
            full["B_axis"] = dict(t12=t12, t192=t192,
                                  spectral_is_B_dependent=
                                  bool(spectral_slower_smallB))
    out = os.path.join(ROOT, "results", "SPECTRUM_REPORT.json")
    json.dump(full, open(out, "w"), indent=2, default=float)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
