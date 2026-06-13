"""ITEM A1 -- sub-edge spectral mode tracking (zero GPU).

Question: does ANY tracked sub-edge preconditioned mode (lp1..lp16, not just
lp1) relax monotonically toward a plateau with a time constant within 3x of
the arm's tau_loss on the drop arms (spec_e10/spec_e40, m and l)?

Protocol
--------
- tau_loss: EXACTLY analyze_spectrum.tau_loss (analyze_floor exp protocol,
  a + b*t + amp*exp(-t/tau) on the smoothed eval curve) -- protocol identity
  with the shipped Attempt-3 record, no refit of anything MPL (G5 untouched).
- Probe gate: res_p <= 0.05 (the prereg Ritz gate); failing probes dropped
  whole-row, as in analyze_spectrum.
- Mode association across probes, two ways:
    rank    : column k of the per-probe descending-sorted Ritz values.
    nearest : chained one-to-one optimal assignment between consecutive kept
              probes in log-value space (scipy linear_sum_assignment).
- Spike mitigation (AUDIT-D: P(v_hat)-state spikes contaminate ranks).  Three
  variants per trajectory, all reported:
    raw     : as recorded.
    medfilt : 3-point rolling median along the probe axis (edge-truncated;
              kills single-probe spikes).  PRIMARY variant for the verdict.
    ratio   : lp_k / bulk, bulk = per-probe median of ranks 9..16 (cancels
              global P(v_hat)-state rescaling), then the same 3-pt median.
              CONFIRMATION variant.
- Per-trajectory fit: y(dS) = c + a*exp(-dS/tau), tau in [5, 8000],
  multistart tau0 {30,100,300,1000}.
- HIT criteria (pre-stated, applied identically everywhere):
    r2 >= 0.8;  tau_mode/tau_loss in [1/3, 3];  |a|/c >= 0.05;
    monotone approach: spearman rho(y, dS) over 0 < dS <= 3*tau_mode
    (>= 4 pts, padded with earliest probes if fewer) with |rho| >= 0.7 and
    sign(rho) = -sign(a);
    LOO stability: >= 80% of leave-one-probe-out refits keep tau in the 3x
    band with r2 >= 0.7.
- Verdict-bearing arms: (m,l) x (spec_e10, spec_e40).  spec_e10_b12/b192 are
  descriptive.  spec_nodrop arms run through the SAME pipeline as a
  false-positive control (band taken from the same scale's drop-arm
  tau_losses): any "hits" there calibrate how easily volatility fakes a
  relaxation at the matching time constant.
- Headline call: CONFIRMED hit = HIT under medfilt AND HIT at the same
  (rank, association) under ratio.  Raw-only hits are flagged spike-suspect.

Output: results/formula_lab/A1_MODE_TRACKING.json
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import curve_fit, linear_sum_assignment
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "represent", "repro"))
import analyze_spectrum as ASP  # noqa: E402  (tau_loss protocol identity)

RES_GATE = 0.05
NRANK = 16
TAU_BOUNDS = (5.0, 8000.0)
ARMS = {"m": ["spec_e10", "spec_e40", "spec_e10_b12", "spec_e10_b192",
              "spec_nodrop"],
        "l": ["spec_e10", "spec_e40", "spec_nodrop"]}
VERDICT_ARMS = [("m", "spec_e10"), ("m", "spec_e40"),
                ("l", "spec_e10"), ("l", "spec_e40")]


def load_spec(scale, tag):
    base = os.path.join(ROOT, "represent", "results", f"curves_spectrum_{scale}")
    rows = np.genfromtxt(os.path.join(base, f"{tag}_spec.csv"),
                         delimiter=",", names=True)
    ds = np.atleast_1d(rows["dS"]).astype(float)
    res = np.atleast_1d(rows["res_p"]).astype(float)
    lp = np.column_stack([np.atleast_1d(rows[f"lp{k}"]).astype(float)
                          for k in range(1, NRANK + 1)])
    lp = -np.sort(-lp, axis=1)               # defensive: descending per probe
    keep = res <= RES_GATE
    return ds[keep], lp[keep], int((~keep).sum()), \
        os.path.join(base, f"{tag}.csv")


def rolling_median3(y):
    out = np.empty_like(y)
    for i in range(len(y)):
        out[i] = np.median(y[max(0, i - 1):i + 2])
    return out


def tracks_rank(lp):
    return lp.copy()                          # (n_probes, 16), column = rank


def tracks_nearest(lp):
    """Chained optimal 1-1 nearest assignment in log space."""
    tr = np.empty_like(lp)
    tr[0] = lp[0]
    cur = np.log(lp[0])
    for i in range(1, lp.shape[0]):
        v = np.log(lp[i])
        cost = np.abs(cur[:, None] - v[None, :])
        r, c = linear_sum_assignment(cost)
        tr[i, r] = lp[i][c]
        cur = np.log(tr[i])
    return tr


def exp_fit(ds, y):
    def mdl(t, c, a, tau):
        return c + a * np.exp(-t / tau)
    c0 = float(np.mean(y[-3:]))
    a0 = float(y[0] - c0)
    best = None
    for tau0 in (30.0, 100.0, 300.0, 1000.0):
        try:
            po, _ = curve_fit(mdl, ds, y, p0=[max(c0, 1e-9), a0, tau0],
                              maxfev=50000,
                              bounds=([0, -np.inf, TAU_BOUNDS[0]],
                                      [np.inf, np.inf, TAU_BOUNDS[1]]))
        except Exception:
            continue
        pred = mdl(ds, *po)
        sse = float(np.sum((y - pred) ** 2))
        if best is None or sse < best[0]:
            best = (sse, po)
    if best is None:
        return None
    sse, po = best
    r2 = 1 - sse / max(float(np.sum((y - y.mean()) ** 2)), 1e-30)
    return dict(c=float(po[0]), a=float(po[1]), tau=float(po[2]),
                r2=float(r2), relamp=float(abs(po[1]) / max(po[0], 1e-30)))


def monotone_check(ds, y, tau):
    m = (ds > 0) & (ds <= 3 * tau)
    idx = np.where(m)[0]
    if len(idx) < 4:                          # pad with earliest probes
        idx = np.where(ds > 0)[0][:max(4, len(idx))]
    if len(idx) < 4:
        return None
    rho = spearmanr(ds[idx], y[idx]).statistic
    return float(rho) if np.isfinite(rho) else None


def assess(ds, y, tau_loss):
    fit = exp_fit(ds, y)
    if fit is None:
        return dict(fit_failed=True, hit=False)
    rho = monotone_check(ds, y, fit["tau"])
    band = (tau_loss / 3.0 <= fit["tau"] <= 3.0 * tau_loss)
    mono = (rho is not None and abs(rho) >= 0.7
            and np.sign(rho) == -np.sign(fit["a"]))
    core = fit["r2"] >= 0.8 and band and fit["relamp"] >= 0.05 and mono
    out = dict(**fit, rho=rho, tau_ratio=float(fit["tau"] / tau_loss),
               in_band=bool(band), monotone=bool(mono))
    if core:                                   # LOO only for core candidates
        taus, ok = [], 0
        for j in range(len(ds)):
            sub = np.delete(np.arange(len(ds)), j)
            f = exp_fit(ds[sub], y[sub])
            if f is None:
                continue
            taus.append(f["tau"])
            if f["r2"] >= 0.7 and tau_loss / 3 <= f["tau"] <= 3 * tau_loss:
                ok += 1
        out["loo_tau_range"] = [float(min(taus)), float(max(taus))] if taus else None
        out["loo_ok_frac"] = float(ok / len(ds))
        out["hit"] = out["loo_ok_frac"] >= 0.8
    else:
        out["hit"] = False
    return out


def analyze_arm(scale, tag, tau_loss):
    ds, lp, ndrop, _ = load_spec(scale, tag)
    bulk = np.median(lp[:, 8:16], axis=1)     # per-probe bulk reference
    res = dict(scale=scale, tag=tag, tau_loss=tau_loss,
               n_probes=len(ds), dropped_probes=ndrop, modes={})
    # Diagnostic: does the BULK itself relax?  Decides how to read ratio-only
    # hits: bulk flat -> ratio hit IS a mode relaxation (up to a constant);
    # bulk relaxing in-band -> the denoised median sub-edge mode relaxes.
    bf = assess(ds, rolling_median3(bulk), tau_loss)
    bf["span_frac"] = float((bulk.max() - bulk.min()) / np.median(bulk))
    res["bulk_fit"] = bf
    for assoc, tfun in (("rank", tracks_rank), ("nearest", tracks_nearest)):
        tr = tfun(lp)
        for k in range(NRANK):
            yraw = tr[:, k]
            variants = {"raw": yraw,
                        "medfilt": rolling_median3(yraw),
                        "ratio": rolling_median3(yraw / bulk)}
            for vname, y in variants.items():
                a = assess(ds, y, tau_loss)
                if a.get("hit") or (a.get("in_band") and a.get("r2", 0) >= 0.8
                                    and a.get("relamp", 0) >= 0.05):
                    res["modes"][f"{assoc}/lp{k + 1}/{vname}"] = a
    # channels: medfilt = absolute trajectory (de-spiked);
    #           ratio   = common-mode-rejected (P(v_hat) cancelled);
    #           confirmed = both at the same (assoc, rank).
    confirmed, medfilt_hits, ratio_hits, raw_only = [], [], [], []
    for key, a in res["modes"].items():
        assoc, rk, v = key.split("/")
        if not a.get("hit"):
            continue
        if v == "medfilt":
            medfilt_hits.append(key)
            r = res["modes"].get(f"{assoc}/{rk}/ratio")
            if r and r.get("hit"):
                confirmed.append(f"{assoc}/{rk}")
        elif v == "ratio":
            ratio_hits.append(key)
        elif v == "raw" and not res["modes"].get(
                f"{assoc}/{rk}/medfilt", {}).get("hit"):
            raw_only.append(key)
    res["medfilt_hits"] = medfilt_hits
    res["ratio_hits"] = ratio_hits
    res["confirmed_hits"] = confirmed
    res["raw_only_spike_suspect"] = raw_only
    return res


def main():
    report = {"protocol": __doc__.strip().split("\n")[0], "arms": {}}
    tau_cache = {}
    for scale, tags in ARMS.items():
        base = os.path.join(ROOT, "represent", "results",
                            f"curves_spectrum_{scale}")
        for tag in tags:
            curve = os.path.join(base, f"{tag}.csv")
            tl, amp, r2 = ASP.tau_loss(curve)
            tau_cache[(scale, tag)] = (tl, r2)
    for scale, tags in ARMS.items():
        for tag in tags:
            if tag == "spec_nodrop":
                continue
            tl, r2l = tau_cache[(scale, tag)]
            arm = analyze_arm(scale, tag, tl)
            arm["tau_loss_r2"] = r2l
            report["arms"][f"{scale}/{tag}"] = arm
    # false-positive control: nodrop arms against each drop arm's band
    report["controls"] = {}
    for scale in ARMS:
        for ref in ["spec_e10", "spec_e40"]:
            tl, _ = tau_cache[(scale, ref)]
            ctrl = analyze_arm(scale, "spec_nodrop", tl)
            report["controls"][f"{scale}/nodrop@tau({ref})={tl:.0f}"] = dict(
                medfilt_hits=ctrl["medfilt_hits"],
                ratio_hits=ctrl["ratio_hits"],
                confirmed_hits=ctrl["confirmed_hits"],
                bulk_fit=ctrl["bulk_fit"],
                n_inband_candidates=len(ctrl["modes"]))
    # verdict
    vb = {f"{s}/{t}": report["arms"][f"{s}/{t}"] for s, t in VERDICT_ARMS}
    any_confirmed = {k: v["confirmed_hits"] for k, v in vb.items()
                     if v["confirmed_hits"]}
    any_medfilt = {k: v["medfilt_hits"] for k, v in vb.items()
                   if v["medfilt_hits"]}
    any_ratio = {k: v["ratio_hits"] for k, v in vb.items()
                 if v["ratio_hits"]}
    bulk_hits = {k: v["bulk_fit"] for k, v in vb.items()
                 if v["bulk_fit"].get("hit")}
    ctrl_fp_med = sum(len(c["medfilt_hits"])
                      for c in report["controls"].values())
    ctrl_fp_ratio = sum(len(c["ratio_hits"])
                        for c in report["controls"].values())
    if any_confirmed or bulk_hits:
        call = ("OVERTURN-CANDIDATE (absolute channel): sub-edge mode/bulk "
                "relaxes with tau within 3x tau_loss")
    elif any_ratio and ctrl_fp_ratio == 0:
        call = ("QUALIFIED YES (ratio channel only): sub-edge modes relax "
                "toward a plateau with tau in [tau_loss/3, 3*tau_loss] AFTER "
                "common-mode P(v_hat) rejection (lp_k / bulk median ranks "
                "9-16); invisible in absolute Ritz values; zero control FPs")
    else:
        call = ("NO: no spike-robust sub-edge mode tracks tau_loss; "
                "decoupling strengthens")
    report["verdict"] = dict(
        confirmed_hits=any_confirmed, medfilt_hits=any_medfilt,
        ratio_hits=any_ratio, bulk_hits=bulk_hits,
        control_false_positives_medfilt=ctrl_fp_med,
        control_false_positives_ratio=ctrl_fp_ratio,
        call=call)
    out = os.path.join(ROOT, "results", "formula_lab", "A1_MODE_TRACKING.json")
    json.dump(report, open(out, "w"), indent=2, default=float)
    # console summary
    for key, arm in report["arms"].items():
        bf = arm["bulk_fit"]
        print(f"== {key}  tau_loss={arm['tau_loss']:.0f} "
              f"(r2={arm['tau_loss_r2']:.3f}) probes={arm['n_probes']} "
              f"dropped={arm['dropped_probes']}")
        print(f"   BULK: tau={bf.get('tau', float('nan')):7.1f} "
              f"r2={bf.get('r2', float('nan')):.3f} "
              f"relamp={bf.get('relamp', float('nan')):.3f} "
              f"span_frac={bf['span_frac']:.3f} hit={bf.get('hit')}")
        for mk, a in sorted(arm["modes"].items(),
                            key=lambda x: -x[1].get("r2", 0)):
            print(f"   {mk:24s} tau={a['tau']:7.1f} ratio={a['tau_ratio']:5.2f} "
                  f"r2={a['r2']:.3f} relamp={a['relamp']:.3f} "
                  f"rho={a['rho'] if a['rho'] is None else round(a['rho'], 2)} "
                  f"hit={a.get('hit')} loo={a.get('loo_ok_frac')}")
    print("\ncontrols (nodrop, false-positive yardstick):")
    for k, c in report["controls"].items():
        print(f"   {k}: medfilt_hits={c['medfilt_hits']} "
              f"ratio_hits={c['ratio_hits']} "
              f"bulk_hit={c['bulk_fit'].get('hit')} "
              f"inband_cands={c['n_inband_candidates']}")
    print(f"\nVERDICT: {report['verdict']['call']}")
    print(f"  confirmed(abs): {any_confirmed}")
    print(f"  medfilt: {any_medfilt}")
    print(f"  ratio: { {k: len(v) for k, v in any_ratio.items()} }")
    print(f"  bulk hits: {list(bulk_hits)}")
    print(f"  control FPs medfilt/ratio: {ctrl_fp_med}/{ctrl_fp_ratio}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
