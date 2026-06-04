"""
make_figs.py -- figures for the report from real-curve analysis + NQM experiments.
Reads results/REAL_CURVE_REPORT.json and results/curves/*; robust to missing pieces.
Usage: python repro/make_figs.py [curves_dir]
"""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import droprelaxS, cumS, fit_powerlaw
import analyze_curves as AC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "figs")
os.makedirs(FIGS, exist_ok=True)


def fig_residual(all_curves, report):
    scales = list(all_curves.keys())
    n = len(scales)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for j, sc in enumerate(scales):
        ax = axes[0][j]
        cv = all_curves[sc]
        params = report["per_scale"][sc]["params"]
        for s, col in [("cosine", "tab:green"), ("wsd", "tab:red"), ("wsdld", "tab:orange")]:
            if s not in cv:
                continue
            step, r, _ = AC.residual(cv, params, s)
            ax.plot(step, r, color=col, label=f"{s} residual", lw=1.4)
        # DropRelaxS overlay on wsd
        drs = report["per_scale"][sc].get("droprelaxS", {}).get("wsd")
        if drs and "wsd" in cv:
            step = cv["wsd"]["step"]
            K = droprelaxS(cv["wsd"]["lr"], drs["lam"])[step]
            ax.plot(step, drs["kappa"] * K, "k--", lw=1.6, label=f"DropRelaxS (R²={drs['R2']:.2f})")
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_title(f"scale {sc}: MPL residual")
        ax.set_xlabel("step"); ax.set_ylabel("L_true - L_MPL")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig1_residual.png"), dpi=130)
    print("wrote fig1_residual.png")


def fig_tau(all_curves, report):
    scales = list(all_curves.keys())
    fig, ax = plt.subplots(figsize=(6, 5))
    for sc in scales:
        tv = report["per_scale"][sc].get("tau_vs_eta", {})
        pts = [p for p in tv.get("points", []) if p.get("used")]
        if len(pts) < 2:
            continue
        e = np.array([p["eta"] for p in pts]); t = np.array([p["tau"] for p in pts])
        ax.loglog(e, t, "o", label=f"{sc} (p={tv.get('p'):.2f})")
        xx = np.geomspace(e.min(), e.max(), 50)
        ax.loglog(xx, tv["c"] * xx ** (-tv["p"]), "-", lw=1)
    # reference slope -1
    if scales:
        e0 = 1e-4
        ax.loglog([e0, e0 * 16], [1e3, 1e3 / 16], "k--", lw=1.2, label="slope -1 (τ∝1/η)")
    ax.set_xlabel("stage-2 LR  η"); ax.set_ylabel("relaxation time τ (steps)")
    ax.set_title("τ ∝ 1/η  (wsdcon transients)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig2_tau.png"), dpi=130)
    print("wrote fig2_tau.png")


def fig_crossscale(report):
    rows = report.get("cross_scale", [])
    if not rows:
        return
    labels = [f"{r['scale']}\n{r['target']}" for r in rows]
    mpl = [r["mae_mpl"] for r in rows]; ours = [r["mae_ours"] for r in rows]
    x = np.arange(len(rows)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(rows)), 4))
    ax.bar(x - w / 2, mpl, w, label="MPL", color="tab:gray")
    ax.bar(x + w / 2, ours, w, label="MPL + DropRelaxS", color="tab:blue")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("held-out MAE")
    cs = report.get("cross_scale_summary", {})
    ax.set_title(f"Parameter-free cross-scale prediction (avg {cs.get('avg_delta',0)*100:+.0f}%, wins {cs.get('wins','?')}/{cs.get('n','?')})")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "fig3_crossscale.png"), dpi=130)
    print("wrote fig3_crossscale.png")


def main():
    cdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "curves")
    AC.CURVEDIR = cdir
    rep_path = os.path.join(os.path.dirname(cdir), "REAL_CURVE_REPORT.json") if cdir.endswith("curves") else os.path.join(cdir, "REAL_CURVE_REPORT.json")
    if not os.path.exists(rep_path):
        rep_path = os.path.join(ROOT, "results", "REAL_CURVE_REPORT.json")
    report = json.load(open(rep_path))
    scales = report["scales"]
    all_curves = {sc: AC.load_scale(sc) for sc in scales}
    fig_residual(all_curves, report)
    fig_tau(all_curves, report)
    fig_crossscale(report)


if __name__ == "__main__":
    main()
