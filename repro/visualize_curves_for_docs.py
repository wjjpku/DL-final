#!/usr/bin/env python3
"""Create curve visualizations used by the paper and slides."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "dl_final_mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "dl_final_xdg_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATA_ROOT = ROOT / "external" / "MultiPowerLaw" / "loss_curve_repo"
PRED_ROOT = ROOT / "results" / "official_compare" / "predictions"
PAPER_FIGS = ROOT / "paper" / "figs"
SLIDE_FIGS = ROOT / "slides" / "figs"

SCALE = "100"
SCALES = ["25", "100", "400"]
SCHEDULES = [
    ("cosine_24000", "cosine 24k", "#4C78A8", "-"),
    ("cosine_72000", "cosine 72k", "#72B7B2", "-"),
    ("constant_24000", "constant 24k", "#54A24B", "--"),
    ("constant_72000", "constant 72k", "#B279A2", "--"),
    ("wsd_20000_24000", "WSD", "#E45756", "-"),
    ("wsdld_20000_24000", "WSD long-decay", "#F58518", "-"),
    ("wsdcon_3", "WSD const tail 3e-5", "#9D755D", ":"),
    ("wsdcon_9", "WSD const tail 9e-5", "#BAB0AC", ":"),
    ("wsdcon_18", "WSD const tail 18e-5", "#B7E075", ":"),
]
FOCUS = ["cosine_72000", "wsd_20000_24000", "wsdld_20000_24000"]


def load_csv(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True)


def curve_path(name: str, scale: str = SCALE) -> Path:
    return DATA_ROOT / f"csv_{scale}" / f"{name}.csv"


def pred_path(name: str, scale: str = SCALE) -> Path:
    return PRED_ROOT / f"{scale}_compare_{name}.csv"


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#D0D0D0", linewidth=0.7, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_all(fig: plt.Figure, name: str, dpi: int = 220) -> None:
    PAPER_FIGS.mkdir(parents=True, exist_ok=True)
    SLIDE_FIGS.mkdir(parents=True, exist_ok=True)
    png = PAPER_FIGS / f"{name}.png"
    pdf = PAPER_FIGS / f"{name}.pdf"
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    shutil.copy2(png, SLIDE_FIGS / png.name)


def make_public_curve_gallery() -> None:
    fig = plt.figure(figsize=(10.8, 6.1), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9])
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_lr = fig.add_subplot(gs[0, 1])
    ax_zoom = fig.add_subplot(gs[1, 0])
    ax_final = fig.add_subplot(gs[1, 1])

    final_names: list[str] = []
    final_losses: list[float] = []
    final_colors: list[str] = []

    for name, label, color, linestyle in SCHEDULES:
        data = load_csv(curve_path(name))
        step = data["step"] / 1000.0
        loss = data["loss"]
        lr = data["lr"]
        ax_loss.plot(step, loss, color=color, linestyle=linestyle, linewidth=1.8, label=label)
        ax_lr.plot(step, lr * 1e4, color=color, linestyle=linestyle, linewidth=1.8)

        if name in FOCUS:
            mask = (data["step"] >= 18000) & (data["step"] <= 26000)
            ax_zoom.plot(step[mask], loss[mask], color=color, linewidth=2.2, label=label)

        final_names.append(label)
        final_losses.append(float(loss[-1]))
        final_colors.append(color)

    ax_loss.set_title("(a) 100M public loss curves")
    ax_loss.set_xlabel("training step (k)")
    ax_loss.set_ylabel("validation loss")
    ax_loss.legend(fontsize=7.0, ncol=2, frameon=False)
    style_axes(ax_loss)

    ax_lr.set_title("(b) learning-rate schedules")
    ax_lr.set_xlabel("training step (k)")
    ax_lr.set_ylabel(r"LR $\times 10^4$")
    style_axes(ax_lr)

    ax_zoom.axvspan(20.0, 24.0, color="#F3C4B5", alpha=0.28, label="sharp WSD decay window")
    ax_zoom.set_title("(c) cooldown zoom: gradual vs sharp decay")
    ax_zoom.set_xlabel("training step (k)")
    ax_zoom.set_ylabel("validation loss")
    ax_zoom.legend(fontsize=7.5, frameon=False)
    style_axes(ax_zoom)

    order = np.argsort(final_losses)
    y = np.arange(len(order))
    ax_final.barh(y, np.array(final_losses)[order], color=np.array(final_colors)[order], alpha=0.88)
    ax_final.set_yticks(y, np.array(final_names)[order], fontsize=7.2)
    ax_final.invert_yaxis()
    ax_final.set_xlabel("final validation loss")
    ax_final.set_title("(d) endpoint differs by schedule")
    xmin = min(final_losses) - 0.01
    xmax = max(final_losses) + 0.015
    ax_final.set_xlim(xmin, xmax)
    style_axes(ax_final)

    fig.suptitle("Public 100M curves used for setup, reproduction, and residual tests", fontsize=13)
    save_all(fig, "fig_public_curve_gallery")
    plt.close(fig)


def make_prediction_overlay() -> None:
    curves = [
        ("wsd_20000_24000", "WSD sharp decay"),
        ("wsdld_20000_24000", "WSD long decay"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 5.6), sharex="col", constrained_layout=True)

    for col, (name, title) in enumerate(curves):
        data = load_csv(pred_path(name))
        step = data["step"] / 1000.0
        loss = data["loss"]
        mpl = data["mpl_pred"]
        tissue = data["tissue_pred"]
        resid = loss - mpl

        ax = axes[0, col]
        ax.plot(step, loss, color="#222222", linewidth=2.3, label="ground truth")
        ax.plot(step, mpl, color="#E45756", linewidth=2.0, linestyle="--", label="MPL")
        ax.plot(step, tissue, color="#4C78A8", linewidth=2.0, linestyle="-.", label="Tissue")
        ax.axvspan(20.0, 24.0, color="#F3C4B5", alpha=0.28)
        ax.set_title(f"(a{col + 1}) {SCALE}M {title}: predictions")
        ax.set_ylabel("validation loss")
        if col == 0:
            ax.legend(fontsize=8.0, frameon=False)
        style_axes(ax)

        axr = axes[1, col]
        axr.plot(step, resid * 1e3, color="#E45756", linewidth=2.0)
        axr.axhline(0, color="#555555", linewidth=0.9)
        axr.axvspan(20.0, 24.0, color="#F3C4B5", alpha=0.28)
        axr.set_title(f"(b{col + 1}) MPL residual")
        axr.set_xlabel("training step (k)")
        axr.set_ylabel(r"$L_{\rm true}-L_{\rm MPL}$  ($\times 10^3$)")
        style_axes(axr)

    fig.suptitle("Curve-level reproduction example: fit quality and non-adiabatic residual", fontsize=13)
    save_all(fig, "fig_curve_prediction_overlay")
    plt.close(fig)


def make_data_wsd_focus() -> None:
    """Show the exact WSD schedules used for the non-adiabatic question."""
    selected = [
        ("cosine_72000", "cosine 72k", "#72B7B2", "-"),
        ("wsd_20000_24000", "WSD sharp", "#E45756", "-"),
        ("wsdld_20000_24000", "WSD long-decay", "#F58518", "-"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 5.7), constrained_layout=True)
    ax_lr, ax_loss = axes[0]
    ax_lr_zoom, ax_loss_zoom = axes[1]

    for name, label, color, linestyle in selected:
        data = load_csv(curve_path(name))
        step = data["step"] / 1000.0
        lr = data["lr"] * 1e4
        loss = data["loss"]
        zoom = (data["step"] >= 18000) & (data["step"] <= 26000)

        ax_lr.plot(step, lr, color=color, linestyle=linestyle, linewidth=2.0, label=label)
        ax_loss.plot(step, loss, color=color, linestyle=linestyle, linewidth=2.0, label=label)
        ax_lr_zoom.plot(step[zoom], lr[zoom], color=color, linewidth=2.4, label=label)
        ax_loss_zoom.plot(step[zoom], loss[zoom], color=color, linewidth=2.4, label=label)

    for ax in [ax_lr, ax_loss, ax_lr_zoom, ax_loss_zoom]:
        ax.axvspan(20.0, 24.0, color="#F3C4B5", alpha=0.25)
        style_axes(ax)

    ax_lr.set_title("(a) LR schedules: same destination, different speed")
    ax_lr.set_xlabel("training step (k)")
    ax_lr.set_ylabel(r"LR $\times 10^4$")
    ax_lr.legend(fontsize=8.0, frameon=False)

    ax_loss.set_title("(b) validation loss curves")
    ax_loss.set_xlabel("training step (k)")
    ax_loss.set_ylabel("validation loss")
    ax_loss.legend(fontsize=8.0, frameon=False)

    ax_lr_zoom.set_title("(c) cooldown LR zoom")
    ax_lr_zoom.set_xlabel("training step (k)")
    ax_lr_zoom.set_ylabel(r"LR $\times 10^4$")
    ax_lr_zoom.set_xlim(18, 26)

    ax_loss_zoom.set_title("(d) loss zoom after LR change")
    ax_loss_zoom.set_xlabel("training step (k)")
    ax_loss_zoom.set_ylabel("validation loss")
    ax_loss_zoom.set_xlim(18, 26)

    fig.suptitle("Data focus: WSD tests whether fast LR change leaves a transient loss lag", fontsize=13)
    save_all(fig, "fig_data_wsd_focus")
    plt.close(fig)


def make_wsd_fit_zoom() -> None:
    """Curve-level WSD fit with local cooldown zoom and residual."""
    data = load_csv(pred_path("wsd_20000_24000"))
    step = data["step"] / 1000.0
    loss = data["loss"]
    mpl = data["mpl_pred"]
    tissue = data["tissue_pred"]
    resid = loss - mpl
    zoom = (data["step"] >= 18500) & (data["step"] <= 24500)
    stable = (data["step"] >= 10000) & (data["step"] < 20000)
    cooldown = data["step"] >= 20000

    stable_mae = np.mean(np.abs(resid[stable])) * 1e3
    cooldown_mae = np.mean(np.abs(resid[cooldown])) * 1e3
    final_resid = resid[-1] * 1e3

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), constrained_layout=True)
    ax_full, ax_zoom, ax_resid = axes

    ax_full.plot(step, loss, color="#222222", linewidth=2.2, label="ground truth")
    ax_full.plot(step, mpl, color="#E45756", linewidth=2.0, linestyle="--", label="MPL")
    ax_full.plot(step, tissue, color="#4C78A8", linewidth=1.9, linestyle="-.", label="Tissue")
    ax_full.axvspan(20.0, 24.0, color="#F3C4B5", alpha=0.26)
    ax_full.set_title("(a) full WSD curve")
    ax_full.set_xlabel("training step (k)")
    ax_full.set_ylabel("validation loss")
    ax_full.legend(fontsize=8.0, frameon=False)
    style_axes(ax_full)

    ax_zoom.plot(step[zoom], loss[zoom], color="#222222", linewidth=2.4, label="ground truth")
    ax_zoom.plot(step[zoom], mpl[zoom], color="#E45756", linewidth=2.2, linestyle="--", label="MPL")
    ax_zoom.plot(step[zoom], tissue[zoom], color="#4C78A8", linewidth=2.0, linestyle="-.", label="Tissue")
    ax_zoom.axvspan(20.0, 24.0, color="#F3C4B5", alpha=0.26)
    ax_zoom.set_title("(b) local zoom after LR drop")
    ax_zoom.set_xlabel("training step (k)")
    ax_zoom.set_ylabel("validation loss")
    ax_zoom.set_xlim(18.5, 24.5)
    style_axes(ax_zoom)

    ax_resid.plot(step[zoom], resid[zoom] * 1e3, color="#E45756", linewidth=2.3)
    ax_resid.axhline(0, color="#555555", linewidth=0.9)
    ax_resid.axvspan(20.0, 24.0, color="#F3C4B5", alpha=0.26)
    ax_resid.set_title("(c) MPL residual in cooldown")
    ax_resid.set_xlabel("training step (k)")
    ax_resid.set_ylabel(r"$L_{\rm true}-L_{\rm MPL}$  ($\times 10^3$)")
    ax_resid.set_xlim(18.5, 24.5)
    ax_resid.text(
        0.03,
        0.94,
        f"stable MAE={stable_mae:.1f}\npost-drop MAE={cooldown_mae:.1f}\nfinal residual={final_resid:.1f}",
        transform=ax_resid.transAxes,
        va="top",
        ha="left",
        fontsize=8.2,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9},
    )
    style_axes(ax_resid)

    fig.suptitle("100M WSD: fitting is good globally, but the LR-drop region exposes a positive lag", fontsize=13)
    save_all(fig, "fig_wsd_fit_zoom")
    plt.close(fig)


def make_wsd_residual_by_scale() -> None:
    """Show the WSD residual around the LR drop across all public scales."""
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4), sharey=True, constrained_layout=True)
    colors = {"25": "#4C78A8", "100": "#E45756", "400": "#54A24B"}
    for ax, scale in zip(axes, SCALES):
        data = load_csv(pred_path("wsd_20000_24000", scale))
        step = data["step"] / 1000.0
        resid = (data["loss"] - data["mpl_pred"]) * 1e3
        zoom = (data["step"] >= 18500) & (data["step"] <= 24500)
        post = data["step"] >= 20000
        ax.plot(step[zoom], resid[zoom], color=colors[scale], linewidth=2.3)
        ax.axhline(0, color="#555555", linewidth=0.9)
        ax.axvspan(20.0, 24.0, color="#F3C4B5", alpha=0.26)
        ax.set_title(f"{scale}M WSD residual")
        ax.set_xlabel("training step (k)")
        ax.set_xlim(18.5, 24.5)
        ax.text(
            0.04,
            0.93,
            f"final={resid[-1]:.1f}\npost mean={np.mean(resid[post]):.1f}",
            transform=ax.transAxes,
            va="top",
            fontsize=8.2,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9},
        )
        style_axes(ax)
    axes[0].set_ylabel(r"$L_{\rm true}-L_{\rm MPL}$  ($\times 10^3$)")
    fig.suptitle("The WSD cooldown residual is positive across public model scales", fontsize=13)
    save_all(fig, "fig_wsd_residual_by_scale")
    plt.close(fig)


def main() -> None:
    make_public_curve_gallery()
    make_prediction_overlay()
    make_data_wsd_focus()
    make_wsd_fit_zoom()
    make_wsd_residual_by_scale()
    print("wrote paper/figs/fig_public_curve_gallery.{png,pdf}")
    print("wrote paper/figs/fig_curve_prediction_overlay.{png,pdf}")
    print("wrote paper/figs/fig_data_wsd_focus.{png,pdf}")
    print("wrote paper/figs/fig_wsd_fit_zoom.{png,pdf}")
    print("wrote paper/figs/fig_wsd_residual_by_scale.{png,pdf}")
    print("copied PNGs to slides/figs/")


if __name__ == "__main__":
    main()
