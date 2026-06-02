#!/usr/bin/env python3
"""Plot reconstructed Pythia loss-vs-tokens curves and print a summary table."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "results" / "pythia_curves" / "pythia_loss_curves.csv"


def load(csv_path: Path):
    by_scale = defaultdict(list)
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            by_scale[r["scale"]].append(
                (int(r["step"]), float(r["tokens_seen"]), float(r["loss"]), float(r["perplexity"]))
            )
    for s in by_scale:
        by_scale[s].sort()
    return by_scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    args = ap.parse_args()
    csv_path = Path(args.csv)
    by_scale = load(csv_path)

    # ordering small -> large
    order = ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"]
    scales = [s for s in order if s in by_scale]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for s in scales:
        rows = by_scale[s]
        toks = [r[1] / 1e9 for r in rows]
        loss = [r[2] for r in rows]
        axes[0].plot(toks, loss, "o-", label=f"pythia-{s}")
        axes[1].plot(toks, loss, "o-", label=f"pythia-{s}")
    for ax in axes:
        ax.set_xlabel("tokens seen (B)")
        ax.set_ylabel("eval loss (cross-entropy, Pile sample)")
        ax.legend()
        ax.grid(alpha=0.3)
    axes[1].set_xscale("log")
    axes[0].set_title("Pythia loss vs tokens (linear x)")
    axes[1].set_title("Pythia loss vs tokens (log x)")
    fig.tight_layout()
    out = csv_path.parent / "pythia_loss_curves.png"
    fig.savefig(out, dpi=130)
    print(f"saved {out}")

    # summary table
    print("\nscale   final_loss  final_ppl   tokens(B)   n_pts")
    for s in scales:
        rows = by_scale[s]
        last = rows[-1]
        print(f"{s:6s}  {last[2]:9.4f}  {last[3]:8.2f}  {last[1]/1e9:9.2f}   {len(rows)}")

    # scale ordering check at the final shared step
    print("\nloss ordering at final step (should be monotone decreasing in scale):")
    print("  " + " > ".join(f"{s}:{by_scale[s][-1][2]:.3f}" for s in scales))


if __name__ == "__main__":
    main()
