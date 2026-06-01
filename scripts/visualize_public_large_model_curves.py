#!/usr/bin/env python3
"""Download and visualize publicly available large-model lr/loss curves."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import certifi
import matplotlib.pyplot as plt
import numpy as np
import requests


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "public_large_model_curves"
DOC_PATH = ROOT / "docs" / "public_large_model_curves.md"

REQUEST_TIMEOUT = 90
HEADERS = {"User-Agent": "DL-final-public-curve-fetcher/1.0"}
LOG_PATTERN = re.compile(r"Step:\s*(\d+),\s*LR:\s*([-+0-9eE.]+),\s*Loss:\s*([-+0-9eE.]+)")


@dataclass
class SourceSpec:
    key: str
    display_name: str
    task_type: str
    param_label: str
    source_url: str
    parser_type: str


SOURCES = [
    SourceSpec(
        key="olmoe_sft",
        display_name="OLMoE SFT",
        task_type="SFT",
        param_label="1.3B active / 6.9B total",
        source_url="https://raw.githubusercontent.com/allenai/OLMoE/main/logs/olmoe-sft-logs.txt",
        parser_type="olmoe_log",
    ),
    SourceSpec(
        key="olmoe_dpo",
        display_name="OLMoE DPO",
        task_type="DPO",
        param_label="1.3B active / 6.9B total",
        source_url="https://raw.githubusercontent.com/allenai/OLMoE/main/logs/olmoe-dpo-logs.txt",
        parser_type="olmoe_log",
    ),
    SourceSpec(
        key="mistral_7b_sft_beta",
        display_name="Mistral-7B SFT beta",
        task_type="SFT",
        param_label="7B",
        source_url="https://huggingface.co/HuggingFaceH4/mistral-7b-sft-beta/raw/main/README.md",
        parser_type="hf_model_card",
    ),
    SourceSpec(
        key="zephyr_7b_alpha",
        display_name="Zephyr-7B alpha",
        task_type="DPO",
        param_label="7B",
        source_url="https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha/raw/main/README.md",
        parser_type="hf_model_card",
    ),
    SourceSpec(
        key="zephyr_7b_beta",
        display_name="Zephyr-7B beta",
        task_type="DPO",
        param_label="7B",
        source_url="https://huggingface.co/HuggingFaceH4/zephyr-7b-beta/raw/main/README.md",
        parser_type="hf_model_card",
    ),
]


def fetch_text(url: str) -> str:
    session = requests.Session()
    session.verify = certifi.where()
    response = session.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS)
    response.raise_for_status()
    return response.text


def parse_olmoe_log(text: str) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    steps, lrs, losses = [], [], []
    for match in LOG_PATTERN.finditer(text):
        step = int(match.group(1))
        lr = float(match.group(2))
        loss = float(match.group(3))
        if not math.isfinite(lr) or not math.isfinite(loss):
            continue
        steps.append(step)
        lrs.append(lr)
        losses.append(loss)

    seen = {}
    for step, lr, loss in zip(steps, lrs, losses):
        seen[step] = (lr, loss)
    ordered_steps = np.array(sorted(seen.keys()), dtype=np.int64)
    ordered_lrs = np.array([seen[s][0] for s in ordered_steps], dtype=np.float64)
    ordered_losses = np.array([seen[s][1] for s in ordered_steps], dtype=np.float64)
    meta = {
        "lr_scheduler_type": "logged",
        "warmup": "logged",
        "total_steps_estimate": str(int(ordered_steps[-1])) if len(ordered_steps) else "0",
    }
    return np.column_stack([ordered_steps, ordered_lrs, ordered_losses]), ordered_steps, meta


def extract_bullet_value(text: str, field: str) -> str:
    pattern = re.compile(rf"-\s*{re.escape(field)}:\s*([^\n]+)")
    match = pattern.search(text)
    if not match:
        raise ValueError(f"missing field: {field}")
    return match.group(1).strip()


def parse_markdown_training_table(text: str) -> list[dict[str, float]]:
    lines = text.splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if "| Training Loss |" in line and "| Step |" in line:
            start_idx = idx
            break
    if start_idx is None:
        raise ValueError("training table not found")

    rows = []
    for line in lines[start_idx + 2 :]:
        if not line.strip().startswith("|"):
            break
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) < 4:
            break
        try:
            rows.append(
                {
                    "train_loss": float(parts[0]),
                    "epoch": float(parts[1]),
                    "step": float(parts[2]),
                    "val_loss": float(parts[3]),
                }
            )
        except ValueError:
            continue
    if not rows:
        raise ValueError("no rows parsed")
    return rows


def reconstruct_scheduler(steps: np.ndarray, peak_lr: float, scheduler: str, warmup_ratio: float, total_steps: int) -> np.ndarray:
    warmup_steps = max(1, int(round(total_steps * warmup_ratio))) if warmup_ratio > 0 else 0
    lrs = np.zeros(len(steps), dtype=np.float64)
    for i, step in enumerate(steps):
        if warmup_steps > 0 and step <= warmup_steps:
            lrs[i] = peak_lr * step / warmup_steps
            continue
        if scheduler == "linear":
            if total_steps <= warmup_steps:
                lrs[i] = peak_lr
            else:
                progress = min(max((step - warmup_steps) / (total_steps - warmup_steps), 0.0), 1.0)
                lrs[i] = peak_lr * (1.0 - progress)
        elif scheduler == "cosine":
            if total_steps <= warmup_steps:
                lrs[i] = peak_lr
            else:
                progress = min(max((step - warmup_steps) / (total_steps - warmup_steps), 0.0), 1.0)
                lrs[i] = 0.5 * peak_lr * (1.0 + math.cos(math.pi * progress))
        else:
            lrs[i] = peak_lr
    return lrs


def parse_hf_model_card(text: str) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    learning_rate = float(extract_bullet_value(text, "learning_rate"))
    scheduler_type = extract_bullet_value(text, "lr_scheduler_type")
    warmup_ratio = float(extract_bullet_value(text, "lr_scheduler_warmup_ratio"))
    num_epochs = float(extract_bullet_value(text, "num_epochs"))
    rows = parse_markdown_training_table(text)
    last_row = rows[-1]
    total_steps = max(int(round(last_row["step"] / last_row["epoch"] * num_epochs)), int(last_row["step"]))
    steps = np.array([int(r["step"]) for r in rows], dtype=np.int64)
    losses = np.array([float(r["train_loss"]) for r in rows], dtype=np.float64)
    lrs = reconstruct_scheduler(steps, learning_rate, scheduler_type, warmup_ratio, total_steps)
    meta = {
        "lr_scheduler_type": scheduler_type,
        "warmup_ratio": f"{warmup_ratio}",
        "total_steps_estimate": str(total_steps),
    }
    return np.column_stack([steps, lrs, losses]), steps, meta


def plot_source(spec: SourceSpec, curve: np.ndarray) -> Path:
    steps = curve[:, 0]
    lrs = curve[:, 1]
    losses = curve[:, 2]

    fig, ax1 = plt.subplots(figsize=(9.2, 5.2))
    ax1.plot(steps, losses, color="#E45756", linewidth=2.0, label="Loss")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss", color="#E45756")
    ax1.tick_params(axis="y", labelcolor="#E45756")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(steps, lrs, color="#4C78A8", linewidth=2.0, linestyle="--", label="LR")
    ax2.set_ylabel("Learning Rate", color="#4C78A8")
    ax2.tick_params(axis="y", labelcolor="#4C78A8")

    title = f"{spec.task_type} | {spec.display_name} | {spec.param_label}"
    ax1.set_title(title)
    fig.tight_layout()

    out_path = OUT_DIR / f"{spec.key}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def write_doc(rows: list[dict[str, str]]) -> None:
    lines = [
        "# 公开可下载的大参数模型曲线可视化",
        "",
        "本次只保留当前可以直接结构化下载并重建的公开曲线，统一画成同图双轴：左轴为 loss，右轴为 learning rate。",
        "",
        "| Source | Task | Params | Parsed Points | LR Source | Output |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['display_name']} | {row['task_type']} | {row['param_label']} | {row['points']} | {row['lr_source']} | `{row['output_file']}` |"
        )
    lines += [
        "",
        "## 说明",
        "",
        "- `OLMoE SFT` 与 `OLMoE DPO` 直接来自官方 GitHub 原始日志，`LR` 和 `Loss` 都是日志中逐步记录的真实值。",
        "- `Mistral-7B SFT beta`、`Zephyr-7B alpha`、`Zephyr-7B beta` 来自 Hugging Face 原始 `README.md` 训练结果表；`Loss` 取训练结果表中的 `Training Loss`，`LR` 根据模型卡中公开的 `learning_rate`、`lr_scheduler_type`、`lr_scheduler_warmup_ratio` 和估计总步数重建。",
        "- 这里的 `total_steps_estimate` 是用 `last_step / last_epoch * num_epochs` 从模型卡表格反推得到，因此 Hugging Face 这几条的 `LR` 是公开超参数下的可重建 schedule，不是站点直接提供的逐步学习率历史。",
        "",
    ]
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table_rows = []

    for spec in SOURCES:
        text = fetch_text(spec.source_url)
        if spec.parser_type == "olmoe_log":
            curve, _, meta = parse_olmoe_log(text)
            lr_source = "official log"
        elif spec.parser_type == "hf_model_card":
            curve, _, meta = parse_hf_model_card(text)
            lr_source = f"reconstructed {meta['lr_scheduler_type']}"
        else:
            raise ValueError(f"unknown parser type: {spec.parser_type}")

        csv_path = OUT_DIR / f"{spec.key}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["step", "lr", "loss"])
            writer.writerows(curve.tolist())

        plot_path = plot_source(spec, curve)
        table_rows.append(
            {
                "key": spec.key,
                "display_name": spec.display_name,
                "task_type": spec.task_type,
                "param_label": spec.param_label,
                "points": str(len(curve)),
                "lr_source": lr_source,
                "output_file": plot_path.name,
                "source_url": spec.source_url,
                "meta": meta,
            }
        )

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(table_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_doc(table_rows)
    print(f"Saved public large model curve visualizations to {OUT_DIR}")


if __name__ == "__main__":
    main()
