#!/usr/bin/env python3
"""Lightweight release checks for the GitHub-facing project package.

The verifier checks file presence, expected headline numbers, data coverage,
PDF page counts, Python syntax for the main scripts, and large-file risks.
It does not rerun the full audit; use schedule_response_robustness_audit.py for
that.
"""
from __future__ import annotations

import argparse
import os
import csv
import py_compile
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_GITHUB_FILE_BYTES = 95 * 1024 * 1024


REQUIRED_FILES = [
    "README.md",
    ".gitignore",
    "FINAL_DELIVERABLES.md",
    "REPRODUCIBILITY.md",
    "DATA_MANIFEST.md",
    "RELEASE_CHECKLIST.md",
    "requirements.txt",
    "docs/README.md",
    "repro/README.md",
    "results/README.md",
    "slides/README.md",
    "paper/README.md",
    "slides/main_zh.tex",
    "slides/main_zh.pdf",
    "slides/main.tex",
    "slides/main.pdf",
    "results/tables/cosine_to_wsd_metrics.csv",
    "results/tables/fitted_params.json",
    "results/figures/avg_test_mae.png",
    "results/figures/avg_test_rmse.png",
    "repro/schedule_response_robustness_audit.py",
    "repro/reproduce_cosine_to_wsd.py",
    "repro/interpretable_error_model.py",
    "repro/interpretable_nuisance_origin_audit.py",
    "repro/interpretable_observation_bracket_audit.py",
    "results/schedule_response_robustness/REPORT.md",
    "results/schedule_response_robustness/LEAKAGE_AUDIT.md",
    "results/schedule_response_robustness/lambda_sensitivity_summary.csv",
    "results/schedule_response_robustness/kernel_ablation_summary.csv",
    "results/schedule_response_robustness/cross_scale_summary.csv",
    "results/schedule_response_robustness/projection_ablation_summary.csv",
    "results/schedule_response_robustness/window_rule.csv",
    "results/schedule_response_robustness/wsdcon_failure_slice.csv",
    "slides/figs/fig_mpl_residual_anomaly_100M.png",
    "slides/figs/fig_projection_decomposition_cosine_100M.png",
    "slides/figs/fig_projection_ablation_time_errors_100M.png",
    "slides/figs/fig_schedule_response_mae_heatmap.png",
    "slides/figs/fig_schedule_response_time_errors_100M.png",
    "slides/figs/fig_kappa_clean_scatter.png",
]


DATA_FILES = [
    "constant_24000.csv",
    "constant_72000.csv",
    "cosine_24000.csv",
    "cosine_72000.csv",
    "wsd_20000_24000.csv",
    "wsdld_20000_24000.csv",
    "wsdcon_3.csv",
    "wsdcon_9.csv",
    "wsdcon_18.csv",
]


MAIN_SCRIPTS = [
    "repro/schedule_response_robustness_audit.py",
    "repro/reproduce_cosine_to_wsd.py",
    "repro/interpretable_error_model.py",
    "repro/interpretable_nuisance_origin_audit.py",
    "repro/interpretable_observation_bracket_audit.py",
]


FORBIDDEN_MAIN_TEXT = [
    "DropRelax",
    "KDrop",
    "顶会",
    "生病",
    "AI生成",
    "ai生成",
    "新版",
    "旧版",
    "MPL+old",
]


MAIN_TEXT_FILES = [
    "README.md",
    "FINAL_DELIVERABLES.md",
    "REPRODUCIBILITY.md",
    "DATA_MANIFEST.md",
    "docs/README.md",
    "repro/README.md",
    "results/README.md",
    "slides/README.md",
    "paper/README.md",
    "slides/main.tex",
    "slides/main_zh.tex",
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)


def fail(errors: list[str], message: str) -> None:
    errors.append(message)
    print(f"[FAIL] {message}")


def warn(warnings: list[str], message: str) -> None:
    warnings.append(message)
    print(f"[WARN] {message}")


def ok(message: str) -> None:
    print(f"[ OK ] {message}")


def check_required_files(errors: list[str]) -> None:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    if missing:
        fail(errors, "missing required files: " + ", ".join(missing))
    else:
        ok(f"required file set present ({len(REQUIRED_FILES)} files)")


def check_data_files(errors: list[str]) -> None:
    missing: list[str] = []
    for scale in ("25", "100", "400"):
        for name in DATA_FILES:
            path = ROOT / "external" / "MultiPowerLaw" / "loss_curve_repo" / f"csv_{scale}" / name
            if not path.is_file():
                missing.append(rel(path))
    if missing:
        fail(errors, "missing public-curve data files: " + ", ".join(missing))
    else:
        ok("public-curve data complete for 25M/100M/400M")


def check_report_numbers(errors: list[str]) -> None:
    report = (ROOT / "results" / "schedule_response_robustness" / "REPORT.md").read_text(
        encoding="utf-8"
    )
    required = ["-30.88%", "-4.67%", "15/15", "+0.910", "+625.92%", "0/15"]
    missing = [item for item in required if item not in report]
    if missing:
        fail(errors, "main REPORT.md is missing headline values: " + ", ".join(missing))
    else:
        ok("headline values present in main audit report")


def check_baseline_reproduction_numbers(errors: list[str]) -> None:
    path = ROOT / "results" / "tables" / "cosine_to_wsd_metrics.csv"
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except OSError as exc:
        fail(errors, f"cannot read baseline metrics CSV: {exc}")
        return

    counts: dict[tuple[str, str], int] = {}
    sums: dict[tuple[str, str, str], float] = {}
    for row in rows:
        split = row.get("split", "")
        model = row.get("model", "")
        counts[(split, model)] = counts.get((split, model), 0) + 1
        for metric in ("mae", "rmse", "r2"):
            try:
                value = float(row[metric])
            except (KeyError, TypeError, ValueError):
                fail(errors, f"bad baseline metric row in {rel(path)}")
                return
            key = (split, model, metric)
            sums[key] = sums.get(key, 0.0) + value

    expected_counts = {
        ("train", "tissue"): 6,
        ("train", "mpl"): 6,
        ("test", "tissue"): 15,
        ("test", "mpl"): 15,
    }
    bad_counts = [
        f"{split}/{model}={counts.get((split, model), 0)} expected {expected}"
        for (split, model), expected in expected_counts.items()
        if counts.get((split, model), 0) != expected
    ]
    if bad_counts:
        fail(errors, "baseline reproduction row counts mismatch: " + ", ".join(bad_counts))
        return

    expected_means = {
        ("train", "tissue", "mae"): 0.002242,
        ("train", "mpl", "mae"): 0.003517,
        ("test", "tissue", "mae"): 0.007493,
        ("test", "tissue", "rmse"): 0.010415,
        ("test", "tissue", "r2"): 0.995223,
        ("test", "mpl", "mae"): 0.006292,
        ("test", "mpl", "rmse"): 0.009848,
        ("test", "mpl", "r2"): 0.995760,
    }
    mismatches: list[str] = []
    for key, expected in expected_means.items():
        split, model, metric = key
        mean = sums[key] / counts[(split, model)]
        if abs(mean - expected) > 5e-6:
            mismatches.append(f"{split}/{model}/{metric}={mean:.6f} expected {expected:.6f}")
    if mismatches:
        fail(errors, "baseline reproduction headline values mismatch: " + ", ".join(mismatches))
    else:
        ok("baseline Tissue/Momentum and MPL reproduction metrics match expected values")


def check_pdf_pages(errors: list[str], warnings: list[str]) -> None:
    for path, expected in [("slides/main_zh.pdf", "36"), ("slides/main.pdf", "38")]:
        proc = run(["pdfinfo", path])
        if proc.returncode != 0:
            warn(warnings, f"pdfinfo failed for {path}: {proc.stderr.strip()}")
            continue
        page_line = next((line for line in proc.stdout.splitlines() if line.startswith("Pages:")), "")
        pages = page_line.split()[-1] if page_line else ""
        if pages != expected:
            fail(errors, f"{path} has {pages or 'unknown'} pages, expected {expected}")
        else:
            ok(f"{path} page count is {expected}")


def check_python_compile(errors: list[str]) -> None:
    bad: list[str] = []
    for script in MAIN_SCRIPTS:
        try:
            py_compile.compile(str(ROOT / script), doraise=True)
        except py_compile.PyCompileError as exc:
            bad.append(f"{script}: {exc.msg}")
    if bad:
        fail(errors, "main script syntax errors: " + " | ".join(bad))
    else:
        ok("main Python scripts compile")


def check_forbidden_text(errors: list[str]) -> None:
    hits: list[str] = []
    for name in MAIN_TEXT_FILES:
        text = (ROOT / name).read_text(encoding="utf-8", errors="replace")
        for token in FORBIDDEN_MAIN_TEXT:
            if token in text:
                hits.append(f"{name}: {token}")
    if hits:
        fail(errors, "forbidden main-text tokens found: " + ", ".join(hits))
    else:
        ok("main text contains no forbidden old-story / private tokens")


def tracked_files() -> set[str]:
    proc = run(["git", "ls-files", "-z"])
    if proc.returncode != 0:
        return set()
    return {item for item in proc.stdout.split("\0") if item}


def check_large_files(errors: list[str], warnings: list[str]) -> None:
    tracked = tracked_files()
    large_tracked: list[str] = []
    large_untracked_unignored: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size <= MAX_GITHUB_FILE_BYTES:
            continue
        r = rel(path)
        if r in tracked:
            large_tracked.append(f"{r} ({size / 1024 / 1024:.1f} MB)")
        else:
            ignored = run(["git", "check-ignore", "-q", r]).returncode == 0
            if not ignored:
                large_untracked_unignored.append(f"{r} ({size / 1024 / 1024:.1f} MB)")
    if large_tracked:
        fail(errors, "tracked files exceed GitHub-safe size: " + ", ".join(large_tracked))
    else:
        ok("no tracked files exceed GitHub-safe size")
    if large_untracked_unignored:
        fail(
            errors,
            "large untracked files are not ignored: " + ", ".join(large_untracked_unignored),
        )
    else:
        ok("large untracked files are ignored or absent")
    ignored_raw = run(["git", "check-ignore", "represent/data/wiki_train.u8"])
    if ignored_raw.returncode != 0:
        warn(warnings, "represent/data/wiki_train.u8 is not currently ignored")
    else:
        ok("represent/data raw training bytes are ignored")


def check_required_files_not_ignored(errors: list[str]) -> None:
    ignored: list[str] = []
    for name in REQUIRED_FILES:
        if run(["git", "check-ignore", "-q", name]).returncode == 0:
            ignored.append(name)
    if ignored:
        fail(errors, "required files are ignored by git: " + ", ".join(ignored))
    else:
        ok("required files are not ignored")


def check_required_files_known_to_git(
    errors: list[str], warnings: list[str], *, strict: bool
) -> None:
    missing: list[str] = []
    for name in REQUIRED_FILES:
        proc = run(["git", "ls-files", "--error-unmatch", "--", name])
        if proc.returncode != 0:
            missing.append(name)
    if missing:
        message = "required files are not in the git index: " + ", ".join(missing)
        if strict:
            fail(errors, message)
        else:
            warn(warnings, message + " (run with --require-index before push)")
    else:
        ok("required files are present in the git index")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the GitHub-facing release package.")
    parser.add_argument(
        "--require-index",
        action="store_true",
        help="Fail if any required release file is not already tracked/staged in the git index.",
    )
    parser.add_argument(
        "--print-files",
        action="store_true",
        help="Print the required release file list and exit.",
    )
    parser.add_argument(
        "--print-git-add",
        action="store_true",
        help="Print a git-add command for the required release file list and exit.",
    )
    args = parser.parse_args(argv)

    if args.print_files:
        print("\n".join(REQUIRED_FILES))
        return 0
    if args.print_git_add:
        quoted = " ".join(shlex.quote(path) for path in REQUIRED_FILES)
        print(f"git add -- {quoted}")
        return 0

    errors: list[str] = []
    warnings: list[str] = []
    os.chdir(ROOT)

    check_required_files(errors)
    check_required_files_not_ignored(errors)
    check_required_files_known_to_git(errors, warnings, strict=args.require_index)
    check_data_files(errors)
    check_report_numbers(errors)
    check_baseline_reproduction_numbers(errors)
    check_pdf_pages(errors, warnings)
    check_python_compile(errors)
    check_forbidden_text(errors)
    check_large_files(errors, warnings)

    print()
    print(f"release verification: {len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
