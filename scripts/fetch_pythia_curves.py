#!/usr/bin/env python3
"""Reconstruct Pythia loss-vs-step curves by evaluating public checkpoints.

Pythia ships no clean per-step training-loss CSV (raw curves live in a messy W&B
project). The robust, reproducible alternative is to evaluate the public
HuggingFace checkpoints on a *fixed* held-out text and record mean cross-entropy.
Shared tokenizer + shared eval text => curves are directly comparable across
scale and step, which is exactly the controlled scale axis the MPL data lacks.

Design choices that make *dense* sampling affordable:
  * --purge: each checkpoint is downloaded into a temp cache, evaluated, then
    deleted. Disk peak stays at ~one checkpoint regardless of density.
  * fp16 .bin weights (use_safetensors=False) cut download ~40% vs safetensors.
  * Eval defaults to a Pile sample (NeelNanda/pile-10k), in-domain for Pythia,
    so curves stay monotone -- unlike wikitext, which is out-of-domain and makes
    the late-training tail spuriously rise.

All Pythia models share data order and LR schedule (cosine + warmup). Global
batch = 1024 seqs * 2048 tokens = 2,097,152 tokens/step; 143000 steps ~= 300B.
"""

from __future__ import annotations

import argparse
import csv as csvmod
import json
import shutil
import tempfile
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "pythia_curves"
TOKENS_PER_STEP = 1024 * 2048

SCALE_PARAMS = {
    "70m": 70e6, "160m": 160e6, "410m": 410e6, "1b": 1.0e9,
    "1.4b": 1.4e9, "2.8b": 2.8e9, "6.9b": 6.9e9, "12b": 12e9,
}

LOG_STEPS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def checkpoint_steps(preset: str) -> list[int]:
    if preset == "full":          # all 153 checkpoints
        return LOG_STEPS + list(range(1000, 143001, 1000))
    if preset == "dense":         # ~71: dense early, coarser tail
        return LOG_STEPS + list(range(1000, 20001, 1000)) + list(range(22000, 143001, 3000))
    if preset == "sparse":        # ~28: cheap scan for big scales
        return LOG_STEPS + list(range(1000, 143001, 8000)) + [143000]
    raise ValueError(preset)


def build_eval_tokens(tokenizer, dataset, config, split, text_key, n_windows, win):
    ds = load_dataset(dataset, config, split=split) if config else load_dataset(dataset, split=split)
    text = "\n\n".join(t for t in ds[text_key] if t and t.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0][: n_windows * win]
    return ids.view(-1, win)


# Flaky-network signatures: hf_transfer raises RuntimeError; SSL EOFs surface as
# httpx ConnectError/SSLError; a torn-down shared client says "client has been closed".
TRANSIENT_NAMES = ("ConnectError", "ReadError", "ReadTimeout", "ConnectionError",
                   "SSLError", "ProtocolError", "RemoteDisconnected",
                   "ChunkedEncodingError", "RuntimeError", "HfHubHTTPError")


def is_transient(e: Exception) -> bool:
    name, msg = type(e).__name__, str(e)
    return (name in TRANSIENT_NAMES or "client has been closed" in msg
            or "SSL" in msg or "EOF" in msg or "Connection" in msg or "timed out" in msg)


@torch.no_grad()
def _eval_once(repo, revision, eval_ids, device, use_safetensors):
    tmp = Path(tempfile.mkdtemp(prefix="pyckpt_"))
    try:
        model = AutoModelForCausalLM.from_pretrained(
            repo, revision=revision, dtype=torch.float32,
            use_safetensors=use_safetensors, cache_dir=str(tmp),
        ).to(device).eval()
        losses = [model(eval_ids[i:i + 1].to(device), labels=eval_ids[i:i + 1].to(device)).loss.item()
                  for i in range(eval_ids.shape[0])]
        del model
        return float(sum(losses) / len(losses))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # purge: bound disk to one ckpt


def eval_checkpoint(repo, revision, eval_ids, device, use_safetensors, retries=4):
    """Eval with in-process retry + backoff; each attempt is a fresh download."""
    last = None
    for attempt in range(retries):
        try:
            return _eval_once(repo, revision, eval_ids, device, use_safetensors)
        except Exception as e:  # noqa: BLE001
            if not is_transient(e):
                raise
            last = e
            wait = 5 * (attempt + 1)
            print(f"    retry {revision} ({attempt + 1}/{retries}) after {type(e).__name__}; "
                  f"sleep {wait}s", flush=True)
            time.sleep(wait)
    raise last


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", nargs="+", default=["70m", "160m"])
    ap.add_argument("--preset", choices=["full", "dense", "sparse"], default="full")
    ap.add_argument("--steps", nargs="+", type=int, help="override preset with explicit steps")
    ap.add_argument("--deduped", action="store_true")
    ap.add_argument("--safetensors", action="store_true", help="use fp32 safetensors (default: fp16 .bin)")
    ap.add_argument("--eval-dataset", default="NeelNanda/pile-10k")
    ap.add_argument("--eval-config", default=None)
    ap.add_argument("--eval-split", default="train")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--n-windows", type=int, default=32)
    ap.add_argument("--win", type=int, default=2048)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(OUT_DIR / "pythia_loss_curves.csv"))
    args = ap.parse_args()

    steps = args.steps or checkpoint_steps(args.preset)
    use_st = bool(args.safetensors)
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    eval_ids = build_eval_tokens(tok, args.eval_dataset, args.eval_config,
                                 args.eval_split, args.text_key, args.n_windows, args.win)
    print(f"eval: {eval_ids.shape[0]} x {eval_ids.shape[1]} = {eval_ids.numel()} tokens "
          f"from {args.eval_dataset} | {len(steps)} checkpoints x {len(args.scales)} scales", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["model_family", "run_id", "scale", "params", "step", "tokens_seen",
            "loss", "perplexity", "schedule_type", "eval_set"]
    suffix = "-deduped" if args.deduped else ""

    # --- resume: collect already-evaluated (scale, step) and append ---
    done: set[tuple[str, int]] = set()
    if out.exists():
        with out.open() as f:
            for r in csvmod.DictReader(f):
                done.add((r["scale"], int(r["step"])))
    new_file = not out.exists()
    f = out.open("a")
    if new_file:
        f.write(",".join(cols) + "\n"); f.flush()

    # in-process retries are exhausted by eval_checkpoint; if a transient error
    # still propagates, exit(2) so the outer loop restarts with a fresh client
    # (resume skips what's done). Non-transient -> skip that point and continue.
    n_new = 0
    for scale in args.scales:
        repo = f"EleutherAI/pythia-{scale}{suffix}"
        for step in steps:
            if (scale, step) in done:
                continue
            try:
                loss = eval_checkpoint(repo, f"step{step}", eval_ids, args.device, use_st)
            except Exception as e:  # noqa: BLE001
                if is_transient(e):
                    print(f"  NETFAIL {scale}@{step}: {type(e).__name__} -> restart", flush=True)
                    f.close()
                    raise SystemExit(2)
                print(f"  SKIP {scale}@{step}: {type(e).__name__}: {e}", flush=True)
                continue
            ppl = float(torch.exp(torch.tensor(loss)))
            r = {"model_family": "pythia", "run_id": f"pythia-{scale}{suffix}", "scale": scale,
                 "params": SCALE_PARAMS.get(scale), "step": step, "tokens_seen": step * TOKENS_PER_STEP,
                 "loss": round(loss, 6), "perplexity": round(ppl, 4),
                 "schedule_type": "cosine", "eval_set": args.eval_dataset}
            f.write(",".join(str(r[c]) for c in cols) + "\n"); f.flush()
            n_new += 1
            print(f"  {scale}@{step}: loss={loss:.4f} ppl={ppl:.2f} "
                  f"tok={r['tokens_seen']/1e9:.2f}B", flush=True)
    f.close()
    print(f"DONE: {n_new} new rows this pass ({len(done)+n_new} total) -> {out}", flush=True)


if __name__ == "__main__":
    main()
