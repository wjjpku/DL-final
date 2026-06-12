"""Smoke test for train_spectrum.py internals (run locally before launch).

1. lanczos_topk vs numpy on a known dense symmetric operator.
2. Full probe_spectrum on a tiny GPT (sdpa MATH double-backward, precond
   plumbing, optimizer-state bias correction).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train as T
import train_spectrum as S


def test_lanczos():
    n = 400
    rng = np.random.default_rng(0)
    A = rng.standard_normal((n, n))
    # separated top edge (outliers) over a dense bulk -- the Hessian regime
    spikes = np.zeros(n)
    spikes[:5] = [120, 95, 80, 70, 62]
    A = (A + A.T) / 2 + np.diag(np.linspace(0, 30, n) + spikes)
    ref = np.sort(np.linalg.eigvalsh(A))[::-1][:S.K_TOP]
    At = torch.tensor(A, dtype=torch.float32, device=T.DEV)
    like = [torch.zeros(n, device=T.DEV)]
    op = lambda v: [At @ v[0]]
    top, res = S.lanczos_topk(op, like, 64, 7)
    err = np.abs(top[:4] - ref[:4]) / np.abs(ref[:4])
    print("lanczos top-4 rel err:", np.round(err, 8), "res:", res)
    assert err.max() < 1e-4, "Lanczos disagrees with numpy on separated top"
    print("[PASS] lanczos vs numpy")


def test_probe():
    torch.manual_seed(0)
    model = T.GPT(vocab=T.VOCAB, d=64, nh=2, nl=2, block=32).to(T.DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95),
                            weight_decay=0.1, eps=1e-8)
    data = np.frombuffer(rngbytes(200000), dtype=np.uint8)
    gen = torch.Generator().manual_seed(1)
    from train_floor2 import step_once
    for _ in range(20):
        step_once(model, opt, gen, data, 1e-3, 8, 32)
    params = [p for p in model.parameters() if p.requires_grad]
    pb = torch.Generator().manual_seed(S.PROBE_SEED)
    batches = [T.get_batch(data, 32, 8, pb) for _ in range(2)]
    S.M_PRE, S.M_RAW = 24, 16
    ploss, lp, resp, lr_, resr = S.probe_spectrum(model, opt, params, batches)
    print(f"probe_loss={ploss:.3f} S_pre={lp[0]:.2f} lam_raw={lr_[0]:.2f} "
          f"res_p={resp:.3f} res_r={resr:.3f}")
    assert np.all(np.diff(lp) <= 1e-6) and lp[0] > 0, "spectrum not sorted/pos"
    assert resp < 0.5, "Ritz residual too large on tiny model"
    print("[PASS] probe_spectrum end-to-end")


def rngbytes(n):
    return np.random.default_rng(3).integers(0, 256, n, dtype=np.uint8) \
        .tobytes()


if __name__ == "__main__":
    test_lanczos()
    test_probe()
    print("SMOKE OK")
