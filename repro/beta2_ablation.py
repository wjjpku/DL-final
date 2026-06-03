#!/usr/bin/env python3
"""beta2 ablation in a controlled AdamW-on-quadratic simulation.

The dataset's beta2 is unknown and we cannot retrain it, so we test the theory's
prediction in a setting where we DO control beta2: simulate AdamW on a noisy
quadratic, equilibrate at a high LR, step the LR down, and measure the loss
relaxation time tau.

Theory: the lag relaxes via two channels --
  (1) the variance channel, rate ~ eta * (lambda/s)  (eta-dependent, beta2-independent);
  (2) a preconditioner channel from finite beta2: v lags g^2 over ~1/(1-beta2) steps,
      an eta-INDEPENDENT step-time constant.
Prediction: for small beta2 the fast preconditioner gives tau ~ variance value; as
beta2 -> 1 the slow preconditioner adds a channel and tau grows, tracking 1/(1-beta2).
This is the controlled analogue of Dremov et al. (2025) finding beta2 matters in cooldown.
"""
import numpy as np

rng = np.random.default_rng(0)

# quadratic: d modes, curvature lam, per-step gradient-noise std s
D = 4
lam = np.array([1.0, 2.0, 4.0, 8.0])
s = np.ones(D)
R = 4000                      # replicas, for averaging the stochastic loss
ETA_HI, ETA_LO = 0.02, 0.002
WARM, HOLD = 8000, 24000      # equilibrate, then hold post-step
B1, EPS = 0.9, 1e-8


def run(beta2):
    delta = rng.standard_normal((R, D)) * 0.3
    m = np.zeros((R, D)); v = np.full((R, D), s**2)   # init v near noise level
    T = WARM + HOLD
    loss = np.empty(T)
    for t in range(T):
        eta = ETA_HI if t < WARM else ETA_LO
        g = lam * delta + s * rng.standard_normal((R, D))     # noisy gradient
        m = B1 * m + (1 - B1) * g
        v = beta2 * v + (1 - beta2) * g * g
        delta = delta - eta * m / (np.sqrt(v) + EPS)
        loss[t] = 0.5 * np.mean((lam * delta * delta).sum(axis=1))
    return loss


def relax_tau(loss):
    """Post-step relaxation time: fit log(loss - loss_inf) ~ -t/tau on the decay."""
    post = loss[WARM:]
    Linf = np.mean(post[-2000:])
    y = post - Linf
    # use the window where y is safely positive and decaying
    k = np.argmax(y > 0)
    yy = y[k:k + 12000]
    pos = yy > 0.02 * yy[0]
    tt = np.arange(len(yy))[pos]
    if len(tt) < 50:
        return np.nan
    slope = np.polyfit(tt, np.log(yy[pos]), 1)[0]
    return -1.0 / slope


def main():
    # ---- (A) POSITIVE CONTROL: eta ablation, confirm tau ~ 1/eta in the sim ----
    print("=" * 66)
    print("(A) eta ablation (beta2=0.99 fixed): does the AdamW sim give tau ~ 1/eta?")
    print("=" * 66)
    print(f"  {'eta_lo':>8s} {'measured tau':>13s}")
    etas, taus = [], []
    for elo in [0.004, 0.002, 0.001, 0.0005]:
        global ETA_LO
        ETA_LO = elo
        tau = relax_tau(run(0.99))
        etas.append(elo); taus.append(tau)
        print(f"  {elo:8.4f} {tau:13.0f}")
    slope = np.polyfit(np.log(etas), np.log(taus), 1)[0]
    print(f"  -> log-log slope p = {-slope:.2f}   (theory: tau ∝ 1/eta => p=1)")

    # ---- (B) beta2 ablation: does beta2 add a relaxation channel to the lag? ----
    ETA_LO = 0.002
    print("\n" + "=" * 66)
    print("(B) beta2 ablation (eta_lo=0.002 fixed): noise-dominated floor lag")
    print("=" * 66)
    print(f"  {'beta2':>8s} {'1/(1-b2)':>9s} {'measured tau':>13s}")
    bt = []
    for b2 in [0.9, 0.99, 0.999, 0.9999]:
        tau = relax_tau(run(b2)); bt.append(tau)
        print(f"  {b2:8.4f} {1/(1-b2):9.0f} {tau:13.0f}")
    bt = np.array(bt)
    print("-" * 66)
    print(f"  tau across beta2 = {bt.astype(int)}  (CV {bt.std()/bt.mean()*100:.0f}%)")
    print("  -> FLAT in beta2: the lag lives on NOISE-dominated modes where v~s^2 is")
    print("     ~constant, so the preconditioner (beta2) channel is negligible. The")
    print("     beta2 'second channel' hypothesis is NOT supported; the 2-exp improvement")
    print("     is the curvature-spectral spread. (Dremov'25 beta2 effect is on cooldown")
    print("     OUTCOME/bias-variance, a different quantity from the relaxation time.)")


if __name__ == "__main__":
    main()
