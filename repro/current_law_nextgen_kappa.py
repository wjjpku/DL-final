#!/usr/bin/env python3
"""Reusable next-generation kappa estimator.

This module centralizes the formula currently supported by the next-generation
audits:

    kappa_safe
      = 1{R_target(lambda) >= 0.01}
        * n/(n+0.5)
        * sqrt(l2_S / full_l2_S)
        * max(0, dot_S / (l2_S + tau^2)).

The implementation uses schedule curves only through their response feature,
observed calibration residuals, and target feature retention. It does not use
schedule-family labels for estimation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import current_law_amplitude_kappa_search as amp  # noqa: E402
import current_law_continuous_kappa_search as base  # noqa: E402
import current_law_eb_kappa_search as eb  # noqa: E402
import current_law_soft_spectral_multicurve_selection_audit as spectral  # noqa: E402
from deep_stime import stime_feature  # noqa: E402


DEFAULT_RHO = 0.5
DEFAULT_TARGET_RETENTION_FLOOR = 0.01
DEFAULT_RULE = "inner_cv_band_mean"


@dataclass(frozen=True)
class NextGenEstimate:
    scale: str
    train_id: str
    train_size: int
    target_curve: str
    selected_lambda: float
    tau: float
    dot_s: float
    l2_s: float
    full_l2_s: float
    pooled_retention: float
    raw_map: float
    kappa_pool: float
    shrink: float
    rho: float
    kappa_transfer: float
    target_retention: float
    target_factor: float
    kappa_safe: float


class NextGenKappaEstimator:
    """Train-only next-generation kappa estimator with target gate."""

    def __init__(
        self,
        *,
        rho: float = DEFAULT_RHO,
        target_retention_floor: float = DEFAULT_TARGET_RETENTION_FLOOR,
        rule: str = DEFAULT_RULE,
    ) -> None:
        self.rho = float(rho)
        self.target_retention_floor = float(target_retention_floor)
        self.rule = rule
        self.feats = base.feature_cache()
        self.tau_rows = spectral.base_tau_rows(self.feats)
        self._stats_cache: dict[tuple[str, str, float], dict[str, float]] = {}
        self._inner_score_cache: dict[tuple[tuple[str, ...], float], dict[str, float]] = {}
        self._target_cache: dict[tuple[str, str, float], dict[str, object]] = {}
        self._train_estimate_cache: dict[tuple[str, tuple[str, ...]], dict[str, float]] = {}
        for scale in base.SCALES:
            for curve, _ in base.CURVES:
                for lam in spectral.LAMBDAS:
                    self._stats_cache[(scale, curve, lam)] = spectral.stats_for(scale, curve, self.feats, lam)

    def _stats_for_train(self, scale: str, curve: str, lam: float) -> dict[str, float]:
        key = (scale, curve, lam)
        if key not in self._stats_cache:
            self._stats_cache[key] = spectral.stats_for(scale, curve, self.feats, lam)
        return self._stats_cache[key]

    def _target_stats(self, scale: str, curve_name: str, lam: float) -> dict[str, object]:
        key = (scale, curve_name, lam)
        if key not in self._target_cache:
            curve = base.load_curve(scale, curve_name)
            phi = stime_feature(curve, base.LAMBDA)
            baseline = base.mpl_predict(base.MPL_PRECOMPUTED_INIT[scale], curve)
            q = spectral.dct_basis(len(curve.step), spectral.MAX_MODE)
            a = spectral.smoother_matrix(q, lam)
            phi_o = spectral.soft_residualize(phi, q, a)
            phi_l2 = float(np.dot(phi, phi))
            retention = 0.0 if phi_l2 <= 1e-18 else float(np.dot(phi_o, phi_o) / phi_l2)
            self._target_cache[key] = {
                "curve": curve,
                "phi": phi,
                "baseline": baseline,
                "base_mae": base.metrics(curve.loss, baseline)["mae"],
                "target_retention": retention,
            }
        return self._target_cache[key]

    def select_lambda(self, train_curves: tuple[str, ...]) -> float:
        selected, _ = spectral.select_lambda(
            train_curves,
            self.rule,
            self._stats_cache,
            self._inner_score_cache,
            self.tau_rows,
            self.feats,
        )
        return float(selected)

    def estimate(self, scale: str, train_curves: tuple[str, ...], target_curve: str) -> NextGenEstimate:
        train_curves = tuple(train_curves)
        train_key = (scale, train_curves)
        if train_key not in self._train_estimate_cache:
            selected_lambda = self.select_lambda(train_curves)
            tau_pool = [row for row in self.tau_rows if row["train_curve"] in train_curves]
            tau = float(eb.estimate_tau(tau_pool, "q75")["tau"])
            train_stats = [self._stats_for_train(scale, curve, selected_lambda) for curve in train_curves]
            pooled = spectral.pooled_kappa(train_stats, tau)
            train_size = len(train_curves)
            shrink = float(train_size / max(train_size + self.rho, 1e-12))
            self._train_estimate_cache[train_key] = {
                "selected_lambda": selected_lambda,
                "tau": tau,
                "train_size": train_size,
                "shrink": shrink,
                "kappa_transfer": float(pooled["kappa"]) * shrink,
                **pooled,
            }
        cached_train = self._train_estimate_cache[train_key]
        selected_lambda = float(cached_train["selected_lambda"])
        tau = float(cached_train["tau"])
        train_size = len(train_curves)
        shrink = float(cached_train["shrink"])
        kappa_transfer = float(cached_train["kappa_transfer"])
        target = self._target_stats(scale, target_curve, selected_lambda)
        target_retention = float(target["target_retention"])
        target_factor = 1.0 if target_retention >= self.target_retention_floor else 0.0
        return NextGenEstimate(
            scale=scale,
            train_id="|".join(train_curves),
            train_size=train_size,
            target_curve=target_curve,
            selected_lambda=selected_lambda,
            tau=tau,
            dot_s=float(cached_train["pooled_dot"]),
            l2_s=float(cached_train["pooled_orth_l2"]),
            full_l2_s=float(cached_train["pooled_full_l2"]),
            pooled_retention=float(cached_train["pooled_retention"]),
            raw_map=float(cached_train["raw_map"]),
            kappa_pool=float(cached_train["kappa"]),
            shrink=shrink,
            rho=self.rho,
            kappa_transfer=kappa_transfer,
            target_retention=target_retention,
            target_factor=target_factor,
            kappa_safe=kappa_transfer * target_factor,
        )

    def score(self, estimate: NextGenEstimate) -> dict[str, object]:
        target = self._target_stats(estimate.scale, estimate.target_curve, estimate.selected_lambda)
        curve = target["curve"]
        pred = target["baseline"] + estimate.kappa_safe * target["phi"]
        corr_mae = base.metrics(curve.loss, pred)["mae"]
        base_mae = float(target["base_mae"])
        return {
            "base_mae": base_mae,
            "corr_mae": corr_mae,
            "delta_pct": 100.0 * (corr_mae / base_mae - 1.0),
            "win": int(corr_mae < base_mae),
        }
