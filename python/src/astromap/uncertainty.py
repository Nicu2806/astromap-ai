"""Photometric and model parameter uncertainty quantification.

Computes 1-sigma empirical parameter uncertainties (Period, Depth, Duration,
and Hybrid Confidence) via residual noise perturbation and multi-pass
stochastic inference grounded in the host star's intrinsic photometric scatter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from astromap.inference import InferenceEngine
    from astromap.pipeline import ProcessedCurve


def estimate_photometric_uncertainties(
    engine: InferenceEngine,
    processed: ProcessedCurve,
    nominal_period: float,
    nominal_depth: float,
    nominal_duration: float,
    nominal_probability: float,
    samples: int = 8,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Estimate 1-sigma parameter uncertainties from photometric scatter.

    Parameters
    ----------
    engine : InferenceEngine
        Inference engine instance.
    processed : ProcessedCurve
        Detrended and normalized light curve with scalogram.
    nominal_period, nominal_depth, nominal_duration, nominal_probability : float
        Nominal estimates from initial consensus pass.
    samples : int
        Number of perturbation passes.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[float, float, float, float]
        (prob_unc, depth_unc, duration_unc, period_unc)
    """
    flux = processed.flux[processed.mask]
    if len(flux) < 8 or samples <= 1:
        # Fallback conservative uncertainties based on point-to-point scatter
        scatter = float(np.median(np.abs(np.diff(flux))) / 0.9539) if len(flux) >= 4 else 1e-3
        return (
            max(0.01, nominal_probability * 0.05),
            max(scatter * 0.1, nominal_depth * 0.05),
            max(0.005, nominal_duration * 0.05),
            max(0.001, nominal_period * 0.01),
        )

    # Robust MAD scatter
    diffs = np.abs(np.diff(flux))
    point_noise = float(np.median(diffs) / 0.9539)
    rng = np.random.default_rng(seed)

    prob_samples: list[float] = [nominal_probability]
    depth_samples: list[float] = [nominal_depth]
    duration_samples: list[float] = [nominal_duration]
    period_samples: list[float] = [nominal_period]

    for _ in range(samples - 1):
        jitter = rng.normal(0.0, point_noise * 0.4, size=len(processed.flux))
        perturbed_flux = processed.flux + jitter

        # Quick evaluation of model or BLS
        if engine.trained:
            # Multi-scale model inference on slightly perturbed input
            from astromap.pipeline import ProcessedCurve, ricker_scalogram

            p_scalogram = ricker_scalogram(perturbed_flux - 1.0)
            p_curve = ProcessedCurve(processed.time, perturbed_flux, processed.mask, p_scalogram)
            prob, (d, dur, p) = engine._model_prediction(p_curve)
        else:
            from astromap.pipeline import ProcessedCurve

            p_curve = ProcessedCurve(processed.time, perturbed_flux, processed.mask, processed.scalogram)
            prob, (d, dur, p) = engine._bls_estimate(p_curve)

        prob_samples.append(prob)
        depth_samples.append(d)
        duration_samples.append(dur)
        period_samples.append(p)

    prob_unc = float(np.std(prob_samples))
    depth_unc = float(np.std(depth_samples))
    duration_unc = float(np.std(duration_samples))
    period_unc = float(np.std(period_samples))

    # Guarantee minimal physical floor
    prob_unc = float(np.clip(prob_unc, 0.005, 0.25))
    depth_unc = float(np.clip(depth_unc, point_noise * 0.01, nominal_depth * 0.50 if nominal_depth > 0 else 0.05))
    duration_unc = float(np.clip(duration_unc, 0.001, 0.50))
    period_unc = float(np.clip(period_unc, 0.0001, 0.50))

    return prob_unc, depth_unc, duration_unc, period_unc
