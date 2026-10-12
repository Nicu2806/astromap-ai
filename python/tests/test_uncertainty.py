"""Unit tests for parameter uncertainty estimation."""

from __future__ import annotations

from astromap.inference import InferenceEngine
from astromap.pipeline import preprocess, synthetic_transit_curve
from astromap.uncertainty import estimate_photometric_uncertainties


def test_uncertainty_estimation_smoke() -> None:
    curve = synthetic_transit_curve(150428135, points=256)
    processed = preprocess(curve, output_points=256)
    engine = InferenceEngine()

    prob_unc, depth_unc, dur_unc, per_unc = estimate_photometric_uncertainties(
        engine=engine,
        processed=processed,
        nominal_period=3.4,
        nominal_depth=0.012,
        nominal_duration=0.16,
        nominal_probability=0.85,
        samples=3,
        seed=42,
    )

    assert 0.0 < prob_unc < 1.0
    assert 0.0 < depth_unc < 0.1
    assert 0.0 < dur_unc < 1.0
    assert 0.0 < per_unc < 1.0
