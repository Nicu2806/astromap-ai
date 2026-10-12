"""Unit tests for Transit Timing Variations (TTV) analysis module."""

from __future__ import annotations

import numpy as np

from astromap.pipeline import synthetic_transit_curve
from astromap.ttv import detect_transit_midtimes


def test_ttv_detection_constant_period() -> None:
    # A perfectly periodic transit with realistic cadence
    curve = synthetic_transit_curve(tic_id=1001, points=4000, period_days=3.0, depth=0.015, duration_days=0.15)
    result = detect_transit_midtimes(
        time=curve.time,
        flux=curve.flux,
        period_days=3.0,
        duration_days=0.15,
    )
    assert result.transits_detected >= 5
    # Strict periodic signal: RMS TTV should be well below 15 minutes threshold
    assert result.rms_ttv_minutes < 15.0
    assert result.has_significant_ttv is False
    assert len(result.measurements) == result.transits_detected


def test_ttv_detection_with_induced_timing_variations() -> None:
    # Synthesize light curve with deliberate 35-minute sinusoidal timing shifts
    period = 4.0
    duration = 0.20
    depth = 0.02
    time = np.linspace(0.0, 28.0, 1500)
    flux = np.ones_like(time)

    # Injected 30-minute shifts
    for k in range(1, 7):
        t_center = k * period + (35.0 / 1440.0) * np.sin(k * 1.2)
        in_transit = np.abs(time - t_center) <= (duration / 2.0)
        flux[in_transit] -= depth

    # Add minor noise
    rng = np.random.default_rng(42)
    flux += rng.normal(0.0, 0.0005, len(flux))

    result = detect_transit_midtimes(
        time=time,
        flux=flux,
        period_days=period,
        duration_days=duration,
        t0=0.0,
    )
    assert result.transits_detected >= 4
    # RMS TTV should be detected as significant (> 15 minutes)
    assert result.rms_ttv_minutes > 15.0
    assert result.has_significant_ttv is True


def test_ttv_empty_and_edge_cases() -> None:
    empty_time = np.array([])
    empty_flux = np.array([])
    res = detect_transit_midtimes(empty_time, empty_flux, period_days=2.0, duration_days=0.1)
    assert res.transits_detected == 0
    assert res.has_significant_ttv is False
    assert res.to_dict()["transits_detected"] == 0
