"""Transit Timing Variation (TTV) detection and O-C analysis engine.

In multi-planet resonant systems (e.g. TOI-216, TRAPPIST-1), mutual gravitational
interactions cause transits to deviate from strict linear periodicity (O - C shifts).
This module measures individual transit mid-times and quantifies TTV amplitudes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TransitTimingMeasurement:
    """Individual transit observation and its linear ephemeris deviation."""

    epoch_index: int
    observed_time_days: float
    calculated_time_days: float
    o_minus_c_minutes: float
    measured_depth: float
    snr: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TtvAnalysisResult:
    """Comprehensive O-C transit timing analysis for a light curve."""

    nominal_period_days: float
    epoch_zero_days: float
    transits_detected: int
    mean_o_minus_c_minutes: float
    rms_ttv_minutes: float
    max_ttv_minutes: float
    has_significant_ttv: bool
    measurements: list[TransitTimingMeasurement]

    def to_dict(self) -> dict[str, object]:
        return {
            "nominal_period_days": self.nominal_period_days,
            "epoch_zero_days": self.epoch_zero_days,
            "transits_detected": self.transits_detected,
            "mean_o_minus_c_minutes": self.mean_o_minus_c_minutes,
            "rms_ttv_minutes": self.rms_ttv_minutes,
            "max_ttv_minutes": self.max_ttv_minutes,
            "has_significant_ttv": self.has_significant_ttv,
            "measurements": [m.to_dict() for m in self.measurements],
        }


def detect_transit_midtimes(
    time: np.ndarray,
    flux: np.ndarray,
    period_days: float,
    duration_days: float,
    t0: float | None = None,
    min_snr: float = 2.0,
) -> TtvAnalysisResult:
    """Extract individual transit midtimes and compute Observed - Calculated (O - C) offsets.

    Parameters
    ----------
    time : np.ndarray
        Cadence timestamps in days.
    flux : np.ndarray
        Normalized relative flux.
    period_days : float
        Mean linear orbital period.
    duration_days : float
        Expected transit duration in days.
    t0 : float, optional
        Reference epoch. If None, detected from deepest transit.
    min_snr : float
        Minimum signal-to-noise ratio to include an individual transit.

    Returns
    -------
    TtvAnalysisResult
        Extracted mid-times, deviations in minutes, and statistical significance.
    """
    if len(time) < 16 or period_days <= 0 or duration_days <= 0:
        return TtvAnalysisResult(
            nominal_period_days=period_days,
            epoch_zero_days=t0 or 0.0,
            transits_detected=0,
            mean_o_minus_c_minutes=0.0,
            rms_ttv_minutes=0.0,
            max_ttv_minutes=0.0,
            has_significant_ttv=False,
            measurements=[],
        )

    t_start, t_end = float(time[0]), float(time[-1])

    if t0 is None:
        min_idx = int(np.nanargmin(flux))
        t_deep = float(time[min_idx])
        t0 = float(t_start + ((t_deep - t_start) % period_days))

    # Calculate integer transit numbers spanning the observation window
    k_min = int(np.floor((t_start - t0) / period_days)) - 1
    k_max = int(np.ceil((t_end - t0) / period_days)) + 1
    half_window = max(duration_days * 1.5, 0.05)
    noise_mad = float(np.nanmedian(np.abs(flux - np.nanmedian(flux)))) * 1.4826
    noise_sigma = max(noise_mad, 1e-4)

    raw_measurements: list[tuple[int, float, float, float]] = []

    for k in range(k_min, k_max + 1):
        t_calc = t0 + k * period_days
        if t_calc < t_start or t_calc > t_end:
            continue

        # Window around this specific transit
        window_mask = np.abs(time - t_calc) <= half_window
        if np.count_nonzero(window_mask) < 4:
            continue

        w_time = time[window_mask]
        w_flux = flux[window_mask]

        # Find transit center via half-depth ingress and egress midpoint
        baseline = float(np.nanmedian(w_flux))
        min_flux = float(np.nanmin(w_flux))
        local_depth = max(baseline - min_flux, 1e-4)
        half_flux_thresh = baseline - 0.40 * local_depth

        in_transit_cadences = w_flux < half_flux_thresh
        if not np.any(in_transit_cadences):
            continue

        transit_times = w_time[in_transit_cadences]
        t_observed = float((transit_times[0] + transit_times[-1]) / 2.0)

        depth_obs = float(baseline - np.nanmedian(w_flux[in_transit_cadences]))
        snr = float(depth_obs / noise_sigma)

        if snr >= min_snr:
            raw_measurements.append((k, t_observed, depth_obs, snr))

    if not raw_measurements:
        return TtvAnalysisResult(
            nominal_period_days=period_days,
            epoch_zero_days=t0,
            transits_detected=0,
            mean_o_minus_c_minutes=0.0,
            rms_ttv_minutes=0.0,
            max_ttv_minutes=0.0,
            has_significant_ttv=False,
            measurements=[],
        )

    # Standard astronomical linear ephemeris fit: t_calc = T0_fit + k * P_fit
    k_arr = np.array([m[0] for m in raw_measurements], dtype=np.float64)
    t_arr = np.array([m[1] for m in raw_measurements], dtype=np.float64)

    if len(k_arr) >= 2:
        p_fit, t0_fit = np.polyfit(k_arr, t_arr, deg=1)
    else:
        p_fit, t0_fit = period_days, t0

    measurements: list[TransitTimingMeasurement] = []
    for k_val, t_obs, depth_obs, snr in raw_measurements:
        t_expected = float(t0_fit + k_val * p_fit)
        o_minus_c_min = float((t_obs - t_expected) * 24.0 * 60.0)
        measurements.append(
            TransitTimingMeasurement(
                epoch_index=int(k_val),
                observed_time_days=t_obs,
                calculated_time_days=t_expected,
                o_minus_c_minutes=o_minus_c_min,
                measured_depth=depth_obs,
                snr=snr,
            )
        )

    if not measurements:
        return TtvAnalysisResult(
            nominal_period_days=period_days,
            epoch_zero_days=t0,
            transits_detected=0,
            mean_o_minus_c_minutes=0.0,
            rms_ttv_minutes=0.0,
            max_ttv_minutes=0.0,
            has_significant_ttv=False,
            measurements=[],
        )

    offsets = np.array([m.o_minus_c_minutes for m in measurements])
    mean_ttv = float(np.mean(offsets))
    rms_ttv = float(np.sqrt(np.mean((offsets - mean_ttv) ** 2))) if len(offsets) > 1 else 0.0
    max_ttv = float(np.max(np.abs(offsets)))
    has_significant = rms_ttv >= 15.0 and len(measurements) >= 3

    return TtvAnalysisResult(
        nominal_period_days=period_days,
        epoch_zero_days=t0,
        transits_detected=len(measurements),
        mean_o_minus_c_minutes=mean_ttv,
        rms_ttv_minutes=rms_ttv,
        max_ttv_minutes=max_ttv,
        has_significant_ttv=has_significant,
        measurements=measurements,
    )
