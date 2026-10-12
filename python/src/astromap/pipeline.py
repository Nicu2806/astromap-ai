"""Deterministic light-curve preprocessing and time-frequency transforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import savgol_filter

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LightCurve:
    """A validated photometric time series."""

    time: FloatArray
    flux: FloatArray
    flux_error: FloatArray
    quality: NDArray[np.int64]

    def __post_init__(self) -> None:
        sizes = {len(self.time), len(self.flux), len(self.flux_error), len(self.quality)}
        if len(sizes) != 1 or not sizes.pop():
            raise ValueError("Light-curve arrays must be non-empty and equally sized")


@dataclass(frozen=True)
class ProcessedCurve:
    """Normalized model input plus its observation mask."""

    time: FloatArray
    flux: FloatArray
    mask: NDArray[np.bool_]
    scalogram: FloatArray


def synthetic_transit_curve(
    tic_id: int,
    points: int = 1024,
    period_days: float = 3.4,
    depth: float = 0.012,
    duration_days: float = 0.16,
) -> LightCurve:
    """Create a deterministic noisy transit curve for demos and smoke tests."""

    if points < 64 or period_days <= 0 or not 0 < depth < 1 or duration_days <= 0:
        raise ValueError("Invalid synthetic light-curve parameters")
    rng = np.random.default_rng(tic_id)
    time = np.linspace(0.0, 27.0, points, dtype=np.float64)
    phase = np.mod(time + period_days / 2, period_days) - period_days / 2
    transit = np.abs(phase) <= duration_days / 2
    variability = 0.0015 * np.sin(2 * np.pi * time / 8.3)
    flux = 1.0 + variability + rng.normal(0.0, 0.0012, points)
    flux[transit] -= depth
    flux_error = np.full(points, 0.0012, dtype=np.float64)
    quality = np.zeros(points, dtype=np.int64)
    return LightCurve(time=time, flux=flux, flux_error=flux_error, quality=quality)


def preprocess(curve: LightCurve, output_points: int = 512) -> ProcessedCurve:
    """Clean, detrend, normalize and resample a light curve."""

    valid = (
        np.isfinite(curve.time)
        & np.isfinite(curve.flux)
        & np.isfinite(curve.flux_error)
        & (curve.quality == 0)
    )
    if np.count_nonzero(valid) < 32:
        raise ValueError("Too few valid cadences after quality masking")

    time = curve.time[valid]
    flux = curve.flux[valid]
    order = np.argsort(time)
    time, flux = time[order], flux[order]
    unique = np.concatenate(([True], np.diff(time) > 0))
    time, flux = time[unique], flux[unique]

    window = min(101, len(flux) - (1 - len(flux) % 2))
    window = max(11, window if window % 2 else window - 1)
    trend = savgol_filter(flux, window_length=window, polyorder=2, mode="interp")
    flattened = flux / np.clip(trend, 1e-8, None)
    flattened /= np.median(flattened)

    target_time = np.linspace(time[0], time[-1], output_points, dtype=np.float64)
    target_flux = np.interp(target_time, time, flattened)
    cadence = np.median(np.diff(time))
    nearest = np.minimum(
        np.searchsorted(time, target_time, side="left"), len(time) - 1
    )
    mask = np.abs(time[nearest] - target_time) <= 2.5 * cadence
    target_flux = np.where(mask, target_flux, 1.0)
    scalogram = ricker_scalogram(target_flux - 1.0)
    return ProcessedCurve(target_time, target_flux, mask, scalogram)


def ricker_scalogram(signal: FloatArray, scales: int = 32) -> FloatArray:
    """Compute a compact CWT-like Ricker scalogram without hidden global state."""

    widths = np.geomspace(1.5, 48.0, scales)
    output = np.empty((scales, len(signal)), dtype=np.float64)
    for index, width in enumerate(widths):
        max_radius = max(2, (len(signal) - 1) // 2)
        radius = min(max_radius, max(4, int(5 * width)))
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        ratio = x / width
        wavelet = (1.0 - ratio**2) * np.exp(-(ratio**2) / 2.0)
        wavelet -= wavelet.mean()
        norm = np.linalg.norm(wavelet)
        conv = np.convolve(signal, wavelet / max(norm, 1e-12), mode="same")
        if len(conv) > len(signal):
            start = (len(conv) - len(signal)) // 2
            conv = conv[start : start + len(signal)]
        output[index] = conv
    scale = np.std(output)
    return output / max(scale, 1e-8)
