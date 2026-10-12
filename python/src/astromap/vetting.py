"""Automated astrophysical vetting battery for exoplanet candidates.

Implements the standard diagnostic tests required by top astronomy journals
and NASA Robovetter to distinguish genuine exoplanets from astrophysical false positives:
1. Odd-Even Transit Depth Consistency Test (Eclipsing Binary discrimination)
2. Secondary Eclipse / Occultation Search at Phase 0.5
3. Stellar Density Consistency Test (Kepler-3 vs. Gaia / TIC physical parameters)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from astromap.pipeline import LightCurve


@dataclass(frozen=True)
class VettingReport:
    """Diagnostic outcome of the automated vetting battery."""

    tic_id: int
    odd_depth: float
    even_depth: float
    odd_even_sigma_diff: float
    odd_even_flag: bool  # True if passes (depths match within 3 sigma)
    secondary_depth: float
    secondary_snr: float
    secondary_eclipse_flag: bool  # True if passes (no significant dip at phase 0.5)
    stellar_density_transit_gcc: float
    stellar_density_star_gcc: float
    stellar_density_ratio: float
    stellar_density_flag: bool  # True if geometrically plausible
    final_disposition: str  # PLANET_CANDIDATE, ECLIPSING_BINARY_ODD_EVEN, etc.
    summary_notes: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def calculate_transit_epoch(time: np.ndarray, flux: np.ndarray, period_days: float) -> float:
    """Find the central epoch t0 of the deepest transit dip."""
    min_idx = int(np.nanargmin(flux))
    t_min = float(time[min_idx])
    # Fold relative to first timestamp
    t0 = float(time[0] + (t_min - time[0]) % period_days)
    return t0


def perform_astrophysical_vetting(
    curve: LightCurve,
    period_days: float,
    duration_days: float,
    depth: float,
    tic_id: int = 0,
    stellar_radius_solar: float = 1.0,
    stellar_mass_solar: float = 1.0,
    epoch: float | None = None,
) -> VettingReport:
    """Execute complete battery of false-positive vetting tests on a light curve."""
    time = curve.time
    flux = curve.flux

    if epoch is None:
        t0 = calculate_transit_epoch(time, flux, period_days)
    else:
        t0 = epoch

    notes: list[str] = []

    # 1. Odd-Even Transit Depth Test
    # Group transit cadences by transit number k = round((t - t0) / P)
    transit_index = np.round((time - t0) / period_days).astype(int)
    time_from_transit = time - (t0 + transit_index * period_days)
    in_transit = np.abs(time_from_transit) <= (duration_days / 2.0)

    even_mask = in_transit & (transit_index % 2 == 0)
    odd_mask = in_transit & (transit_index % 2 != 0)
    out_mask = ~in_transit

    baseline_flux = float(np.nanmedian(flux[out_mask])) if np.any(out_mask) else 1.0
    baseline_err = float(np.nanstd(flux[out_mask])) if np.any(out_mask) else 0.001

    even_depth = (baseline_flux - float(np.nanmedian(flux[even_mask]))) if np.any(even_mask) else depth
    odd_depth = (baseline_flux - float(np.nanmedian(flux[odd_mask]))) if np.any(odd_mask) else depth

    n_even = max(1, int(np.count_nonzero(even_mask)))
    n_odd = max(1, int(np.count_nonzero(odd_mask)))
    combined_err = baseline_err * np.sqrt(1.0 / n_even + 1.0 / n_odd)

    odd_even_sigma = float(abs(even_depth - odd_depth) / max(combined_err, 1e-6))
    odd_even_flag = odd_even_sigma < 3.5  # < 3.5 sigma difference passes

    if not odd_even_flag:
        notes.append(f"Odd-even depth disparity ({odd_even_sigma:.1f} sigma) indicates eclipsing binary.")

    # 2. Secondary Eclipse Search at Anti-Phase (phi ~ 0.5)
    phase = ((time - t0) / period_days) % 1.0
    secondary_mask = np.abs(phase - 0.50) <= (duration_days / (2.0 * period_days))

    if np.any(secondary_mask):
        sec_flux = float(np.nanmedian(flux[secondary_mask]))
        sec_depth = max(0.0, baseline_flux - sec_flux)
        sec_n = max(1, int(np.count_nonzero(secondary_mask)))
        sec_err = baseline_err / np.sqrt(sec_n)
        sec_snr = float(sec_depth / max(sec_err, 1e-6))
    else:
        sec_depth = 0.0
        sec_snr = 0.0

    # Fails if secondary eclipse is detected at > 4.0 sigma and depth is significant
    secondary_flag = not (sec_snr > 4.0 and sec_depth > (depth * 0.15))
    if not secondary_flag:
        notes.append(f"Secondary eclipse detected at phase 0.5 with SNR {sec_snr:.1f} (self-luminous companion).")

    # 3. Stellar Density Consistency Test (Kepler's Third Law)
    # rho_* [g/cm^3] for circular transit with b=0:
    # a/R_* ~ (P / (pi * T_dur)) * sqrt((1 + sqrt(depth))^2)
    radius_ratio = np.sqrt(max(depth, 1e-6))
    a_over_r = (period_days / (np.pi * max(duration_days, 1e-4))) * (1.0 + radius_ratio)
    # rho_* = (3 * pi / (G * P^2)) * (a/R_*)^3
    # In solar units, rho_solar = 1.408 g/cm^3
    # rho_* / rho_solar = (a/R_*)^3 / (215.0)^3 * (365.25 / P)^2 ... or standard constant:
    # rho_* [g/cm^3] approx 0.0134 * (a/R_*)^3 / (P_days^2)
    rho_transit_gcc = float(0.0134 * (a_over_r**3) / (period_days**2))

    # Density from catalog parameters: M / R^3 * 1.408 g/cm^3
    rho_star_gcc = float((stellar_mass_solar / (stellar_radius_solar**3)) * 1.408)

    density_ratio = rho_transit_gcc / max(rho_star_gcc, 1e-6)
    density_flag = 0.02 <= density_ratio <= 35.0  # Conservative physical bounds for eccentricities <= 0.4
    if not density_flag:
        notes.append(
            f"Stellar density mismatch (ratio {density_ratio:.2f}) indicates unphysical duration or background blending."
        )

    # Final Disposition Logic
    if not odd_even_flag:
        disposition = "ECLIPSING_BINARY_ODD_EVEN"
    elif not secondary_flag:
        disposition = "ECLIPSING_BINARY_SECONDARY"
    elif not density_flag:
        disposition = "UNPHYSICAL_STELLAR_DENSITY"
    else:
        disposition = "PLANET_CANDIDATE"
        notes.append("Target passes all astrophysical false-positive screening tests.")

    return VettingReport(
        tic_id=tic_id,
        odd_depth=float(odd_depth),
        even_depth=float(even_depth),
        odd_even_sigma_diff=odd_even_sigma,
        odd_even_flag=odd_even_flag,
        secondary_depth=float(sec_depth),
        secondary_snr=sec_snr,
        secondary_eclipse_flag=secondary_flag,
        stellar_density_transit_gcc=rho_transit_gcc,
        stellar_density_star_gcc=rho_star_gcc,
        stellar_density_ratio=float(density_ratio),
        stellar_density_flag=density_flag,
        final_disposition=disposition,
        summary_notes=notes,
    )
