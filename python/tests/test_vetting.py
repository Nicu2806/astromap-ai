"""Unit tests for the automated exoplanet vetting battery."""

import numpy as np

from astromap.pipeline import synthetic_transit_curve
from astromap.vetting import calculate_transit_epoch, perform_astrophysical_vetting


def test_vetting_passes_genuine_planet() -> None:
    # Clean synthetic transit curve
    curve = synthetic_transit_curve(
        tic_id=150428135, points=1024, period_days=3.4, depth=0.012, duration_days=0.16
    )

    report = perform_astrophysical_vetting(
        curve=curve,
        period_days=3.4,
        duration_days=0.16,
        depth=0.012,
        tic_id=150428135,
        stellar_radius_solar=1.0,
        stellar_mass_solar=1.0,
    )

    assert report.odd_even_flag is True
    assert report.secondary_eclipse_flag is True
    assert report.stellar_density_flag is True
    assert report.final_disposition == "PLANET_CANDIDATE"
    assert "Target passes all astrophysical false-positive screening tests." in report.summary_notes


def test_vetting_catches_eclipsing_binary_odd_even() -> None:
    curve = synthetic_transit_curve(
        tic_id=999, points=1024, period_days=4.0, depth=0.02, duration_days=0.20
    )

    # Artificially alter odd transits to have smaller depth (secondary eclipse on odd)
    t0 = calculate_transit_epoch(curve.time, curve.flux, 4.0)
    transit_index = np.round((curve.time - t0) / 4.0).astype(int)
    odd_in_transit = (np.abs(curve.time - (t0 + transit_index * 4.0)) <= 0.10) & (transit_index % 2 != 0)

    flux_modified = np.copy(curve.flux)
    flux_modified[odd_in_transit] += 0.015  # Reduce depth from 0.02 to 0.005 on odd transits

    from astromap.pipeline import LightCurve

    eb_curve = LightCurve(curve.time, flux_modified, curve.flux_error, curve.quality)

    report = perform_astrophysical_vetting(
        curve=eb_curve,
        period_days=4.0,
        duration_days=0.20,
        depth=0.02,
        tic_id=999,
    )

    assert report.odd_even_flag is False
    assert report.final_disposition == "ECLIPSING_BINARY_ODD_EVEN"


def test_vetting_catches_secondary_eclipse() -> None:
    curve = synthetic_transit_curve(
        tic_id=888, points=1024, period_days=5.0, depth=0.02, duration_days=0.20
    )
    t0 = calculate_transit_epoch(curve.time, curve.flux, 5.0)

    # Inject a secondary eclipse at phase 0.5
    phase = ((curve.time - t0) / 5.0) % 1.0
    sec_mask = np.abs(phase - 0.50) <= (0.20 / (2.0 * 5.0))

    flux_modified = np.copy(curve.flux)
    flux_modified[sec_mask] -= 0.008  # Noticeable secondary dip

    from astromap.pipeline import LightCurve

    sec_curve = LightCurve(curve.time, flux_modified, curve.flux_error, curve.quality)

    report = perform_astrophysical_vetting(
        curve=sec_curve,
        period_days=5.0,
        duration_days=0.20,
        depth=0.02,
        tic_id=888,
    )

    assert report.secondary_eclipse_flag is False
    assert report.final_disposition == "ECLIPSING_BINARY_SECONDARY"
