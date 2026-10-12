"""Production adapter for public TESS light curves from MAST."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from astromap.pipeline import LightCurve

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RealObservation:
    """A downloaded observation with auditable archive provenance."""

    curve: LightCurve
    mission: str
    sector: int
    cadence_count: int
    source_uri: str


@dataclass(frozen=True)
class ObservationProduct:
    """A preferred SPOC light-curve product for one TESS sector."""

    sector: int
    mission: str
    year: int
    exposure_seconds: float
    source_uri: str


@lru_cache(maxsize=128)
def list_tess_observations(tic_id: int) -> tuple[ObservationProduct, ...]:
    """List one preferred SPOC light-curve product per observed sector."""

    if tic_id <= 0:
        raise ValueError("tic_id must be positive")
    try:
        import lightkurve as lk
    except ImportError as error:
        raise RuntimeError("Lightkurve is required to query MAST") from error

    search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC")
    if len(search) == 0:
        raise LookupError(f"No SPOC light curve found for TIC {tic_id}")

    preferred: dict[int, ObservationProduct] = {}
    for row in search.table:
        sector = int(row["sequence_number"])
        exposure = float(row["exptime"])
        product = ObservationProduct(
            sector=sector,
            mission=str(row["mission"]),
            year=int(row["year"]),
            exposure_seconds=exposure,
            source_uri=str(row["dataURI"]),
        )
        current = preferred.get(sector)
        if current is None or abs(exposure - 120.0) < abs(current.exposure_seconds - 120.0):
            preferred[sector] = product
    return tuple(preferred[sector] for sector in sorted(preferred))


def download_tess_lightcurve(tic_id: int, sector: int | None = None) -> RealObservation:
    """Download and combine SPOC light curves for a TIC target.

    If no sector is supplied, the earliest available SPOC observation is used.
    """

    if tic_id <= 0:
        raise ValueError("tic_id must be positive")
    try:
        import lightkurve as lk
        import numpy as np
    except ImportError as error:
        raise RuntimeError("Lightkurve is required to download MAST data") from error

    search = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC", sector=sector)
    if len(search) == 0:
        sector_text = f" in sector {sector}" if sector is not None else ""
        raise LookupError(f"No SPOC light curve found for TIC {tic_id}{sector_text}")

    selected = search[0:1]
    downloaded = selected.download(quality_bitmask="default")
    if downloaded is None:
        raise RuntimeError(f"MAST download failed for TIC {tic_id}")

    normalized = downloaded.normalize()
    quality = getattr(normalized, "quality", np.zeros(len(normalized.time), dtype=np.int64))
    flux_error = getattr(normalized, "flux_err", np.full(len(normalized.time), np.nan))
    curve = LightCurve(
        time=np.asarray(normalized.time.value, dtype=np.float64),
        flux=np.asarray(normalized.flux.value, dtype=np.float64),
        flux_error=np.asarray(flux_error.value, dtype=np.float64),
        quality=np.asarray(quality, dtype=np.int64),
    )
    selected_sector = int(selected.table["sequence_number"][0])
    source_uri = str(selected.table["dataURI"][0])
    return RealObservation(
        curve=curve,
        mission="TESS/SPOC",
        sector=selected_sector,
        cadence_count=len(curve.time),
        source_uri=source_uri,
    )

def load_tess_observation(
    tic_id: int,
    sector: int | None = None,
    cache_dir: str = "data/real_curves",
) -> RealObservation:
    """Load real TESS SPOC observation from local disk cache if available, or fetch from MAST."""
    import numpy as np

    cache_path = Path(cache_dir) / f"tic_{tic_id}_s{sector or 1}.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        time = np.asarray(data["time"], dtype=np.float64)
        flux = np.asarray(data["flux"], dtype=np.float64)
        curve = LightCurve(
            time=time,
            flux=flux,
            flux_error=np.full(len(time), 0.001, dtype=np.float64),
            quality=np.zeros(len(time), dtype=np.int64),
        )
        return RealObservation(
            curve=curve,
            mission="TESS/SPOC",
            sector=sector or 1,
            cadence_count=len(time),
            source_uri=str(cache_path.as_posix()),
        )

    obs = download_tess_lightcurve(tic_id, sector=sector)
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        from astromap.pipeline import preprocess

        processed = preprocess(obs.curve)
        np.savez_compressed(
            cache_path,
            time=processed.time,
            flux=processed.flux,
            mask=processed.mask,
            scalogram=processed.scalogram,
        )
    except (OSError, RuntimeError, ValueError) as err:
        logger.debug("Failed to cache preprocessed curve for TIC %d: %s", tic_id, err)
    return obs

