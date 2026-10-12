"""Tests for Parquet storage serialization and AWS S3 helper."""

from pathlib import Path

import numpy as np
import pytest

from astromap.pipeline import preprocess, synthetic_transit_curve
from astromap.storage import (
    load_lightcurve_parquet,
    load_processed_parquet,
    save_lightcurve_parquet,
    save_processed_parquet,
    tess_s3_path_prefix,
)


def test_lightcurve_parquet_roundtrip(tmp_path: Path) -> None:
    raw = synthetic_transit_curve(150428135, points=256)
    target = tmp_path / "lightcurve.parquet"

    meta_in = {"tic_id": "150428135", "sector": "1", "mission": "TESS"}
    saved_path = save_lightcurve_parquet(raw, target, extra_meta=meta_in)
    assert saved_path.is_file()

    loaded, meta_out = load_lightcurve_parquet(saved_path)
    assert len(loaded.time) == len(raw.time)
    np.testing.assert_allclose(loaded.time, raw.time, rtol=1e-7)
    np.testing.assert_allclose(loaded.flux, raw.flux, rtol=1e-7)
    np.testing.assert_allclose(loaded.flux_error, raw.flux_error, rtol=1e-7)
    np.testing.assert_array_equal(loaded.quality, raw.quality)

    assert meta_out["astromap_type"] == "LightCurve"
    assert meta_out["tic_id"] == "150428135"


def test_processed_curve_parquet_roundtrip(tmp_path: Path) -> None:
    raw = synthetic_transit_curve(261136679, points=256)
    processed = preprocess(raw, output_points=128)
    target = tmp_path / "processed.parquet"

    saved_path = save_processed_parquet(
        processed, target, extra_meta={"source": "test_synthetic"}
    )
    assert saved_path.is_file()

    loaded, meta_out = load_processed_parquet(saved_path)
    assert len(loaded.time) == 128
    np.testing.assert_allclose(loaded.time, processed.time, rtol=1e-7)
    np.testing.assert_allclose(loaded.flux, processed.flux, rtol=1e-7)
    np.testing.assert_array_equal(loaded.mask, processed.mask)
    assert loaded.scalogram.shape == processed.scalogram.shape
    np.testing.assert_allclose(loaded.scalogram, processed.scalogram, rtol=1e-7)
    assert meta_out["source"] == "test_synthetic"


def test_load_nonexistent_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_lightcurve_parquet(tmp_path / "does_not_exist.parquet")


def test_tess_s3_path_prefix() -> None:
    prefix = tess_s3_path_prefix(tic_id=150428135, sector=1)
    # 150428135 -> 0000 0001 5042 8135
    assert prefix == "s3://stpubdata/tess/public/tid/s0001/0000/0001/5042/8135/"
