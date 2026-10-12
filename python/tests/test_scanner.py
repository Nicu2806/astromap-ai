"""Unit tests for the batch sky scanner."""

from __future__ import annotations

from pathlib import Path

from astromap.inference import InferenceEngine
from astromap.mast import RealObservation
from astromap.pipeline import synthetic_transit_curve
from astromap.scanner import scan_targets


def fixture_loader(tic_id: int, sector: int | None) -> RealObservation:
    return RealObservation(
        curve=synthetic_transit_curve(tic_id, points=256),
        mission="TESS/SPOC",
        sector=sector or 1,
        cadence_count=256,
        source_uri="mast:TESS/product/test.fits",
    )


def test_batch_scan_and_catalog_generation(tmp_path: Path) -> None:
    engine = InferenceEngine(observation_loader=fixture_loader)
    tic_ids = [150428135, 261136679, 307210830]
    csv_file = tmp_path / "catalog.csv"
    json_file = tmp_path / "summary.json"

    predictions, summary = scan_targets(
        tic_ids=tic_ids,
        engine=engine,
        output_csv=csv_file,
        output_json=json_file,
    )

    assert len(predictions) == 3
    assert summary.total_scanned == 3
    assert csv_file.exists()
    assert json_file.exists()

    csv_text = csv_file.read_text(encoding="utf-8")
    assert "tic_id,label,hybrid_confidence" in csv_text
    assert "150428135" in csv_text
