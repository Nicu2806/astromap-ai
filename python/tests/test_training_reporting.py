from pathlib import Path

from astromap.inference import InferenceEngine
from astromap.mast import RealObservation
from astromap.pipeline import synthetic_transit_curve
from astromap.reporting import write_report
from astromap.training import build_batch


def fixture_observation(tic_id: int, sector: int | None) -> RealObservation:
    return RealObservation(
        curve=synthetic_transit_curve(tic_id),
        mission="TESS/SPOC",
        sector=sector or 1,
        cadence_count=1024,
        source_uri="mast:TESS/product/test-lc.fits",
    )


def test_training_batch_contract() -> None:
    temporal, scalogram, labels, parameters = build_batch(42, size=2)
    assert temporal.shape == (2, 2, 512)
    assert scalogram.shape == (2, 1, 32, 512)
    assert labels.tolist() == [1.0, 0.0]
    assert parameters.shape == (2, 3)


def test_report_is_printable(tmp_path: Path) -> None:
    prediction = InferenceEngine(observation_loader=fixture_observation).predict(150428135)
    output = write_report(prediction, tmp_path / "candidate.html")
    text = output.read_text(encoding="utf-8")
    assert "TIC 150428135" in text
    assert "does not confirm an exoplanet" in text


def test_scientific_training_and_model_card(tmp_path: Path) -> None:
    from astromap.training import train_manifest

    manifest_path = tmp_path / "manifest.json"
    checkpoint_path = tmp_path / "model.pt"
    card_path = tmp_path / "card.json"
    card = train_manifest(
        manifest_path=manifest_path,
        output_checkpoint=checkpoint_path,
        model_card_path=card_path,
        epochs=1,
        batch_size=8,
        seed=42,
    )
    assert checkpoint_path.is_file()
    assert card_path.is_file()
    assert card["model_name"] == "PINN-TransNet-v1"
    assert "scientific_summary" in card["evaluation_on_locked_test_set"]