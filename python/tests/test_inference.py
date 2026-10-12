from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import astromap.api as api_module
from astromap.inference import InferenceEngine
from astromap.mast import ObservationProduct, RealObservation
from astromap.pipeline import synthetic_transit_curve


def fixture_observation(tic_id: int, sector: int | None) -> RealObservation:
    return RealObservation(
        curve=synthetic_transit_curve(tic_id),
        mission="TESS/SPOC",
        sector=sector or 1,
        cadence_count=1024,
        source_uri="mast:TESS/product/test-lc.fits",
    )


def test_inference_contract() -> None:
    prediction = InferenceEngine(observation_loader=fixture_observation).predict(150428135)
    assert prediction.tic_id == 150428135
    assert 0.0 <= prediction.probability <= 1.0
    assert prediction.depth > 0
    assert prediction.score_kind == "bls_screening_score"
    assert prediction.data_source.startswith("mast:")
    assert len(prediction.time) == len(prediction.normalized_flux)


def test_api_health_and_inference(monkeypatch) -> None:
    test_engine = InferenceEngine(observation_loader=fixture_observation)
    monkeypatch.setattr(api_module, "engine", lambda: test_engine)
    client = TestClient(api_module.app)
    assert client.get("/health").json()["status"] == "ok"
    response = client.post("/v1/infer/150428135?sector=3")
    assert response.status_code == 200
    assert response.json()["model_version"] == "bls-real-0.1.0"
    assert response.json()["sector"] == 3


def test_api_lists_observation_products(monkeypatch) -> None:
    products = [
        ObservationProduct(
            sector=3,
            mission="TESS Sector 03",
            year=2018,
            exposure_seconds=120.0,
            source_uri="mast:TESS/product/sector-3-lc.fits",
        )
    ]
    monkeypatch.setattr(api_module, "list_tess_observations", lambda _: products)
    response = TestClient(api_module.app).get("/v1/observations/150428135")
    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert response.json()["items"][0]["sector"] == 3


def test_inference_with_trained_checkpoint(tmp_path: Path) -> None:
    from astromap.training import train_smoke

    checkpoint_path = tmp_path / "test_model.pt"
    train_smoke(checkpoint_path, epochs=1)
    engine = InferenceEngine(checkpoint=checkpoint_path, observation_loader=fixture_observation)
    assert engine.trained is True
    prediction = engine.predict(150428135)
    assert prediction.score_kind == "hybrid_pinn_bls_consensus"
    assert prediction.model_version == "test_model"
    assert 0.0 <= prediction.probability <= 1.0
    assert prediction.ai_probability is not None
    assert prediction.bls_score is not None
    assert prediction.harmonic_consistency is not None
    assert prediction.period_days > 0
    assert prediction.duration_days > 0