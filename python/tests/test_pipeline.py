import numpy as np
import pytest

from astromap.pipeline import LightCurve, preprocess, synthetic_transit_curve


def test_pipeline_is_deterministic_and_finite() -> None:
    first = preprocess(synthetic_transit_curve(261136679))
    second = preprocess(synthetic_transit_curve(261136679))

    np.testing.assert_allclose(first.flux, second.flux)
    assert first.scalogram.shape == (32, 512)
    assert np.isfinite(first.scalogram).all()
    assert 0.97 < np.median(first.flux) < 1.03


def test_pipeline_rejects_insufficient_valid_data() -> None:
    curve = LightCurve(
        time=np.arange(40, dtype=np.float64),
        flux=np.ones(40),
        flux_error=np.ones(40),
        quality=np.ones(40, dtype=np.int64),
    )
    with pytest.raises(ValueError, match="Too few valid"):
        preprocess(curve)
