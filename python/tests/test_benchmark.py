"""Unit tests for the AstroMap-AI validation and benchmark engine."""

from __future__ import annotations

from pathlib import Path

from astromap.benchmark import (
    generate_benchmark_curve,
    generate_population_catalog,
    get_curated_benchmark_catalog,
    run_full_benchmark,
)


def test_benchmark_catalog_integrity() -> None:
    catalog = get_curated_benchmark_catalog()
    assert len(catalog) == 50
    positives = [t for t in catalog if t.label == 1]
    negatives = [t for t in catalog if t.label == 0]
    assert len(positives) == 32
    assert len(negatives) == 18

    categories = {t.category for t in catalog}
    assert "HOT_JUPITER" in categories
    assert "TERRESTRIAL" in categories
    assert "SUB_NEPTUNE" in categories
    assert "TTV_SYSTEM" in categories
    assert "ECLIPSING_BINARY" in categories
    assert "QUIET_STAR" in categories


def test_population_generator_scaling() -> None:
    catalog_100 = generate_population_catalog(count=100)
    assert len(catalog_100) == 100
    positives = [t for t in catalog_100 if t.label == 1]
    negatives = [t for t in catalog_100 if t.label == 0]
    assert len(positives) > 50
    assert len(negatives) > 20


def test_generate_benchmark_curve() -> None:
    catalog = get_curated_benchmark_catalog()
    planet_target = next(t for t in catalog if t.name == "WASP-18 b")
    curve = generate_benchmark_curve(planet_target, points=256)
    assert len(curve.time) == 256
    assert len(curve.flux) == 256
    assert curve.flux.min() < 1.0


def test_run_benchmark_and_generate_artifacts(tmp_path: Path) -> None:
    csv_out = tmp_path / "benchmark_results.csv"
    json_out = tmp_path / "benchmark_metrics.json"
    html_out = tmp_path / "benchmark_report.html"

    # Fast smoke test with 10 targets
    results, metrics = run_full_benchmark(
        count=10,
        output_csv=csv_out,
        output_json=json_out,
        output_html=html_out,
    )

    assert len(results) == 10
    assert metrics.total_targets == 10
    assert csv_out.exists()
    assert json_out.exists()
    assert html_out.exists()

    csv_text = csv_out.read_text(encoding="utf-8")
    assert "WASP-18 b" in csv_text or "TOI" in csv_text

    html_text = html_out.read_text(encoding="utf-8")
    assert "AstroMap-AI" in html_text
    assert "Methodological Ablation" in html_text

