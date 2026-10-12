"""Unit tests for the scientific ablation study (RQ1, RQ2)."""

from pathlib import Path

from astromap.ablation import run_ablation_study


def test_ablation_study_structure(tmp_path: Path) -> None:
    manifest_path = Path("data/manifest.json")
    if not manifest_path.is_file():
        return

    out_file = tmp_path / "ablation_test.json"
    results = run_ablation_study(
        manifest_path=manifest_path,
        checkpoint_path="artifacts/pinn-transnet-v1.pt",
        output_path=out_file,
    )

    assert "variants" in results
    variants = results["variants"]
    assert "astropy_bls" in variants
    assert "temporal_1d_only" in variants
    assert "dual_domain_no_physics" in variants
    assert "full_pinn_transnet" in variants

    assert "research_questions_conclusions" in results
    rq = results["research_questions_conclusions"]
    assert "RQ1_CWT_Wavelet_Impact" in rq
    assert "RQ2_Physics_Loss_Impact" in rq
    assert out_file.is_file()

