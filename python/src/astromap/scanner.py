"""High-throughput batch sky scanner and candidate catalog compiler."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astromap.inference import InferenceEngine, Prediction


@dataclass(frozen=True)
class ScanSummary:
    """Statistical summary of a batch sky scan."""

    total_scanned: int
    candidates_count: int
    false_positives_count: int
    unlikely_count: int
    top_candidates: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        return {
            "total_scanned": self.total_scanned,
            "candidates_count": self.candidates_count,
            "false_positives_count": self.false_positives_count,
            "unlikely_count": self.unlikely_count,
            "top_candidates": self.top_candidates,
        }


def scan_targets(
    tic_ids: list[int],
    engine: InferenceEngine | None = None,
    checkpoint: Path | None = None,
    sector: int | None = None,
    min_score: float = 0.50,
    output_csv: Path | None = None,
    output_json: Path | None = None,
    verbose: bool = True,
) -> tuple[list[Prediction], ScanSummary]:
    """Scan a list of TIC targets, evaluate hybrid detections and compile a discovery catalog.

    Parameters
    ----------
    tic_ids : list[int]
        Target identifiers to screen.
    engine : InferenceEngine, optional
        Pre-instantiated inference engine.
    checkpoint : Path, optional
        Path to PINN-TransNet model weights.
    sector : int, optional
        Specific TESS sector to query.
    min_score : float
        Threshold for candidate classification.
    output_csv : Path, optional
        Destination CSV path.
    output_json : Path, optional
        Destination JSON summary path.
    verbose : bool
        Whether to print progress indicators.

    Returns
    -------
    tuple[list[Prediction], ScanSummary]
        Predictions list and summary statistics.
    """
    if engine is None:
        from astromap.inference import InferenceEngine

        engine = InferenceEngine(checkpoint=checkpoint)

    predictions: list[Prediction] = []
    candidates_count = 0
    false_positives_count = 0
    unlikely_count = 0

    total = len(tic_ids)
    for index, tic_id in enumerate(tic_ids, 1):
        if verbose:
            sector_info = f" (Sector {sector})" if sector is not None else ""
            print(f"[{index}/{total}] Ingesting TIC {tic_id}{sector_info} from MAST...", flush=True)
        try:
            pred = engine.predict(tic_id, sector=sector)
            predictions.append(pred)
            if verbose:
                print(
                    f"       -> Score: {pred.probability:.3f} | Vetting: {pred.vetting_disposition} | Label: {pred.label}",
                    flush=True,
                )
            if pred.label == "candidate":
                candidates_count += 1
            elif pred.label == "false_positive":
                false_positives_count += 1
            else:
                unlikely_count += 1
        except (ValueError, RuntimeError, OSError) as error:
            if verbose:
                print(f"       -> Skipped (Error: {error})", flush=True)
            continue

    # Sort predictions by hybrid probability descending
    predictions.sort(key=lambda p: p.probability, reverse=True)

    top_candidates = [
        {
            "tic_id": p.tic_id,
            "probability": round(p.probability, 4),
            "period_days": round(p.period_days, 4),
            "depth_ppm": round(p.depth * 1e6, 1),
            "duration_hours": round(p.duration_days * 24, 2),
            "vetting_disposition": p.vetting_disposition,
            "label": p.label,
        }
        for p in predictions
        if p.probability >= min_score and p.vetting_disposition == "PLANET_CANDIDATE"
    ]

    summary = ScanSummary(
        total_scanned=len(predictions),
        candidates_count=candidates_count,
        false_positives_count=false_positives_count,
        unlikely_count=unlikely_count,
        top_candidates=top_candidates,
    )

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "tic_id",
                "label",
                "hybrid_confidence",
                "period_days",
                "depth_ppm",
                "duration_hours",
                "vetting_disposition",
                "odd_even_flag",
                "secondary_eclipse_flag",
                "stellar_density_flag",
                "mission",
                "sector",
            ])
            for p in predictions:
                writer.writerow([
                    p.tic_id,
                    p.label,
                    round(p.probability, 4),
                    round(p.period_days, 4),
                    round(p.depth * 1e6, 1),
                    round(p.duration_days * 24, 2),
                    p.vetting_disposition,
                    p.odd_even_flag,
                    p.secondary_eclipse_flag,
                    p.stellar_density_flag,
                    p.mission,
                    p.sector,
                ])

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")

    return predictions, summary

