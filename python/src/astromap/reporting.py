"""Self-contained candidate report generation."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

from astromap.inference import Prediction


def write_report(prediction: Prediction, output: Path) -> Path:
    """Write a printable HTML candidate report without external templates."""

    output.parent.mkdir(parents=True, exist_ok=True)
    content = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>TIC {prediction.tic_id}</title>
<style>body{{font:16px Georgia,serif;max-width:800px;margin:48px auto;color:#17212b}}
h1{{border-bottom:3px solid #007c7a;padding-bottom:12px}} table{{border-collapse:collapse;width:100%}}
td,th{{padding:10px;border-bottom:1px solid #ccc;text-align:left}} .notice{{background:#f2f0ea;padding:14px}}</style>
</head><body><h1>AstroMap-AI Candidate Report</h1><p>TIC {prediction.tic_id}</p>
<table><tr><th>Metric</th><th>Value</th></tr>
<tr><td>Classification</td><td>{html.escape(prediction.label)}</td></tr>
<tr><td>Transit probability</td><td>{prediction.probability:.3f}</td></tr>
<tr><td>Depth</td><td>{prediction.depth:.7f}</td></tr>
<tr><td>Duration</td><td>{prediction.duration_days:.4f} days</td></tr>
<tr><td>Period</td><td>{prediction.period_days:.4f} days</td></tr>
<tr><td>Method</td><td>{html.escape(prediction.score_kind)}</td></tr>
<tr><td>Model</td><td>{html.escape(prediction.model_version)}</td></tr></table>
<h2>Observation provenance</h2><table>
<tr><td>Mission</td><td>{html.escape(prediction.mission)}</td></tr>
<tr><td>Sector</td><td>{prediction.sector}</td></tr>
<tr><td>Cadences</td><td>{prediction.cadence_count}</td></tr>
<tr><td>MAST product</td><td>{html.escape(prediction.data_source)}</td></tr></table>
<p class="notice">Research screening output only. This report does not confirm an exoplanet.</p>
</body></html>"""
    output.write_text(content, encoding="utf-8")
    return output


def main() -> None:
    from astromap.inference import InferenceEngine

    parser = argparse.ArgumentParser(description="Generate a printable candidate report")
    parser.add_argument("--tic-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    write_report(InferenceEngine().predict(arguments.tic_id), arguments.output)
    print(arguments.output)


if __name__ == "__main__":
    main()
