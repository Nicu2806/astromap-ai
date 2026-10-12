"""Command-line entry point for reproducible local inference, scanning, and benchmarking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from astromap.inference import InferenceEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="AstroMap-AI Scientific Workbench CLI")
    subparsers = parser.add_subparsers(dest="subcommand")

    # Subcommand: scan
    scan_parser = subparsers.add_parser("scan", help="Batch scan TIC targets for exoplanet candidates")
    scan_parser.add_argument("--tic-ids", type=int, nargs="+", required=True, help="List of TIC IDs to scan")
    scan_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/pinn-transnet-v1.pt"),
        help="Path to PINN-TransNet model weights",
    )
    scan_parser.add_argument(
        "--min-score",
        type=float,
        default=0.50,
        help="Minimum candidate classification score",
    )
    scan_parser.add_argument("--sector", type=int, default=None, help="Target TESS sector to analyze")
    scan_parser.add_argument("--output-csv", type=Path, default=None, help="Destination CSV file")
    scan_parser.add_argument("--output-json", type=Path, default=None, help="Destination JSON file")

    # Subcommand: benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run multi-target statistical benchmark")
    bench_parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of targets in benchmark population",
    )
    bench_parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/pinn-transnet-v1.pt"),
        help="Path to PINN-TransNet model weights",
    )
    bench_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Artifacts directory",
    )

    # Root flags for single target inference (backwards compatible)
    parser.add_argument("--tic-id", type=int, default=None, help="Analyze a single TIC target")
    parser.add_argument("--sector", type=int, default=None, help="Specific TESS sector to analyze")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/pinn-transnet-v1.pt"),
        help="Path to PINN-TransNet model weights",
    )

    args = parser.parse_args()

    if args.subcommand == "scan":
        from astromap.scanner import scan_targets

        engine = InferenceEngine(checkpoint=args.checkpoint)
        _predictions, summary = scan_targets(
            tic_ids=args.tic_ids,
            engine=engine,
            sector=args.sector,
            min_score=args.min_score,
            output_csv=args.output_csv,
            output_json=args.output_json,
        )
        print(json.dumps(summary.to_dict(), indent=2))
        return

    if args.subcommand == "benchmark":
        from astromap.benchmark import run_large_scale_benchmark

        metrics = run_large_scale_benchmark(
            checkpoint=args.checkpoint,
            target_count=args.count,
            artifacts_dir=args.output_dir,
        )
        print(json.dumps(metrics.to_dict(), indent=2))
        return

    if args.tic_id is not None:
        if args.tic_id <= 0:
            parser.error("--tic-id must be positive")
        engine = InferenceEngine(checkpoint=args.checkpoint)
        pred = engine.predict(args.tic_id, sector=args.sector)
        print(json.dumps(pred.to_dict()))
        return

    parser.print_help()


if __name__ == "__main__":
    main()

