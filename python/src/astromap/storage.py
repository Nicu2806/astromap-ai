"""High-performance Parquet storage and cloud serialization for light curves.

Part of AstroMap-AI Scaling Pillar 1 (Big Data & Parquet Columnar Storage).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from astromap.pipeline import LightCurve, ProcessedCurve


def save_lightcurve_parquet(
    curve: LightCurve,
    target_path: Path | str,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Save a raw or calibrated LightCurve to a compressed Parquet file.

    Parameters
    ----------
    curve : LightCurve
        Validated photometric series containing time, flux, flux_error, quality.
    target_path : Path | str
        Destination path on disk.
    extra_meta : dict[str, Any] | None
        Optional dictionary of metadata (e.g. TIC ID, sector, mission).

    Returns
    -------
    Path
        The destination path.
    """
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, str] = {}
    if extra_meta:
        metadata = {str(k): json.dumps(v) if not isinstance(v, str) else v for k, v in extra_meta.items()}
    metadata["astromap_type"] = "LightCurve"
    metadata["n_points"] = str(len(curve.time))

    schema_metadata = {k.encode("utf-8"): v.encode("utf-8") for k, v in metadata.items()}

    arrays = [
        pa.array(curve.time, type=pa.float64()),
        pa.array(curve.flux, type=pa.float64()),
        pa.array(curve.flux_error, type=pa.float64()),
        pa.array(curve.quality, type=pa.int64()),
    ]
    names = ["time", "flux", "flux_error", "quality"]
    table = pa.Table.from_arrays(arrays, names=names)
    table = table.replace_schema_metadata(schema_metadata)

    pq.write_table(table, str(path), compression="ZSTD")
    return path


def load_lightcurve_parquet(source_path: Path | str) -> tuple[LightCurve, dict[str, str]]:
    """Load a LightCurve and its metadata from a Parquet file.

    Parameters
    ----------
    source_path : Path | str
        Path to the Parquet file.

    Returns
    -------
    tuple[LightCurve, dict[str, str]]
        Reconstructed LightCurve dataclass and the stored metadata dictionary.
    """
    path = Path(source_path)
    if not path.is_file():
        raise FileNotFoundError(f"Parquet light-curve file not found: {path}")

    table = pq.read_table(str(path))
    raw_meta = table.schema.metadata or {}
    meta = {k.decode("utf-8"): v.decode("utf-8") for k, v in raw_meta.items()}

    time = np.asarray(table["time"].to_numpy(), dtype=np.float64)
    flux = np.asarray(table["flux"].to_numpy(), dtype=np.float64)
    flux_error = np.asarray(table["flux_error"].to_numpy(), dtype=np.float64)
    quality = np.asarray(table["quality"].to_numpy(), dtype=np.int64)

    curve = LightCurve(time=time, flux=flux, flux_error=flux_error, quality=quality)
    return curve, meta


def save_processed_parquet(
    processed: ProcessedCurve,
    target_path: Path | str,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Save a ProcessedCurve and its 2D scalogram to a compressed Parquet file.

    Parameters
    ----------
    processed : ProcessedCurve
        Preprocessed curve with time, flux, mask, and 2D Ricker scalogram.
    target_path : Path | str
        Destination path.
    extra_meta : dict[str, Any] | None
        Optional metadata dictionary.

    Returns
    -------
    Path
        Destination path.
    """
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, str] = {}
    if extra_meta:
        metadata = {str(k): json.dumps(v) if not isinstance(v, str) else v for k, v in extra_meta.items()}
    metadata["astromap_type"] = "ProcessedCurve"
    metadata["scalogram_shape"] = json.dumps(list(processed.scalogram.shape))
    metadata["scalogram_dtype"] = str(processed.scalogram.dtype)

    schema_metadata = {k.encode("utf-8"): v.encode("utf-8") for k, v in metadata.items()}

    arrays = [
        pa.array(processed.time, type=pa.float64()),
        pa.array(processed.flux, type=pa.float64()),
        pa.array(processed.mask, type=pa.bool_()),
    ]
    names = ["time", "flux", "mask"]

    # Store scalogram rows as individual columns: sc_0, sc_1, ...
    scales, _ = processed.scalogram.shape
    for s in range(scales):
        names.append(f"scalogram_scale_{s:02d}")
        arrays.append(pa.array(processed.scalogram[s], type=pa.float64()))

    table = pa.Table.from_arrays(arrays, names=names)
    table = table.replace_schema_metadata(schema_metadata)

    pq.write_table(table, str(path), compression="ZSTD")
    return path


def load_processed_parquet(source_path: Path | str) -> tuple[ProcessedCurve, dict[str, str]]:
    """Load a ProcessedCurve and reconstruct its 2D scalogram from a Parquet file.

    Parameters
    ----------
    source_path : Path | str
        Path to the Parquet file.

    Returns
    -------
    tuple[ProcessedCurve, dict[str, str]]
        Reconstructed ProcessedCurve and metadata.
    """
    path = Path(source_path)
    if not path.is_file():
        raise FileNotFoundError(f"Processed curve Parquet file not found: {path}")

    table = pq.read_table(str(path))
    raw_meta = table.schema.metadata or {}
    meta = {k.decode("utf-8"): v.decode("utf-8") for k, v in raw_meta.items()}

    time = np.asarray(table["time"].to_numpy(), dtype=np.float64)
    flux = np.asarray(table["flux"].to_numpy(), dtype=np.float64)
    mask = np.asarray(table["mask"].to_numpy(), dtype=np.bool_)

    shape = json.loads(meta.get("scalogram_shape", "[32, 512]"))
    scales, points = shape[0], shape[1]

    scalogram = np.empty((scales, points), dtype=np.float64)
    for s in range(scales):
        col_name = f"scalogram_scale_{s:02d}"
        if col_name in table.column_names:
            scalogram[s] = np.asarray(table[col_name].to_numpy(), dtype=np.float64)
        else:
            scalogram[s] = 0.0

    return ProcessedCurve(time=time, flux=flux, mask=mask, scalogram=scalogram), meta


def tess_s3_path_prefix(tic_id: int, sector: int) -> str:
    """Compute the AWS Open Data S3 path prefix for a TESS SPOC light curve.

    AWS Registry of Open Data format:
    s3://stpubdata/tess/public/tid/s{sector:04d}/{tic_prefix}/{tic_id}/
    """
    tic_str = f"{tic_id:016d}"
    part1, part2, part3, part4 = (
        tic_str[0:4],
        tic_str[4:8],
        tic_str[8:12],
        tic_str[12:16],
    )
    return (
        f"s3://stpubdata/tess/public/tid/s{sector:04d}/"
        f"{part1}/{part2}/{part3}/{part4}/"
    )
