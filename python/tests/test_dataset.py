"""Tests for dataset manifest, anti-leakage grouping by TIC ID, and PyTorch dataset."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from astromap.dataset import (
    CatalogManifest,
    TessTransitDataset,
    get_curated_seed_records,
)


def test_seed_records_coverage() -> None:
    records = get_curated_seed_records()
    assert len(records) >= 20
    positive_count = sum(1 for r in records if r.label == 1)
    negative_count = sum(1 for r in records if r.label == 0)
    assert positive_count > 10
    assert negative_count > 5


def test_split_by_tic_prevents_data_leakage() -> None:
    records = get_curated_seed_records()
    manifest = CatalogManifest(records)
    split_manifest = manifest.split_by_tic(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)

    train_items = split_manifest.filter_split("train")
    val_items = split_manifest.filter_split("validation")
    test_items = split_manifest.filter_split("test")

    assert len(train_items) > 0
    assert len(val_items) > 0
    assert len(test_items) > 0

    train_tics = {item.tic_id for item in train_items}
    val_tics = {item.tic_id for item in val_items}
    test_tics = {item.tic_id for item in test_items}

    # Strict anti-leakage assertion: No TIC ID exists in more than one partition!
    assert len(train_tics.intersection(val_tics)) == 0
    assert len(train_tics.intersection(test_tics)) == 0
    assert len(val_tics.intersection(test_tics)) == 0


def test_manifest_serialization_json_and_parquet() -> None:
    records = get_curated_seed_records()[:6]
    manifest = CatalogManifest(records)
    checksum = manifest.compute_checksum()
    assert len(checksum) == 64

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "manifest.json"
        parquet_path = Path(tmpdir) / "manifest.parquet"

        manifest.save_json(json_path)
        loaded_json = CatalogManifest.load_json(json_path)
        assert len(loaded_json.items) == len(manifest.items)
        assert loaded_json.compute_checksum() == checksum

        manifest.save_parquet(parquet_path)
        loaded_parquet = CatalogManifest.load_parquet(parquet_path)
        assert len(loaded_parquet.items) == len(manifest.items)
        assert loaded_parquet.compute_checksum() == checksum


def test_tess_transit_dataset_item_shapes() -> None:
    records = get_curated_seed_records()[:4]
    dataset = TessTransitDataset(records, base_seed=123)
    assert len(dataset) == 4

    sample = dataset[0]
    assert "temporal" in sample
    assert "scalogram" in sample
    assert "label" in sample
    assert "parameters" in sample

    temporal = sample["temporal"]
    scalogram = sample["scalogram"]
    label = sample["label"]
    params = sample["parameters"]

    assert isinstance(temporal, torch.Tensor)
    assert temporal.shape == (2, 512)
    assert isinstance(scalogram, torch.Tensor)
    assert scalogram.shape == (1, 32, 512)
    assert isinstance(label, torch.Tensor)
    assert isinstance(params, torch.Tensor)
    assert params.shape == (3,)
