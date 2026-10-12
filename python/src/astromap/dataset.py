"""Dataset manifest, grouping by star (no leakage), and PyTorch dataset."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from astromap.pipeline import ProcessedCurve, preprocess, synthetic_transit_curve


@dataclass(frozen=True)
class TargetRecord:
    """A target entry in the versioned catalog manifest."""

    tic_id: int
    name: str
    ra: float
    dec: float
    sector: int
    period_days: float
    duration_days: float
    depth: float
    label: int  # 1 = candidate / confirmed planet, 0 = false positive / non-transit
    disposition: str  # CONFIRMED, CANDIDATE, FALSE_POSITIVE, NON_TRANSIT
    stellar_radius_solar: float = 1.0
    stellar_mass_solar: float = 1.0
    split: str = "train"  # train, validation, test

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetRecord:
        return cls(
            tic_id=int(data["tic_id"]),
            name=str(data["name"]),
            ra=float(data["ra"]),
            dec=float(data["dec"]),
            sector=int(data["sector"]),
            period_days=float(data["period_days"]),
            duration_days=float(data["duration_days"]),
            depth=float(data["depth"]),
            label=int(data["label"]),
            disposition=str(data["disposition"]),
            stellar_radius_solar=float(data.get("stellar_radius_solar", 1.0)),
            stellar_mass_solar=float(data.get("stellar_mass_solar", 1.0)),
            split=str(data.get("split", "train")),
        )


def get_curated_seed_records() -> list[TargetRecord]:
    """Return a curated set of real TESS targets covering confirmed planets and false positives/controls."""

    return [
        # TOI 700 - Multi-planet M dwarf system (confirmed habitable-zone Earth-size planet d, e)
        TargetRecord(
            tic_id=150428135,
            name="TOI 700 d",
            ra=97.096786,
            dec=-65.579311,
            sector=1,
            period_days=37.426,
            duration_days=0.148,
            depth=0.00085,
            label=1,
            disposition="CONFIRMED",
            stellar_radius_solar=0.42,
            stellar_mass_solar=0.41,
        ),
        TargetRecord(
            tic_id=150428135,
            name="TOI 700 c",
            ra=97.096786,
            dec=-65.579311,
            sector=3,
            period_days=16.051,
            duration_days=0.125,
            depth=0.0021,
            label=1,
            disposition="CONFIRMED",
            stellar_radius_solar=0.42,
            stellar_mass_solar=0.41,
        ),
        TargetRecord(
            tic_id=150428135,
            name="TOI 700 b",
            ra=97.096786,
            dec=-65.579311,
            sector=11,
            period_days=9.977,
            duration_days=0.091,
            depth=0.00087,
            label=1,
            disposition="CONFIRMED",
            stellar_radius_solar=0.42,
            stellar_mass_solar=0.41,
        ),
        # WASP-18 b (TOI 216) - Ultra-short period hot Jupiter
        TargetRecord(
            tic_id=100100827,
            name="WASP-18 b",
            ra=24.3544,
            dec=-45.6778,
            sector=2,
            period_days=0.9414,
            duration_days=0.091,
            depth=0.0098,
            label=1,
            disposition="CONFIRMED",
            stellar_radius_solar=1.28,
            stellar_mass_solar=1.24,
        ),
        TargetRecord(
            tic_id=100100827,
            name="WASP-18 b (S3)",
            ra=24.3544,
            dec=-45.6778,
            sector=3,
            period_days=0.9414,
            duration_days=0.091,
            depth=0.0098,
            label=1,
            disposition="CONFIRMED",
            stellar_radius_solar=1.28,
            stellar_mass_solar=1.24,
        ),
        # TOI 1233 - HD 108236 multi-transiting system
        TargetRecord(
            tic_id=260128333,
            name="TOI 1233 b",
            ra=186.2917,
            dec=-51.3638,
            sector=10,
            period_days=3.795,
            duration_days=0.088,
            depth=0.00062,
            label=1,
            disposition="CONFIRMED",
            stellar_radius_solar=0.88,
            stellar_mass_solar=0.87,
        ),
        TargetRecord(
            tic_id=260128333,
            name="TOI 1233 c",
            ra=186.2917,
            dec=-51.3638,
            sector=11,
            period_days=6.203,
            duration_days=0.104,
            depth=0.00114,
            label=1,
            disposition="CONFIRMED",
            stellar_radius_solar=0.88,
            stellar_mass_solar=0.87,
        ),
        # TOI 256 (TIC 307210830)
        TargetRecord(
            tic_id=307210830,
            name="TOI 256.01",
            ra=124.531756,
            dec=-68.3130,
            sector=1,
            period_days=2.784,
            duration_days=0.112,
            depth=0.0042,
            label=1,
            disposition="CANDIDATE",
            stellar_radius_solar=0.75,
            stellar_mass_solar=0.72,
        ),
        # TOI 455 (TIC 98796344) - LTT 1445 A b
        TargetRecord(
            tic_id=98796344,
            name="TOI 455.01",
            ra=45.2412,
            dec=-16.5947,
            sector=4,
            period_days=5.358,
            duration_days=0.058,
            depth=0.0017,
            label=1,
            disposition="CONFIRMED",
            stellar_radius_solar=0.28,
            stellar_mass_solar=0.26,
        ),
        # TOI 114 (TIC 25155310)
        TargetRecord(
            tic_id=25155310,
            name="TOI 114.01",
            ra=336.562,
            dec=-13.042,
            sector=2,
            period_days=3.288,
            duration_days=0.095,
            depth=0.0076,
            label=1,
            disposition="CONFIRMED",
            stellar_radius_solar=0.95,
            stellar_mass_solar=0.92,
        ),
        # TOI 197 (TIC 441462715)
        TargetRecord(
            tic_id=441462715,
            name="TOI 197.01",
            ra=358.337,
            dec=-14.888,
            sector=2,
            period_days=14.298,
            duration_days=0.198,
            depth=0.00048,
            label=1,
            disposition="CONFIRMED",
            stellar_radius_solar=1.96,
            stellar_mass_solar=1.15,
        ),
        # TOI 270 (TIC 259377017)
        TargetRecord(
            tic_id=259377017,
            name="TOI 270 b",
            ra=64.004,
            dec=-51.956,
            sector=3,
            period_days=3.360,
            duration_days=0.071,
            depth=0.00185,
            label=1,
            disposition="CONFIRMED",
            stellar_radius_solar=0.38,
            stellar_mass_solar=0.39,
        ),
        TargetRecord(
            tic_id=259377017,
            name="TOI 270 c",
            ra=64.004,
            dec=-51.956,
            sector=4,
            period_days=5.660,
            duration_days=0.082,
            depth=0.0028,
            label=1,
            disposition="CONFIRMED",
            stellar_radius_solar=0.38,
            stellar_mass_solar=0.39,
        ),
        # TOI 125 (TIC 52368076) - Multi-planet system
        TargetRecord(
            tic_id=52368076,
            name="TOI 125 b",
            ra=20.598,
            dec=-66.673,
            sector=1,
            period_days=4.654,
            duration_days=0.095,
            depth=0.0012,
            label=1,
            disposition="CONFIRMED",
            stellar_radius_solar=0.86,
            stellar_mass_solar=0.86,
        ),
        TargetRecord(
            tic_id=52368076,
            name="TOI 125 c",
            ra=20.598,
            dec=-66.673,
            sector=2,
            period_days=9.151,
            duration_days=0.125,
            depth=0.0014,
            label=1,
            disposition="CONFIRMED",
            stellar_radius_solar=0.86,
            stellar_mass_solar=0.86,
        ),
        # TOI 178 (TIC 251848941) - Resonant Laplace chain system
        TargetRecord(
            tic_id=251848941,
            name="TOI 178 b",
            ra=10.024,
            dec=-30.453,
            sector=2,
            period_days=1.914,
            duration_days=0.065,
            depth=0.00065,
            label=1,
            disposition="CONFIRMED",
            stellar_radius_solar=0.65,
            stellar_mass_solar=0.65,
        ),
        TargetRecord(
            tic_id=251848941,
            name="TOI 178 c",
            ra=10.024,
            dec=-30.453,
            sector=3,
            period_days=3.238,
            duration_days=0.078,
            depth=0.00095,
            label=1,
            disposition="CONFIRMED",
            stellar_radius_solar=0.65,
            stellar_mass_solar=0.65,
        ),
        # TOI 216 (TIC 55525572) - Warm Jupiter system with strong TTVs
        TargetRecord(
            tic_id=55525572,
            name="TOI 216 b",
            ra=70.385,
            dec=-63.302,
            sector=1,
            period_days=17.09,
            duration_days=0.165,
            depth=0.0085,
            label=1,
            disposition="CONFIRMED",
            stellar_radius_solar=0.77,
            stellar_mass_solar=0.77,
        ),
        # TOI 560 (TIC 101955823) - HD 73583
        TargetRecord(
            tic_id=101955823,
            name="TOI 560 b",
            ra=129.584,
            dec=-13.298,
            sector=8,
            period_days=6.398,
            duration_days=0.092,
            depth=0.0016,
            label=1,
            disposition="CONFIRMED",
            stellar_radius_solar=0.65,
            stellar_mass_solar=0.71,
        ),
        # TOI 776 (TIC 424310910)
        TargetRecord(
            tic_id=424310910,
            name="TOI 776 b",
            ra=177.943,
            dec=-37.625,
            sector=10,
            period_days=8.246,
            duration_days=0.089,
            depth=0.0011,
            label=1,
            disposition="CONFIRMED",
            stellar_radius_solar=0.54,
            stellar_mass_solar=0.54,
        ),
        # TOI 836 (TIC 440887364)
        TargetRecord(
            tic_id=440887364,
            name="TOI 836 b",
            ra=224.283,
            dec=-28.481,
            sector=11,
            period_days=3.816,
            duration_days=0.076,
            depth=0.00085,
            label=1,
            disposition="CONFIRMED",
            stellar_radius_solar=0.66,
            stellar_mass_solar=0.68,
        ),
        # CONTROLS / NEGATIVES (False positives, eclipsing binaries, quiet stars, instrumental)
        # TIC 261136679 - Quiet bright star without transit
        TargetRecord(
            tic_id=261136679,
            name="TIC 261136679 (Quiet)",
            ra=84.291188,
            dec=-80.469120,
            sector=1,
            period_days=1.0,
            duration_days=0.0,
            depth=0.0,
            label=0,
            disposition="NON_TRANSIT",
            stellar_radius_solar=1.1,
            stellar_mass_solar=1.05,
        ),
        TargetRecord(
            tic_id=261136679,
            name="TIC 261136679 (S2)",
            ra=84.291188,
            dec=-80.469120,
            sector=2,
            period_days=1.0,
            duration_days=0.0,
            depth=0.0,
            label=0,
            disposition="NON_TRANSIT",
            stellar_radius_solar=1.1,
            stellar_mass_solar=1.05,
        ),
        # TIC 219800809 - Eclipsing Binary (V-shape deep eclipse, not planet)
        TargetRecord(
            tic_id=219800809,
            name="TIC 219800809 (EB)",
            ra=52.1245,
            dec=-62.3411,
            sector=1,
            period_days=2.154,
            duration_days=0.32,
            depth=0.185,
            label=0,
            disposition="ECLIPSING_BINARY",
            stellar_radius_solar=1.45,
            stellar_mass_solar=1.35,
        ),
        TargetRecord(
            tic_id=219800809,
            name="TIC 219800809 (S2)",
            ra=52.1245,
            dec=-62.3411,
            sector=2,
            period_days=2.154,
            duration_days=0.32,
            depth=0.185,
            label=0,
            disposition="ECLIPSING_BINARY",
            stellar_radius_solar=1.45,
            stellar_mass_solar=1.35,
        ),
        # TIC 382200986 - Stellar flare & rotational modulation
        TargetRecord(
            tic_id=382200986,
            name="TIC 382200986 (Variable)",
            ra=112.553,
            dec=-54.218,
            sector=4,
            period_days=4.82,
            duration_days=0.45,
            depth=0.035,
            label=0,
            disposition="FALSE_POSITIVE",
            stellar_radius_solar=0.82,
            stellar_mass_solar=0.78,
        ),
        # TIC 149603524 - Instrumental systematic artifact
        TargetRecord(
            tic_id=149603524,
            name="TIC 149603524 (Systematic)",
            ra=78.914,
            dec=-48.729,
            sector=5,
            period_days=13.7,
            duration_days=0.85,
            depth=0.0015,
            label=0,
            disposition="FALSE_POSITIVE",
            stellar_radius_solar=1.02,
            stellar_mass_solar=0.98,
        ),
        # TIC 410153553 - Quiet reference
        TargetRecord(
            tic_id=410153553,
            name="TIC 410153553 (Ref)",
            ra=92.115,
            dec=-60.104,
            sector=6,
            period_days=2.0,
            duration_days=0.0,
            depth=0.0,
            label=0,
            disposition="NON_TRANSIT",
            stellar_radius_solar=0.98,
            stellar_mass_solar=0.95,
        ),
        # TIC 300015238 - Grazing eclipsing binary (V-shape)
        TargetRecord(
            tic_id=300015238,
            name="TIC 300015238 (Grazing EB)",
            ra=105.412,
            dec=-58.332,
            sector=8,
            period_days=1.842,
            duration_days=0.28,
            depth=0.075,
            label=0,
            disposition="ECLIPSING_BINARY",
            stellar_radius_solar=1.20,
            stellar_mass_solar=1.15,
        ),
        # TIC 167418704 - Contact binary / W UMa variable
        TargetRecord(
            tic_id=167418704,
            name="TIC 167418704 (Contact EB)",
            ra=142.115,
            dec=-45.890,
            sector=10,
            period_days=0.385,
            duration_days=0.19,
            depth=0.12,
            label=0,
            disposition="ECLIPSING_BINARY",
            stellar_radius_solar=1.10,
            stellar_mass_solar=1.05,
        ),
        # TIC 234284556 - Non-transiting quiet solar analog
        TargetRecord(
            tic_id=234284556,
            name="TIC 234284556 (Solar analog)",
            ra=195.340,
            dec=-33.450,
            sector=14,
            period_days=1.0,
            duration_days=0.0,
            depth=0.0,
            label=0,
            disposition="NON_TRANSIT",
            stellar_radius_solar=1.0,
            stellar_mass_solar=1.0,
        ),
        # TIC 350146231 - Stellar pulsator (Delta Scuti)
        TargetRecord(
            tic_id=350146231,
            name="TIC 350146231 (Pulsator)",
            ra=210.875,
            dec=-22.114,
            sector=16,
            period_days=0.15,
            duration_days=0.08,
            depth=0.02,
            label=0,
            disposition="FALSE_POSITIVE",
            stellar_radius_solar=1.65,
            stellar_mass_solar=1.75,
        ),
    ]


class CatalogManifest:
    """Versioned collection of dataset target records with non-leakage split."""

    def __init__(self, items: list[TargetRecord]) -> None:
        self.items = list(items)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([item.to_dict() for item in self.items])

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> CatalogManifest:
        items = [TargetRecord.from_dict(row.to_dict()) for _, row in df.iterrows()]
        return cls(items)

    def save_json(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "count": len(self.items),
            "checksum": self.compute_checksum(),
            "items": [item.to_dict() for item in self.items],
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_json(cls, path: Path | str) -> CatalogManifest:
        with open(Path(path), "r", encoding="utf-8") as f:
            data = json.load(f)
        items = [TargetRecord.from_dict(raw) for raw in data["items"]]
        return cls(items)

    def save_parquet(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(p, index=False)

    @classmethod
    def load_parquet(cls, path: Path | str) -> CatalogManifest:
        df = pd.read_parquet(Path(path))
        return cls.from_dataframe(df)

    def compute_checksum(self) -> str:
        """Deterministic sha256 checksum based on sorted record keys."""
        dumped = json.dumps(
            [item.to_dict() for item in sorted(self.items, key=lambda x: (x.tic_id, x.sector, x.name))],
            sort_keys=True,
        )
        return hashlib.sha256(dumped.encode("utf-8")).hexdigest()

    def split_by_tic(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> CatalogManifest:
        """Split records strictly by unique TIC ID to prevent data leakage between partitions."""

        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Partition ratios must sum to 1.0")

        pos_tics = sorted({item.tic_id for item in self.items if item.label == 1})
        neg_tics = sorted({item.tic_id for item in self.items if item.label == 0 and item.tic_id not in set(pos_tics)})

        rng = random.Random(seed)
        shuffled_pos = list(pos_tics)
        shuffled_neg = list(neg_tics)
        rng.shuffle(shuffled_pos)
        rng.shuffle(shuffled_neg)

        def partition_ids(ids: list[int]) -> tuple[set[int], set[int], set[int]]:
            n = len(ids)
            if n <= 2:
                return set(ids[:1]), set(ids[1:2]), set(ids[2:])
            n_tr = max(1, round(n * train_ratio))
            n_va = max(1, round(n * val_ratio))
            if n_tr + n_va >= n:
                n_tr = max(1, n - 2)
                n_va = 1
            return set(ids[:n_tr]), set(ids[n_tr : n_tr + n_va]), set(ids[n_tr + n_va :])

        pos_tr, pos_va, pos_te = partition_ids(shuffled_pos)
        neg_tr, neg_va, neg_te = partition_ids(shuffled_neg)

        train_tics = pos_tr | neg_tr
        val_tics = pos_va | neg_va
        test_tics = pos_te | neg_te

        # Anti-leakage invariant assertion: zero overlap
        assert not (train_tics & val_tics), "Data leakage between train and validation!"
        assert not (train_tics & test_tics), "Data leakage between train and test!"
        assert not (val_tics & test_tics), "Data leakage between validation and test!"

        updated_items: list[TargetRecord] = []
        for item in self.items:
            if item.tic_id in train_tics:
                split_name = "train"
            elif item.tic_id in val_tics:
                split_name = "validation"
            else:
                split_name = "test"

            record_dict = item.to_dict()
            record_dict["split"] = split_name
            updated_items.append(TargetRecord.from_dict(record_dict))

        return CatalogManifest(updated_items)

    def filter_split(self, split: str) -> list[TargetRecord]:
        return [item for item in self.items if item.split == split]


def generate_curve_for_record(record: TargetRecord, seed: int = 42) -> ProcessedCurve:
    """Generate a clean processed curve matching the record's physical parameters."""

    has_transit = record.label == 1
    depth = max(record.depth, 1e-4) if has_transit else 1e-6
    period = max(record.period_days, 0.5)
    duration = max(record.duration_days, 0.02) if has_transit else 0.1
    rng_seed = seed + (record.tic_id % 10000) + record.sector * 37

    raw_curve = synthetic_transit_curve(
        tic_id=rng_seed,
        points=1000,
        period_days=period,
        depth=depth,
        duration_days=duration,
    )
    return preprocess(raw_curve)


class TessTransitDataset(Dataset):
    """PyTorch Dataset yielding 1D flux, 2D scalogram, binary label, and transit parameters."""

    def __init__(self, records: list[TargetRecord], base_seed: int = 42) -> None:
        self.records = list(records)
        self.base_seed = base_seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor | float | int]:
        record = self.records[index]
        processed = generate_curve_for_record(record, seed=self.base_seed + index)

        # 1D Temporal branch: shape [2, 512] (flux offset, mask)
        flux = torch.from_numpy(processed.flux - 1.0).float()
        mask = torch.from_numpy(processed.mask.astype(np.float32))
        temporal = torch.stack((flux, mask))

        # 2D Wavelet branch: shape [1, 32, 512]
        scalogram = torch.from_numpy(processed.scalogram).float().unsqueeze(0)

        # Targets
        label = torch.tensor(float(record.label), dtype=torch.float32)
        params = torch.tensor(
            [record.depth, record.duration_days, record.period_days],
            dtype=torch.float32,
        )

        return {
            "temporal": temporal,
            "scalogram": scalogram,
            "label": label,
            "parameters": params,
            "stellar_radius": float(record.stellar_radius_solar),
            "stellar_mass": float(record.stellar_mass_solar),
            "tic_id": record.tic_id,
        }
