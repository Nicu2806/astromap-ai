"""Internal inference API consumed by the Rust gateway."""

from __future__ import annotations

import os
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException

from astromap.inference import InferenceEngine
from astromap.mast import list_tess_observations

app = FastAPI(title="AstroMap-AI Inference", version="0.1.0")


@lru_cache(maxsize=1)
def engine() -> InferenceEngine:
    checkpoint_value = os.getenv("ASTROMAP_CHECKPOINT")
    checkpoint = Path(checkpoint_value) if checkpoint_value else None
    if checkpoint is not None and not checkpoint.is_file():
        raise RuntimeError(f"Configured checkpoint does not exist: {checkpoint}")
    return InferenceEngine(checkpoint=checkpoint)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "inference"}


@app.get("/v1/observations/{tic_id}")
def observations(tic_id: int) -> dict[str, object]:
    if tic_id <= 0:
        raise HTTPException(status_code=422, detail="tic_id must be positive")
    try:
        items = [asdict(product) for product in list_tess_observations(tic_id)]
        return {"items": items, "total": len(items)}
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/v1/infer/{tic_id}")
def infer(tic_id: int, sector: int | None = None) -> dict[str, object]:
    if tic_id <= 0:
        raise HTTPException(status_code=422, detail="tic_id must be positive")
    if sector is not None and sector <= 0:
        raise HTTPException(status_code=422, detail="sector must be positive")
    try:
        return engine().predict(tic_id, sector).to_dict()
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
