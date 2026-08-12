"""The HTTP surface: four reads and a `latest` alias.

Run it locally with

    uv run python -m americast.api.app          # http://localhost:8000

Interactive docs at /docs, generated from the models in `models.py`,
which are the contract a frontend builds against.

Read-only by design. Nothing here writes to the stores, so a client
cannot corrupt anything, and the whole service can be restarted or
thrown away without consequence.
"""

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from americast.api import frames
from americast.api.models import PlantFrames, PlantList, RunList, Totals
from americast.ingest.hrrr import HRRR_DIR
from americast.region import CAISO_CA

# Where a React dev server lives. Deployment, and any origin policy
# that survives it, is deliberately a separate decision.
DEV_ORIGINS = ["http://localhost:5173", "http://localhost:3000"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Fail at startup on a missing registry, not on the first request.

    Every endpoint needs it, so a missing file is a broken deployment
    rather than a bad request, and it should be visible the moment the
    process starts.
    """
    registry = Path(CAISO_CA.plant_registry_path)
    if not registry.exists():
        raise RuntimeError(f"plant registry missing at {registry}")
    yield


app = FastAPI(
    title="americast",
    summary="California utility-scale solar forecast, per plant and in total.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=DEV_ORIGINS,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/runs", response_model=RunList)
def list_runs() -> RunList:
    """Every stored model run, newest first."""
    return frames.runs(HRRR_DIR)


@app.get("/plants", response_model=PlantList)
def list_plants() -> PlantList:
    """Plant metadata. Static — fetch once and cache for the session."""
    return frames.plants(CAISO_CA)


@app.get("/runs/latest/plants", response_model=PlantFrames)
def latest_plant_frames() -> PlantFrames:
    """Per-plant values for the newest stored run."""
    return plant_frames(_latest())


@app.get("/runs/latest/totals", response_model=Totals)
def latest_totals() -> Totals:
    """Aggregated curves for the newest stored run."""
    return run_totals(_latest())


@app.get("/runs/{run_time}/plants", response_model=PlantFrames)
def plant_frames(run_time: datetime) -> PlantFrames:
    """Per-plant megawatts and clearness across one run's hours."""
    try:
        return frames.frames(run_time, HRRR_DIR, CAISO_CA)
    except FileNotFoundError as missing:
        raise HTTPException(status_code=404, detail=str(missing)) from missing


@app.get("/runs/{run_time}/totals", response_model=Totals)
def run_totals(run_time: datetime) -> Totals:
    """State, zone and county curves across one run's hours."""
    try:
        return frames.totals(run_time, HRRR_DIR, CAISO_CA)
    except FileNotFoundError as missing:
        raise HTTPException(status_code=404, detail=str(missing)) from missing


def _latest() -> datetime:
    """The newest stored run, or 404 if the store is empty."""
    stored = frames.runs(HRRR_DIR).runs
    if not stored:
        raise HTTPException(status_code=404, detail="no runs are stored")
    return stored[0]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
