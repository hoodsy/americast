"""Turning one stored run into the payloads the API serves.

Nothing here is stored. `estimate` rebuilds a run's per-plant megawatts
in about 1.4 seconds, and 1100 runs at that resolution would be ~39M
rows, so the Gate 4 ruling stands: compute on demand, keep the answer
in memory, and let the weather store remain the only copy.

The cache is what makes that comfortable. A client scrubbing through a
run's 47 hours hits one run repeatedly; only the first request pays.
"""

from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from americast.api.models import (
    GRADED_LEVEL,
    LevelSeries,
    Plant,
    PlantFrames,
    PlantList,
    PlantSeries,
    RunList,
    Totals,
)
from americast.features.features import fleet, hourly
from americast.features.power import clearness, estimate
from americast.ingest.hrrr import HRRR_DIR
from americast.region import CAISO_CA, RegionConfig
from americast.schemas import HRRR_WEATHER

# How many runs to keep computed. Each is ~37k rows of seven columns,
# a couple of megabytes, so this is small next to the ~1.5 s it saves.
CACHED_RUNS = 8

# Megawatts to one decimal, clearness to three. The payload carries
# 788 plants x 47 hours of each, and neither is meaningful past that:
# a plant's output is not known to the kilowatt, and a colour scale
# cannot show a thousandth.
MW_PLACES = 1
RATIO_PLACES = 3


def runs(hrrr_dir: Path = HRRR_DIR) -> RunList:
    """Stored runs, newest first.

    A file whose schema does not match the current HRRR_WEATHER is left
    out rather than served. The store is refetched in place after a
    schema change, so during one a stale file would otherwise be read
    as silent nulls — the same check the golden tests use.
    """
    found = []
    for path in sorted(hrrr_dir.glob("hrrr_*.parquet")):
        if pq.read_schema(path).equals(HRRR_WEATHER):
            found.append(_run_time(path))
    return RunList(runs=sorted(found, reverse=True))


def plants(region: RegionConfig = CAISO_CA) -> PlantList:
    """Every modelled plant, as it always is.

    The CISO filter happens here, so the list matches exactly the
    plants that appear in a run's payload. A frontend joining the two
    on plant_id must never find one missing from the other.
    """
    registry = fleet(pd.read_parquet(region.plant_registry_path))
    rows = [
        Plant(
            plant_id=int(row.plant_id),
            name=row.plant_name,
            latitude=float(row.latitude),
            longitude=float(row.longitude),
            capacity_mw_ac=float(row.capacity_mw_ac),
            dc_capacity_mw=float(row.dc_capacity_mw),
            county=row.county,
            zone=row.zone,
        )
        for row in registry.itertuples()
    ]
    return PlantList(plants=rows)


def frames(
    run_time: datetime,
    hrrr_dir: Path = HRRR_DIR,
    region: RegionConfig = CAISO_CA,
) -> PlantFrames:
    """One run's per-plant megawatts and clearness."""
    aligned = _aligned(pd.Timestamp(run_time), hrrr_dir, region)
    hours = _valid_times(aligned)

    series = []
    for plant_id, part in aligned.groupby("plant_id", sort=True):
        ordered = part.sort_values("valid_time")
        series.append(
            PlantSeries(
                plant_id=int(plant_id),
                mw=[round(v, MW_PLACES) for v in ordered["ac_mw"]],
                clearness=[
                    None if pd.isna(v) else round(v, RATIO_PLACES)
                    for v in ordered["clearness"]
                ],
            )
        )
    return PlantFrames(run_time=run_time, valid_times=hours, plants=series)


def totals(
    run_time: datetime,
    hrrr_dir: Path = HRRR_DIR,
    region: RegionConfig = CAISO_CA,
) -> Totals:
    """One run's state, zone and county curves.

    Plain sums, because a megawatt at one plant is a megawatt at
    another. Everything below the state is marked unvalidated: it is a
    physical estimate that adds up to the graded number, and no hourly
    truth exists to check it against.
    """
    aligned = _aligned(pd.Timestamp(run_time), hrrr_dir, region)
    hours = _valid_times(aligned)

    levels = [_level("state", region.iso, aligned)]
    for name, group in aligned.groupby("zone", sort=True):
        levels.append(_level("zone", str(name), group))
    for name, group in aligned.groupby("county", sort=True):
        levels.append(_level("county", str(name), group))
    return Totals(run_time=run_time, valid_times=hours, levels=levels)


def _level(level: str, name: str, rows: pd.DataFrame) -> LevelSeries:
    summed = rows.groupby("valid_time", sort=True)[["ac_mw", "clear_mw"]].sum()
    return LevelSeries(
        level=level,
        name=name,
        validated=level == GRADED_LEVEL,
        mw=[round(v, MW_PLACES) for v in summed["ac_mw"]],
        clear_mw=[round(v, MW_PLACES) for v in summed["clear_mw"]],
    )


def _valid_times(aligned: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted(aligned["valid_time"].unique())


@lru_cache(maxsize=CACHED_RUNS)
def _aligned(
    run_time: pd.Timestamp, hrrr_dir: Path, region: RegionConfig
) -> pd.DataFrame:
    """A run's per-plant estimate, aligned to hour means. Cached.

    The alignment is the same trapezoid the training table uses, and it
    has to be applied here too: HRRR reports instants, the graded
    forecast is an hour mean, and a map showing instants would disagree
    with the statewide curve by a factor of 2.6 at dusk. Applied within
    each plant, so 47 hours come out of 48.

    Clearness is computed after the averaging, because the quantity
    wanted is the ratio of the hour's output to the hour's ceiling, not
    the average of two instantaneous ratios.

    Raises FileNotFoundError if the run is not stored; the route turns
    that into a 404.
    """
    path = hrrr_dir / f"hrrr_{run_time:%Y%m%d}_{run_time:%H}z.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no stored run at {run_time.isoformat()}")

    registry = fleet(pd.read_parquet(region.plant_registry_path))
    weather = pd.read_parquet(path)
    mine = weather[weather["plant_id"].isin(registry["plant_id"])]

    estimated = estimate(mine, registry)
    kept = estimated[
        ["run_time", "valid_time", "lead_hours", "plant_id", "ac_mw", "clear_mw", "zenith"]
    ]
    aligned = hourly(kept, within=("plant_id",))
    aligned["clearness"] = clearness(aligned)

    places = registry[["plant_id", "county", "zone"]]
    return aligned.merge(places, on="plant_id", how="left")


def _run_time(path: Path) -> pd.Timestamp:
    """`hrrr_20240615_06z.parquet` -> 2024-06-15 06:00 UTC."""
    day, hour = path.stem.split("_")[1:3]
    return pd.Timestamp(f"{day} {hour[:2]}:00", tz="UTC")
