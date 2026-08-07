"""HRRR ingestion: forecast weather extracted at plant locations.

HRRR is NOAA's 3 km hourly-updating weather model. We touch only the
00/06/12/18z runs (the ones that extend to 48 forecast hours) and, per
run and forecast hour, byte-range download five GRIB messages — DSWRF,
TCDC, TMP:2m, UGRD/VGRD:10m — sample them at plant coordinates, and
discard the grid. Grids are never stored.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
from herbie import Herbie

from americast.region import CAISO_CA
from americast.schemas import HRRR_WEATHER

# One regex alternative per GRIB message we keep (5 of ~170 in the file).
SEARCH = (
    ":DSWRF:surface"
    "|:TCDC:entire atmosphere"
    "|:TMP:2 m above ground"
    "|:UGRD:10 m above ground"
    "|:VGRD:10 m above ground"
)

# Scratch space for in-flight GRIB subsets; remove_grib deletes each
# file right after decode, so nothing accumulates here.
GRIB_TMP = Path("data/tmp/herbie")

# One parquet per run; the directory listing doubles as the backfill
# manifest (a file exists == that run is done).
HRRR_DIR = Path("data/hrrr")


def run_path(run_time: pd.Timestamp, root: Path = HRRR_DIR) -> Path:
    return root / f"hrrr_{run_time:%Y%m%d_%Hz}.parquet"


# cfgrib's names for the five messages; fetch must deliver exactly these.
EXPECTED_VARS = {"sdswrf", "tcc", "t2m", "u10", "v10"}


def fetch(run_time: pd.Timestamp, fhour: int) -> list[xr.Dataset]:
    """Download the five fields for one (run, forecast hour) as xarray.

    run_time must be tz-aware UTC. cfgrib groups variables by level
    type, so the result is a list of Datasets (a lone Dataset is
    normalized into a one-element list).

    An interrupted earlier run can leave a truncated GRIB subset in
    GRIB_TMP, and Herbie trusts local files — so when fields come back
    short we force exactly one fresh download, then fail loudly.
    """
    naive_utc = run_time.tz_convert("UTC").tz_localize(None)
    h = Herbie(
        naive_utc.to_pydatetime(),
        model="hrrr",
        product="sfc",
        fxx=fhour,
        save_dir=GRIB_TMP,
    )
    dss = _as_list(h.xarray(SEARCH, remove_grib=True))
    if _fields(dss) != EXPECTED_VARS:
        dss = _as_list(h.xarray(SEARCH, remove_grib=True, overwrite=True))
    missing = EXPECTED_VARS - _fields(dss)
    if missing:
        raise ValueError(
            f"{run_time:%Y-%m-%d %Hz} f{fhour:02d} missing fields {sorted(missing)}"
        )
    return dss


def _as_list(dss: xr.Dataset | list[xr.Dataset]) -> list[xr.Dataset]:
    return [dss] if isinstance(dss, xr.Dataset) else dss


def _fields(dss: list[xr.Dataset]) -> set[str]:
    names: set[str] = set()
    for ds in dss:
        names |= set(ds.data_vars)
    return names


def extract(dss: list[xr.Dataset], plants: pd.DataFrame) -> pd.DataFrame:
    """Sample each gridded field at the plant coordinates.

    Nearest-gridpoint on HRRR's 3 km Lambert grid via Herbie's
    pick_points (the grid is projected, so lat/lon lookup is a
    nearest-neighbor search, not an index). Returns one row per plant
    with cfgrib's variable names — renaming and wind math are
    finalize's job.
    """
    pts = plants[["plant_id", "latitude", "longitude"]].reset_index(drop=True)
    out = pts[["plant_id"]].copy()
    for ds in dss:
        picked = ds.herbie.pick_points(pts, method="nearest")
        if picked.sizes["point"] != len(pts):
            raise ValueError(
                f"picked {picked.sizes['point']} points for {len(pts)} plants"
            )
        # > half a grid diagonal from every cell center means a plant is
        # off-grid or a coordinate is garbage — fail loudly.
        max_km = float(picked["point_grid_distance"].max())
        if max_km > 5.0:
            raise ValueError(f"plant {max_km:.1f} km from nearest gridpoint")
        frame = picked.to_dataframe()
        for var in ds.data_vars:
            out[var] = frame[var].to_numpy()
    return out


def finalize(
    raw: pd.DataFrame, run_time: pd.Timestamp, fhour: int
) -> pd.DataFrame:
    """Pure reshaping of one extracted forecast hour into schema form.

    cfgrib names → schema names, wind components → scalar speed, and the
    three time facts stamped on every row. lead_hours >= 1 by
    construction — the "no row where valid_time <= run_time" invariant
    is born here, not enforced later.
    """
    return pd.DataFrame(
        {
            "run_time": run_time,
            "valid_time": run_time + pd.Timedelta(hours=fhour),
            "lead_hours": np.int32(fhour),
            "plant_id": raw["plant_id"],
            "dswrf": raw["sdswrf"],
            "tcdc": raw["tcc"],
            "t2m": raw["t2m"],
            "w10m": np.sqrt(raw["u10"] ** 2 + raw["v10"] ** 2),
        }
    )


def build(run_time: pd.Timestamp, plants: pd.DataFrame) -> pd.DataFrame:
    """One run's full table: forecast hours f01-f48 × all plants.

    Network-bound — 48 sequential byte-range fetches, ~10 minutes.
    """
    frames = []
    for fhour in range(1, 49):
        grids = fetch(run_time, fhour)
        at_plants = extract(grids, plants)
        frames.append(finalize(at_plants, run_time, fhour))
    return pd.concat(frames, ignore_index=True)


def write(df: pd.DataFrame, root: Path = HRRR_DIR) -> Path:
    """Write one run's frame as one schema-enforced parquet.

    Rewriting the same run replaces its file — idempotent by
    construction, like the CAISO store.
    """
    if df["run_time"].nunique() != 1:
        raise ValueError("one file per run: frame must hold a single run_time")
    path = run_path(df["run_time"].iloc[0], root)
    root.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, schema=HRRR_WEATHER, preserve_index=False)
    pq.write_table(table, path)
    return path


def pilot(root: Path = HRRR_DIR, plants: pd.DataFrame | None = None) -> int:
    """Gate 3a pilot: June 2024, 06z runs only, resumable.

    A run is skipped when its file already exists, so interrupting and
    re-running continues where it stopped. Returns runs fetched this
    call.
    """
    if plants is None:
        plants = pd.read_parquet(CAISO_CA.plant_registry_path)
    fetched = 0
    for day in range(1, 31):
        run_time = pd.Timestamp(2024, 6, day, 6, tz="UTC")
        if run_path(run_time, root).exists():
            continue
        frame = build(run_time, plants)
        path = write(frame, root)
        fetched += 1
        print(f"{path.name} written ({fetched} this session)", flush=True)
    return fetched


if __name__ == "__main__":
    n = pilot()
    print(f"pilot complete: {n} runs fetched into {HRRR_DIR}")
