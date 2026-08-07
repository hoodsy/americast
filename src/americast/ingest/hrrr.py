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
import xarray as xr
from herbie import Herbie

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


def fetch_fields(run_time: pd.Timestamp, fhour: int) -> list[xr.Dataset]:
    """Download the five fields for one (run, forecast hour) as xarray.

    run_time must be tz-aware UTC. cfgrib groups variables by level
    type, so the result is a list of Datasets (a lone Dataset is
    normalized into a one-element list).
    """
    naive_utc = run_time.tz_convert("UTC").tz_localize(None)
    h = Herbie(
        naive_utc.to_pydatetime(),
        model="hrrr",
        product="sfc",
        fxx=fhour,
        save_dir=GRIB_TMP,
    )
    dss = h.xarray(SEARCH, remove_grib=True)
    return [dss] if isinstance(dss, xr.Dataset) else dss


def extract_at_plants(dss: list[xr.Dataset], plants: pd.DataFrame) -> pd.DataFrame:
    """Sample each gridded field at the plant coordinates.

    Nearest-gridpoint on HRRR's 3 km Lambert grid via Herbie's
    pick_points (the grid is projected, so lat/lon lookup is a
    nearest-neighbor search, not an index). Returns one row per plant
    with cfgrib's variable names — renaming and wind math are
    build_run_frame's job.
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


def finalize_fhour(
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


def build_run_frame(run_time: pd.Timestamp, plants: pd.DataFrame) -> pd.DataFrame:
    """One run's full table: forecast hours f01-f48 × all plants.

    Network-bound — 48 sequential byte-range fetches, ~10 minutes.
    """
    frames = [
        finalize_fhour(
            extract_at_plants(fetch_fields(run_time, fhour), plants),
            run_time,
            fhour,
        )
        for fhour in range(1, 49)
    ]
    return pd.concat(frames, ignore_index=True)
