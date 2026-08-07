"""HRRR ingestion: forecast weather extracted at plant locations.

HRRR is NOAA's 3 km hourly-updating weather model. We touch only the
00/06/12/18z runs (the ones that extend to 48 forecast hours) and, per
run and forecast hour, byte-range download five GRIB messages — DSWRF,
TCDC, TMP:2m, UGRD/VGRD:10m — sample them at plant coordinates, and
discard the grid. Grids are never stored.
"""

from pathlib import Path

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
