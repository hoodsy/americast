"""HRRR ingestion: forecast weather extracted at plant locations.

HRRR is NOAA's 3 km hourly-updating weather model. We touch only the
00/06/12/18z runs (the ones that extend to 48 forecast hours) and, per
run and forecast hour, byte-range download seven GRIB messages — DSWRF,
VBDSF, VDDSF, TCDC, TMP:2m, UGRD/VGRD:10m — sample them at plant
coordinates, and discard the grid. Grids are never stored.
"""

import os
import shutil
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xarray as xr
from herbie import Herbie

from americast.region import CAISO_CA
from americast.schemas import HRRR_WEATHER

# One regex alternative per GRIB message we keep (7 of ~170 in the file).
#
# VBDSF and VDDSF are named "Visible Beam/Diffuse Downward Solar Flux",
# which is misleading on both counts: they are broadband, and VBDSF is
# normal to the beam (DNI), not on a horizontal surface. Verified at 788
# plant points over zenith 13.5-54.6 degrees — DSWRF = VBDSF*cos(zenith)
# + VDDSF closes to within 0.04%. We take the split from the model
# instead of estimating it, because a tracker aims at the beam and 18.2
# of the fleet's 21.5 GW tracks.
SEARCH = (
    ":DSWRF:surface"
    "|:VBDSF:surface"
    "|:VDDSF:surface"
    "|:TCDC:entire atmosphere"
    "|:TMP:2 m above ground"
    "|:UGRD:10 m above ground"
    "|:VGRD:10 m above ground"
)

# Scratch space for in-flight GRIB subsets; remove_grib deletes each
# file right after decode, so nothing accumulates here.
GRIB_TMP = Path("data/tmp/herbie")

# One parquet per run. A file that exists means that run is stored;
# the manifest beside it records the runs that produced no file.
HRRR_DIR = Path("data/hrrr")

# One line per attempted run: run_time, status, fhours, rows, seconds.
# status is ok (48 hours) | partial (1-47) | missing (archive hole) |
# failed (our bug). Only the driver writes it, so lines never interleave.
MANIFEST_COLUMNS = ["run_time", "status", "fhours", "rows", "seconds"]

# Tries per forecast hour before we accept it is not coming.
TRIES = 3


def run_path(run_time: pd.Timestamp, root: Path = HRRR_DIR) -> Path:
    return root / f"hrrr_{run_time:%Y%m%d_%Hz}.parquet"


def manifest_path(root: Path = HRRR_DIR) -> Path:
    return root / "manifest.csv"


# cfgrib's names for the seven messages; fetch must deliver exactly these.
EXPECTED_VARS = {"sdswrf", "vbdsf", "vddsf", "tcc", "t2m", "u10", "v10"}


def fetch(
    run_time: pd.Timestamp, fhour: int, scratch: Path = GRIB_TMP
) -> list[xr.Dataset]:
    """Download the seven fields for one (run, forecast hour) as xarray.

    run_time must be tz-aware UTC. cfgrib groups variables by level
    type, so the result is a list of Datasets (a lone Dataset is
    normalized into a one-element list).

    An interrupted earlier run can leave a truncated GRIB subset in
    scratch, and Herbie trusts local files — so when fields come back
    short we force exactly one fresh download, then fail loudly.

    scratch is the Herbie save directory. The backfill gives each
    worker process its own, so two processes can never write the same
    file — the failure mode seen on 2026-08-07.
    """
    naive_utc = run_time.tz_convert("UTC").tz_localize(None)
    h = Herbie(
        naive_utc.to_pydatetime(),
        model="hrrr",
        product="sfc",
        fxx=fhour,
        save_dir=scratch,
        verbose=False,
    )
    dss = _open(h)
    if _fields(dss) != EXPECTED_VARS:
        dss = _open(h, overwrite=True)
    missing = EXPECTED_VARS - _fields(dss)
    if missing:
        raise ValueError(
            f"{run_time:%Y-%m-%d %Hz} f{fhour:02d} missing fields {sorted(missing)}"
        )
    return dss


def _open(h: Herbie, overwrite: bool = False) -> list[xr.Dataset]:
    """Decode the GRIB subset, without cfgrib's merge warning.

    cfgrib merges its hypercubes using xarray's old combine defaults and
    warns about the coming change. Our subset holds one message per
    variable, so the overlap the warning describes cannot arise — and
    the backfill would otherwise print 62,000 copies of it. The filter
    covers only this call.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return _as_list(h.xarray(SEARCH, remove_grib=True, overwrite=overwrite))


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
            "dni": raw["vbdsf"],
            "dhi": raw["vddsf"],
            "tcdc": raw["tcc"],
            "t2m": raw["t2m"],
            "w10m": np.sqrt(raw["u10"] ** 2 + raw["v10"] ** 2),
        }
    )


def attempt(
    run_time: pd.Timestamp,
    fhour: int,
    scratch: Path = GRIB_TMP,
    tries: int = TRIES,
) -> list[xr.Dataset] | None:
    """Fetch one forecast hour, or return None after `tries` failures.

    A dropped connection and a genuine hole in the AWS archive look the
    same from here, so both spend the same tries. Backing off between
    tries lets a brief S3 problem clear.
    """
    for n in range(1, tries + 1):
        try:
            return fetch(run_time, fhour, scratch)
        # Blind on purpose: S3, eccodes and cfgrib each fail their own
        # way, and every one of them means the same thing here — try
        # again, then move on.
        except Exception as err:  # noqa: BLE001
            last = err
            if n < tries:
                time.sleep(2 * n)
    print(
        f"  {run_time:%Y-%m-%d %Hz} f{fhour:02d} skipped after "
        f"{tries} tries: {type(last).__name__}: {last}",
        flush=True,
    )
    return None


def build(
    run_time: pd.Timestamp, plants: pd.DataFrame, scratch: Path = GRIB_TMP
) -> pd.DataFrame:
    """One run's full table: forecast hours f01-f48 × all plants.

    Network-bound — 48 sequential byte-range fetches, ~7 minutes.

    A forecast hour that never arrives is skipped, not fatal: the
    archive has real holes, and one hole must not cost the other 47
    hours. When the first three hours all fail we treat the whole run
    as absent and stop, so a missing run costs 3 tries and not 144.

    Returns an empty frame when nothing downloaded at all.
    """
    frames = []
    for fhour in range(1, 49):
        grids = attempt(run_time, fhour, scratch)
        if grids is None:
            if not frames and fhour >= 3:
                break
            continue
        at_plants = extract(grids, plants)
        frames.append(finalize(at_plants, run_time, fhour))
    if not frames:
        return pd.DataFrame(columns=[f.name for f in HRRR_WEATHER])
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


# The Gate 3a pilot: June 2024, 06z only. Named once here because two
# callers need the same 30 run_times — pilot() to fetch them, and
# feature work and tests to read exactly them while the backfill keeps
# adding other runs to the same directory.
PILOT_RUNS = [pd.Timestamp(2024, 6, day, 6, tz="UTC") for day in range(1, 31)]


def load(
    root: Path = HRRR_DIR, run_times: list[pd.Timestamp] | None = None
) -> pd.DataFrame:
    """Runs stored in `root`, concatenated into one frame.

    run_times selects a subset; the default reads every stored run.
    Pass PILOT_RUNS to hold the pilot month steady while the backfill
    grows the directory underneath. A requested run with no file is
    skipped rather than fatal — the archive has real holes, and
    `pending` is the function whose job is reporting them.

    The schema check is the point. Writes go through HRRR_WEATHER, but
    a file written before a schema change reads back with silent NaN
    columns instead of an error. pq.read_schema touches only the
    parquet footer, so checking every file costs no data read.

    This holds its result in memory — fine for the pilot month (30
    files, 1.3M rows). The full backfill is ~5,000 files and will need
    reading in slices instead.
    """
    if run_times is None:
        paths = sorted(root.glob("hrrr_*.parquet"))
    else:
        wanted = [run_path(run_time, root) for run_time in sorted(run_times)]
        paths = [path for path in wanted if path.exists()]
    if not paths:
        raise FileNotFoundError(f"no run files in {root}")
    stale = []
    for path in paths:
        if not pq.read_schema(path).equals(HRRR_WEATHER):
            stale.append(path.name)
    if stale:
        raise ValueError(f"files do not match HRRR_WEATHER: {stale}")
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def pilot(root: Path = HRRR_DIR, plants: pd.DataFrame | None = None) -> int:
    """Gate 3a pilot: June 2024, 06z runs only, resumable.

    A run is skipped when its file already exists, so interrupting and
    re-running continues where it stopped. Returns runs fetched this
    call.
    """
    if plants is None:
        plants = pd.read_parquet(CAISO_CA.plant_registry_path)
    fetched = 0
    for run_time in PILOT_RUNS:
        if run_path(run_time, root).exists():
            continue
        frame = build(run_time, plants)
        path = write(frame, root)
        fetched += 1
        print(f"{path.name} written ({fetched} this session)", flush=True)
    return fetched


def read_manifest(root: Path = HRRR_DIR) -> pd.DataFrame:
    """The record of every attempted run. Empty frame when none yet."""
    path = manifest_path(root)
    if not path.exists():
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    out = pd.read_csv(path)
    out["run_time"] = pd.to_datetime(out["run_time"], utc=True)
    return out


def record(row: dict, root: Path = HRRR_DIR) -> None:
    """Append one line to the manifest, writing the header if new."""
    path = manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    line = pd.DataFrame([row], columns=MANIFEST_COLUMNS)
    line.to_csv(path, mode="a", header=new, index=False)


def runs(start: str, end: str, hours: tuple[int, ...] = (6,)) -> list[pd.Timestamp]:
    """Every target run_time from start to end, one per listed UTC hour."""
    days = pd.date_range(start, end, freq="D", tz="UTC")
    return [day + pd.Timedelta(hours=h) for day in days for h in sorted(hours)]


def pending(
    targets: list[pd.Timestamp], root: Path = HRRR_DIR
) -> list[pd.Timestamp]:
    """The targets that still need work.

    A run is finished when its parquet exists — which also covers the
    30 pilot files, written before the manifest existed. A run the
    manifest calls `missing` is a hole in the archive and is never
    retried. `partial` and `failed` runs do come back, because those
    usually mean a bad afternoon on the network, not absent data.
    """
    manifest = read_manifest(root)
    absent = set(manifest.loc[manifest["status"] == "missing", "run_time"])
    todo = []
    for run_time in targets:
        if run_path(run_time, root).exists():
            continue
        if run_time in absent:
            continue
        todo.append(run_time)
    return todo


def one(run_time: pd.Timestamp, root: Path = HRRR_DIR) -> dict:
    """Worker entry point: build one run, store it, describe the result.

    Each worker owns a scratch directory named for its own pid and
    empties it after every run. Two processes therefore never share a
    Herbie file, and scratch stays under a megabyte per worker instead
    of growing to gigabytes across the backfill.

    This never raises. A worker that dies takes the whole pool with it,
    so every failure comes back as a status instead.
    """
    scratch = GRIB_TMP / f"w{os.getpid()}"
    started = time.perf_counter()
    status, fhours, rows = "failed", 0, 0
    try:
        plants = pd.read_parquet(CAISO_CA.plant_registry_path)
        frame = build(run_time, plants, scratch)
        fhours = int(frame["lead_hours"].nunique())
        rows = len(frame)
        if frame.empty:
            status = "missing"
        else:
            write(frame, root)
            status = "ok" if fhours == 48 else "partial"
    # Blind on purpose: an exception escaping a worker kills the pool
    # and ends a 15-hour job. Every failure becomes a status instead.
    except Exception as err:  # noqa: BLE001
        print(
            f"  {run_time:%Y-%m-%d %Hz} failed: {type(err).__name__}: {err}",
            flush=True,
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return {
        "run_time": run_time.isoformat(),
        "status": status,
        "fhours": fhours,
        "rows": rows,
        "seconds": round(time.perf_counter() - started, 1),
    }


def backfill(
    start: str,
    end: str,
    hours: tuple[int, ...] = (6,),
    workers: int = 12,
    root: Path = HRRR_DIR,
) -> int:
    """Fetch every pending run from start to end with a pool of workers.

    Resumable: re-running skips what is already stored. The driver is
    the only process that touches the manifest, so its lines never
    interleave. Each fetch waits on S3 far more than it uses the CPU,
    so workers can exceed the core count.
    """
    targets = runs(start, end, hours)
    todo = pending(targets, root)
    print(
        f"{len(targets)} targets, {len(targets) - len(todo)} done, "
        f"{len(todo)} pending, {workers} workers",
        flush=True,
    )
    if not todo:
        return 0
    shutil.rmtree(GRIB_TMP, ignore_errors=True)
    started = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, run_time, root) for run_time in todo]
        for future in as_completed(futures):
            row = future.result()
            record(row, root)
            done += 1
            hours_spent = (time.perf_counter() - started) / 3600
            rate = done / hours_spent
            left = (len(todo) - done) / rate
            print(
                f"[{done}/{len(todo)}] {row['run_time'][:13]}z {row['status']:7s} "
                f"f={row['fhours']:2d} {row['seconds']:6.1f}s | "
                f"{rate:5.1f} runs/h, ~{left:4.1f} h left",
                flush=True,
            )
    return done


def verify(root: Path = HRRR_DIR) -> pd.DataFrame:
    """Audit what is actually stored: one row per parquet on disk.

    Reads only the three columns needed to check shape, so auditing
    thousands of files stays cheap. The files are the truth; the
    manifest is the log of how they got there.
    """
    rows = []
    for path in sorted(root.glob("hrrr_*.parquet")):
        frame = pd.read_parquet(path, columns=["run_time", "lead_hours", "plant_id"])
        rows.append(
            {
                "run_time": frame["run_time"].iloc[0],
                "fhours": int(frame["lead_hours"].nunique()),
                "plants": int(frame["plant_id"].nunique()),
                "rows": len(frame),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    yesterday = (pd.Timestamp.now("UTC") - pd.Timedelta(days=1)).date()
    parser = argparse.ArgumentParser(description="HRRR backfill")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=str(yesterday))
    parser.add_argument("--hours", default="6", help="UTC run hours, comma separated")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    run_hours = tuple(int(h) for h in args.hours.split(","))
    n = backfill(args.start, args.end, run_hours, args.workers)
    print(f"backfill session complete: {n} runs attempted")
