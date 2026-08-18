"""The public archive: one directory per run, and an index over them.

Every object written here is a **projection of a store that already
exists** — `live/forecasts.parquet` for the curve, `live/scores.parquet`
for what happened, the weather store and the registry for the map. This
module computes nothing of its own.

That property is the whole design. Re-running is safe, a sealed run can
be rebuilt byte-identically when a bug is found, and the JSON never
becomes a second source of truth that can quietly disagree with the
parquet it came from.

The map halves come from `api.frames`, the same functions the local
FastAPI app serves, so the bucket and the API cannot drift into two
contracts sharing one name.

## Sealing

A run's forecast object gains actuals for a day or two after it is
issued, then stops changing. `sealed` says which state it is in, and the
cache header follows: a sealed object is `immutable` for a year, an open
one for five minutes. An object sent `immutable` and rewritten later is
invisible to every reader that cached it, so the flag is a correctness
concern rather than a performance one.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from americast import storage
from americast.api import frames
from americast.daily import grade_daily, run_daily
from americast.features.baselines import DAYLIGHT_MW
from americast.ingest.hrrr import HRRR_DIR
from americast.region import CAISO_CA, RegionConfig
from americast.schemas import LIVE_SCORES

# A published run covers 47 hours: the 48th has no successor to average
# with and is dropped upstream.
RUN_HOURS = 47

# After this an open run seals whatever its grading looks like. Some
# hours never grade — `grade_daily.MIN_INTERVALS` drops an hour backed by
# short CAISO telemetry, and CAISO does not re-send it — so a rule of
# "all 47 graded" alone would rewrite those runs every morning forever.
SEAL_AFTER_DAYS = 4

# Header values, written onto the objects themselves. `immutable` goes
# only where the bytes will never change again; see the module note.
IMMUTABLE = "public, max-age=31536000, immutable"
BRIEF = "public, max-age=300"
DAILY = "public, max-age=86400"


def run_key(run_time: pd.Timestamp) -> str:
    """`20260817T06z` — the weather file's spelling, not a second one."""
    return f"{run_time:%Y%m%dT%Hz}"


def sealed(
    run_time: pd.Timestamp, graded: pd.DataFrame, now: pd.Timestamp | None = None
) -> bool:
    """Will this run's forecast object ever be written again?"""
    moment = now or pd.Timestamp(datetime.now(tz=UTC))
    if len(graded) >= RUN_HOURS:
        return True
    return moment - run_time >= pd.Timedelta(days=SEAL_AFTER_DAYS)


def caching(sealed: bool) -> str:
    """The header a run's forecast object carries, given its state."""
    return IMMUTABLE if sealed else BRIEF


def curve(
    run_time: pd.Timestamp,
    forecasts: pd.DataFrame,
    scores: pd.DataFrame,
    region: RegionConfig = CAISO_CA,
    accuracy: dict | None = None,
    generated_at: pd.Timestamp | None = None,
    updated_at: pd.Timestamp | None = None,
) -> dict:
    """One run's statewide object: the curve as issued, plus how it did.

    `error` is this run's own record over its own graded hours.
    `accuracy` is the rolling 30-day figure as of `updated_at`. They are
    different numbers and both belong: one says how this forecast did,
    the other says how the model has been doing lately.
    """
    issued = forecasts[forecasts["run_time"] == run_time]
    if issued.empty:
        raise ValueError(f"no stored forecast for {run_time.isoformat()}")

    contract = run_daily.to_json(issued, region, accuracy, generated_at)
    graded = scores[scores["run_time"] == run_time]
    stamped = updated_at or pd.Timestamp(datetime.now(tz=UTC))

    return {
        **contract,
        "updated_at": stamped.isoformat(),
        "observed_mw": _observed(contract["valid_times"], graded),
        "error": _error(graded),
        "sealed": sealed(run_time, graded, now=stamped),
    }


def _observed(valid_times: list[str], graded: pd.DataFrame) -> list[float | None]:
    """CAISO's actuals, parallel to valid_times, None where not graded.

    None rather than zero, and the distinction is the point: zero claims
    the fleet made nothing, None says nobody has checked. An hour CAISO
    never published whole stays None forever.
    """
    by_time = {
        stamp.isoformat(): round(float(value), 1)
        for stamp, value in zip(graded["valid_time"], graded["solar_mw"], strict=True)
    }
    return [by_time.get(stamp) for stamp in valid_times]


def _error(graded: pd.DataFrame) -> dict | None:
    """This run's own score, over daylight hours only.

    Daylight on the same cut every other score in this project uses.
    Night is trivially correct and would flatter the number by about half
    while saying nothing.

    None before anything is graded, which is a real state on the morning
    a run is published and must not be faked with zeroes.
    """
    lit = graded[graded["p90_mw"] > DAYLIGHT_MW]
    if lit.empty:
        return None
    return {
        "mae_mw": round(float(lit["error_mw"].abs().mean()), 1),
        "bias_mw": round(float(lit["error_mw"].mean()), 1),
        "coverage": round(float(lit["inside_band"].mean()), 3),
        "graded_hours": len(lit),
    }


def run_prefix(run_time: pd.Timestamp, region: RegionConfig = CAISO_CA) -> Path | str:
    """Where one run's objects live."""
    return storage.public(f"{region.id}/runs/{run_key(run_time)}")


def index_path(region: RegionConfig = CAISO_CA) -> Path | str:
    """Where the run index lives."""
    return storage.public(f"{region.id}/runs.json")


def write(
    run_time: pd.Timestamp,
    region: RegionConfig = CAISO_CA,
    hrrr_dir: Path | str = HRRR_DIR,
    accuracy: dict | None = None,
    now: pd.Timestamp | None = None,
    forecasts: pd.DataFrame | None = None,
    scores: pd.DataFrame | None = None,
) -> dict[str, Path | str]:
    """One run's three objects. Safe to re-run.

    The forecast object is rewritten while actuals land. The two map
    objects are physics over an immutable weather file, so they are
    written once and never touched again — see `_map_objects`.

    `forecasts` and `scores` default to the live stores. They are
    arguments because every store path in this project is a module
    constant resolved at import, so a caller that moves the data root
    after import — a test, mostly — cannot reach the stores any other
    way. Passing them also keeps this function a projection of whatever
    it is handed rather than of global state.
    """
    prefix = run_prefix(run_time, region)
    forecast_path = storage.child(prefix, "forecast.json")

    issued = run_daily.load() if forecasts is None else forecasts
    graded = _scores() if scores is None else scores
    object_ = curve(
        run_time,
        issued,
        graded,
        region,
        accuracy,
        generated_at=_first_written(forecast_path),
        updated_at=now,
    )
    storage.write_text(
        forecast_path,
        json.dumps(object_, indent=2),
        cache_control=caching(object_["sealed"]),
    )

    written: dict[str, Path | str] = {"forecast": forecast_path}
    written.update(_map_objects(run_time, prefix, region, hrrr_dir) or {})
    return written


def _map_objects(
    run_time: pd.Timestamp,
    prefix: Path | str,
    region: RegionConfig,
    hrrr_dir: Path | str,
) -> dict[str, Path | str]:
    """The per-level and per-plant objects, written at most once.

    Both are `immutable` from the moment they land, because both are
    physics over a weather file that never changes. Writing one a second
    time would be invisible to every reader that cached the first, so the
    guard is correctness rather than a saved round trip — and it also
    keeps `refresh` from rebuilding 400 KB a morning for nothing.

    **A run with no stored weather publishes its curve and no map.**
    Every run issued before the daily job began storing its weather is
    in that state, and so is any run whose fetch failed. Holding the
    forecast hostage to the map would mean publishing nothing at all for
    those mornings, which is worse than publishing half. `verify`
    reports the gap; it is not silently fine.
    """
    totals_path = storage.child(prefix, "totals.json")
    plants_path = storage.child(prefix, "plants.json.gz")

    try:
        if not storage.exists(totals_path):
            levels = frames.totals(run_time, hrrr_dir, region)
            storage.write_text(
                totals_path, levels.model_dump_json(indent=2), cache_control=IMMUTABLE
            )

        if not storage.exists(plants_path):
            series = frames.frames(run_time, hrrr_dir, region)
            storage.write_gzip(
                plants_path, series.model_dump_json(), cache_control=IMMUTABLE
            )
    except FileNotFoundError:
        return {}

    return {"totals": totals_path, "plants": plants_path}


def metadata(region: RegionConfig = CAISO_CA) -> Path | str:
    """The static plant list: names, coordinates, capacities, counties.

    Changes when the registry is rebuilt, not when a run lands, so it
    sits beside the runs rather than inside one. Compressed because it is
    142 KB of mostly-repeated text and 25 KB once packed.
    """
    path = storage.public(f"{region.id}/plants.json.gz")
    storage.write_gzip(
        path, frames.plants(region).model_dump_json(), cache_control=DAILY
    )
    return path


def catalogue(
    region: RegionConfig = CAISO_CA,
    forecasts: pd.DataFrame | None = None,
    scores: pd.DataFrame | None = None,
    now: pd.Timestamp | None = None,
) -> dict:
    """The run index: every published run, newest first.

    Built from the stores rather than from a listing of the bucket. The
    store knows every run that was published; a listing knows every
    object that happens to be there, which is a different question and
    the wrong one.

    `peak_mw` and `mae_mw` ride along so a run picker can show which days
    were sunny and which days the model missed, without fetching 47 run
    objects to find out.

    **This object grows without bound**, by about 150 bytes a run and so
    about 55 KB a year. That is fine for years at a five-minute TTL, and
    it is said here rather than capped, because a silently truncated
    archive reads exactly like a complete one.
    """
    issued = run_daily.load() if forecasts is None else forecasts
    graded = _scores() if scores is None else scores
    stamped = now or pd.Timestamp(datetime.now(tz=UTC))

    peaks = issued.groupby("run_time")["p50_mw"].max()
    lit = graded[graded["p90_mw"] > DAYLIGHT_MW]
    errors = lit.groupby("run_time")["error_mw"].apply(lambda e: e.abs().mean())

    runs = []
    for run_time in sorted(peaks.index, reverse=True):
        scored = graded[graded["run_time"] == run_time]
        runs.append(
            {
                "run_time": run_time.isoformat(),
                "path": f"{region.id}/runs/{run_key(run_time)}/",
                "sealed": sealed(run_time, scored, now=stamped),
                "peak_mw": round(float(peaks[run_time]), 1),
                "mae_mw": (
                    round(float(errors[run_time]), 1)
                    if run_time in errors.index
                    else None
                ),
            }
        )

    return {
        "schema_version": run_daily.SCHEMA_VERSION,
        "generated_at": stamped.isoformat(),
        "region": region.id,
        "runs": runs,
    }


def refresh(
    region: RegionConfig = CAISO_CA,
    hrrr_dir: Path | str = HRRR_DIR,
    now: pd.Timestamp | None = None,
    forecasts: pd.DataFrame | None = None,
    scores: pd.DataFrame | None = None,
) -> list[pd.Timestamp]:
    """Rewrite every run that is still open, then the index.

    Every open run, not a fixed window: `SEAL_AFTER_DAYS` is 4, so at
    most four runs are ever open and the cost is bounded without a second
    rule that could drift out of step with the sealing rule.

    This does **not** rewrite `public/{region}/forecast.json`. That object
    is the newest run as issued, written by `run_daily` and replaced the
    following morning. Its actuals arrive after it has stopped being the
    newest, so patching them in would be work nobody reads. A reader who
    wants a graded run opens it from the index.
    """
    listing = catalogue(region, forecasts, scores, now)
    open_runs = [
        pd.Timestamp(entry["run_time"])
        for entry in listing["runs"]
        if not entry["sealed"]
    ]

    accuracy = run_daily.recent_accuracy()
    for run_time in open_runs:
        write(run_time, region, hrrr_dir, accuracy, now, forecasts, scores)

    storage.write_text(
        index_path(region), json.dumps(listing, indent=2), cache_control=BRIEF
    )
    return open_runs


def _first_written(path: Path | str) -> pd.Timestamp | None:
    """When this run was computed, carried forward from an earlier write.

    `generated_at` describes the forecast, not the object, so a rewrite
    that adds actuals must not move it — `docs/web_handoff.md` tells a
    consumer to read it as the cron's heartbeat. None the first time,
    which lets the caller stamp now.
    """
    if not storage.exists(path):
        return None
    stored = json.loads(storage.read_text(path))
    return pd.Timestamp(stored["generated_at"])


def _scores() -> pd.DataFrame:
    """The scoreboard, or an empty frame shaped like it.

    Built from the declared schema rather than a bare DataFrame, so the
    dtypes are right on day one and a comparison against DAYLIGHT_MW does
    not meet an object column.
    """
    if not storage.exists(grade_daily.STORE_PATH):
        return LIVE_SCORES.empty_table().to_pandas()
    return grade_daily.load()


def verify(
    region: RegionConfig = CAISO_CA,
    now: pd.Timestamp | None = None,
    forecasts: pd.DataFrame | None = None,
    scores: pd.DataFrame | None = None,
) -> dict:
    """Checks on the published archive that the schema cannot express.

    - `missing_objects`: an indexed run whose three objects are not all
      there. Usually a job that died between writing the forecast and
      building the map.
    - `short_runs`: a run object holding fewer than 47 hours, which means
      the weather archive had holes and the page will have gaps in it.
    - `sealed` / `open`: how the archive splits, so a morning where
      nothing sealed is visible rather than inferred.

    Reports. Decides nothing. Raises nothing.
    """
    listing = catalogue(region, forecasts, scores, now)
    missing, short = [], []

    for entry in listing["runs"]:
        prefix = run_prefix(pd.Timestamp(entry["run_time"]), region)
        for name in ("forecast.json", "totals.json", "plants.json.gz"):
            if not storage.exists(storage.child(prefix, name)):
                missing.append(f"{entry['path']}{name}")

        forecast_path = storage.child(prefix, "forecast.json")
        if storage.exists(forecast_path):
            stored = json.loads(storage.read_text(forecast_path))
            if len(stored["valid_times"]) != RUN_HOURS:
                short.append(entry["run_time"])

    return {
        "runs": len(listing["runs"]),
        "sealed": sum(1 for entry in listing["runs"] if entry["sealed"]),
        "open": sum(1 for entry in listing["runs"] if not entry["sealed"]),
        "missing_objects": missing,
        "short_runs": short,
    }


if __name__ == "__main__":
    reopened = refresh()
    static = metadata()
    audit = verify()

    print(f"published archive -> {index_path()}")
    print(f"  {audit['runs']} runs, {audit['sealed']} sealed, {audit['open']} open")
    print(f"  rewrote {len(reopened)} open run(s)")
    for run_time in reopened:
        print(f"    {run_time:%Y-%m-%d %H}z")
    print(f"  metadata {static}")
    if audit["missing_objects"]:
        print(f"  MISSING  {len(audit['missing_objects'])} object(s)")
        for name in audit["missing_objects"][:5]:
            print(f"    {name}")
    if audit["short_runs"]:
        print(f"  SHORT    {len(audit['short_runs'])} run(s) under {RUN_HOURS} hours")
