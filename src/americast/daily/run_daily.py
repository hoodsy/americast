"""The morning run: fetch one HRRR run, forecast 48 hours, publish.

This is the product. Everything before it was built so that this could
be trusted; this is the part somebody actually opens.

## Why the 06z run

HRRR's 06z run reaches 48 forecast hours, and its f01-f48 span 07:00
UTC to 06:00 UTC two days later. In Pacific time that is exactly
midnight today through 23:00 the day after tomorrow — **two whole local
days, with no partial edges**. No other run hour lands so cleanly on
the days a Californian reader means by "today and tomorrow".

It is also the run the model was trained on. Every stored historical
run is 06z, so lead time and time of day are locked together in the
training data; feeding the model a 12z run at inference would present
it with a lead/hour combination it has never seen. Using 06z in
production keeps inference inside the distribution the fit came from.
The cost is freshness — by 9am Pacific the forecast is eight hours old
— and the fix is pass 3 of the backfill, not a different run today.

## Idempotence

Re-running a day must change nothing. `append` keys on
(run_time, valid_time) and keeps the last write, so a cron that fires
twice, or a manual re-run after a failure, converges rather than
duplicating. The golden test asserts it by running the whole loop
twice and comparing bytes.

## What it does not do

No baselines. Those exist to be beaten during evaluation, and
recomputing clear-sky persistence daily would mean carrying the label
history into the forecast path for no operational purpose. `grade_daily`
has the labels; if a live baseline is ever wanted, that is where it
belongs.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa

from americast import storage
from americast.features.features import fleet
from americast.features.table import one_run
from americast.ingest import hrrr
from americast.model import calibrate
from americast.model import model as boosters
from americast.model.split import design
from americast.region import CAISO_CA, RegionConfig
from americast.schemas import LIVE_FORECASTS

STORE_PATH = storage.key("live/forecasts.parquet")
# Under the public prefix, filed by region id: this is the object a
# browser fetches. The region segment is here from the first day rather
# than added later, because these are public URLs and moving one breaks
# every client that saved it.
JSON_PATH = storage.public(f"{CAISO_CA.id}/forecast.json")
INDEX_PATH = storage.public("regions.json")

# Bumped only when a consumer would break. Additive fields do not.
SCHEMA_VERSION = 1

# The run hour the model was trained on, and the one that spans two
# whole Pacific days. See the module docstring.
RUN_HOUR = 6

# HRRR's 06z run finishes uploading to the AWS archive about ninety
# minutes after its nominal time. A cron that fires earlier finds a
# half-written run and gives up on hours that exist.
ARCHIVE_LAG = pd.Timedelta(hours=2)


def latest(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """The newest 06z run whose archive should be complete.

    Takes `now` rather than reading the clock, so a test can ask what
    the loop would have chosen at any instant and the answer never
    depends on when the suite runs.
    """
    stamp = now or pd.Timestamp(datetime.now(tz=UTC))
    today = stamp.normalize() + pd.Timedelta(hours=RUN_HOUR)
    if stamp >= today + ARCHIVE_LAG:
        return today
    return today - pd.Timedelta(days=1)


def fetch(
    run_time: pd.Timestamp,
    region: RegionConfig = CAISO_CA,
    root: Path | str = hrrr.HRRR_DIR,
) -> pd.DataFrame:
    """This run's weather at every plant, stored on the way past.

    The storing is the point. Until this existed the daily loop built its
    run in memory and dropped it, so the weather archive stopped growing
    the day the backfill finished, and the next retrain would have had a
    hole in it exactly where the live period is.

    It is also what lets the publisher read the run back through
    `api.frames` rather than being handed a frame, so the bucket and the
    local API compute the map from the same file.

    `root` is explicit because `HRRR_DIR` is resolved at import, so a
    caller that moves the data root afterwards would otherwise write into
    the real store — which a test did, once.
    """
    plants = fleet(storage.read_parquet(region.plant_registry_path))
    weather = hrrr.build(run_time, plants)
    if weather.empty:
        raise RuntimeError(f"HRRR archive has nothing for {run_time:%Y-%m-%d %Hz}")
    hrrr.write(weather, root)
    return weather


def forecast(
    run_time: pd.Timestamp,
    models: dict,
    region: RegionConfig = CAISO_CA,
    band: tuple[float, float] | None = None,
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One run, fetched and turned into a 48-hour forecast.

    Returns one row per (run_time, valid_time) carrying the band and
    the two physical columns the frontend draws beneath it. Raises when
    the archive has nothing, because a morning with no forecast is a
    failure to report rather than an empty file to publish.

    `band` re-aims p10 and p90 from recently graded hours; None
    publishes the boosters' own band. The trained band covers 63.6% of
    hours against a promised 80%, because uncertainty here is seasonal
    and the fit inherits whichever season it saw. See
    `model/calibrate.py`.

    Featurization goes through `features.table.one_run`, the same
    function the training table uses. Two copies of that ordering would
    drift apart silently — the columns would still be present, holding
    subtly different numbers — and the model would be fed something it
    was never fitted on.
    """
    plants = fleet(storage.read_parquet(region.plant_registry_path))
    if weather is None:
        weather = hrrr.build(run_time, plants)
    if weather.empty:
        raise RuntimeError(f"HRRR archive has nothing for {run_time:%Y-%m-%d %Hz}")

    featurized = design(one_run(weather, plants, region))
    predicted = boosters.attach(models, featurized)
    columns = [field.name for field in LIVE_FORECASTS]
    forecast_rows = predicted[columns].sort_values("valid_time", ignore_index=True)
    return calibrate.apply(forecast_rows, band)


def append(frame: pd.DataFrame, path: Path | str = STORE_PATH) -> int:
    """Add a run to the store, replacing any earlier copy of it.

    Returns the number of rows the store gained, which is zero on a
    re-run. Keyed on (run_time, valid_time) with the last write
    winning: a re-run after a partial archive should upgrade the day,
    not sit beside it.
    """
    before = 0
    combined = frame
    present = storage.exists(path)
    if present:
        stored = storage.read_parquet(path)
        before = len(stored)
        combined = pd.concat([stored, frame], ignore_index=True)

    combined = combined.drop_duplicates(
        ["run_time", "valid_time"], keep="last"
    ).sort_values(["run_time", "valid_time"], ignore_index=True)

    table = pa.Table.from_pandas(
        combined[[field.name for field in LIVE_FORECASTS]],
        schema=LIVE_FORECASTS,
        preserve_index=False,
    )

    # Nothing new: leave the file alone rather than rewriting identical
    # content. Parquet is not byte-stable across writes, so a rewrite
    # would churn the artifact on every cron tick and make "nothing
    # changed today" indistinguishable from "something did".
    #
    # The comparison is between encoded tables, not DataFrames. The
    # in-memory frame and the parquet round-trip disagree about integer
    # width, so `.equals` on the frames reports a difference that the
    # stored bytes do not have.
    if present and table.equals(
        pa.Table.from_pandas(
            storage.read_parquet(path), schema=LIVE_FORECASTS, preserve_index=False
        ),
        check_metadata=False,
    ):
        return 0

    storage.write_parquet(table, path)
    return len(combined) - before


def load(path: Path | str = STORE_PATH) -> pd.DataFrame:
    """Read the published forecasts."""
    return storage.read_parquet(path)


def to_json(
    frame: pd.DataFrame,
    region: RegionConfig = CAISO_CA,
    accuracy: dict | None = None,
    generated_at: pd.Timestamp | None = None,
) -> dict:
    """The frozen contract the frontend consumes.

    Parallel arrays, not an array of objects, matching the map API's
    shape: `valid_times` and every series are the same length, so a
    client indexes them together and never has to match on a timestamp.

    Timestamps are ISO-8601 in UTC. The reader converts for display —
    the same rule the rest of the project follows, and the reason a
    forecast published at 06z is not secretly a Pacific document.
    """
    latest_run = frame["run_time"].max()
    current = frame[frame["run_time"] == latest_run].sort_values("valid_time")
    stamped = generated_at or pd.Timestamp(datetime.now(tz=UTC))
    peak_row = current.loc[current["p50_mw"].idxmax()]

    return {
        "schema_version": SCHEMA_VERSION,
        "region": _region_block(region),
        "units": "MW",
        "level": "state",
        "validated": region.graded,
        "run_time": latest_run.isoformat(),
        "generated_at": stamped.isoformat(),
        "valid_times": [stamp.isoformat() for stamp in current["valid_time"]],
        "lead_hours": current["lead_hours"].astype(int).tolist(),
        "p50_mw": current["p50_mw"].round(1).tolist(),
        "p10_mw": current["p10_mw"].round(1).tolist(),
        "p90_mw": current["p90_mw"].round(1).tolist(),
        "physical_mw": current["fleet_ac_mw"].round(1).tolist(),
        "clear_sky_mw": current["fleet_clear_mw"].round(1).tolist(),
        "peak": {
            "valid_time": peak_row["valid_time"].isoformat(),
            "p50_mw": round(float(peak_row["p50_mw"]), 1),
        },
        "accuracy": accuracy,
    }


def _region_block(region: RegionConfig) -> dict:
    """How a region identifies itself to a consumer.

    An object rather than a bare string. A national map needs a display
    name, a timezone to render local hours, and — above all — whether
    the forecast is graded, because a region with no public actuals feed
    is a different product from one that publishes its own error.
    """
    return {
        "id": region.id,
        "name": region.name,
        "kind": region.kind,
        "timezone": region.timezone,
        "graded": region.graded,
    }


def index(regions: list[RegionConfig] | None = None, generated_at=None) -> dict:
    """The catalogue: what exists and where to fetch it.

    Lets a client draw a picker or a map without fetching every
    region's payload. One entry today; the shape is what matters.
    """
    stamped = generated_at or pd.Timestamp(datetime.now(tz=UTC))
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": stamped.isoformat(),
        "regions": [
            {
                **_region_block(region),
                "forecast": f"{region.id}/forecast.json",
                "scoreboard": f"{region.id}/scoreboard.json",
                "runs": f"{region.id}/runs.json",
                "plants": f"{region.id}/plants.json.gz",
            }
            for region in (regions or [CAISO_CA])
        ],
    }


def publish_index(path: Path | str = INDEX_PATH) -> None:
    """Write the catalogue beside the regions it lists."""
    storage.write_text(path, json.dumps(index(), indent=2))


def recent_accuracy(days: int = 30) -> dict | None:
    """The rolling error, read from the scoreboard the grader wrote.

    Embedded in the forecast so the number always travels with the
    thing it describes: a reader sees today's curve and how wrong last
    month's curves were, together, on one request. Most published solar
    forecasts do not show this, and it is the part of this project worth
    publishing.

    Returns None before the first grading, which is a real state on day
    one and must not be faked with zeroes.
    """
    from americast.daily import grade_daily

    if not storage.exists(grade_daily.STORE_PATH):
        return None
    summary = grade_daily.rolling(grade_daily.load(), days=days)
    if not summary.get("n"):
        return None
    return {
        "window_days": summary["days"],
        "mae_mw": round(summary["mae_mw"], 1),
        "bias_mw": round(summary["bias_mw"], 1),
        "coverage": round(summary["coverage"], 3),
        "graded_hours": summary["n"],
    }


def publish(frame: pd.DataFrame, path: Path | str = JSON_PATH) -> None:
    """Write the JSON contract."""
    storage.write_text(
        path, json.dumps(to_json(frame, accuracy=recent_accuracy()), indent=2)
    )


def verify(frame: pd.DataFrame) -> dict:
    """Checks on a published forecast, before anyone reads it.

    - `hours`: forecast hours published. 47 is a whole run; fewer means
      the archive had holes and the page will have gaps in it.
    - `band_inverted` / `negative`: the same physical guards the model's
      own verify applies, re-run here because this is the last point
      before publication.
    - `night_not_zero`: a forecast of generation where the ceiling is
      zero.
    - `predicts_the_past`: a valid_time at or before its run.
    """
    return {
        "run_time": frame["run_time"].max(),
        "hours": len(frame),
        "span": (frame["valid_time"].min(), frame["valid_time"].max()),
        "band_inverted": int(
            ((frame["p10_mw"] > frame["p50_mw"]) | (frame["p50_mw"] > frame["p90_mw"])).sum()
        ),
        "negative": int((frame["p10_mw"] < 0.0).sum()),
        "night_not_zero": int(
            ((frame["fleet_clear_mw"] <= 0.0) & (frame["p50_mw"] > 0.0)).sum()
        ),
        "predicts_the_past": int((frame["valid_time"] <= frame["run_time"]).sum()),
        "peak_mw": float(frame["p50_mw"].max()),
    }


if __name__ == "__main__":
    # Imported here, and aliased. `publish` is already a function in this
    # module, so importing the module under its own name would shadow it
    # and `publish(load())` below would stop being callable. The import
    # is lazy for a second reason: `publish` imports this module, so a
    # module-level import would close the cycle.
    from americast.daily import grade_daily
    from americast.daily import publish as archive

    models, meta = boosters.load()
    run_time = latest()

    # Re-aim the band from hours already graded. Nothing here looks
    # forward: yesterday's scoreboard is the only input.
    band = None
    if storage.exists(grade_daily.STORE_PATH):
        band = calibrate.offsets(grade_daily.load())
    weather = fetch(run_time)
    published = forecast(run_time, models, band=band, weather=weather)
    gained = append(published)
    publish(load())
    publish_index()

    audit = verify(published)
    print(f"forecast for {run_time:%Y-%m-%d %H}z -> {STORE_PATH}")
    print(f"  {audit['hours']} hours, {gained} new rows in the store")
    print(f"  span {audit['span'][0]} to {audit['span'][1]}")
    print(f"  peak p50 {audit['peak_mw']:,.0f} MW")
    if band is None:
        print("  band: uncalibrated (not enough graded history yet)")
    else:
        print(f"  band: recalibrated from the last {calibrate.WINDOW_DAYS} days "
              f"({band[0]:+.3f} / {band[1]:+.3f} x clear_mw)")
    print(f"  published {JSON_PATH}")
    print(f"  index     {INDEX_PATH}")

    # The archive. `metadata` belongs here rather than only in the
    # publisher's own driver: the workflow runs this module and
    # grade_daily and nothing else, so without it plants.json.gz is
    # never written and the map has no plants to draw.
    run_objects = archive.write(run_time, accuracy=recent_accuracy())
    static = archive.metadata()
    print(f"  archived  {archive.run_prefix(run_time)}")
    for name, path in run_objects.items():
        print(f"    {name:9} {path}")
    print(f"  metadata  {static}")
