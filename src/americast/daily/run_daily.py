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
from americast.model import model as boosters
from americast.model.split import design
from americast.region import CAISO_CA, RegionConfig
from americast.schemas import LIVE_FORECASTS

STORE_PATH = storage.key("live/forecasts.parquet")
# Under the public prefix: this is the object a browser fetches.
JSON_PATH = storage.public("forecast.json")

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


def forecast(
    run_time: pd.Timestamp,
    models: dict,
    region: RegionConfig = CAISO_CA,
) -> pd.DataFrame:
    """One run, fetched and turned into a 48-hour forecast.

    Returns one row per (run_time, valid_time) carrying the band and
    the two physical columns the frontend draws beneath it. Raises when
    the archive has nothing, because a morning with no forecast is a
    failure to report rather than an empty file to publish.

    Featurization goes through `features.table.one_run`, the same
    function the training table uses. Two copies of that ordering would
    drift apart silently — the columns would still be present, holding
    subtly different numbers — and the model would be fed something it
    was never fitted on.
    """
    plants = fleet(pd.read_parquet(region.plant_registry_path))
    weather = hrrr.build(run_time, plants)
    if weather.empty:
        raise RuntimeError(f"HRRR archive has nothing for {run_time:%Y-%m-%d %Hz}")

    featurized = design(one_run(weather, plants, region))
    predicted = boosters.attach(models, featurized)
    columns = [field.name for field in LIVE_FORECASTS]
    return predicted[columns].sort_values("valid_time", ignore_index=True)


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


def to_json(frame: pd.DataFrame) -> dict:
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
    return {
        "run_time": latest_run.isoformat(),
        "valid_times": [stamp.isoformat() for stamp in current["valid_time"]],
        "lead_hours": current["lead_hours"].astype(int).tolist(),
        "p10_mw": current["p10_mw"].round(1).tolist(),
        "p50_mw": current["p50_mw"].round(1).tolist(),
        "p90_mw": current["p90_mw"].round(1).tolist(),
        "physical_mw": current["fleet_ac_mw"].round(1).tolist(),
        "clear_sky_mw": current["fleet_clear_mw"].round(1).tolist(),
        "units": "MW",
        "level": "state",
        "validated": True,
        "region": CAISO_CA.name,
    }


def publish(frame: pd.DataFrame, path: Path | str = JSON_PATH) -> None:
    """Write the JSON contract."""
    storage.write_text(path, json.dumps(to_json(frame), indent=2))


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
    models, meta = boosters.load()
    run_time = latest()

    published = forecast(run_time, models)
    gained = append(published)
    publish(load())

    audit = verify(published)
    print(f"forecast for {run_time:%Y-%m-%d %H}z -> {STORE_PATH}")
    print(f"  {audit['hours']} hours, {gained} new rows in the store")
    print(f"  span {audit['span'][0]} to {audit['span'][1]}")
    print(f"  peak p50 {audit['peak_mw']:,.0f} MW")
    print(f"  published {JSON_PATH}")
