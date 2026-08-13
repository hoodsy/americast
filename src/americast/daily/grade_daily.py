"""The next morning: join yesterday's forecast to what actually happened.

A forecast nobody scores is a claim nobody can check. This is the half
of the loop that makes the other half evidence.

## Grading never touches the forecast

`forecasts.parquet` is append-only and is never rewritten here. Scores
live in their own file, keyed back to (run_time, valid_time). That
separation is what lets grading be re-run — a label that arrived late,
a CAISO revision, a fixed bug in the scorer — without the published
forecast changing underneath the record of how it did. A scoreboard
that can quietly edit its own questions is not a scoreboard.

## Only complete hours are graded

The label is a mean over twelve 5-minute readings. An hour backed by
three of them is not a worse measurement, it is a different one, and
scoring against it would charge the model for CAISO's telemetry.
`n_intervals` carries the count and `MIN_INTERVALS` is the cut.

## The rolling summary is what the page shows

A single day's MAE is noise — one cloudy afternoon moves it hundreds of
megawatts. Thirty days is long enough to be stable and short enough to
notice a model going stale, which given the fleet's drift is the
failure this project should expect.
"""

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa

from americast import storage
from americast.daily.run_daily import load as load_forecasts
from americast.features.baselines import DAYLIGHT_MW
from americast.ingest.caiso import STORE_PATH as CAISO_STORE
from americast.ingest.caiso import to_hourly
from americast.region import CAISO_CA
from americast.schemas import LIVE_SCORES

STORE_PATH = storage.key("live/scores.parquet")
# Under the public prefix: this is the object a browser fetches.
JSON_PATH = storage.public(f"{CAISO_CA.id}/scoreboard.json")

# Twelve 5-minute readings make a whole hour. Fewer is a different
# measurement, not a worse one.
MIN_INTERVALS = 12

# The window the published summary covers.
ROLLING_DAYS = 30


def grade(
    forecasts: pd.DataFrame, labels: pd.DataFrame, min_intervals: int = MIN_INTERVALS
) -> pd.DataFrame:
    """Score every forecast hour the label store has reached.

    An inner join, deliberately: an ungraded forecast hour is not a
    score of zero, and it must not enter the file at all until CAISO
    has published the hour it predicted. Re-running tomorrow picks it
    up.
    """
    truth = labels.rename(columns={"utc_time": "valid_time"})
    whole = truth[truth["n_intervals"] >= min_intervals]

    joined = forecasts.merge(
        whole[["valid_time", "solar_mw"]], on="valid_time", how="inner"
    )
    joined["error_mw"] = joined["p50_mw"] - joined["solar_mw"]
    joined["inside_band"] = (joined["solar_mw"] >= joined["p10_mw"]) & (
        joined["solar_mw"] <= joined["p90_mw"]
    )
    columns = [field.name for field in LIVE_SCORES]
    return joined[columns].sort_values(
        ["run_time", "valid_time"], ignore_index=True
    )


def append(frame: pd.DataFrame, path: Path | str = STORE_PATH) -> int:
    """Add scores, replacing any earlier grading of the same hours.

    Last write wins on (run_time, valid_time), so re-grading a day
    after a label revision updates it rather than storing both verdicts.
    Returns rows gained, which is zero on a re-run.
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
        combined[[field.name for field in LIVE_SCORES]],
        schema=LIVE_SCORES,
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
            storage.read_parquet(path), schema=LIVE_SCORES, preserve_index=False
        ),
        check_metadata=False,
    ):
        return 0

    storage.write_parquet(table, path)
    return len(combined) - before


def load(path: Path | str = STORE_PATH) -> pd.DataFrame:
    """Read the scoreboard."""
    return storage.read_parquet(path)


def rolling(scores: pd.DataFrame, days: int = ROLLING_DAYS) -> dict:
    """Accuracy over the last `days` of graded hours.

    Daylight hours only, on the same cut every other score in this
    project uses. Night is trivially correct and including it would
    flatter the number by roughly half without saying anything.

    `coverage` is the fraction of hours the p10-p90 band contained. It
    is here rather than in a report because it is the number that
    degrades first when a model goes stale, and the loop should show it
    going wrong before the MAE does.
    """
    if scores.empty:
        return {"n": 0, "days": days}

    cutoff = scores["valid_time"].max() - pd.Timedelta(days=days)
    window = scores[scores["valid_time"] > cutoff]
    lit = window[window["p90_mw"] > DAYLIGHT_MW]
    if lit.empty:
        return {"n": 0, "days": days}

    return {
        "days": days,
        "n": len(lit),
        "span": (lit["valid_time"].min(), lit["valid_time"].max()),
        "mae_mw": float(lit["error_mw"].abs().mean()),
        "rmse_mw": float((lit["error_mw"] ** 2).mean() ** 0.5),
        "bias_mw": float(lit["error_mw"].mean()),
        "coverage": float(lit["inside_band"].mean()),
        "peak_error_mw": float(lit["error_mw"].abs().max()),
    }


def to_json(scores: pd.DataFrame, days: int = ROLLING_DAYS) -> dict:
    """The scoreboard contract: a summary, and a daily series behind it."""
    summary = rolling(scores, days)
    lit = scores[scores["p90_mw"] > DAYLIGHT_MW].copy()
    lit["day"] = lit["valid_time"].dt.tz_convert("UTC").dt.date
    per_day = lit.groupby("day").agg(
        mae_mw=("error_mw", lambda e: e.abs().mean()),
        coverage=("inside_band", "mean"),
        n=("error_mw", "size"),
    )
    return {
        "region": {"id": CAISO_CA.id, "name": CAISO_CA.name},
        "units": "MW",
        "rolling": {
            key: (value.isoformat() if hasattr(value, "isoformat") else value)
            for key, value in summary.items()
            if key != "span"
        },
        "days": [day.isoformat() for day in per_day.index],
        "daily_mae_mw": per_day["mae_mw"].round(1).tolist(),
        "daily_coverage": per_day["coverage"].round(3).tolist(),
        "daily_hours": per_day["n"].astype(int).tolist(),
    }


def publish(scores: pd.DataFrame, path: Path | str = JSON_PATH) -> None:
    """Write the scoreboard contract."""
    storage.write_text(path, json.dumps(to_json(scores), indent=2))


def verify(scores: pd.DataFrame) -> dict:
    """Checks the schema cannot express.

    - `graded_future`: a scored hour whose valid_time is still ahead of
      now. Impossible unless the label store was joined on the wrong key.
    - `perfect`: hours with exactly zero error. A handful is chance; a
      block of them means the forecast was joined to itself.
    - `duplicated`: more than one verdict for a forecast hour.
    """
    return {
        "n_rows": len(scores),
        "span": (scores["valid_time"].min(), scores["valid_time"].max()),
        "duplicated": int(scores.duplicated(["run_time", "valid_time"]).sum()),
        "perfect": int((scores["error_mw"] == 0.0).sum()),
        "coverage": float(scores["inside_band"].mean()),
        "mae_mw": float(scores["error_mw"].abs().mean()),
    }


if __name__ == "__main__":
    forecasts = load_forecasts()
    labels = to_hourly(pd.read_parquet(CAISO_STORE))

    fresh = grade(forecasts, labels)
    gained = append(fresh)
    scores = load()
    publish(scores)

    summary = rolling(scores)
    print(f"graded {len(fresh):,} hours, {gained} new -> {STORE_PATH}")
    if summary["n"]:
        print(f"  last {summary['days']} days, {summary['n']:,} daylight hours")
        print(f"  MAE {summary['mae_mw']:,.0f} MW   bias {summary['bias_mw']:+,.0f} MW")
        print(f"  band coverage {summary['coverage']:.1%}")
    print(f"  published {JSON_PATH}")
