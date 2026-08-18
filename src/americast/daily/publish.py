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

from datetime import UTC, datetime

import pandas as pd

from americast.daily import run_daily
from americast.features.baselines import DAYLIGHT_MW
from americast.region import CAISO_CA, RegionConfig

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
