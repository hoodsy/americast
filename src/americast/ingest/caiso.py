"""CAISO fuel-mix ingestion: 5-minute utility-scale solar actuals.

The feed behind caiso.com "Today's Outlook": one CSV per local (Pacific)
calendar day, 288 five-minute rows, MW by fuel type. Solar counts
grid-connected utility-scale plants only — rooftop solar never appears
here (it shows up as suppressed demand instead). Night values dip a few
tens of MW negative (panels drawing station service power); they are real
and kept as-is.
"""

import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import gridstatus
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from americast.region import CAISO_CA, RegionConfig
from americast.schemas import CAISO_SOLAR_5MIN

STORE_PATH = Path("data/caiso/solar_5min.parquet")


def fetch_day(day: date) -> pd.DataFrame:
    """One local (Pacific) calendar day of 5-minute solar MW, in UTC."""
    iso = gridstatus.CAISO()
    raw = iso.get_fuel_mix(date=day.isoformat(), verbose=False)
    return _normalize_fuel_mix(raw)


def _normalize_fuel_mix(raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "utc_time": raw["Interval Start"].dt.tz_convert("UTC"),
            "solar_mw": raw["Solar"].astype("float64"),
        }
    )
    return out.sort_values("utc_time").reset_index(drop=True)


def to_hourly(df_5min: pd.DataFrame) -> pd.DataFrame:
    """Derive hourly labels: mean MW per hour, labeled by hour start (UTC).

    n_intervals counts the 5-minute rows behind each hour (12 = complete).
    Hours with no data inside the frame's span appear with n_intervals == 0
    and solar_mw NaN, so gaps are visible rather than silently absent.
    """
    grouped = df_5min.set_index("utc_time")["solar_mw"].resample("1h")
    return pd.DataFrame(
        {"solar_mw": grouped.mean(), "n_intervals": grouped.count()}
    ).reset_index()


def append_to_store(df: pd.DataFrame, path: Path) -> None:
    """Merge rows into the parquet store; idempotent by construction.

    Duplicate utc_times keep the newest fetch. The tz guard exists because
    pyarrow silently assumes naive timestamps are UTC (pinned in
    test_schemas), which would let a local-time bug through unnoticed.
    """
    if df["utc_time"].dt.tz is None:
        raise ValueError("utc_time must be tz-aware")
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df = (
        df.drop_duplicates("utc_time", keep="last")
        .sort_values("utc_time")
        .reset_index(drop=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, schema=CAISO_SOLAR_5MIN, preserve_index=False)
    pq.write_table(table, path)


def backfill(
    start: date,
    end: date,
    region: RegionConfig,
    path: Path,
    sleep_s: float = 1.0,
) -> int:
    """Fetch every local day in [start, end) missing from the store.

    Resumable: completeness is read from the store itself, so interrupting
    and re-running picks up where it left off and never duplicates rows.
    Returns the number of days fetched.
    """
    done = _complete_days(path, region)
    fetched = 0
    day = start
    while day < end:
        if day not in done:
            append_to_store(fetch_day(day), path)
            fetched += 1
            time.sleep(sleep_s)
        day += timedelta(days=1)
    return fetched


def _complete_days(path: Path, region: RegionConfig) -> set[date]:
    """Local calendar days already fully present in the store.

    A day counts as complete with >= 276 intervals: 288 normally, 276 on
    the spring-forward day, 300 on the fall-back day. The most recent
    stored day is never considered complete — it may have been fetched
    mid-day — so backfill always refetches it.
    """
    if not path.exists():
        return set()
    utc_time = pd.read_parquet(path, columns=["utc_time"])["utc_time"]
    local_days = utc_time.dt.tz_convert(region.timezone).dt.date
    counts = local_days.value_counts()
    days = set(counts[counts >= 276].index)
    days.discard(local_days.max())
    return days


def qa_report(df_5min: pd.DataFrame, region: RegionConfig) -> dict:
    """Value-level checks the schema can't express. Reports, decides nothing.

    - missing_intervals: holes in the expected 5-minute grid
    - odd_days: local days whose interval count isn't 288 (DST days are
      expected here: 276 spring-forward, 300 fall-back)
    - large_negatives: below -100 MW (small night negatives are normal)
    - frozen_runs: the same positive value repeated for >= 1 hour
    """
    ts = df_5min["utc_time"]
    expected = pd.date_range(ts.min(), ts.max(), freq="5min", tz="UTC")
    missing = expected.difference(ts)

    local_days = ts.dt.tz_convert(region.timezone).dt.date
    day_counts = local_days.value_counts().sort_index()
    odd_days = day_counts[day_counts != 288]

    run_ids = (df_5min["solar_mw"].diff() != 0).cumsum()
    run_sizes = df_5min.groupby(run_ids)["solar_mw"].transform("size")
    frozen = df_5min[(run_sizes >= 12) & (df_5min["solar_mw"] > 0)]

    return {
        "n_rows": len(df_5min),
        "span": (ts.min(), ts.max()),
        "missing_intervals": missing,
        "odd_days": odd_days,
        "large_negatives": df_5min[df_5min["solar_mw"] < -100],
        "frozen_runs": frozen,
    }


if __name__ == "__main__":
    # End at *tomorrow* in the region's timezone so today's partial day is
    # included; the next run refetches it (most recent day is never
    # considered complete).
    today_local = datetime.now(tz=ZoneInfo(CAISO_CA.timezone)).date()
    end_date = today_local + timedelta(days=1)
    start_date = date(2023, 1, 1)

    num_days = backfill(
        start=start_date, end=end_date, region=CAISO_CA, path=STORE_PATH
    )

    print(f"backfill complete: {num_days} days fetched into {STORE_PATH}")
