"""CAISO curtailment: the sunlight the grid told the fleet not to make.

CAISO publishes a daily wind-and-solar curtailment report. When supply
exceeds what the system can take — mild spring days with low load, high
hydro and full batteries — the operator instructs plants to reduce
output. That generation never happens, so it never appears in the fuel
mix, and the label this project predicts is quietly smaller than the
weather would suggest.

This matters because the physical model estimates **what the panels
could produce**, while the label measures **what the grid accepted**.
The gap between them is not a modelling error. It is an economic
decision no irradiance model can see, and until it is measured it hides
inside every residual as if it were physics.

## Two sources, one series

CAISO changed its publication format in mid-2025, and gridstatus
exposes the two separately. `get_curtailment_legacy` covers 2016-06-30
to 2025-05-31 and reports only the intervals where something was
actually curtailed. `get_curtailment` takes over afterwards and reports
every interval, zeros included. `fetch_day` picks by date and fills the
legacy gaps with zeros, so the stored series has the same meaning
throughout.

## Only the energy column is trusted

CAISO reports MWh and MW for each hour. MWh is the energy withheld
across the interval, so over a one-hour interval it is a mean power,
directly comparable to the hourly label. MW is the instantaneous peak
reduction — and the legacy report leaves it blank on some System-reason
rows while still reporting their energy. Summing categories then yields
an hour whose "peak" sits below its own mean, which is not a rounding
problem but a column that means something different before and after
mid-2025. Only the energy figure is stored.

Both columns also arrive as **strings**. Summing object dtype
concatenates digits rather than adding them, so "17" and "8" become
178 — a number large enough to look like a real megawatt reading and
small enough not to announce itself. Everything is coerced before any
arithmetic.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import gridstatus
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from americast.schemas import CAISO_CURTAILMENT

STORE_PATH = Path("data/caiso/curtailment_hourly.parquet")

# The last day CAISO published in the old format. gridstatus refuses
# `get_curtailment_legacy` after it and returns nothing useful from
# `get_curtailment` before it.
LEGACY_END = date(2025, 5, 31)

FUEL = "Solar"

# No hour can curtail more solar than the fleet can generate. Set above
# installed AC capacity, so it flags an arithmetic failure rather than a
# busy spring afternoon.
FLEET_CEILING_MW = 30_000.0

# Concurrent day fetches. Network-bound on CAISO's side, so this is a
# politeness limit rather than a resource one.
WORKERS = 8
SLEEP_S = 0.3


def fetch_day(day: date) -> pd.DataFrame:
    """One local day of solar curtailment, as hourly mean MW.

    Returns utc_time and curtailed_mw. Wind rows are dropped and every
    curtailment reason is summed away: the question here is how much
    solar the grid refused, not why, and a reason breakdown that nothing
    reads is a column that will rot.
    """
    raw = _raw_day(day)
    if raw.empty:
        return _empty()

    solar = raw[raw["Fuel Type"] == FUEL].copy()
    if solar.empty:
        return _empty()

    energy = _energy_column(solar)

    # The legacy report arrives as strings, and a blank cell where a
    # category was reported with no power reading. Summing object dtype
    # concatenates digits instead of adding them -- "17" + "8" becomes
    # 178 -- which reads as a plausible megawatt figure and is how this
    # first shipped wrong. Coerce before any arithmetic.
    solar["_energy"] = pd.to_numeric(solar[energy], errors="coerce").fillna(0.0)
    solar["utc_time"] = pd.to_datetime(solar["Interval Start"], utc=True)

    # Summed across curtailment type and reason. Economic and
    # self-scheduled reductions happen at the same instant on different
    # plants, so the fleet's total is their sum.
    hourly = solar.groupby("utc_time", as_index=False).agg(
        curtailed_mw=("_energy", "sum"),
    )
    return hourly.astype({"curtailed_mw": "float64"}).sort_values(
        "utc_time", ignore_index=True
    )


def backfill(
    start: date,
    end: date,
    path: Path = STORE_PATH,
    workers: int = WORKERS,
) -> int:
    """Fetch every missing day between start and end, inclusive.

    Resumable in the same way as the fuel-mix ingest: days already in
    the store are skipped, so re-running costs one read and no requests.
    A day CAISO never published stays missing rather than being written
    as zeros — an absent report and a day with no curtailment are
    different claims, and only one of them is safe to add to a label.

    **Threads, not processes.** Each legacy day is a PDF download that
    takes about seventeen seconds, almost all of it waiting on CAISO
    rather than parsing. Serially that is six hours for the record;
    eight threads bring it under an hour. Unlike the HRRR backfill this
    holds no grids, so there is no memory ceiling to respect — the limit
    is politeness to a public server.

    One write at the end, not one per day. The store is small enough to
    rebuild in a moment, and concurrent appends to a single parquet
    would race.
    """
    done = _stored_days(path)
    wanted = [
        start + timedelta(days=offset) for offset in range((end - start).days + 1)
    ]
    todo = [day for day in wanted if day not in done]
    if not todo:
        return 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        fetched = list(pool.map(_polite_day, todo))

    got = [frame for frame in fetched if not frame.empty]
    if got:
        _append(pd.concat(got, ignore_index=True), path)
    return len(got)


def _polite_day(day: date) -> pd.DataFrame:
    """One day, with a pause so a thread pool does not hammer CAISO."""
    frame = fetch_day(day)
    time.sleep(SLEEP_S)
    return frame


def load(path: Path = STORE_PATH) -> pd.DataFrame:
    """Read the stored hourly curtailment series."""
    return pd.read_parquet(path)


def verify(frame: pd.DataFrame) -> dict:
    """Value checks the schema cannot express.

    - `negative`: curtailment below zero. Any is a parsing bug.
    - `implausible`: an hour curtailing more than the fleet could
      produce. The check that catches string columns being concatenated
      instead of summed, which is how this first shipped wrong.
    - `missing_days`: calendar days inside the span with no rows, which
      is normal — CAISO omits days it curtailed nothing — but the count
      should track the season rather than jumping.
    - `by_year`: mean curtailed MW per year. The number the drift
      investigation actually wants.
    """
    stamps = frame["utc_time"]
    days = pd.to_datetime(stamps).dt.tz_convert("UTC").dt.normalize()
    span = pd.date_range(days.min(), days.max(), freq="1D", tz="UTC")

    return {
        "n_rows": len(frame),
        "span": (stamps.min(), stamps.max()),
        "negative": int((frame["curtailed_mw"] < 0.0).sum()),
        "implausible": int((frame["curtailed_mw"] > FLEET_CEILING_MW).sum()),
        "missing_days": len(span.difference(days.unique())),
        "total_gwh": frame["curtailed_mw"].sum() / 1000.0,
        "by_year": frame.groupby(stamps.dt.year)["curtailed_mw"].mean().round(1),
    }


def _raw_day(day: date) -> pd.DataFrame:
    """The right gridstatus call for the date, or an empty frame.

    A missing report raises out of gridstatus rather than returning
    nothing, and CAISO has gaps. One absent day must not end a backfill
    of thirteen hundred.
    """
    caiso = gridstatus.CAISO()
    stamp = pd.Timestamp(day)
    try:
        if day <= LEGACY_END:
            return caiso.get_curtailment_legacy(stamp)
        return caiso.get_curtailment(stamp)
    except Exception:  # noqa: BLE001 - gridstatus raises whatever the source did
        return pd.DataFrame()


def _energy_column(frame: pd.DataFrame) -> str:
    """Locate the MWh column across both report formats.

    Legacy spells it "Curtailment (MWh)"; the current format drops the
    brackets and shouts the unit. Matching on a substring rather than
    listing both spellings means a third format with the same quantity
    keeps working.
    """
    return next(c for c in frame.columns if "MWH" in c.upper().replace(" ", ""))


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "utc_time": pd.Series([], dtype="datetime64[us, UTC]"),
            "curtailed_mw": pd.Series([], dtype="float64"),
        }
    )


def _stored_days(path: Path) -> set[date]:
    """Local days already in the store."""
    if not path.exists():
        return set()
    stored = pd.read_parquet(path, columns=["utc_time"])
    if stored.empty:
        return set()
    return set(stored["utc_time"].dt.tz_convert("UTC").dt.date)


def _append(frame: pd.DataFrame, path: Path) -> None:
    """Append, de-duplicate on utc_time, and re-write under the schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    combined = frame
    if path.exists():
        combined = pd.concat([pd.read_parquet(path), frame], ignore_index=True)
    combined = (
        combined.drop_duplicates("utc_time", keep="last")
        .sort_values("utc_time", ignore_index=True)
    )
    table = pa.Table.from_pandas(
        combined[[f.name for f in CAISO_CURTAILMENT]],
        schema=CAISO_CURTAILMENT,
        preserve_index=False,
    )
    pq.write_table(table, path)


if __name__ == "__main__":
    import sys

    start = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2023, 1, 1)
    end = (
        date.fromisoformat(sys.argv[2])
        if len(sys.argv) > 2
        else datetime.now(tz=UTC).date()
    )

    written = backfill(start, end)
    stored = load()
    audit = verify(stored)
    print(f"{written} new days -> {STORE_PATH}")
    print(f"{audit['n_rows']:,} hourly rows, {audit['total_gwh']:,.0f} GWh curtailed")
    print(f"span {audit['span'][0]} to {audit['span'][1]}")
    print("\nmean curtailed MW per year:")
    print(audit["by_year"].to_string())
