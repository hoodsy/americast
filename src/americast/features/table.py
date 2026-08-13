"""The training table: every stored HRRR run, joined to what happened.

One row per (run_time, valid_time), holding zone weather means, the
physical model's megawatts, calendar columns, and the CAISO label. This
is the only thing the model in Gate 5 reads.

The build is a fold over run files. Each run collapses from 37,824
plant rows to 47 hour rows before anything is kept, so the whole table
is small enough to hold in memory and rebuild from scratch whenever a
feature changes. Nothing here is incremental, on purpose: features get
edited far more often than the weather store grows, and a table that
can only be appended to is a table that quietly holds two definitions
of the same column.
"""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from americast.features.baselines import attach
from americast.features.features import (
    aggregate,
    calendar,
    fleet,
    hourly,
    physical,
)
from americast.ingest.caiso import STORE_PATH as CAISO_STORE
from americast.ingest.caiso import to_hourly
from americast.ingest.hrrr import HRRR_DIR
from americast.region import CAISO_CA, RegionConfig
from americast.schemas import TRAIN_TABLE

STORE_PATH = Path("data/train/table.parquet")


def build(
    region: RegionConfig = CAISO_CA,
    hrrr_dir: Path = HRRR_DIR,
    caiso_path: Path = CAISO_STORE,
) -> pd.DataFrame:
    """Fold every stored HRRR run into one table and join the label.

    Returns the frame; `write` puts it on disk. Runs are read in time
    order so that the output is sorted without a final sort of the
    whole thing.

    A run that is missing forecast hours still contributes the hours it
    has. `hourly` drops each run's last hour, so a run with a hole in
    the middle loses the hour before the hole as well — the average
    that hour needs has no partner. That is the honest answer: the
    alternative is a column that means an instant on some rows and a
    mean on others.
    """
    plants = fleet(pd.read_parquet(region.plant_registry_path))
    label = to_hourly(pd.read_parquet(caiso_path))

    runs = sorted(hrrr_dir.glob("hrrr_*.parquet"))
    if not runs:
        raise FileNotFoundError(f"no HRRR runs in {hrrr_dir}")

    rows = [one_run(pd.read_parquet(path), plants, region) for path in runs]
    stacked = pd.concat(rows, ignore_index=True)
    labelled = _attach_label(stacked, label)
    return attach(labelled, region)


def write(frame: pd.DataFrame, path: Path = STORE_PATH) -> None:
    """Write the table, letting the declared schema check every column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame[[field.name for field in TRAIN_TABLE]]
    table = pa.Table.from_pandas(ordered, schema=TRAIN_TABLE, preserve_index=False)
    pq.write_table(table, path)


def load(path: Path = STORE_PATH) -> pd.DataFrame:
    """Read the table back."""
    return pd.read_parquet(path)


def verify(table: pd.DataFrame) -> dict:
    """Value-level checks the schema cannot express. Reports, decides nothing.

    The schema catches a wrong column, dtype or null. It cannot catch a
    table that is correctly shaped and wrong, which is what this looks
    for.

    - `predicts_the_past`: rows where valid_time <= run_time. Any is a bug.
    - `short_runs`: runs holding fewer than 47 hours, meaning the archive
      had a hole or the fetch was interrupted.
    - `missing_days`: calendar days in the span with no run at all.
    - `unlabelled`: forecast hours the label store has not reached.
    - `physical_mae` / `physical_bias`: how far the unfitted physics sits
      from CAISO over daylight hours, in MW. Bias is the number to watch:
      it moves with SYSTEM_LOSSES and with nothing else in the chain.
    """
    lit = table[(table["fleet_clear_mw"] > 0.0) & table["solar_mw"].notna()]
    error = lit["fleet_ac_mw"] - lit["solar_mw"]
    per_run = table.groupby("run_time").size()
    run_days = pd.to_datetime(table["run_time"].unique()).normalize()
    span = pd.date_range(run_days.min(), run_days.max(), freq="1D")

    return {
        "n_rows": len(table),
        "n_runs": table["run_time"].nunique(),
        "span": (table["valid_time"].min(), table["valid_time"].max()),
        "predicts_the_past": int((table["valid_time"] <= table["run_time"]).sum()),
        "short_runs": per_run[per_run < 47],
        "missing_days": span.difference(run_days),
        "unlabelled": int(table["solar_mw"].isna().sum()),
        "physical_mae": error.abs().mean(),
        "physical_bias": error.mean(),
    }


def one_run(
    weather: pd.DataFrame, plants: pd.DataFrame, region: RegionConfig
) -> pd.DataFrame:
    """One run file -> its forecast hours, aligned and complete.

    Public because the daily loop featurizes exactly one run and must
    do it identically to the training table. Two copies of this
    ordering would drift, and the drift would be invisible: the columns
    would still be there, holding subtly different numbers.

    The order matters. Aggregation comes first because it is far
    cheaper on 37,824 rows than on the table, and averaging is linear,
    so averaging zone sums gives the same answer as summing averaged
    plants. Calendar comes last because it reads valid_time, which
    `hourly` does not change.
    """
    weather_means = aggregate(weather, plants)
    power = physical(weather, plants)
    joined = weather_means.merge(power, on=["run_time", "valid_time"], how="inner")
    return calendar(hourly(joined), region.timezone)


def _attach_label(frame: pd.DataFrame, label: pd.DataFrame) -> pd.DataFrame:
    """Left-join CAISO's hourly actuals onto the forecast hours.

    Left, not inner: a forecast hour the label store has not reached
    yet is still a valid feature row, and Gate 6 will grade it
    tomorrow. It arrives with a null solar_mw, which the schema allows
    and which training must drop explicitly rather than by accident.
    """
    named = label.rename(columns={"utc_time": "valid_time"})
    return frame.merge(named, on="valid_time", how="left")


if __name__ == "__main__":
    built = build()
    write(built)

    labelled = built["solar_mw"].notna().sum()
    graded = built["baseline_clear_sky_mw"].notna().sum()
    print(f"{len(built):,} rows from {built['run_time'].nunique()} runs -> {STORE_PATH}")
    print(f"  {labelled:,} rows carry a label ({labelled / len(built):.1%})")
    print(f"  {graded:,} rows carry baselines ({graded / len(built):.1%})")
    print(f"  span {built['valid_time'].min()} to {built['valid_time'].max()}")
