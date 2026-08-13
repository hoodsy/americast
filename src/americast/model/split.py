"""The time split, and the design matrix the boosters read.

Two jobs, kept together because they share one rule: nothing the model
learns from may come from a period it is graded on.

The split is by wall-clock time and never shuffled. Solar forecasting is
a time-series problem wearing a tabular disguise — a random split would
put a Tuesday afternoon in training and the same Tuesday morning in
test, and the resulting score would measure interpolation between
neighbouring hours rather than the ability to forecast a day nobody has
seen.

The design matrix adds the ratio columns a tree cannot build for
itself. A gradient-boosted tree splits on thresholds of single columns,
so it can learn "dswrf above 600" but never "ac_mw is 70% of clear_mw"
— a quotient of two columns is not reachable by any number of axis-
aligned cuts. The physics already computes both halves, so handing over
the quotient costs one multiplication and saves the model from
approximating a division with a staircase.
"""

from itertools import pairwise

import pandas as pd

from americast.features.baselines import DAYLIGHT_MW
from americast.features.county import ZONES

# The boundaries, from the build plan. Cut on UTC instants rather than
# dates: every stored timestamp is UTC, and a Pacific-midnight boundary
# would slice a California afternoon away from its own morning.
TRAIN_END = pd.Timestamp("2025-01-01", tz="UTC")
VAL_END = pd.Timestamp("2025-07-01", tz="UTC")

PERIODS = ("train", "validate", "test")

# What the model is allowed to read. run_time and valid_time are
# deliberately absent: a tree splitting on a raw timestamp learns "after
# March 2024, add 900 MW", which is fleet growth memorised as a date and
# is worth exactly nothing on a future it has not seen. Growth reaches
# the model through the physics columns instead, which are rebuilt from
# the registry and know when each plant switched on.
#
# The two baseline columns are also absent, though they would be legal —
# both are keyed on run_time and read no future. They are held out so
# that the skill score compares two genuinely separate forecasts. A
# model handed smart persistence as a feature would beat smart
# persistence, and would have proved nothing about whether the weather
# forecast carries information.
_WEATHER = ("dswrf", "tcdc", "t2m", "w10m")

FEATURES = (
    *[f"{zone}_{var}" for zone in ZONES for var in _WEATHER],
    *[f"fleet_{var}" for var in _WEATHER],
    *[f"{zone}_{var}" for zone in ZONES for var in ("ac_mw", "clear_mw")],
    "fleet_ac_mw",
    "fleet_clear_mw",
    *[f"{zone}_clearness" for zone in ZONES],
    "fleet_clearness",
    "fleet_cos_zenith",
    "local_hour",
    "day_of_year",
    "lead_hours",
)

# What the model actually predicts, and what it is multiplied by to get
# back to megawatts.
TARGET = "ratio"
SCALE = "fleet_clear_mw"


def design(table: pd.DataFrame) -> pd.DataFrame:
    """Add the clearness ratios and the training target.

    `{zone}_clearness` is the physical model's own answer divided by its
    clear-sky ceiling: how much of the possible sunlight HRRR thinks
    reaches that zone. It is the single most informative number in the
    table and the one a tree cannot derive.

    `ratio` is the label in the same currency — how much of the ceiling
    CAISO actually delivered. It is null wherever the label is null, and
    it is meaningless where the ceiling is near zero, which is why
    `graded` cuts the night away before anything is fitted.

    Division by zero is left as an infinity here rather than being
    filled. Every consumer downstream cuts on `fleet_clear_mw` first, so
    a fill would only hide a night row that escaped the cut.

    The target is added only when a label is present. A live forecast
    has no `solar_mw` — tomorrow has not happened — and it still needs
    every feature column. Training reads `ratio`; inference never does.
    """
    out = table.copy()
    for zone in (*ZONES, "fleet"):
        ceiling = out[f"{zone}_clear_mw"]
        lit = ceiling > 0.0
        out[f"{zone}_clearness"] = (out[f"{zone}_ac_mw"] / ceiling).where(lit, 0.0)
    if "solar_mw" in out.columns:
        out[TARGET] = out["solar_mw"] / out[SCALE]
    return out


def graded(frame: pd.DataFrame) -> pd.DataFrame:
    """Daylight rows carrying a label and both baselines.

    The same cut Gate 4's report uses, for the same reason: every
    predictor must be scored on identical rows. A baseline that is null
    for the first week of the record would otherwise be graded on an
    easier subset than the model, and would look better for having
    skipped the hard days.

    Night is excluded from training as well as from grading. The ratio
    is undefined there, and a fleet that is switched off is a fact of
    astronomy that needs no model.
    """
    columns = ["solar_mw", "baseline_clear_sky_mw", "baseline_smart_mw"]
    complete = frame.dropna(subset=columns)
    return complete[complete[SCALE] > DAYLIGHT_MW].copy()


def split(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Cut the table into train, validate and test, by time only.

    A row joins a period when **both** its run_time and its valid_time
    fall inside it. Assigning by valid_time alone would let the last
    training run — which forecasts 47 hours ahead — carry validation
    labels into the fit, and the model would be graded partly on hours
    it had already been shown. Assigning by run_time alone has the
    mirror fault.

    Rows straddling a boundary are dropped rather than reassigned. They
    cost 47, 36 and 36 rows of 61,899, and every alternative buys those
    rows back with a leak.
    """
    edges = {
        "train": (table["valid_time"].min(), TRAIN_END),
        "validate": (TRAIN_END, VAL_END),
        "test": (VAL_END, table["valid_time"].max() + pd.Timedelta(hours=1)),
    }
    built = design(table)

    parts = {}
    for name, (start, end) in edges.items():
        inside = (
            (built["valid_time"] >= start)
            & (built["valid_time"] < end)
            & (built["run_time"] >= start)
            & (built["run_time"] < end)
        )
        parts[name] = built[inside].copy()
    return parts


def verify(parts: dict[str, pd.DataFrame]) -> dict:
    """Check the split cannot leak, and report what it kept.

    The schema cannot express any of this. These are the four ways a
    time split goes wrong while still looking like a table:

    - `overlap`: any run_time or valid_time shared across two periods.
      Must be zero, or the test score is partly a memory test.
    - `out_of_order`: a period whose rows start before the previous
      period ends.
    - `graded_rows`: how many daylight, fully-labelled rows survive in
      each period — the number the scores are actually computed on.
    - `label_ceiling`: the highest label in train against the highest in
      test. When test exceeds train, a model predicting megawatts
      directly cannot reach the top of its own test period. This is the
      measurement that put the target in ratio space; it is reported
      every run so that the reason stays visible.
    """
    spans = {name: (part["valid_time"].min(), part["valid_time"].max()) for name, part in parts.items()}

    overlap = 0
    for column in ("run_time", "valid_time"):
        seen = [set(part[column]) for part in parts.values()]
        overlap += len(seen[0] & seen[1]) + len(seen[1] & seen[2]) + len(seen[0] & seen[2])

    ordered = [spans[name] for name in PERIODS]
    out_of_order = sum(
        1 for earlier, later in pairwise(ordered) if later[0] <= earlier[1]
    )

    train_max = parts["train"]["solar_mw"].max()
    test_rows = parts["test"]
    above = int((test_rows["solar_mw"] > train_max).sum())

    return {
        "rows": {name: len(part) for name, part in parts.items()},
        "graded_rows": {name: len(graded(part)) for name, part in parts.items()},
        "spans": spans,
        "overlap": overlap,
        "out_of_order": out_of_order,
        "label_ceiling": {
            "train_max_mw": train_max,
            "test_max_mw": test_rows["solar_mw"].max(),
            "test_rows_above_train_max": above,
            "test_share_above": above / len(test_rows) if len(test_rows) else 0.0,
        },
    }
