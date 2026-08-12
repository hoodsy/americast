"""Scoring: the model against the physics and both baselines.

Everything here is computed on the test period, on daylight rows where
every predictor exists, and never on a row any booster was fitted on.

Two conventions run through the module.

**Errors are reported in megawatts, never as percentages.** A percentage
needs a denominator, and every available denominator lies: percent of
capacity flatters the model at midday, percent of actual explodes at
dawn, percent of the mean hides the fact that the mean is not a
quantity anybody dispatches. Megawatts are what the grid runs on.

**Skill is always stated against a named reference.** `1 - error/
reference` is a number between minus infinity and one, and it is
meaningless without saying what it beat. A model can post 40% skill
against smart persistence and 5% against the physics, and reporting
only the first would be a choice, not a measurement.
"""

import numpy as np
import pandas as pd

from americast.model.split import PERIODS as PERIODS_ORDER
from americast.model.split import graded

# Below this the physics predicts near-nothing, and dividing the label
# by it turns a rounding difference into a large ratio. Only `drift`
# needs the cut: it is the one place a ratio appears in a denominator.
_DRIFT_FLOOR_MW = 500.0

# Scored in this fixed order. The model first because it is the subject,
# the physics second because it is the honest bar — an unfitted
# calculation the model must justify itself against — and the two
# persistence baselines last, in the order the build plan names them.
PREDICTORS = {
    "p50_mw": "Model (p50)",
    "fleet_ac_mw": "Physical model",
    "baseline_clear_sky_mw": "Clear-sky persistence",
    "baseline_smart_mw": "Smart persistence",
}

# The reference the build plan's exit criterion names.
REFERENCE = "baseline_clear_sky_mw"

# The lead at and beyond which the exit criterion applies. Below it the
# forecast is nowcasting, where persistence is genuinely hard to beat
# and beating it proves little.
CRITERION_LEAD = 4

LEAD_BUCKETS = [0, 6, 12, 18, 24, 30, 36, 42, 48]

# What the p10-p90 band claims about itself.
NOMINAL_COVERAGE = 0.80


def score(rows: pd.DataFrame) -> pd.DataFrame:
    """MAE, RMSE and bias for every predictor, on one set of rows.

    RMSE beside MAE because they answer different questions. MAE is the
    average miss and is what a daily scoreboard should quote. RMSE
    squares before averaging, so it is dominated by the worst hours —
    two forecasts with equal MAE and very different RMSE differ in
    whether they are steadily mediocre or occasionally terrible, and for
    a grid operator that difference is the whole story.
    """
    scored = []
    for column, name in PREDICTORS.items():
        error = rows[column] - rows["solar_mw"]
        scored.append(
            {
                "predictor": name,
                "column": column,
                "mae": error.abs().mean(),
                "rmse": np.sqrt((error**2).mean()),
                "bias": error.mean(),
                "n": len(rows),
            }
        )
    naive = -rows["solar_mw"]
    scored.append(
        {
            "predictor": "Naive zero",
            "column": None,
            "mae": naive.abs().mean(),
            "rmse": np.sqrt((naive**2).mean()),
            "bias": naive.mean(),
            "n": len(rows),
        }
    )
    return pd.DataFrame(scored)


def skill(rows: pd.DataFrame, reference: str = REFERENCE) -> pd.DataFrame:
    """Skill score of every predictor against one named reference.

    `1 - error/reference_error`. One is a perfect forecast, zero is the
    reference exactly, and negative means the reference won. Reported
    for MAE and RMSE separately, because a model can improve the average
    miss while making the worst hours worse.
    """
    scored = score(rows).set_index("predictor")
    base = scored.loc[PREDICTORS[reference]]
    out = pd.DataFrame(
        {
            "mae": scored["mae"],
            "mae_skill": 1.0 - scored["mae"] / base["mae"],
            "rmse": scored["rmse"],
            "rmse_skill": 1.0 - scored["rmse"] / base["rmse"],
        }
    )
    out.attrs["reference"] = PREDICTORS[reference]
    return out


def by_lead(rows: pd.DataFrame) -> pd.DataFrame:
    """Error per predictor, bucketed by how far ahead the forecast reached.

    **Read this against the 06z warning.** While only the 06z run is
    stored, every lead hour lands on one fixed time of day, so this
    chart and `by_hour` are two views of a single axis. A bucket that
    looks easy is not a lead the model handles well; it is a lead that
    happens to fall at night. `confounded` measures how badly.
    """
    bucketed = rows.copy()
    bucketed["bucket"] = pd.cut(bucketed["lead_hours"], LEAD_BUCKETS)
    return _grouped(bucketed, "bucket")


def by_hour(rows: pd.DataFrame) -> pd.DataFrame:
    """Error per predictor, by hour of the local day."""
    return _grouped(rows, "local_hour")


def coverage(rows: pd.DataFrame) -> dict:
    """Does the p10-p90 band hold the truth 80% of the time?

    A band is a claim, and this is the only place it gets tested. Three
    numbers, because the band can fail in two directions independently:
    a band that is too narrow misses on both sides, while one that is
    correctly wide but shifted misses on one.

    `width` is reported beside them because coverage alone can be bought
    by making the band enormous. A band that spans the whole fleet is
    100% correct and completely useless.
    """
    inside = (rows["solar_mw"] >= rows["p10_mw"]) & (rows["solar_mw"] <= rows["p90_mw"])
    return {
        "coverage": float(inside.mean()),
        "nominal": NOMINAL_COVERAGE,
        "below_p10": float((rows["solar_mw"] < rows["p10_mw"]).mean()),
        "above_p90": float((rows["solar_mw"] > rows["p90_mw"]).mean()),
        "width_mw": float((rows["p90_mw"] - rows["p10_mw"]).mean()),
        "n": len(rows),
    }


def criterion(rows: pd.DataFrame, reference: str = REFERENCE) -> dict:
    """The build plan's definition of done, computed rather than eyeballed.

    "Model beats clear-sky persistence at lead times of 4h+ on the test
    period." Reported per lead bucket as well as overall, because a
    model that wins on average while losing at every long lead has not
    met the criterion in any useful sense — and that is exactly the
    shape a model takes when it is really doing nowcasting.
    """
    long_lead = rows[rows["lead_hours"] >= CRITERION_LEAD]
    model_error = (long_lead["p50_mw"] - long_lead["solar_mw"]).abs().mean()
    base_error = (long_lead[reference] - long_lead["solar_mw"]).abs().mean()

    buckets = by_lead(long_lead)
    model_row = buckets[buckets["predictor"] == PREDICTORS["p50_mw"]].set_index("group")
    base_row = buckets[buckets["predictor"] == PREDICTORS[reference]].set_index("group")
    won = model_row["mae"] < base_row["mae"]

    return {
        "lead_floor": CRITERION_LEAD,
        "reference": PREDICTORS[reference],
        "n": len(long_lead),
        "model_mae": float(model_error),
        "reference_mae": float(base_error),
        "skill": float(1.0 - model_error / base_error),
        "buckets_won": int(won.sum()),
        "buckets_total": len(won),
        "buckets_lost": [str(name) for name, ok in won.items() if not ok],
        "passed": bool(model_error < base_error and won.all()),
    }


def confounded(rows: pd.DataFrame) -> dict:
    """How much lead time and time of day are the same axis in this store.

    With one model run per day, `lead_hours` determines `local_hour`
    exactly, and every statement about lead time is really a statement
    about the clock. This measures it instead of asserting it, so the
    warning on the report carries a number and will shrink honestly when
    the 00z, 12z and 18z runs land.

    `local_hours_per_lead` is the mean number of distinct local hours
    each lead hour reaches. Four run times would push it toward four.

    **The floor is two, not one, and daylight saving is why.** A fixed
    06z run lands at 22:00 Pacific in winter and 23:00 in summer, so
    even a single daily run reaches two local hours per lead. Those two
    are one hour apart and are picked by the calendar rather than by
    anything the forecast knows, so two here still means the axes are
    the same axis. The threshold is set accordingly, and it will only
    read False once the other three runs land.
    """
    per_lead = rows.groupby("lead_hours")["local_hour"].nunique()
    per_hour = rows.groupby("local_hour")["lead_hours"].nunique()
    return {
        "run_hours": sorted(rows["run_time"].dt.hour.unique().tolist()),
        "local_hours_per_lead": float(per_lead.mean()),
        "leads_per_local_hour": float(per_hour.mean()),
        "confounded": bool(per_lead.max() <= 2),
    }


def drift(parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """How far CAISO has moved from the physics, period by period.

    This is the diagnostic that explains the model's bias, and it has to
    be measured rather than assumed, because two very different things
    produce the same symptom.

    `ratio` is what the fleet delivered against its clear-sky ceiling.
    It moves with the weather, and a run of cloudy winters would move it
    without anything being wrong.

    `residual` divides out the weather by comparing against the physical
    model's own answer instead of the ceiling. HRRR's clouds are already
    inside the denominator, so what is left drifts only if the fleet
    itself has changed — capacity the registry does not know about, or
    curtailment behaving differently than it used to.

    A residual that climbs across the three periods means the model is
    being graded on a fleet that is not the one it was fitted to. The
    median is reported beside the mean because curtailment produces a
    long low tail that drags a mean around.
    """
    out = []
    for name in PERIODS_ORDER:
        rows = graded(parts[name])
        lit = rows[rows["fleet_ac_mw"] > _DRIFT_FLOOR_MW]
        ratio = lit["solar_mw"] / lit["fleet_clear_mw"]
        residual = lit["solar_mw"] / lit["fleet_ac_mw"]
        out.append(
            {
                "period": name,
                "start": rows["valid_time"].min(),
                "end": rows["valid_time"].max(),
                "ratio_mean": ratio.mean(),
                "residual_mean": residual.mean(),
                "residual_median": residual.median(),
                "n": len(lit),
            }
        )
    return pd.DataFrame(out)


def days(rows: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """Mean model error per local day, sorted best to worst.

    Local days, so a day is a day: a UTC date cuts a California
    afternoon away from its own morning. Only days holding a full
    complement of daylight hours are ranked — a partial day at the edge
    of the record wins "best" for having fewer hours to be wrong in.
    """
    local_date = rows["valid_time"].dt.tz_convert(timezone).dt.date
    error = (rows["p50_mw"] - rows["solar_mw"]).abs()
    per_day = error.groupby(local_date).agg(["mean", "size"])
    whole = per_day[per_day["size"] >= per_day["size"].max() - 1]
    return whole.sort_values("mean")


def verify(rows: pd.DataFrame) -> dict:
    """Everything the gate claims, in one dict, computed from one slice.

    This is what `__main__` prints and what the golden tests read. It
    exists so that no claim in the report or the docs is typed by hand
    from a number somebody remembers seeing.
    """
    scored = score(rows)
    return {
        "n_rows": len(rows),
        "span": (rows["valid_time"].min(), rows["valid_time"].max()),
        "scores": scored,
        "skill_vs_clear_sky": skill(rows, "baseline_clear_sky_mw"),
        "skill_vs_physics": skill(rows, "fleet_ac_mw"),
        "coverage": coverage(rows),
        "criterion": criterion(rows),
        "confounded": confounded(rows),
    }


def _grouped(rows: pd.DataFrame, key: str) -> pd.DataFrame:
    """Long-form MAE and RMSE per predictor per group.

    Long rather than wide — one row per (group, predictor) — because
    every consumer either plots it, which wants long, or pivots it,
    which is one call. A wide frame with a column per predictor forces
    the opposite conversion on the plotting code.
    """
    out = []
    for column, name in PREDICTORS.items():
        error = rows[column] - rows["solar_mw"]
        grouped = error.groupby(rows[key], observed=True)
        frame = pd.DataFrame(
            {
                "group": grouped.size().index,
                "predictor": name,
                "mae": grouped.apply(lambda e: e.abs().mean()).to_numpy(),
                "rmse": grouped.apply(lambda e: np.sqrt((e**2).mean())).to_numpy(),
                "n": grouped.size().to_numpy(),
            }
        )
        out.append(frame)
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    from americast.features.table import load
    from americast.model import model
    from americast.model.split import split

    parts = split(load())
    models, _ = model.load()
    test = model.attach(models, graded(parts["test"]))

    report = verify(test)
    print(f"test period {report['span'][0]:%Y-%m-%d} to {report['span'][1]:%Y-%m-%d}")
    print(f"{report['n_rows']:,} graded daylight hours\n")
    print(report["scores"].round(1).to_string(index=False))
    print(f"\nskill vs clear-sky persistence:\n{report['skill_vs_clear_sky'].round(3).to_string()}")
    print(f"\ncoverage: {report['coverage']}")
    print(f"criterion: {report['criterion']}")
