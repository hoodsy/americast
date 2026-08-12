"""The split: does it cut on time, and can anything cross the cut?

Everything here runs on a synthetic table small enough to reason about.
Whether the numbers describe California is the golden tests' job.
"""

import numpy as np
import pandas as pd
import pytest

from americast.features.county import ZONES
from americast.model.split import (
    FEATURES,
    TRAIN_END,
    VAL_END,
    design,
    graded,
    split,
    verify,
)

TIMEZONE = "America/Los_Angeles"


def table(start: str = "2023-01-01", days: int = 1200, seed: int = 0) -> pd.DataFrame:
    """A synthetic training table with the real column set and shape.

    One 06z run per day, forecast hours 1-47, a sine-shaped clear-sky
    ceiling peaking at local noon, and a label that is the ceiling times
    a random clearness. Enough structure that a booster can learn
    something, and enough days to span all three periods.

    The default length is not arbitrary: it has to reach past VAL_END,
    or the test period comes back empty and every assertion about it
    passes vacuously.
    """
    rng = np.random.default_rng(seed)
    runs = pd.date_range(start, periods=days, freq="1D", tz="UTC") + pd.Timedelta(hours=6)
    leads = np.arange(1, 48)

    run_time = np.repeat(runs, len(leads))
    lead_hours = np.tile(leads, len(runs))
    valid_time = run_time + pd.to_timedelta(lead_hours, unit="h")

    frame = pd.DataFrame(
        {
            "run_time": run_time,
            "valid_time": valid_time,
            "lead_hours": lead_hours.astype("int32"),
        }
    )
    local = frame["valid_time"].dt.tz_convert(TIMEZONE)
    frame["local_hour"] = local.dt.hour.astype("int32")
    frame["day_of_year"] = local.dt.dayofyear.astype("int32")

    # A day-shaped ceiling: zero at night, peaking at local 13:00.
    angle = np.clip(np.sin((frame["local_hour"] - 6.0) * np.pi / 13.0), 0.0, None)
    season = 1.0 + 0.25 * np.sin((frame["day_of_year"] - 80.0) * 2 * np.pi / 365.0)
    ceiling = 20_000.0 * angle * season
    clearness = np.clip(rng.normal(0.85, 0.15, len(frame)), 0.05, 1.05)

    frame["fleet_cos_zenith"] = angle
    frame["fleet_clear_mw"] = ceiling
    frame["fleet_ac_mw"] = ceiling * clearness
    for index, zone in enumerate(ZONES):
        share = 0.2 + 0.02 * index
        frame[f"{zone}_clear_mw"] = ceiling * share
        frame[f"{zone}_ac_mw"] = ceiling * share * clearness
        frame[f"{zone}_dswrf"] = 900.0 * angle * clearness
        frame[f"{zone}_tcdc"] = (1.0 - clearness) * 100.0
        frame[f"{zone}_t2m"] = 290.0 + 10.0 * angle
        frame[f"{zone}_w10m"] = 3.0 + rng.normal(0.0, 0.5, len(frame))
    for var in ("dswrf", "tcdc", "t2m", "w10m"):
        frame[f"fleet_{var}"] = frame[[f"{zone}_{var}" for zone in ZONES]].mean(axis=1)

    frame["solar_mw"] = frame["fleet_ac_mw"] * rng.normal(0.97, 0.05, len(frame))
    frame["n_intervals"] = np.int32(12)
    frame["baseline_clear_sky_mw"] = ceiling * 0.85
    frame["baseline_smart_mw"] = ceiling * 0.83
    return frame


@pytest.fixture(scope="module")
def synthetic() -> pd.DataFrame:
    return table()


@pytest.fixture(scope="module")
def parts(synthetic: pd.DataFrame) -> dict:
    return split(synthetic)


# --- design ---------------------------------------------------------


def test_design_adds_a_clearness_for_every_zone(synthetic: pd.DataFrame) -> None:
    built = design(synthetic)
    for zone in (*ZONES, "fleet"):
        assert f"{zone}_clearness" in built.columns


def test_clearness_is_the_quotient_it_claims_to_be(synthetic: pd.DataFrame) -> None:
    built = design(synthetic)
    lit = built[built["fleet_clear_mw"] > 0.0]
    expected = lit["fleet_ac_mw"] / lit["fleet_clear_mw"]
    assert lit["fleet_clearness"].to_numpy() == pytest.approx(expected.to_numpy())


def test_a_dark_hour_gets_a_clearness_of_zero_not_an_infinity(
    synthetic: pd.DataFrame,
) -> None:
    """Dividing by a zero ceiling must not produce inf or NaN."""
    built = design(synthetic)
    dark = built[built["fleet_clear_mw"] == 0.0]
    assert len(dark) > 0
    assert np.isfinite(dark["fleet_clearness"]).all()
    assert (dark["fleet_clearness"] == 0.0).all()


# --- the feature list ------------------------------------------------


def test_the_model_never_sees_the_answer() -> None:
    """The label, its metadata, and the target must not be features."""
    for forbidden in ("solar_mw", "n_intervals", "ratio"):
        assert forbidden not in FEATURES


def test_the_model_never_sees_a_baseline() -> None:
    """Held out so the skill score compares two separate forecasts."""
    assert "baseline_clear_sky_mw" not in FEATURES
    assert "baseline_smart_mw" not in FEATURES


def test_the_model_never_sees_a_raw_timestamp() -> None:
    """A tree splitting on a date memorises fleet growth as a calendar."""
    assert "run_time" not in FEATURES
    assert "valid_time" not in FEATURES


def test_every_feature_exists_in_the_table(synthetic: pd.DataFrame) -> None:
    built = design(synthetic)
    missing = set(FEATURES) - set(built.columns)
    assert not missing


# --- graded ----------------------------------------------------------


def test_graded_drops_the_night(synthetic: pd.DataFrame) -> None:
    rows = graded(design(synthetic))
    assert (rows["fleet_clear_mw"] > 0.0).all()
    assert len(rows) < len(synthetic)


def test_graded_drops_a_row_missing_any_predictor(synthetic: pd.DataFrame) -> None:
    holed = design(synthetic).copy()
    holed.loc[holed.index[:5000], "baseline_smart_mw"] = np.nan
    rows = graded(holed)
    assert rows["baseline_smart_mw"].notna().all()
    assert rows["solar_mw"].notna().all()


# --- the split itself ------------------------------------------------


def test_the_three_periods_land_where_the_plan_says(parts: dict) -> None:
    assert parts["train"]["valid_time"].max() < TRAIN_END
    assert parts["validate"]["valid_time"].min() >= TRAIN_END
    assert parts["validate"]["valid_time"].max() < VAL_END
    assert parts["test"]["valid_time"].min() >= VAL_END


def test_no_run_appears_in_two_periods(parts: dict) -> None:
    """The leak the whole gate exists to prevent."""
    seen = [set(part["run_time"]) for part in parts.values()]
    assert not seen[0] & seen[1]
    assert not seen[1] & seen[2]
    assert not seen[0] & seen[2]


def test_no_forecast_hour_appears_in_two_periods(parts: dict) -> None:
    seen = [set(part["valid_time"]) for part in parts.values()]
    assert not seen[0] & seen[1]
    assert not seen[1] & seen[2]
    assert not seen[0] & seen[2]


def test_a_run_straddling_a_boundary_is_dropped_entirely(synthetic: pd.DataFrame) -> None:
    """The last training run forecasts into validation. It must not survive.

    Assigning by valid_time alone would keep those rows in training and
    hand the fit 47 hours of validation labels.
    """
    parts = split(synthetic)
    for name, part in parts.items():
        span_start = part["valid_time"].min()
        assert (part["run_time"] >= span_start - pd.Timedelta(hours=48)).all()
        crossing = part[part["run_time"] < span_start.floor("D")]
        assert crossing.empty or name == "train"


def test_the_straddle_costs_only_a_handful_of_rows(synthetic: pd.DataFrame) -> None:
    kept = sum(len(part) for part in split(synthetic).values())
    assert kept > len(synthetic) - 200, "the boundary cut should be nearly free"


def test_nothing_is_shuffled(parts: dict) -> None:
    for part in parts.values():
        assert part["run_time"].is_monotonic_increasing


# --- verify ----------------------------------------------------------


def test_verify_reports_a_clean_split(parts: dict) -> None:
    audit = verify(parts)
    assert audit["overlap"] == 0
    assert audit["out_of_order"] == 0
    assert all(count > 0 for count in audit["graded_rows"].values())


def test_verify_notices_a_deliberately_poisoned_split(parts: dict) -> None:
    """The check has to be able to fail, or it is decoration."""
    poisoned = dict(parts)
    poisoned["test"] = pd.concat([parts["test"], parts["train"].head(10)])
    assert verify(poisoned)["overlap"] > 0
