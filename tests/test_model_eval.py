"""Scoring arithmetic, checked against numbers worked out by hand.

Small frames with values chosen so the right answer is obvious. If MAE
here is wrong, every number on the report is wrong, and no amount of
real data would make that visible.
"""

import numpy as np
import pandas as pd
import pytest
from test_model_split import table

from americast.model.eval import (
    NOMINAL_COVERAGE,
    PREDICTORS,
    by_hour,
    by_lead,
    confounded,
    coverage,
    criterion,
    days,
    drift,
    score,
    skill,
)
from americast.model.split import split

TIMEZONE = "America/Los_Angeles"


def hand_made(
    actual: list[float],
    model: list[float],
    physics: list[float] | None = None,
    clear_sky: list[float] | None = None,
    smart: list[float] | None = None,
) -> pd.DataFrame:
    """A frame carrying every column the scorers read, and nothing else."""
    n = len(actual)
    stamps = pd.date_range("2025-07-01 14:00", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "run_time": stamps - pd.Timedelta(hours=8),
            "valid_time": stamps,
            "lead_hours": np.arange(1, n + 1, dtype="int32"),
            "local_hour": np.int32(7),
            "solar_mw": actual,
            "p50_mw": model,
            "p10_mw": [m - 500.0 for m in model],
            "p90_mw": [m + 500.0 for m in model],
            "fleet_ac_mw": physics if physics is not None else model,
            "fleet_clear_mw": [10_000.0] * n,
            "baseline_clear_sky_mw": clear_sky if clear_sky is not None else model,
            "baseline_smart_mw": smart if smart is not None else model,
        }
    )


# --- score -----------------------------------------------------------


def test_mae_is_the_mean_absolute_miss() -> None:
    rows = hand_made(actual=[1000.0, 2000.0], model=[1100.0, 1700.0])
    mae = score(rows).set_index("column").loc["p50_mw", "mae"]
    assert mae == pytest.approx(200.0)  # (100 + 300) / 2


def test_rmse_punishes_the_worst_hour_harder_than_mae() -> None:
    """Equal MAE, different RMSE: the whole reason both are reported."""
    steady = hand_made(actual=[1000.0, 1000.0], model=[1200.0, 1200.0])
    spiky = hand_made(actual=[1000.0, 1000.0], model=[1000.0, 1400.0])

    steady_score = score(steady).set_index("column").loc["p50_mw"]
    spiky_score = score(spiky).set_index("column").loc["p50_mw"]

    assert steady_score["mae"] == pytest.approx(spiky_score["mae"])
    assert spiky_score["rmse"] > steady_score["rmse"]


def test_bias_keeps_its_sign() -> None:
    """An over-prediction is positive; MAE cannot tell the two apart."""
    high = hand_made(actual=[1000.0], model=[1500.0])
    low = hand_made(actual=[1000.0], model=[500.0])
    assert score(high).set_index("column").loc["p50_mw", "bias"] == pytest.approx(500.0)
    assert score(low).set_index("column").loc["p50_mw", "bias"] == pytest.approx(-500.0)


def test_naive_zero_is_scored_too() -> None:
    rows = hand_made(actual=[1000.0, 3000.0], model=[1000.0, 3000.0])
    naive = score(rows).set_index("predictor").loc["Naive zero"]
    assert naive["mae"] == pytest.approx(2000.0)
    assert naive["bias"] == pytest.approx(-2000.0)


# --- skill -----------------------------------------------------------


def test_skill_against_itself_is_zero() -> None:
    rows = hand_made(actual=[1000.0, 2000.0], model=[1200.0, 1900.0])
    scored = skill(rows, "baseline_clear_sky_mw")
    assert scored.loc["Clear-sky persistence", "mae_skill"] == pytest.approx(0.0)


def test_halving_the_error_is_half_skill() -> None:
    rows = hand_made(
        actual=[1000.0, 1000.0],
        model=[1100.0, 1100.0],
        clear_sky=[1200.0, 1200.0],
    )
    scored = skill(rows, "baseline_clear_sky_mw")
    assert scored.loc["Model (p50)", "mae_skill"] == pytest.approx(0.5)


def test_a_worse_model_scores_negative_skill() -> None:
    rows = hand_made(
        actual=[1000.0], model=[1400.0], clear_sky=[1100.0]
    )
    scored = skill(rows, "baseline_clear_sky_mw")
    assert scored.loc["Model (p50)", "mae_skill"] < 0.0


# --- coverage --------------------------------------------------------


def test_coverage_counts_what_the_band_contains() -> None:
    """Band is model +/- 500 by construction, so this is countable by eye."""
    rows = hand_made(actual=[1000.0, 1000.0, 1000.0, 1000.0], model=[1000.0, 1200.0, 2000.0, 100.0])
    band = coverage(rows)
    assert band["coverage"] == pytest.approx(0.5)  # rows 1 and 2 land inside
    assert band["above_p90"] == pytest.approx(0.25)  # row 4: actual above 600
    assert band["below_p10"] == pytest.approx(0.25)  # row 3: actual below 1500


def test_coverage_reports_width_so_a_wide_band_cannot_hide() -> None:
    rows = hand_made(actual=[1000.0], model=[1000.0])
    assert coverage(rows)["width_mw"] == pytest.approx(1000.0)
    assert coverage(rows)["nominal"] == NOMINAL_COVERAGE


# --- the exit criterion ----------------------------------------------


def test_the_criterion_passes_when_the_model_wins_everywhere() -> None:
    rows = hand_made(
        actual=[1000.0] * 12,
        model=[1050.0] * 12,
        clear_sky=[1300.0] * 12,
    )
    verdict = criterion(rows)
    assert verdict["passed"]
    assert verdict["skill"] > 0.0
    assert not verdict["buckets_lost"]


def test_the_criterion_fails_when_the_baseline_wins() -> None:
    """It has to be able to fail, or the gate grades itself."""
    rows = hand_made(
        actual=[1000.0] * 12,
        model=[1400.0] * 12,
        clear_sky=[1050.0] * 12,
    )
    verdict = criterion(rows)
    assert not verdict["passed"]
    assert verdict["skill"] < 0.0
    assert verdict["buckets_lost"]


def test_the_criterion_ignores_the_shortest_leads() -> None:
    """Below 4h a forecast is nowcasting, where persistence is hard to beat."""
    rows = hand_made(
        actual=[1000.0] * 12,
        model=[5000.0, 5000.0, 5000.0, *[1000.0] * 9],
        clear_sky=[1200.0] * 12,
    )
    assert criterion(rows)["n"] == 9
    assert criterion(rows)["passed"]


# --- grouped views ---------------------------------------------------


def test_by_lead_scores_every_predictor_in_every_bucket() -> None:
    rows = hand_made(actual=[1000.0] * 12, model=[1100.0] * 12)
    table_out = by_lead(rows)
    assert set(table_out["predictor"]) == set(PREDICTORS.values())
    assert (table_out["mae"] >= 0.0).all()
    assert table_out["n"].sum() == len(rows) * len(PREDICTORS)


def test_by_hour_groups_on_the_local_clock() -> None:
    rows = hand_made(actual=[1000.0] * 6, model=[1100.0] * 6)
    rows.loc[rows.index[:3], "local_hour"] = np.int32(9)
    grouped = by_hour(rows)
    assert set(grouped["group"]) == {7, 9}


# --- the diagnostics -------------------------------------------------


def test_drift_reports_one_row_per_period() -> None:
    parts = split(table())
    drifted = drift(parts)
    assert list(drifted["period"]) == ["train", "validate", "test"]
    assert (drifted["n"] > 0).all()
    assert (drifted["residual_median"] > 0.0).all()


def test_drift_sees_a_fleet_that_grew_underneath_the_model() -> None:
    """Inflate the test period's label and the residual must follow."""
    parts = split(table())
    grown = {name: part.copy() for name, part in parts.items()}
    grown["test"]["solar_mw"] = grown["test"]["solar_mw"] * 1.2

    before = drift(parts).set_index("period").loc["test", "residual_median"]
    after = drift(grown).set_index("period").loc["test", "residual_median"]
    assert after == pytest.approx(before * 1.2, rel=1e-6)


def test_confounded_notices_a_single_daily_run() -> None:
    """With 06z alone, lead time and time of day are one axis.

    Two local hours per lead, not one: a fixed UTC run hour shifts by an
    hour across the daylight-saving boundary, so every lead reaches a
    winter local hour and a summer one.
    """
    parts = split(table())
    audit = confounded(parts["test"])
    assert audit["run_hours"] == [6]
    assert audit["local_hours_per_lead"] == pytest.approx(2.0)
    assert audit["confounded"]


def test_confounded_clears_once_the_other_runs_land() -> None:
    """The flag has to be able to read False, or it never says anything."""
    parts = split(table())
    rows = parts["test"].copy()
    shifted = rows.copy()
    shifted["run_time"] = shifted["run_time"] + pd.Timedelta(hours=6)
    shifted["local_hour"] = (shifted["local_hour"] + 6) % 24
    both = pd.concat([rows, shifted], ignore_index=True)

    assert not confounded(both)["confounded"]
    assert confounded(both)["local_hours_per_lead"] > 2.0


def test_days_ranks_best_to_worst() -> None:
    parts = split(table())
    rows = parts["test"].head(2000).copy()
    rows["p50_mw"] = rows["fleet_ac_mw"]
    ranked = days(rows, TIMEZONE)
    assert ranked["mean"].is_monotonic_increasing
