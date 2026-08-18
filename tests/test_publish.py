"""The publisher: does a run project into the object a browser reads?

No network. Every store is a synthetic frame, so the whole path from
parquet to published dict is exercised without leaving the machine.
"""

import pandas as pd
import pytest

from americast.daily import publish
from americast.schemas import LIVE_SCORES

RUN = pd.Timestamp("2026-08-17 06:00", tz="UTC")
NOW = pd.Timestamp("2026-08-17 09:45", tz="UTC")


def forecasts(run_time: pd.Timestamp = RUN, hours: int = publish.RUN_HOURS):
    """One run's published curve, shaped like the real store."""
    valid = [run_time + pd.Timedelta(hours=lead) for lead in range(1, hours + 1)]
    frame = pd.DataFrame(
        {
            "run_time": [run_time] * hours,
            "valid_time": valid,
            "lead_hours": list(range(1, hours + 1)),
            "p10_mw": [800.0] * hours,
            "p50_mw": [1000.0] * hours,
            "p90_mw": [1200.0] * hours,
            "fleet_ac_mw": [900.0] * hours,
            "fleet_clear_mw": [1500.0] * hours,
        }
    )
    return frame.astype({"lead_hours": "int32"})


def scores(run_time: pd.Timestamp = RUN, hours: int = 3):
    """The first `hours` of that run, graded."""
    issued = forecasts(run_time).head(hours).copy()
    issued["solar_mw"] = 950.0
    issued["error_mw"] = issued["p50_mw"] - issued["solar_mw"]
    issued["inside_band"] = True
    return issued[[field.name for field in LIVE_SCORES]]


def empty_scores():
    return LIVE_SCORES.empty_table().to_pandas()


# --- the curve --------------------------------------------------------


def test_the_curve_carries_the_run_as_issued() -> None:
    built = publish.curve(RUN, forecasts(), empty_scores(), updated_at=NOW)
    assert built["run_time"] == RUN.isoformat()
    assert len(built["valid_times"]) == publish.RUN_HOURS
    assert len(built["p50_mw"]) == publish.RUN_HOURS


def test_an_ungraded_hour_is_none_not_zero() -> None:
    """Zero says the fleet made nothing. None says nobody has checked."""
    built = publish.curve(RUN, forecasts(), scores(hours=3), updated_at=NOW)
    assert built["observed_mw"][:3] == [950.0, 950.0, 950.0]
    assert built["observed_mw"][3] is None


def test_observed_runs_parallel_to_valid_times() -> None:
    built = publish.curve(RUN, forecasts(), scores(hours=3), updated_at=NOW)
    assert len(built["observed_mw"]) == len(built["valid_times"])


def test_error_is_absent_before_anything_is_graded() -> None:
    built = publish.curve(RUN, forecasts(), empty_scores(), updated_at=NOW)
    assert built["error"] is None


def test_error_reports_this_run_not_the_rolling_window() -> None:
    built = publish.curve(RUN, forecasts(), scores(hours=3), updated_at=NOW)
    assert built["error"]["graded_hours"] == 3
    assert built["error"]["mae_mw"] == 50.0
    assert built["error"]["bias_mw"] == 50.0
    assert built["error"]["coverage"] == 1.0


def test_an_unknown_run_is_an_error_not_an_empty_object() -> None:
    with pytest.raises(ValueError, match="no stored forecast"):
        publish.curve(
            pd.Timestamp("2020-01-01 06:00", tz="UTC"),
            forecasts(),
            empty_scores(),
            updated_at=NOW,
        )


def test_generated_at_survives_a_rewrite_and_updated_at_moves() -> None:
    """generated_at describes the forecast; updated_at describes the object."""
    issued = pd.Timestamp("2026-08-17 09:45", tz="UTC")
    later = pd.Timestamp("2026-08-19 09:45", tz="UTC")
    built = publish.curve(
        RUN, forecasts(), scores(), generated_at=issued, updated_at=later
    )
    assert built["generated_at"] == issued.isoformat()
    assert built["updated_at"] == later.isoformat()


# --- sealing ----------------------------------------------------------


def test_a_fresh_partly_graded_run_is_open() -> None:
    assert not publish.sealed(RUN, scores(hours=3), now=NOW)


def test_a_fully_graded_run_seals() -> None:
    assert publish.sealed(RUN, scores(hours=publish.RUN_HOURS), now=NOW)


def test_an_old_run_seals_even_with_hours_that_never_graded() -> None:
    """CAISO does not re-send telemetry, so some hours never become gradeable.

    Without the age backstop those runs are rewritten every morning forever.
    """
    old = NOW + pd.Timedelta(days=publish.SEAL_AFTER_DAYS)
    assert publish.sealed(RUN, scores(hours=3), now=old)


def test_the_header_follows_the_seal() -> None:
    assert publish.caching(sealed=True) == publish.IMMUTABLE
    assert publish.caching(sealed=False) == publish.BRIEF


def test_the_run_key_spells_the_weather_file() -> None:
    assert publish.run_key(RUN) == "20260817T06z"
