import numpy as np
import pandas as pd
import pytest

from americast.features.baselines import (
    MIN_DAYLIGHT_HOURS,
    PERSISTENCE_DAYS,
    attach,
    clear_sky,
    smart,
)
from americast.region import CAISO_CA

# A flat-topped daylight window in UTC. 15:00-23:00 UTC is 08:00-16:00
# Pacific, so every hour of it lands on the same local day and no test
# here has to reason about a midnight crossing.
DAY_HOURS = range(15, 23)
CEILING_MW = 10_000.0


def day_rows(date: str, delivered: float, ceiling: float = CEILING_MW) -> pd.DataFrame:
    """One local day of history: a flat ceiling and a flat actual."""
    stamps = [pd.Timestamp(f"{date} {h:02d}:00", tz="UTC") for h in DAY_HOURS]
    return pd.DataFrame(
        {
            "valid_time": stamps,
            "fleet_clear_mw": ceiling,
            "solar_mw": delivered,
        }
    )


def history(days: dict[str, float]) -> pd.DataFrame:
    """Past days, each labelled by the run that would forecast it."""
    frames = []
    for date, delivered in days.items():
        rows = day_rows(date, delivered)
        rows["run_time"] = pd.Timestamp(f"{date} 06:00", tz="UTC")
        frames.append(rows)
    out = pd.concat(frames, ignore_index=True)
    out["lead_hours"] = 1
    return out


def forecast(run: str, valid: str, ceiling: float = CEILING_MW) -> pd.DataFrame:
    """The row being predicted, from a run at 06z on the given day."""
    rows = day_rows(valid, delivered=np.nan, ceiling=ceiling)
    rows["run_time"] = pd.Timestamp(f"{run} 06:00", tz="UTC")
    rows["lead_hours"] = 1
    return rows


def table(past: dict[str, float], run: str, valid: str, **kw) -> pd.DataFrame:
    return pd.concat([history(past), forecast(run, valid, **kw)], ignore_index=True)


# --- clear-sky persistence ------------------------------------------


def test_yesterdays_ratio_scales_todays_ceiling() -> None:
    """Half the ceiling yesterday means half the ceiling forecast today."""
    frame = table({"2024-06-10": 5000.0}, run="2024-06-11", valid="2024-06-11")
    out = clear_sky(frame, CAISO_CA)
    predicted = out[frame["solar_mw"].isna()]
    assert predicted.to_numpy() == pytest.approx(5000.0)


def test_a_clear_day_predicts_the_whole_ceiling() -> None:
    frame = table({"2024-06-10": CEILING_MW}, run="2024-06-11", valid="2024-06-11")
    predicted = clear_sky(frame, CAISO_CA)[frame["solar_mw"].isna()]
    assert predicted.to_numpy() == pytest.approx(CEILING_MW)


def test_the_forecast_follows_the_ceiling_not_yesterdays_megawatts() -> None:
    """A shorter day must shrink the forecast, which naive persistence cannot."""
    frame = table(
        {"2024-06-10": 5000.0}, run="2024-06-11", valid="2024-06-11", ceiling=4000.0
    )
    predicted = clear_sky(frame, CAISO_CA)[frame["solar_mw"].isna()]
    assert predicted.to_numpy() == pytest.approx(2000.0), "half of a smaller ceiling"


def test_no_history_means_no_baseline() -> None:
    frame = forecast("2024-06-11", "2024-06-11")
    assert clear_sky(frame, CAISO_CA).isna().all()


def test_a_day_too_short_to_trust_is_refused() -> None:
    """A ratio from two twilight hours would scale a whole 48-hour run."""
    short = day_rows("2024-06-10", delivered=5000.0).head(MIN_DAYLIGHT_HOURS - 1)
    short["run_time"] = pd.Timestamp("2024-06-10 06:00", tz="UTC")
    short["lead_hours"] = 1
    frame = pd.concat(
        [short, forecast("2024-06-11", "2024-06-11")], ignore_index=True
    )
    assert clear_sky(frame, CAISO_CA).isna().all()


# --- leakage --------------------------------------------------------


def test_a_run_never_reads_a_day_it_has_not_seen_the_end_of() -> None:
    """The run is at 06z; that day's afternoon has not happened yet.

    This is the whole leakage question in one test. A 48-hour forecast
    issued at 06z on the 11th must scale by the 10th, never by the 11th
    — even though the 11th's rows sit in the same table.
    """
    frame = table(
        {"2024-06-10": 5000.0, "2024-06-11": 9000.0},
        run="2024-06-11",
        valid="2024-06-12",
    )
    predicted = clear_sky(frame, CAISO_CA)[frame["solar_mw"].isna()]
    assert predicted.to_numpy() == pytest.approx(5000.0), "the 10th, not the 11th"


def test_a_later_run_may_use_the_newer_day() -> None:
    """The mirror image: once the day has ended, it is fair to use."""
    frame = table(
        {"2024-06-10": 5000.0, "2024-06-11": 9000.0},
        run="2024-06-12",
        valid="2024-06-13",
    )
    predicted = clear_sky(frame, CAISO_CA)[frame["solar_mw"].isna()]
    assert predicted.to_numpy() == pytest.approx(9000.0)


# --- smart persistence ----------------------------------------------


def week(delivered: float, start: str = "2024-06-01") -> dict[str, float]:
    days = pd.date_range(start, periods=PERSISTENCE_DAYS, freq="1D")
    return {str(day.date()): delivered for day in days}


def test_smart_averages_the_same_hour_over_the_week() -> None:
    past = week(4000.0)
    frame = table(past, run="2024-06-08", valid="2024-06-08")
    predicted = smart(frame, CAISO_CA)[frame["solar_mw"].isna()]
    assert predicted.to_numpy() == pytest.approx(4000.0)


def test_smart_ignores_the_ceiling_entirely() -> None:
    """It knows nothing about geometry, which is its weakness."""
    past = week(4000.0)
    frame = table(past, run="2024-06-08", valid="2024-06-08", ceiling=1.0)
    predicted = smart(frame, CAISO_CA)[frame["solar_mw"].isna()]
    assert predicted.to_numpy() == pytest.approx(4000.0)


def test_smart_needs_a_full_week() -> None:
    past = week(4000.0)
    past.pop("2024-06-01")
    frame = table(past, run="2024-06-08", valid="2024-06-08")
    assert smart(frame, CAISO_CA)[frame["solar_mw"].isna()].isna().all()


def test_a_gap_breaks_the_window_instead_of_shortening_it() -> None:
    """Seven rows of a rolling mean must be seven consecutive days."""
    past = week(4000.0, start="2024-06-01")
    past.pop("2024-06-04")
    past["2024-06-08"] = 4000.0
    frame = table(past, run="2024-06-09", valid="2024-06-09")
    assert smart(frame, CAISO_CA)[frame["solar_mw"].isna()].isna().all()


def test_smart_tracks_a_changing_week() -> None:
    past = {str(d.date()): mw for d, mw in zip(
        pd.date_range("2024-06-01", periods=7, freq="1D"),
        [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 7000.0],
    )}
    frame = table(past, run="2024-06-08", valid="2024-06-08")
    predicted = smart(frame, CAISO_CA)[frame["solar_mw"].isna()]
    assert predicted.to_numpy() == pytest.approx(4000.0), "the mean of 1000..7000"


# --- attach ---------------------------------------------------------


def test_attach_adds_both_columns_and_keeps_the_rest() -> None:
    frame = table(week(4000.0), run="2024-06-08", valid="2024-06-08")
    out = attach(frame, CAISO_CA)
    assert {"baseline_clear_sky_mw", "baseline_smart_mw"} <= set(out.columns)
    assert len(out) == len(frame)
    assert out["fleet_clear_mw"].equals(frame["fleet_clear_mw"])
