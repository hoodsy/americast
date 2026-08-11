"""Golden-answer tests over the real training table.

These read the table the build actually wrote and ask whether it
describes California. The unit tests prove each function does what it
says; these prove the whole assembly landed on a real place in a real
year. Skipped where no conforming table is stored.
"""

import pandas as pd
import pyarrow.parquet as pq
import pytest

from americast.features.baselines import DAYLIGHT_MW
from americast.features.county import ZONES
from americast.features.table import STORE_PATH
from americast.schemas import TRAIN_TABLE

# CISO's installed AC capacity in the registry snapshot, in MW. Nothing
# the fleet does can pass it, and the observed CAISO peak is near it.
INSTALLED_MW = 21_520.0


def stored_table() -> pd.DataFrame | None:
    """The stored table if it matches the current schema, else None."""
    if not STORE_PATH.exists():
        return None
    if not pq.read_schema(STORE_PATH).equals(TRAIN_TABLE):
        return None
    return pd.read_parquet(STORE_PATH)


pytestmark = pytest.mark.skipif(
    stored_table() is None, reason="no stored training table matches the schema"
)


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return stored_table()


@pytest.fixture(scope="module")
def daylight(table: pd.DataFrame) -> pd.DataFrame:
    lit = table[table["fleet_clear_mw"] > DAYLIGHT_MW]
    return lit[lit["solar_mw"].notna()]


# --- the table describes forecasts ----------------------------------


def test_no_row_predicts_the_past(table: pd.DataFrame) -> None:
    """A forecast hour at or before its own run is not a forecast."""
    assert (table["valid_time"] > table["run_time"]).all()


def test_lead_hours_matches_the_two_timestamps(table: pd.DataFrame) -> None:
    gap = (table["valid_time"] - table["run_time"]) // pd.Timedelta(hours=1)
    assert (gap == table["lead_hours"]).all()


def test_the_last_hour_of_every_run_was_dropped(table: pd.DataFrame) -> None:
    """`hourly` cannot average the final hour, so it must not appear."""
    assert table["lead_hours"].max() <= 47


def test_one_row_per_run_and_hour(table: pd.DataFrame) -> None:
    assert not table.duplicated(["run_time", "valid_time"]).any()


# --- the numbers are physical ---------------------------------------


def test_no_zone_column_is_missing(table: pd.DataFrame) -> None:
    for zone in ZONES:
        for suffix in ("dswrf", "tcdc", "t2m", "w10m", "ac_mw", "clear_mw"):
            assert f"{zone}_{suffix}" in table.columns


def test_the_fleet_never_beats_its_nameplate(table: pd.DataFrame) -> None:
    assert table["fleet_ac_mw"].max() <= INSTALLED_MW
    assert table["fleet_clear_mw"].max() <= INSTALLED_MW


def test_the_modelled_peak_is_the_right_size(table: pd.DataFrame) -> None:
    """Within sight of CAISO's real peak, or the physics is off."""
    assert 15_000.0 < table["fleet_ac_mw"].max() < INSTALLED_MW


def test_power_is_never_negative(table: pd.DataFrame) -> None:
    assert (table["fleet_ac_mw"] >= 0.0).all()
    assert (table["fleet_clear_mw"] >= 0.0).all()


def test_zones_sum_to_the_fleet(table: pd.DataFrame) -> None:
    zoned = table[[f"{zone}_ac_mw" for zone in ZONES]].sum(axis=1)
    assert zoned.to_numpy() == pytest.approx(table["fleet_ac_mw"].to_numpy(), abs=1e-6)


def test_temperature_is_still_kelvin(table: pd.DataFrame) -> None:
    """A Celsius conversion leaking into storage would show up here."""
    assert 240.0 < table["fleet_t2m"].min() < 290.0
    assert 290.0 < table["fleet_t2m"].max() < 330.0


def test_night_is_dark_everywhere(table: pd.DataFrame) -> None:
    """Power is exactly zero; irradiance is merely negligible.

    cos_zenith is floored at a zenith of 89 degrees rather than 90, so
    these rows include the last strip of twilight. HRRR still reports
    light there — 42 rows of 12,579 carry any at all, and the largest
    is 0.01 W/m². The power columns are held to the stricter test
    because the same cut forces them to zero outright.
    """
    night = table[table["fleet_cos_zenith"] == 0.0]
    assert len(night) > 0
    assert (night["fleet_ac_mw"] == 0.0).all()
    assert (night["fleet_clear_mw"] == 0.0).all()
    assert (night["fleet_dswrf"] < 1.0).all()


def test_the_sun_peaks_in_the_middle_of_the_local_day(table: pd.DataFrame) -> None:
    by_hour = table.groupby("local_hour")["fleet_clear_mw"].mean()
    assert 11 <= by_hour.idxmax() <= 13


def test_summer_out_produces_winter(table: pd.DataFrame) -> None:
    summer = table[table["day_of_year"].between(152, 244)]["fleet_clear_mw"].max()
    winter = table[table["day_of_year"].between(335, 366)]["fleet_clear_mw"].max()
    assert summer > winter


# --- the label joined correctly -------------------------------------


def test_almost_every_forecast_hour_found_its_label(table: pd.DataFrame) -> None:
    assert table["solar_mw"].notna().mean() > 0.95


def test_the_model_tracks_caiso_through_the_day(daylight: pd.DataFrame) -> None:
    """The one check that would catch a whole-table time shift."""
    correlation = daylight["fleet_ac_mw"].corr(daylight["solar_mw"])
    assert correlation > 0.95


def test_the_physical_estimate_is_the_right_size(daylight: pd.DataFrame) -> None:
    """Unfitted physics, so a few percent off is expected; 20% is not."""
    ratio = daylight["fleet_ac_mw"].sum() / daylight["solar_mw"].sum()
    assert 0.85 < ratio < 1.20


# --- the baselines are worth beating --------------------------------


def test_both_baselines_beat_a_naive_zero(daylight: pd.DataFrame) -> None:
    """The bar Gate 5 has to clear, and proof the bar is not trivial."""
    graded = daylight.dropna(subset=["baseline_clear_sky_mw", "baseline_smart_mw"])
    naive = graded["solar_mw"].abs().mean()
    for column in ("baseline_clear_sky_mw", "baseline_smart_mw"):
        mae = (graded[column] - graded["solar_mw"]).abs().mean()
        assert mae < naive / 2.0, f"{column} barely beats predicting nothing"


def test_the_baselines_cover_almost_every_graded_hour(daylight: pd.DataFrame) -> None:
    """Null baselines on the first week are expected; on half is a bug."""
    assert daylight["baseline_clear_sky_mw"].notna().mean() > 0.95
    assert daylight["baseline_smart_mw"].notna().mean() > 0.95


def test_a_baseline_never_reads_its_own_answer(table: pd.DataFrame) -> None:
    """A perfect baseline would mean the future leaked into the past."""
    graded = table.dropna(subset=["solar_mw", "baseline_clear_sky_mw"])
    error = (graded["baseline_clear_sky_mw"] - graded["solar_mw"]).abs()
    assert error.mean() > 100.0, "too good to be honest"


# --- the report renders from the real table -------------------------


def test_the_report_renders(table: pd.DataFrame) -> None:
    """A smoke test over real data: plotly is happy, the page is whole."""
    from americast.features import report

    page = report.render(table)
    assert page.startswith("<!doctype html>")
    assert page.count("plotly-graph-div") == 4, "four figures on the page"


def test_the_report_scores_every_predictor(table: pd.DataFrame) -> None:
    from americast.features import report

    scores = report.summarize(report.graded(table))
    assert set(scores["predictor"]) == {
        "Physical model",
        "Clear-sky persistence",
        "Smart persistence",
        "Naive zero",
    }
    assert (scores["mae"] > 0).all()
    worst = scores.loc[scores["mae"].idxmax(), "predictor"]
    assert worst == "Naive zero", "every predictor must beat predicting nothing"
