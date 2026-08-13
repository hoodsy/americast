"""Golden-answer tests over the real weather store.

The unit tests prove the API assembles what it says it assembles.
These prove the numbers coming out of it describe California. Skipped
where no conforming run is stored.
"""

import pandas as pd
import pyarrow.parquet as pq
import pytest

from americast.api import frames as build
from americast.features.power import CLEARNESS_ZENITH
from americast.ingest.hrrr import HRRR_DIR
from americast.region import CAISO_CA
from americast.schemas import HRRR_WEATHER


# The California bounding box every plant must fall inside.
def _weather_gap() -> int:
    """How many registry plants the weather store has never sampled."""
    try:
        from americast.features.features import fleet
        from americast.ingest.hrrr import uncovered_plants
        from americast.region import CAISO_CA

        registry = pd.read_parquet(CAISO_CA.plant_registry_path)
        return len(uncovered_plants(fleet(registry)))
    except (OSError, ValueError, KeyError):
        return 0


# CISO's footprint, not California's: the balancing authority
# reaches into Arizona and Nevada, and so does the fleet.
LAT_RANGE, LON_RANGE = (32.0, 42.1), (-124.5, -111.5)

# CISO's installed AC capacity in the registry snapshot, in MW.
INSTALLED_MW = 21_520.0


def a_stored_run() -> pd.Timestamp | None:
    for path in sorted(HRRR_DIR.glob("hrrr_*.parquet")):
        if pq.read_schema(path).equals(HRRR_WEATHER):
            return build._run_time(path)
    return None


pytestmark = pytest.mark.skipif(
    a_stored_run() is None, reason="no stored HRRR run matches the current schema"
)


@pytest.fixture(scope="module")
def run_time() -> pd.Timestamp:
    return a_stored_run()


@pytest.fixture(scope="module")
def plants() -> list:
    return build.plants(CAISO_CA).plants


@pytest.fixture(scope="module")
def payload(run_time):
    return build.frames(run_time, HRRR_DIR, CAISO_CA)


@pytest.fixture(scope="module")
def curves(run_time):
    return build.totals(run_time, HRRR_DIR, CAISO_CA)


# --- the fleet is where it should be --------------------------------


def test_every_plant_sits_inside_the_ciso_footprint(plants) -> None:
    for plant in plants:
        assert LAT_RANGE[0] < plant.latitude < LAT_RANGE[1], plant.name
        assert LON_RANGE[0] < plant.longitude < LON_RANGE[1], plant.name


def test_the_fleet_is_the_expected_size(plants) -> None:
    total = sum(plant.capacity_mw_ac for plant in plants)
    assert len(plants) > 700
    assert 23_000 < total < 26_000, "the CISO fleet is about 24.2 GW"


def test_every_plant_carries_more_panel_than_inverter(plants) -> None:
    """The loading ratio is why clipping exists at all."""
    ratios = [p.dc_capacity_mw / p.capacity_mw_ac for p in plants]
    assert min(ratios) > 0.8
    assert sum(ratios) / len(ratios) > 1.0


# --- a run describes a real day -------------------------------------


def test_a_run_covers_47_hours(payload) -> None:
    """48 forecast hours in, 47 hour means out."""
    assert len(payload.valid_times) == 47
    assert payload.valid_times[0] == payload.run_time + pd.Timedelta(hours=1)


def test_the_hours_are_contiguous(payload) -> None:
    gaps = pd.Series(payload.valid_times).diff().dropna().unique()
    assert list(gaps) == [pd.Timedelta(hours=1)]


@pytest.mark.skipif(
    _weather_gap() > 0,
    reason=f"{_weather_gap()} registry plants have no weather yet; "
    "the HRRR store needs refetching after the balancing-authority change",
)
def test_every_plant_appears_once(payload, plants) -> None:
    """A frontend joins these on plant_id and must find no orphans.

    Skipped, not deleted, while the weather store predates the registry.
    Adding Arizona and Nevada to the fleet gave 46 plants a coordinate
    no stored run ever sampled, so `/plants` advertises plants the
    per-run endpoint cannot serve. The skip clears itself the moment the
    store covers the registry again.
    """
    served = [series.plant_id for series in payload.plants]
    assert len(served) == len(set(served))
    assert set(served) == {plant.plant_id for plant in plants}


def test_no_plant_generates_in_the_dark(payload) -> None:
    for series in payload.plants:
        assert min(series.mw) >= 0.0
        assert min(series.mw) == 0.0, "every plant has a night in 47 hours"


def test_no_plant_beats_its_own_nameplate(payload, plants) -> None:
    limits = {plant.plant_id: plant.capacity_mw_ac for plant in plants}
    for series in payload.plants:
        assert max(series.mw) <= limits[series.plant_id] + 0.1


# --- clearness behaves ----------------------------------------------


def test_clearness_is_reported_only_when_the_sun_is_up(payload) -> None:
    every = [c for series in payload.plants for c in series.clearness]
    reported = [c for c in every if c is not None]
    share = len(reported) / len(every)
    assert 0.2 < share < 0.7, f"{share:.0%} reported over a 47-hour run"


def test_a_reported_clearness_is_a_believable_ratio(payload) -> None:
    """The calibration and the elevation floor, checked on real data."""
    reported = [c for s in payload.plants for c in s.clearness if c is not None]
    series = pd.Series(reported)
    assert series.min() >= 0.0
    assert series.median() < 1.15, "a typical hour is not far above its ceiling"
    assert series.quantile(0.99) < 2.0, "the horizon blowup is gone"


def test_the_elevation_floor_is_what_silences_the_rest() -> None:
    """Guards the constant the null rule depends on."""
    assert 60.0 < CLEARNESS_ZENITH < 85.0


# --- the aggregations add up ----------------------------------------


def test_the_state_curve_is_the_right_size(curves) -> None:
    state = next(level for level in curves.levels if level.level == "state")
    assert max(state.mw) < INSTALLED_MW
    assert max(state.mw) > 5_000.0, "a real day reaches a real number"


def test_zones_and_counties_each_sum_to_the_state(curves) -> None:
    state = next(level for level in curves.levels if level.level == "state")
    for kind in ("zone", "county"):
        parts = [level.mw for level in curves.levels if level.level == kind]
        summed = [sum(hour) for hour in zip(*parts)]
        assert summed == pytest.approx(state.mw, abs=1.0), f"{kind} does not add up"


def test_every_level_stays_under_its_ceiling_on_average(curves) -> None:
    """Cloud enhancement lifts single hours; a whole day should not."""
    for level in curves.levels:
        if sum(level.clear_mw) > 0:
            assert sum(level.mw) <= sum(level.clear_mw) * 1.05, level.name


def test_only_the_state_claims_to_be_graded(curves) -> None:
    graded = [level.name for level in curves.levels if level.validated]
    assert graded == [CAISO_CA.iso]


def test_every_zone_is_present(curves) -> None:
    """Read from ZONES rather than listed, so a new zone cannot be missed.

    A hard-coded set of five silently stopped describing the fleet when
    Arizona's `sonoran` was added, and a test that lists what it expects
    is a test that has to be remembered.
    """
    from americast.features.county import ZONES

    zones = {level.name for level in curves.levels if level.level == "zone"}
    assert zones == set(ZONES)


def test_the_counties_are_plausible(curves) -> None:
    counties = [level for level in curves.levels if level.level == "county"]
    assert 30 < len(counties) < 60
    biggest = max(counties, key=lambda level: max(level.mw))
    assert biggest.name in {"Kern", "Riverside"}, "the two largest by capacity"
