"""Golden-answer tests over the real plant registry.

Values frozen from the EIA-860 2025 Early Release build (2026-08-07).
A new EIA vintage will shift these numbers — updating them then is a
conscious act, not a test fix. Skipped where the registry is absent (CI).
"""

import pandas as pd
import pytest

from americast.ingest.eia860 import REGISTRY_PATH

pytestmark = pytest.mark.skipif(
    not REGISTRY_PATH.exists(), reason="local plant registry not present"
)

# CISO's footprint, slightly padded. Not California: the balancing
# authority reaches into Arizona and Nevada, and the registry follows
# the authority rather than the state line. Yuma sits at -114.6, Clark
# County at -114.9, so a California box would fail on real plants.
LAT = (32.0, 42.1)
LON = (-124.6, -111.5)


@pytest.fixture(scope="module")
def registry() -> pd.DataFrame:
    return pd.read_parquet(REGISTRY_PATH)


def test_no_null_coordinates(registry: pd.DataFrame) -> None:
    assert registry[["latitude", "longitude"]].notna().all().all()


def test_all_plants_inside_the_ciso_box(registry: pd.DataFrame) -> None:
    assert registry["latitude"].between(*LAT).all()
    assert registry["longitude"].between(*LON).all()


def test_the_registry_reaches_outside_california(registry: pd.DataFrame) -> None:
    """The correction Gate 5 forced, asserted on the real file.

    CAISO's territory includes Arizona and Nevada solar, and its
    reported number counts that generation. A registry that stopped at
    the state line made the modelled ceiling smaller than the fleet it
    was meant to bound.
    """
    outside = registry[registry["longitude"] > -114.1]
    assert len(outside) >= 10
    assert outside["capacity_mw_ac"].sum() > 1_500.0


def test_golden_totals(registry: pd.DataFrame) -> None:
    assert len(registry) == 833
    total_gw = registry["capacity_mw_ac"].sum() / 1000
    assert total_gw == pytest.approx(24.23, abs=0.05)


def test_every_plant_is_inside_the_balancing_authority(registry: pd.DataFrame) -> None:
    """No slice needed any more: the filter is the balancing authority."""
    assert (registry["balancing_authority"] == "CISO").all()


def test_the_fleet_can_produce_what_caiso_reports(registry: pd.DataFrame) -> None:
    """The check that would have caught the original bug.

    CAISO's observed peak is 23.2 GW. A registry whose whole nameplate
    sits below that is describing a fleet too small to have produced the
    label, whatever else it gets right.
    """
    assert registry["capacity_mw_ac"].sum() > 23_300.0


def test_single_axis_dominates(registry: pd.DataFrame) -> None:
    by_tracking = registry.groupby("tracking")["capacity_mw_ac"].sum()
    assert by_tracking.idxmax() == "single_axis"
    assert by_tracking["single_axis"] / by_tracking.sum() > 0.8
