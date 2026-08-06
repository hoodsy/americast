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

# California bounding box, slightly padded
LAT = (32.4, 42.1)
LON = (-124.6, -114.1)


@pytest.fixture(scope="module")
def registry() -> pd.DataFrame:
    return pd.read_parquet(REGISTRY_PATH)


def test_no_null_coordinates(registry: pd.DataFrame) -> None:
    assert registry[["latitude", "longitude"]].notna().all().all()


def test_all_plants_inside_california_box(registry: pd.DataFrame) -> None:
    assert registry["latitude"].between(*LAT).all()
    assert registry["longitude"].between(*LON).all()


def test_golden_totals(registry: pd.DataFrame) -> None:
    assert len(registry) == 928
    total_gw = registry["capacity_mw_ac"].sum() / 1000
    assert total_gw == pytest.approx(23.88, abs=0.05)


def test_golden_ciso_share(registry: pd.DataFrame) -> None:
    ciso = registry[registry["balancing_authority"] == "CISO"]
    assert len(ciso) == 788
    assert ciso["capacity_mw_ac"].sum() / 1000 == pytest.approx(21.52, abs=0.05)


def test_single_axis_dominates(registry: pd.DataFrame) -> None:
    by_tracking = registry.groupby("tracking")["capacity_mw_ac"].sum()
    assert by_tracking.idxmax() == "single_axis"
    assert by_tracking["single_axis"] / by_tracking.sum() > 0.8
