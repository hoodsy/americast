"""Golden-answer tests against the real CAISO store.

Values were frozen from CAISO's fuel-mix history on first fetch
(2026-08-06) after inspection: curve shape, night station-service
negatives, and daily capacity factor all check out, and the storm day is
the documented Feb 2024 atmospheric river. CAISO history is immutable, so
any drift here means our ingestion changed behavior — not the data.

These tests need the local store and are skipped where it doesn't exist
(e.g. CI, which has no data/ directory).
"""

import pandas as pd
import pytest

from americast.ingest.caiso import STORE_PATH
from americast.region import CAISO_CA

pytestmark = pytest.mark.skipif(
    not STORE_PATH.exists(), reason="local CAISO store not present"
)


def local_day(df: pd.DataFrame, day: str) -> pd.DataFrame:
    local = df["utc_time"].dt.tz_convert(CAISO_CA.timezone)
    return df[local.dt.date.astype(str) == day]


@pytest.fixture(scope="module")
def store() -> pd.DataFrame:
    return pd.read_parquet(STORE_PATH)


def test_golden_clear_summer_day(store: pd.DataFrame) -> None:
    day = local_day(store, "2024-06-15")
    assert len(day) == 288
    energy_gwh = day["solar_mw"].sum() / 12 / 1000
    assert energy_gwh == pytest.approx(195.19, abs=0.05)
    assert day["solar_mw"].max() == 17865.0


def test_golden_storm_day(store: pd.DataFrame) -> None:
    # Feb 5, 2024: atmospheric river over Southern California.
    day = local_day(store, "2024-02-05")
    assert len(day) == 288
    energy_gwh = day["solar_mw"].sum() / 12 / 1000
    assert energy_gwh == pytest.approx(36.63, abs=0.05)
    assert day["solar_mw"].max() == 6748.0
