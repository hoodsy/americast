"""Golden-answer tests over a real stored HRRR run.

These check the download itself, not our reshaping code: they read a
parquet the backfill actually wrote and confirm the numbers inside it
behave like irradiance. Skipped where no conforming run is stored (CI,
or mid-refetch after a schema change).
"""

import numpy as np
import pandas as pd
import pvlib
import pyarrow.parquet as pq
import pytest

from americast.ingest.hrrr import HRRR_DIR
from americast.region import CAISO_CA
from americast.schemas import HRRR_WEATHER


def stored_run() -> pd.DataFrame | None:
    """The first stored run matching the current schema, or None.

    Schema-checked rather than assumed: after a schema change the old
    files are still on disk until they are refetched, and reading one
    would give silent NaN columns instead of a failure.
    """
    for path in sorted(HRRR_DIR.glob("hrrr_*.parquet")):
        if pq.read_schema(path).equals(HRRR_WEATHER):
            return pd.read_parquet(path)
    return None


pytestmark = pytest.mark.skipif(
    stored_run() is None, reason="no stored HRRR run matches the current schema"
)


@pytest.fixture(scope="module")
def run() -> pd.DataFrame:
    return stored_run()


@pytest.fixture(scope="module")
def with_zenith(run: pd.DataFrame) -> pd.DataFrame:
    """The run joined to plant coordinates and the solar zenith angle.

    pvlib wants one (time, latitude, longitude) triple per row, which is
    exactly what the frame already is once the registry is joined on.
    """
    plants = pd.read_parquet(CAISO_CA.plant_registry_path)
    coords = plants[["plant_id", "latitude", "longitude"]]
    rows = run.merge(coords, on="plant_id", how="inner")
    position = pvlib.solarposition.get_solarposition(
        pd.DatetimeIndex(rows["valid_time"]),
        rows["latitude"].to_numpy(),
        rows["longitude"].to_numpy(),
    )
    rows["zenith"] = position["apparent_zenith"].to_numpy()
    return rows


def test_irradiance_identity_closes(with_zenith: pd.DataFrame) -> None:
    """dswrf = dni * cos(zenith) + dhi.

    This is the test that pins what VBDSF actually is. NOAA names it
    "Visible Beam Downward Solar Flux", but it is broadband and normal
    to the beam — i.e. DNI. If a future HRRR version changes that, this
    fails instead of quietly feeding a wrong beam value to the tracker
    model. Daylight only: at night every term is zero and the identity
    is vacuous.
    """
    day = with_zenith[with_zenith["dswrf"] > 20].copy()
    assert len(day) > 1000, "need a meaningful number of daylight rows"
    cos_zenith = np.cos(np.radians(day["zenith"]))
    rebuilt = day["dni"] * cos_zenith + day["dhi"]
    ratio = rebuilt / day["dswrf"]
    assert ratio.mean() == pytest.approx(1.0, abs=0.005)
    assert ratio.std() < 0.01
    assert ratio.between(0.95, 1.05).all()


def test_night_is_dark(with_zenith: pd.DataFrame) -> None:
    """Sun below the horizon means no shortwave of any kind."""
    night = with_zenith[with_zenith["zenith"] > 95]
    assert len(night) > 0
    assert (night["dswrf"] == 0).all()
    assert (night["dni"] == 0).all()
    assert (night["dhi"] == 0).all()


# HRRR's radiation scheme leaves a little numerical noise right at the
# terminator, where dhi can go slightly negative. Measured across the
# whole June 2024 pilot: 2 rows in 1,336,320, at -2.3 and -0.1 W/m²,
# both at a solar zenith near 89.4 degrees where dswrf is under
# 2 W/m² against a daylight median of 690. Real model output, not our
# arithmetic, and physically negligible — but the bound is pinned here
# so that a real sign error could not hide behind it.
NOISE_FLOOR = -5.0


def test_irradiance_ranges_are_physical(run: pd.DataFrame) -> None:
    """Nothing meaningfully negative, and nothing above the solar constant."""
    for column in ["dswrf", "dni", "dhi"]:
        assert run[column].min() >= NOISE_FLOOR, f"{column} went properly negative"
        assert run[column].max() < 1361.0, f"{column} exceeds the solar constant"
    assert run["dswrf"].max() > 800.0, "no clear midday hour in a whole run"


def test_cloud_shifts_light_from_beam_to_diffuse(run: pd.DataFrame) -> None:
    """Cloud moves light out of the beam and into the diffuse sky.

    Stated as a direction rather than a threshold on purpose. tcdc is
    cloud cover over the entire atmospheric column, which says nothing
    about optical thickness: high thin cirrus reads as 100% cover while
    passing almost the whole beam. Under tcdc > 95 across the pilot
    month, rows with a diffuse share below 0.3 carry a median dni of
    813, and rows above 0.9 carry a median dni of 0 — the same "fully
    overcast" label over two different skies.

    So an absolute floor on the overcast share is not a physical fact,
    it is a fact about which day you read. The gap between clear and
    overcast is the robust claim: clear-sky share sits near 0.17 on
    every run measured, and overcast always sits well above it.
    """
    day = run[run["dswrf"] > 20].copy()
    day["share"] = day["dhi"] / day["dswrf"]
    clear = day[day["tcdc"] < 5]["share"]
    overcast = day[day["tcdc"] > 95]["share"]

    assert len(clear) > 500, "need a clear-sky sample to compare against"
    assert clear.median() < 0.25, "clear sky is beam-dominated"

    # A cloudless run is a fact about California in June, not a defect.
    # Seven of the thirty pilot runs hold fewer than 500 overcast rows,
    # the thinnest carrying 118. Skipping beats either failing on a
    # clear day or drawing a median from a handful of rows.
    if len(overcast) < 100:
        pytest.skip(f"only {len(overcast)} overcast rows in this run")
    assert overcast.median() > clear.median() + 0.15, "cloud raises diffuse share"


def test_lead_hours_never_reaches_back(run: pd.DataFrame) -> None:
    """The Gate 3 invariant: a forecast hour is always in the future."""
    assert (run["valid_time"] > run["run_time"]).all()
    assert run["lead_hours"].between(1, 48).all()
