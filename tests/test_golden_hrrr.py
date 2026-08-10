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

from americast.features.power import HORIZON_ZENITH
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

    Asserted over the bulk rather than row by row. HRRR does not keep
    the three fields perfectly consistent everywhere: in one run a
    single forecast hour over Ventura, San Diego and Los Angeles, under
    97-100% cloud, has them disagreeing by 10%, and across 54 runs up
    to 3% of daylight rows fall outside a 5% band. The mean is what
    carries the claim — reading VBDSF as a horizontal flux instead
    lands it at 1.02 to 1.59 depending on sun angle, nowhere near the
    0.01 allowed here.
    """
    day = with_zenith[with_zenith["dswrf"] > 20].copy()
    assert len(day) > 1000, "need a meaningful number of daylight rows"
    cos_zenith = np.cos(np.radians(day["zenith"]))
    rebuilt = day["dni"] * cos_zenith + day["dhi"]
    ratio = rebuilt / day["dswrf"]
    assert ratio.mean() == pytest.approx(1.0, abs=0.01)
    assert ratio.std() < 0.02
    assert ratio.between(0.95, 1.05).mean() > 0.90, "the bulk must close"


def test_night_is_dark(with_zenith: pd.DataFrame) -> None:
    """Sun below the horizon means no shortwave of any kind."""
    night = with_zenith[with_zenith["zenith"] > 95]
    assert len(night) > 0
    assert (night["dswrf"] == 0).all()
    assert (night["dni"] == 0).all()
    assert (night["dhi"] == 0).all()


# HRRR's radiation scheme comes apart in the last fraction of a degree
# before sunset, where it divides by a cosine approaching zero. Between
# 89 and 90 degrees of zenith it emits dni up to 3200 W/m² — more than
# twice what the sun sends — alongside slightly negative dhi, in rows
# carrying under 2 W/m² of actual light. Below 89 degrees it is clean:
# peak dni there is 1047.
#
# HORIZON_ZENITH already excludes that band, so the physical checks
# below run on the data as the model actually uses it, and a separate
# test pins the garbage inside the band it belongs to.
NOISE_FLOOR = -5.0

# Irradiance arriving at the top of the atmosphere, at perihelion in
# early January. The familiar 1361 is the yearly mean, and using it
# would flag clear January days as impossible: Earth is nearest the sun
# then, and the real figure swings from 1320 in July to 1414 in January.
SOLAR_CEILING = 1414.0


def test_irradiance_is_physical_where_the_model_uses_it(
    with_zenith: pd.DataFrame,
) -> None:
    """Sound above the horizon cutoff, which is all the model reads.

    The daylight check counts lit rows rather than testing peak
    irradiance against a fixed number. Peak GHI is a fact about the
    season, not about the data being sound: across runs from January
    2023 to June 2024 it swings from 612 to 1117 W/m², and a midwinter
    run that clears 623 is a perfectly clear day. Counting lit rows
    catches what this is actually for — an all-zero or half-decoded
    file — without asserting that it is summer.
    """
    used = with_zenith[with_zenith["zenith"] < HORIZON_ZENITH]
    for column in ["dswrf", "dni", "dhi"]:
        assert used[column].min() >= NOISE_FLOOR, f"{column} went properly negative"
        assert used[column].max() < SOLAR_CEILING, f"{column} beats the sun"
    lit = used[used["dswrf"] > 200]
    assert len(lit) > 1000, "a run with almost no daylight in it is not a run"


def test_terminator_garbage_stays_below_the_horizon_cutoff(
    with_zenith: pd.DataFrame,
) -> None:
    """The reason HORIZON_ZENITH exists, pinned against real data.

    HRRR can emit a dni of 3200 W/m² in the last degree before sunset.
    Two things must stay true: none of it reaches the daylight the
    model reads, and whatever does appear carries no meaningful light.
    If either broke, the power model would put a plant at full output
    in the dark.
    """
    used = with_zenith[with_zenith["zenith"] < HORIZON_ZENITH]
    assert (used["dni"] < SOLAR_CEILING).all(), "impossible dni reached daylight"

    twilight = with_zenith[with_zenith["zenith"] >= HORIZON_ZENITH]
    wild = twilight[twilight["dni"] >= SOLAR_CEILING]
    if wild.empty:
        pytest.skip("no terminator instability in this run")
    assert wild["dswrf"].max() < 20.0, "garbage rows should carry no real light"


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

    # Whether a run holds enough of either sky to compare is weather,
    # not correctness. Seven of the thirty June runs carry fewer than
    # 500 overcast rows, the thinnest 118; several January runs are the
    # other way about, overcast nearly all day with barely a clear hour.
    # Skipping beats failing on a cloudy week or drawing a median from
    # a handful of rows.
    if len(clear) < 100 or len(overcast) < 100:
        pytest.skip(f"{len(clear)} clear and {len(overcast)} overcast rows")

    # Bounds sit well clear of what 54 runs across 18 months actually
    # do. Clear-sky share runs 0.128 to 0.247 — the top of that range
    # is midwinter, where a low sun drives light through more
    # atmosphere and scatters more of it. The clear-to-overcast gap
    # runs 0.144 to about 0.8. Tighter limits would track the weather
    # rather than the physics.
    assert clear.median() < 0.30, "clear sky is beam-dominated"
    assert overcast.median() > clear.median() + 0.10, "cloud raises diffuse share"


def test_lead_hours_never_reaches_back(run: pd.DataFrame) -> None:
    """The core invariant: a forecast hour is always in the future."""
    assert (run["valid_time"] > run["run_time"]).all()
    assert run["lead_hours"].between(1, 48).all()
