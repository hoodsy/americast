import numpy as np
import pandas as pd
import pytest

from americast.features.power import HORIZON_ZENITH, position

# Kern County, roughly the fleet's centre of mass.
LAT, LON = 35.0, -118.5


def plants(ids=(1,), lat=LAT, lon=LON) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "plant_id": list(ids),
            "latitude": [lat] * len(ids),
            "longitude": [lon] * len(ids),
        }
    )


def hours(times: list[str], plant_ids=(1,)) -> pd.DataFrame:
    stamps = pd.to_datetime(times, utc=True)
    return pd.DataFrame(
        [
            {"valid_time": stamp, "plant_id": pid}
            for stamp in stamps
            for pid in plant_ids
        ]
    )


def test_noon_sun_is_high_and_south() -> None:
    # 2024-06-21 20:00 UTC is 13:00 PDT — near solar noon at this
    # longitude, three days after the solstice.
    out = position(hours(["2024-06-21 20:00"]), plants())
    assert out["zenith"].iloc[0] < 15.0, "midsummer noon sun is nearly overhead"
    assert 150.0 < out["solar_azimuth"].iloc[0] < 210.0, "and roughly due south"
    assert out["cos_zenith"].iloc[0] > 0.95


def test_midnight_sun_is_below_the_horizon() -> None:
    out = position(hours(["2024-06-21 08:00"]), plants())
    assert out["zenith"].iloc[0] > 90.0
    assert out["cos_zenith"].iloc[0] == 0.0, "no beam on a horizontal surface"


def test_cos_zenith_never_goes_negative() -> None:
    """The floor exists so that later stages cannot produce dark power."""
    day = pd.date_range("2024-06-21", periods=24, freq="1h", tz="UTC")
    out = position(hours([str(t) for t in day]), plants())
    assert (out["cos_zenith"] >= 0.0).all()
    assert (out["cos_zenith"] <= 1.0).all()
    night = out[out["zenith"] >= HORIZON_ZENITH]
    assert len(night) > 0, "a June day in California still has a night"
    assert (night["cos_zenith"] == 0.0).all()


def test_sun_rises_in_the_east_and_sets_in_the_west() -> None:
    morning = position(hours(["2024-06-21 14:00"]), plants())
    evening = position(hours(["2024-06-22 02:00"]), plants())
    assert morning["solar_azimuth"].iloc[0] < 130.0, "morning sun in the east"
    assert evening["solar_azimuth"].iloc[0] > 240.0, "evening sun in the west"


def test_longitude_shifts_solar_noon() -> None:
    """Geometry per plant, not one position for the whole fleet.

    The zone centroids span 4.6 degrees of longitude. At a fixed
    instant that must show up as a real difference in sun angle, or
    aggregating per zone would be pointless.
    """
    east = position(hours(["2024-06-21 20:00"]), plants(lat=LAT, lon=-115.7))
    west = position(hours(["2024-06-21 20:00"]), plants(lat=LAT, lon=-120.3))
    assert east["solar_azimuth"].iloc[0] > west["solar_azimuth"].iloc[0], (
        "the eastern plant passed solar noon first, so its sun sits further west"
    )


def test_every_row_keeps_its_own_plant() -> None:
    """Two plants far apart must not be handed the same sun."""
    both = pd.DataFrame(
        {
            "plant_id": [1, 2],
            "latitude": [32.8, 41.0],
            "longitude": [-115.7, -121.0],
        }
    )
    out = position(hours(["2024-06-21 20:00"], plant_ids=(1, 2)), both)
    assert len(out) == 2
    by_plant = out.set_index("plant_id")
    assert by_plant.loc[1, "zenith"] != by_plant.loc[2, "zenith"]
    assert by_plant.loc[1, "zenith"] < by_plant.loc[2, "zenith"], (
        "the southern plant sits closer to the June sun"
    )


def test_row_order_and_columns_are_preserved() -> None:
    frame = hours(["2024-06-21 20:00", "2024-06-21 21:00"], plant_ids=(1, 2))
    frame["dswrf"] = [100.0, 200.0, 300.0, 400.0]
    out = position(frame, plants(ids=(1, 2)))
    assert list(out["plant_id"]) == list(frame["plant_id"])
    assert list(out["dswrf"]) == list(frame["dswrf"]), "carried through untouched"
    assert "latitude" not in out.columns, "coordinates were only borrowed"


def test_unknown_plant_fails_loudly() -> None:
    with pytest.raises(ValueError, match="absent from the registry"):
        position(hours(["2024-06-21 20:00"], plant_ids=(1, 99)), plants(ids=(1,)))


def test_matches_pvlib_directly() -> None:
    """Pins the wiring: right coordinate, right instant, no tz round-trip."""
    import pvlib

    when = pd.Timestamp("2024-06-21 20:00", tz="UTC")
    expected = pvlib.solarposition.get_solarposition(
        pd.DatetimeIndex([when]), LAT, LON
    )
    out = position(hours(["2024-06-21 20:00"]), plants())
    assert out["zenith"].iloc[0] == pytest.approx(
        expected["apparent_zenith"].iloc[0], abs=1e-9
    )
    assert out["cos_zenith"].iloc[0] == pytest.approx(
        np.cos(np.radians(expected["apparent_zenith"].iloc[0])), abs=1e-9
    )
