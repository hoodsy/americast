import numpy as np
import pandas as pd
import pytest

from americast.features.power import (
    HORIZON_ZENITH,
    MAX_ROTATION,
    orient,
    position,
)

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


# --- orient ---------------------------------------------------------


def mounted(tracking: str, tilt: float = 0.0, azimuth: float = 180.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "plant_id": [1],
            "latitude": [LAT],
            "longitude": [LON],
            "tracking": [tracking],
            "tilt": [tilt],
            "azimuth": [azimuth],
        }
    )


def oriented(tracking: str, times: list[str], **kw) -> pd.DataFrame:
    rig = mounted(tracking, **kw)
    return orient(position(hours(times), rig), rig)


def test_fixed_panels_never_move() -> None:
    out = oriented("fixed", ["2024-06-21 15:00", "2024-06-21 20:00"], tilt=22.0, azimuth=170.0)
    assert (out["surface_tilt"] == 22.0).all(), "a fixed panel is fixed"
    assert (out["surface_azimuth"] == 170.0).all()


def test_tracker_follows_the_sun_east_to_west() -> None:
    morning = oriented("single_axis", ["2024-06-21 15:00"])
    evening = oriented("single_axis", ["2024-06-22 02:00"])
    assert morning["surface_azimuth"].iloc[0] < 180.0, "tilted toward the east"
    assert evening["surface_azimuth"].iloc[0] > 180.0, "tilted toward the west"


def test_tracker_lies_flat_at_solar_noon() -> None:
    """A north-south axis has nothing to rotate for when the sun is due south."""
    out = oriented("single_axis", ["2024-06-21 20:00"])
    assert out["surface_tilt"].iloc[0] < 12.0


def test_tracker_stops_at_the_rotation_limit() -> None:
    day = pd.date_range("2024-06-21 12:00", periods=14, freq="1h", tz="UTC")
    out = oriented("single_axis", [str(t) for t in day])
    assert out["surface_tilt"].max() <= MAX_ROTATION + 1e-9, "cannot exceed its limit"
    assert out["surface_tilt"].max() > 30.0, "and does swing a long way"


def test_dual_axis_points_straight_at_the_sun() -> None:
    out = oriented("dual_axis", ["2024-06-21 16:00"])
    assert out["surface_tilt"].iloc[0] == pytest.approx(out["zenith"].iloc[0])
    assert out["surface_azimuth"].iloc[0] == pytest.approx(out["solar_azimuth"].iloc[0])


def test_unknown_tracking_is_treated_as_a_tracker() -> None:
    guess = oriented("unknown", ["2024-06-21 15:00"])
    tracker = oriented("single_axis", ["2024-06-21 15:00"])
    assert guess["surface_tilt"].iloc[0] == pytest.approx(tracker["surface_tilt"].iloc[0])


def test_every_mount_parks_flat_after_dark() -> None:
    """NaN here would spread through the irradiance maths and poison a whole day."""
    for tracking in ["fixed", "single_axis", "dual_axis", "unknown"]:
        out = oriented(tracking, ["2024-06-21 08:00"], tilt=22.0)
        assert out["cos_zenith"].iloc[0] == 0.0, "this hour really is dark"
        assert out["surface_tilt"].iloc[0] == 0.0, f"{tracking} should park flat"


def test_no_nan_survives_a_full_day() -> None:
    day = pd.date_range("2024-06-21", periods=24, freq="1h", tz="UTC")
    for tracking in ["fixed", "single_axis", "dual_axis", "unknown"]:
        out = oriented(tracking, [str(t) for t in day], tilt=22.0)
        assert out["surface_tilt"].notna().all(), f"{tracking} left a NaN tilt"
        assert out["surface_azimuth"].notna().all(), f"{tracking} left a NaN azimuth"


def test_tracker_axis_is_forced_north_south() -> None:
    """The registry's tracker azimuth is not trusted; the axis is always N-S."""
    as_reported = oriented("single_axis", ["2024-06-21 15:00"], azimuth=90.0)
    north_south = oriented("single_axis", ["2024-06-21 15:00"], azimuth=180.0)
    assert as_reported["surface_tilt"].iloc[0] == pytest.approx(
        north_south["surface_tilt"].iloc[0]
    )


def test_unrecognised_tracking_fails_loudly() -> None:
    rig = mounted("bifacial_carport")
    with pytest.raises(ValueError, match="unrecognised tracking types"):
        orient(position(hours(["2024-06-21 20:00"]), rig), rig)
