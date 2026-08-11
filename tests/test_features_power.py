import numpy as np
import pandas as pd
import pytest

from americast.features.power import (
    GAMMA_PDC,
    HORIZON_ZENITH,
    KELVIN_ZERO,
    MAX_ROTATION,
    generate,
    heat,
    orient,
    poa,
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


def mounted(
    tracking: str,
    tilt: float = 0.0,
    azimuth: float = 180.0,
    dc_capacity_mw: float = 127.5,
    capacity_mw_ac: float = 100.0,
    operating_date: str = "2020-01-01",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "plant_id": [1],
            "latitude": [LAT],
            "longitude": [LON],
            "tracking": [tracking],
            "tilt": [tilt],
            "azimuth": [azimuth],
            "dc_capacity_mw": [dc_capacity_mw],
            "capacity_mw_ac": [capacity_mw_ac],
            "operating_date": [pd.Timestamp(operating_date, tz="UTC")],
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


def test_small_reported_tilt_leans_the_axis() -> None:
    """Below the split, the number is what it claims to be: an axis tilt."""
    flat = oriented("single_axis", ["2024-06-21 20:00"], tilt=0.0)
    leaning = oriented("single_axis", ["2024-06-21 20:00"], tilt=20.0)
    assert leaning["surface_tilt"].iloc[0] != pytest.approx(
        flat["surface_tilt"].iloc[0]
    ), "a tilted axis presents a different plane at noon"
    assert leaning["surface_tilt"].iloc[0] == pytest.approx(20.0, abs=2.0)


def test_large_reported_tilt_is_a_rotation_limit() -> None:
    """Above the split, the number caps the swing instead of leaning the axis."""
    day = pd.date_range("2024-06-21 12:00", periods=14, freq="1h", tz="UTC")
    times = [str(t) for t in day]
    tight = oriented("single_axis", times, tilt=45.0)
    loose = oriented("single_axis", times, tilt=0.0)
    assert tight["surface_tilt"].max() == pytest.approx(45.0, abs=1e-6)
    assert loose["surface_tilt"].max() == pytest.approx(MAX_ROTATION, abs=1e-6)
    assert tight["surface_tilt"].max() < loose["surface_tilt"].max()


def test_unreported_tilt_falls_back_to_the_fleet_limit() -> None:
    day = pd.date_range("2024-06-21 12:00", periods=14, freq="1h", tz="UTC")
    out = oriented("single_axis", [str(t) for t in day], tilt=0.0)
    assert out["surface_tilt"].max() == pytest.approx(MAX_ROTATION, abs=1e-6)


def test_mixed_fleet_keeps_each_plant_on_its_own_parameters() -> None:
    """Grouping must not leak one plant's rotation limit onto another."""
    rig = pd.DataFrame(
        {
            "plant_id": [1, 2, 3],
            "latitude": [LAT] * 3,
            "longitude": [LON] * 3,
            "tracking": ["single_axis"] * 3,
            "tilt": [45.0, 0.0, 20.0],
            "azimuth": [180.0] * 3,
        }
    )
    day = pd.date_range("2024-06-21 12:00", periods=14, freq="1h", tz="UTC")
    frame = hours([str(t) for t in day], plant_ids=(1, 2, 3))
    out = orient(position(frame, rig), rig).set_index("plant_id")
    assert out.loc[1, "surface_tilt"].max() == pytest.approx(45.0, abs=1e-6)
    assert out.loc[2, "surface_tilt"].max() == pytest.approx(MAX_ROTATION, abs=1e-6)
    assert out.loc[3, "surface_tilt"].max() > 45.0, "a leaning axis is not capped at 45"


# --- poa ------------------------------------------------------------


def lit_frame(tracking: str, times: list[str], dni, dhi, dswrf, **kw) -> pd.DataFrame:
    rig = mounted(tracking, **kw)
    frame = position(hours(times), rig)
    frame["dni"], frame["dhi"], frame["dswrf"] = dni, dhi, dswrf
    return poa(orient(frame, rig))


def test_poa_is_zero_after_dark() -> None:
    """Even handed absurd irradiance, a dark hour must collect nothing."""
    out = lit_frame("single_axis", ["2024-06-21 08:00"], 3200.0, 500.0, 900.0)
    assert out["cos_zenith"].iloc[0] == 0.0, "this hour really is dark"
    assert out["poa"].iloc[0] == 0.0, "HRRR terminator garbage must not light a panel"


def test_tracker_beats_flat_ground_in_the_morning() -> None:
    """The whole reason 18.2 of the fleet's 21.5 GW tracks."""
    out = lit_frame("single_axis", ["2024-06-21 14:00"], 800.0, 100.0, 350.0)
    assert out["poa"].iloc[0] > 350.0 * 1.2, "a tilted panel collects far more"


def test_flat_panel_at_noon_matches_horizontal_irradiance() -> None:
    """With the panel flat and the sun overhead, poa should track ghi."""
    out = lit_frame("fixed", ["2024-06-21 20:00"], 900.0, 90.0, 960.0, tilt=0.0)
    assert out["poa"].iloc[0] == pytest.approx(960.0, rel=0.05)


def test_poa_never_negative_on_a_full_day() -> None:
    day = pd.date_range("2024-06-21", periods=24, freq="1h", tz="UTC")
    times = [str(t) for t in day]
    for tracking in ["fixed", "single_axis", "dual_axis", "unknown"]:
        out = lit_frame(tracking, times, 700.0, 120.0, 500.0, tilt=22.0)
        assert (out["poa"] >= 0.0).all(), f"{tracking} produced negative light"
        assert out["poa"].notna().all(), f"{tracking} left a NaN"


def test_negative_diffuse_does_not_subtract_light() -> None:
    """HRRR's terminator noise reaches -3.2 W/m2; it must not eat the beam."""
    out = lit_frame("fixed", ["2024-06-21 18:00"], 700.0, -3.2, 400.0, tilt=22.0)
    assert out["poa"].iloc[0] > 0.0


def test_dual_axis_collects_the_most() -> None:
    """Pointing straight at the sun beats a single axis, which beats flat."""
    args = (["2024-06-21 15:00"], 850.0, 110.0, 420.0)
    dual = lit_frame("dual_axis", *args)["poa"].iloc[0]
    single = lit_frame("single_axis", *args)["poa"].iloc[0]
    flat = lit_frame("fixed", *args, tilt=0.0)["poa"].iloc[0]
    assert dual > single > flat


# --- heat -----------------------------------------------------------


def heated(
    times: list[str],
    dni=800.0,
    dhi=100.0,
    dswrf=500.0,
    t2m=303.15,
    w10m=3.0,
    tracking="single_axis",
    **kw,
) -> pd.DataFrame:
    frame = lit_frame(tracking, times, dni, dhi, dswrf, **kw)
    frame["t2m"], frame["w10m"] = t2m, w10m
    return heat(frame)


def test_cell_runs_hotter_than_the_air() -> None:
    """A panel in full sun sits well above the air around it."""
    out = heated(["2024-06-21 20:00"])
    air = out["t2m"].iloc[0] - KELVIN_ZERO
    assert out["poa"].iloc[0] > 500.0, "this hour really is lit"
    assert out["cell_temperature"].iloc[0] > air + 15.0


def test_unlit_cell_sits_at_air_temperature() -> None:
    """With no light to absorb, a panel is just an object in the dark."""
    out = heated(["2024-06-21 08:00"], t2m=290.0)
    assert out["poa"].iloc[0] == 0.0, "this hour really is dark"
    assert out["cell_temperature"].iloc[0] == pytest.approx(290.0 - KELVIN_ZERO)


def test_wind_cools_the_cell() -> None:
    calm = heated(["2024-06-21 20:00"], w10m=0.5)
    breezy = heated(["2024-06-21 20:00"], w10m=8.0)
    assert breezy["cell_temperature"].iloc[0] < calm["cell_temperature"].iloc[0] - 5.0


def test_kelvin_is_converted_to_celsius() -> None:
    """HRRR stores Kelvin; nothing downstream would catch the confusion."""
    out = heated(["2024-06-21 08:00"], t2m=300.0)
    assert out["cell_temperature"].iloc[0] == pytest.approx(26.85, abs=0.01)


def test_cell_temperature_stays_physical_over_a_day() -> None:
    """A day-shaped sky, because a flat one is not a sky at all.

    The beam has to fade as the sun nears the horizon. Held constant
    instead, 800 W/m² of beam at a zenith of 88.9 degrees drives a
    dual-axis panel to 122 C — the transposition step multiplies the
    diffuse by cos(aoi)/cos(zenith), and pvlib floors that denominator
    at cos(89 degrees), so the ratio reaches 57. Real HRRR rows at that
    zenith carry a beam near zero, so this never fires on stored data;
    the ceiling below is measured from it. See docs for the audit.
    """
    day = pd.date_range("2024-06-21", periods=24, freq="1h", tz="UTC")
    times = [str(t) for t in day]
    for tracking in ["fixed", "single_axis", "dual_axis", "unknown"]:
        rig = mounted(tracking, tilt=22.0)
        frame = position(hours(times), rig)
        frame["dni"] = 900.0 * frame["cos_zenith"]
        frame["dhi"] = 100.0 * frame["cos_zenith"]
        frame["dswrf"] = 850.0 * frame["cos_zenith"]
        frame["t2m"], frame["w10m"] = 303.15, 3.0
        out = heat(poa(orient(frame, rig)))
        assert out["cell_temperature"].notna().all(), f"{tracking} left a NaN"
        assert (out["cell_temperature"] > 0.0).all(), f"{tracking} froze a panel"
        assert (out["cell_temperature"] < 85.0).all(), f"{tracking} melted a panel"


def test_every_row_keeps_its_own_weather() -> None:
    """A dark cold hour and a lit warm one, in that order."""
    out = heated(["2024-06-21 08:00", "2024-06-21 20:00"], t2m=[310.0, 280.0])
    assert list(out["t2m"]) == [310.0, 280.0], "carried through untouched"
    assert out["cell_temperature"].iloc[0] == pytest.approx(310.0 - KELVIN_ZERO), (
        "the dark row must not collect the lit row's sun"
    )
    assert out["cell_temperature"].iloc[1] > 280.0 - KELVIN_ZERO + 15.0


# --- generate -------------------------------------------------------


def generated(
    poa_wm2, cell_c, times=("2024-06-21 20:00",), plant_ids=(1,), **kw
) -> pd.DataFrame:
    rig = mounted("single_axis", **kw)
    frame = hours(list(times), plant_ids=plant_ids)
    frame["poa"], frame["cell_temperature"] = poa_wm2, cell_c
    return generate(frame, rig)


def test_standard_conditions_give_the_nameplate() -> None:
    """1000 W/m² on a 25 C cell is what the DC nameplate means."""
    out = generated(1000.0, 25.0, dc_capacity_mw=127.5)
    assert out["dc_mw"].iloc[0] == pytest.approx(127.5)


def test_dc_scales_with_the_light() -> None:
    half = generated(500.0, 25.0, dc_capacity_mw=100.0)["dc_mw"].iloc[0]
    full = generated(1000.0, 25.0, dc_capacity_mw=100.0)["dc_mw"].iloc[0]
    assert half == pytest.approx(full / 2.0)


def test_a_hot_cell_makes_less_power() -> None:
    cool = generated(1000.0, 25.0, dc_capacity_mw=100.0)["dc_mw"].iloc[0]
    hot = generated(1000.0, 65.0, dc_capacity_mw=100.0)["dc_mw"].iloc[0]
    assert hot < cool
    assert hot == pytest.approx(cool * (1 + GAMMA_PDC * 40.0)), "0.35% per degree"


def test_dark_panels_make_nothing() -> None:
    out = generated(0.0, 20.0)
    assert out["dc_mw"].iloc[0] == 0.0


def test_dc_is_allowed_past_the_ac_nameplate() -> None:
    """Clipping belongs to the inverter step; this one must not do it."""
    out = generated(1000.0, 25.0, dc_capacity_mw=127.5, capacity_mw_ac=100.0)
    assert out["dc_mw"].iloc[0] > 100.0, "the loading ratio must survive to here"


def test_a_plant_makes_nothing_before_it_is_built() -> None:
    """The registry is a 2025 snapshot; the weather store starts in 2023."""
    out = generated(
        1000.0,
        25.0,
        times=("2023-01-15 20:00",),
        operating_date="2024-06-01",
    )
    assert out["dc_mw"].iloc[0] == 0.0, "these panels were still a field"


def test_a_plant_generates_from_its_operating_month() -> None:
    out = generated(
        1000.0,
        25.0,
        times=("2024-06-01 20:00", "2024-05-31 20:00"),
        operating_date="2024-06-01",
    )
    assert out["dc_mw"].iloc[0] > 0.0, "the first day counts"
    assert out["dc_mw"].iloc[1] == 0.0, "the day before does not"


def test_every_plant_uses_its_own_capacity_and_start() -> None:
    """A merge that lined up wrongly would be invisible in a one-plant test."""
    rig = pd.DataFrame(
        {
            "plant_id": [1, 2, 3],
            "dc_capacity_mw": [100.0, 250.0, 40.0],
            "operating_date": pd.to_datetime(
                ["2020-01-01", "2025-01-01", "2020-01-01"], utc=True
            ),
        }
    )
    frame = hours(["2024-06-21 20:00"], plant_ids=(1, 2, 3))
    frame["poa"], frame["cell_temperature"] = 1000.0, 25.0
    out = generate(frame, rig).set_index("plant_id")
    assert out.loc[1, "dc_mw"] == pytest.approx(100.0)
    assert out.loc[2, "dc_mw"] == 0.0, "not built until 2025"
    assert out.loc[3, "dc_mw"] == pytest.approx(40.0)


def test_the_whole_chain_runs_end_to_end() -> None:
    """position -> orient -> poa -> heat -> generate, on a full day."""
    day = pd.date_range("2024-06-21", periods=24, freq="1h", tz="UTC")
    rig = mounted("single_axis", dc_capacity_mw=100.0)
    frame = position(hours([str(t) for t in day]), rig)
    frame["dni"] = 900.0 * frame["cos_zenith"]
    frame["dhi"] = 100.0 * frame["cos_zenith"]
    frame["dswrf"] = 850.0 * frame["cos_zenith"]
    frame["t2m"], frame["w10m"] = 303.15, 3.0
    out = generate(heat(poa(orient(frame, rig))), rig)

    assert out["dc_mw"].notna().all()
    assert (out["dc_mw"] >= 0.0).all(), "no plant generates in the dark"
    assert out["dc_mw"].max() > 60.0, "a June day should work the panels hard"
    assert out["dc_mw"].max() < 100.0, "and never beat the nameplate on this sky"
