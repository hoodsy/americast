import pandas as pd
import pytest

from americast.ingest.eia860 import build_registry
from americast.schemas import PLANTS_CA


def solar_sheet() -> pd.DataFrame:
    def gen(code, gid, state="CA", status="OP", tech="Solar Photovoltaic",
            mw=100.0, single="N", dual="N", fixed="N", dc=None, tilt=None,
            az=180.0, year=2020, month=6):
        return {
            "Plant Code": code, "Generator ID": gid, "State": state,
            "Status": status, "Technology": tech,
            "Nameplate Capacity (MW)": mw,
            "Single-Axis Tracking?": single, "Dual-Axis Tracking?": dual,
            "Fixed Tilt?": fixed,
            "DC Net Capacity (MW)": mw * 1.3 if dc is None else dc,
            "Tilt Angle": tilt, "Azimuth Angle": az,
            "Operating Year": year, "Operating Month": month,
        }

    return pd.DataFrame(
        [
            # plant 1: two phases, big single-axis + small fixed. The
            # fixed phase arrived first and tilts steeply; neither fact
            # should reach the plant's geometry, but the earlier date
            # should reach its operating_date.
            gen(1, "A", mw=200.0, single="Y", tilt=0.0, az=180.0, year=2019),
            gen(1, "B", mw=5.0, fixed="Y", tilt=25.0, az=200.0, year=2016),
            # plant 2: fixed only, reports no tilt — takes the fixed default
            gen(2, "A", mw=50.0, fixed="Y", tilt=None),
            # plant 3: not operating — excluded
            gen(3, "A", status="RE", mw=75.0, single="Y"),
            # plant 4: solar thermal — excluded
            gen(4, "A", tech="Solar Thermal without Energy Storage", mw=250.0),
            # plant 5: Nevada — excluded
            gen(5, "A", state="NV", mw=120.0, single="Y"),
            # plant 6: no tracking flags, and no DC rating — takes both
            # the axis-tilt default and the ILR fallback
            gen(6, "A", mw=10.0, dc=None, tilt=None),
            # plant 7: one north-south axis written both ways round.
            # Averaging 0 and 180 would invent an east-west axis. Its
            # reported 60-degree "tilt" is a tracker rotation limit
            # answering a question EIA did not ask.
            gen(7, "A", mw=90.0, single="Y", tilt=60.0, az=180.0),
            gen(7, "B", mw=80.0, single="Y", tilt=60.0, az=0.0),
        ]
    )


def plant_sheet() -> pd.DataFrame:
    def plant(code, name, lat, lon, county="Kern", ba="CISO"):
        return {
            "Plant Code": code, "Plant Name": name,
            "Latitude": lat, "Longitude": lon,
            "County": county, "Balancing Authority Code": ba,
        }

    return pd.DataFrame(
        [
            plant(1, "Big Sun", 35.0, -118.2),
            plant(2, "Fixed Farm", 33.7, -115.4, county="Riverside", ba="LDWP"),
            plant(3, "Retired", 36.0, -119.0),
            plant(4, "Thermal", 35.5, -117.5),
            plant(5, "Vegas", 36.2, -115.1),
            plant(6, "Mystery", 38.0, -121.0, ba=None),
            plant(7, "Two Ways Round", 34.5, -117.0),
        ]
    )


def test_registry_filters_and_aggregates() -> None:
    reg = build_registry(plant_sheet(), solar_sheet())
    assert list(reg["plant_id"]) == [1, 2, 6, 7]
    big_sun = reg.set_index("plant_id").loc[1]
    assert big_sun["capacity_mw_ac"] == 205.0
    assert big_sun["tracking"] == "single_axis", "capacity-dominant, not count"
    assert big_sun["county"] == "Kern"


def test_dc_capacity_sums_and_falls_back() -> None:
    reg = build_registry(plant_sheet(), solar_sheet()).set_index("plant_id")
    assert reg.loc[1, "dc_capacity_mw"] == pytest.approx(205.0 * 1.3)
    assert reg.loc[6, "dc_capacity_mw"] == pytest.approx(10.0 * 1.3)
    assert reg.loc[6, "capacity_mw_ac"] == 10.0, "AC untouched by the DC fallback"


def test_geometry_comes_from_the_dominant_tracking_phase() -> None:
    # Plant 1's fixed phase tilts 25 degrees and faces 200. It is the
    # smaller phase and the wrong tracking type, so neither number
    # should describe the plant.
    reg = build_registry(plant_sheet(), solar_sheet()).set_index("plant_id")
    assert reg.loc[1, "tilt"] == 0.0
    assert reg.loc[1, "azimuth"] == 180.0


def test_azimuth_is_never_averaged() -> None:
    # Plant 7 records one north-south axis as 180 and 0. Their mean is
    # 90 — an east-west axis that exists nowhere on site.
    reg = build_registry(plant_sheet(), solar_sheet()).set_index("plant_id")
    assert reg.loc[7, "azimuth"] == 180.0, "largest phase wins, no mean"


def test_tilt_default_follows_tracking_type() -> None:
    reg = build_registry(plant_sheet(), solar_sheet()).set_index("plant_id")
    assert reg.loc[2, "tilt"] == 22.0, "fixed with no tilt takes the fixed default"
    assert reg.loc[6, "tilt"] == 0.0, "untracked default is a flat axis"


def test_reported_tracker_tilt_is_discarded() -> None:
    # Plant 7's generators both report 60 degrees. On a tracker that is
    # a rotation limit, not an axis tilt, and modelling it as an axis
    # tilt would bend the plant's whole output curve.
    reg = build_registry(plant_sheet(), solar_sheet()).set_index("plant_id")
    assert reg.loc[7, "tracking"] == "single_axis"
    assert reg.loc[7, "tilt"] == 0.0, "trackers are modelled with a flat axis"


def test_operating_date_is_the_first_phase() -> None:
    reg = build_registry(plant_sheet(), solar_sheet()).set_index("plant_id")
    assert reg.loc[1, "operating_date"] == pd.Timestamp("2016-06-01", tz="UTC")
    assert reg.loc[2, "operating_date"] == pd.Timestamp("2020-06-01", tz="UTC")


def test_registry_carries_ba_and_fills_unknown() -> None:
    reg = build_registry(plant_sheet(), solar_sheet()).set_index("plant_id")
    assert reg.loc[2, "balancing_authority"] == "LDWP"
    assert reg.loc[6, "balancing_authority"] == "UNKNOWN"
    assert reg.loc[6, "tracking"] == "unknown"


def test_registry_conforms_to_schema() -> None:
    import pyarrow as pa

    reg = build_registry(plant_sheet(), solar_sheet())
    table = pa.Table.from_pandas(reg, schema=PLANTS_CA, preserve_index=False)
    assert table.num_rows == 4


def test_missing_coordinates_violate_schema() -> None:
    import pyarrow as pa

    plants = plant_sheet()
    plants.loc[plants["Plant Code"] == 2, "Latitude"] = None
    reg = build_registry(plants, solar_sheet())
    with pytest.raises(ValueError, match="non-nullable"):
        pa.Table.from_pandas(reg, schema=PLANTS_CA, preserve_index=False)
