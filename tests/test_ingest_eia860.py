import pandas as pd
import pytest

from americast.ingest.eia860 import build_registry
from americast.schemas import PLANTS_CA


def solar_sheet() -> pd.DataFrame:
    def gen(code, gid, state="CA", status="OP", tech="Solar Photovoltaic",
            mw=100.0, single="N", dual="N", fixed="N"):
        return {
            "Plant Code": code, "Generator ID": gid, "State": state,
            "Status": status, "Technology": tech,
            "Nameplate Capacity (MW)": mw,
            "Single-Axis Tracking?": single, "Dual-Axis Tracking?": dual,
            "Fixed Tilt?": fixed,
        }

    return pd.DataFrame(
        [
            # plant 1: two phases, big single-axis + small fixed
            gen(1, "A", mw=200.0, single="Y"),
            gen(1, "B", mw=5.0, fixed="Y"),
            # plant 2: fixed only
            gen(2, "A", mw=50.0, fixed="Y"),
            # plant 3: not operating — excluded
            gen(3, "A", status="RE", mw=75.0, single="Y"),
            # plant 4: solar thermal — excluded
            gen(4, "A", tech="Solar Thermal without Energy Storage", mw=250.0),
            # plant 5: Nevada — excluded
            gen(5, "A", state="NV", mw=120.0, single="Y"),
            # plant 6: no tracking flags set
            gen(6, "A", mw=10.0),
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
        ]
    )


def test_registry_filters_and_aggregates() -> None:
    reg = build_registry(plant_sheet(), solar_sheet())
    assert list(reg["plant_id"]) == [1, 2, 6]
    big_sun = reg.set_index("plant_id").loc[1]
    assert big_sun["capacity_mw_ac"] == 205.0
    assert big_sun["tracking"] == "single_axis", "capacity-dominant, not count"
    assert big_sun["county"] == "Kern"


def test_registry_carries_ba_and_fills_unknown() -> None:
    reg = build_registry(plant_sheet(), solar_sheet()).set_index("plant_id")
    assert reg.loc[2, "balancing_authority"] == "LDWP"
    assert reg.loc[6, "balancing_authority"] == "UNKNOWN"
    assert reg.loc[6, "tracking"] == "unknown"


def test_registry_conforms_to_schema() -> None:
    import pyarrow as pa

    reg = build_registry(plant_sheet(), solar_sheet())
    table = pa.Table.from_pandas(reg, schema=PLANTS_CA, preserve_index=False)
    assert table.num_rows == 3


def test_missing_coordinates_violate_schema() -> None:
    import pyarrow as pa

    plants = plant_sheet()
    plants.loc[plants["Plant Code"] == 2, "Latitude"] = None
    reg = build_registry(plants, solar_sheet())
    with pytest.raises(ValueError, match="non-nullable"):
        pa.Table.from_pandas(reg, schema=PLANTS_CA, preserve_index=False)
