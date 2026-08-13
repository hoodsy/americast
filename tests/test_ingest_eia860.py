import pandas as pd
import pytest

from americast.ingest.eia860 import build_registry
from americast.schemas import PLANTS_CISO


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


def monthly_sheet() -> pd.DataFrame:
    """The EIA-860M Operating sheet: the spine that decides the fleet."""

    def gen(pid, gid, name, ba="CISO", county="Kern", state="CA",
            tech="Solar Photovoltaic", status="(OP) Operating", mw=100.0,
            dc=None, lat=35.0, lon=-118.2, year=2020, month=6):
        return {
            "Plant ID": pid, "Generator ID": gid, "Plant Name": name,
            "Balancing Authority Code": ba, "County": county,
            "Plant State": state, "Technology": tech, "Status": status,
            "Nameplate Capacity (MW)": mw,
            "DC Net Capacity (MW)": mw * 1.3 if dc is None else dc,
            "Latitude": lat, "Longitude": lon,
            "Operating Year": year, "Operating Month": month,
        }

    return pd.DataFrame(
        [
            # plant 1: two phases; the earlier, smaller one dates it
            gen(1, "A", "Big Sun", mw=200.0, year=2019),
            gen(1, "B", "Big Sun", mw=5.0, year=2016),
            # plant 2: fixed mount reporting no tilt
            gen(2, "A", "Fixed Farm", county="Riverside", mw=50.0),
            # plant 3: retired -- excluded by status
            gen(3, "A", "Retired", status="(RE) Retired", mw=75.0),
            # plant 4: solar thermal -- excluded by technology
            gen(4, "A", "Thermal", tech="Solar Thermal with Energy Storage"),
            # plant 5: Nevada, inside CISO. The whole point of the
            # balancing-authority filter: it must survive.
            gen(5, "A", "Vegas", county="Clark", state="NV", mw=120.0,
                lat=36.2, lon=-115.1),
            # plant 6: no DC rating at all -- takes the ILR fallback
            gen(6, "A", "Mystery", mw=10.0, dc=float("nan")),
            # plant 7: one axis recorded both ways round
            gen(7, "A", "Two Ways Round", mw=90.0),
            gen(7, "B", "Two Ways Round", mw=80.0),
            # plant 8: California, but outside CISO -- excluded, because
            # its output never reaches the label
            gen(8, "A", "Outsider", ba="LDWP", mw=300.0),
            # plant 9: built since the annual vintage, so it has no
            # geometry anywhere. It must still be in the fleet.
            gen(9, "A", "Brand New", county="Yuma", state="AZ", mw=40.0,
                lat=32.9, lon=-113.5, year=2026, month=5),
            # EIA's trailing source note arrives as a null Plant ID
            {"Plant ID": None},
        ]
    )


def built() -> pd.DataFrame:
    return build_registry(solar_sheet(), monthly_sheet()).set_index("plant_id")


# --- the filter ------------------------------------------------------


def test_the_fleet_is_the_balancing_authority_not_the_state() -> None:
    """Nevada is in; a Californian plant outside CISO is out."""
    reg = build_registry(solar_sheet(), monthly_sheet())
    assert list(reg["plant_id"]) == [1, 2, 5, 6, 7, 9]


def test_an_out_of_state_ciso_plant_is_kept() -> None:
    """The 2,478 MW correction, in miniature."""
    reg = built()
    assert 5 in reg.index
    assert reg.loc[5, "county"] == "Clark"
    assert reg.loc[5, "balancing_authority"] == "CISO"


def test_a_californian_plant_outside_ciso_is_dropped() -> None:
    assert 8 not in built().index


def test_retired_and_thermal_are_dropped() -> None:
    reg = built()
    assert 3 not in reg.index
    assert 4 not in reg.index


def test_the_trailing_source_note_is_not_a_plant() -> None:
    """A null Plant ID row would otherwise become a plant called NaN."""
    assert built().index.notna().all()


# --- aggregation -----------------------------------------------------


def test_registry_filters_and_aggregates() -> None:
    big_sun = built().loc[1]
    assert big_sun["capacity_mw_ac"] == 205.0
    assert big_sun["tracking"] == "single_axis", "capacity-dominant, not count"
    assert big_sun["county"] == "Kern"


def test_dc_capacity_sums_and_falls_back() -> None:
    reg = built()
    assert reg.loc[1, "dc_capacity_mw"] == pytest.approx(205.0 * 1.3)
    assert reg.loc[6, "dc_capacity_mw"] == pytest.approx(10.0 * 1.275)
    assert reg.loc[6, "capacity_mw_ac"] == 10.0, "AC untouched by the DC fallback"


def test_operating_date_is_the_first_phase() -> None:
    reg = built()
    assert reg.loc[1, "operating_date"] == pd.Timestamp("2016-06-01", tz="UTC")
    assert reg.loc[2, "operating_date"] == pd.Timestamp("2020-06-01", tz="UTC")


# --- geometry, joined from the annual vintage ------------------------


def test_geometry_comes_from_the_dominant_tracking_phase() -> None:
    # Plant 1's fixed phase tilts 25 degrees and faces 200. It is the
    # smaller phase and the wrong tracking type, so neither number
    # should describe the plant.
    reg = built()
    assert reg.loc[1, "tilt"] == 0.0
    assert reg.loc[1, "azimuth"] == 180.0


def test_azimuth_is_never_averaged() -> None:
    # Plant 7 records one north-south axis as 180 and 0. Their mean is
    # 90 -- an east-west axis that exists nowhere on site.
    assert built().loc[7, "azimuth"] == 180.0, "largest phase wins, no mean"


def test_tilt_default_follows_tracking_type() -> None:
    reg = built()
    assert reg.loc[2, "tilt"] == 22.0, "fixed with no tilt takes the fixed default"
    assert reg.loc[6, "tilt"] == 0.0, "untracked default is a flat axis"


def test_reported_tracker_tilt_is_kept_raw() -> None:
    # Plant 7's generators both report 60 degrees. On a tracker that is
    # a rotation limit rather than an axis tilt, but the registry
    # stores what EIA said and leaves the reading to the power model.
    reg = built()
    assert reg.loc[7, "tracking"] == "single_axis"
    assert reg.loc[7, "tilt"] == 60.0, "stored as reported, not interpreted"


def test_a_plant_too_new_for_the_annual_file_still_joins_the_fleet() -> None:
    """The currency fix. It must not be dropped for lacking a tilt."""
    reg = built()
    assert 9 in reg.index
    assert reg.loc[9, "capacity_mw_ac"] == 40.0
    assert reg.loc[9, "tracking"] == "unknown"
    assert reg.loc[9, "tilt"] == 0.0, "unknown mount reads as a flat axis"
    assert reg.loc[9, "azimuth"] == 180.0


# --- the schema ------------------------------------------------------


def test_registry_conforms_to_schema() -> None:
    import pyarrow as pa

    reg = build_registry(solar_sheet(), monthly_sheet())
    table = pa.Table.from_pandas(reg, schema=PLANTS_CISO, preserve_index=False)
    assert table.num_rows == 6


def test_missing_coordinates_violate_schema() -> None:
    import pyarrow as pa

    monthly = monthly_sheet()
    monthly.loc[monthly["Plant ID"] == 2, "Latitude"] = None
    reg = build_registry(solar_sheet(), monthly)
    with pytest.raises(ValueError, match="non-nullable"):
        pa.Table.from_pandas(reg, schema=PLANTS_CISO, preserve_index=False)


def test_verify_reports_the_fleet_it_built() -> None:
    from americast.ingest.eia860 import verify

    audit = verify(build_registry(solar_sheet(), monthly_sheet()))
    assert audit["n_plants"] == 6
    assert audit["non_ciso"] == 0
    assert audit["default_geometry"] == 2, "plants 6 and 9 have no tracking"
    assert not audit["unknown_county"]
