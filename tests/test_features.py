import pandas as pd
import pytest

from americast.features.county import ZONES
from americast.features.features import (
    aggregate,
    calendar,
    fleet,
    hourly,
    physical,
)

TZ = "America/Los_Angeles"


# Roughly where each zone's capacity sits, so a synthetic plant put in
# a county gets a coordinate that belongs to it.
COORDS = [(35.0, -118.5), (33.0, -115.5), (36.7, -119.8), (33.9, -116.5), (36.0, -120.7)]


def registry(
    plant_ids=(1, 2),
    counties=("Kern", "Imperial"),
    capacities=(100.0, 200.0),
    ba="CISO",
) -> pd.DataFrame:
    n = len(plant_ids)
    places = [COORDS[i % len(COORDS)] for i in range(n)]
    return pd.DataFrame(
        {
            "plant_id": list(plant_ids),
            "latitude": [lat for lat, _ in places],
            "longitude": [lon for _, lon in places],
            "capacity_mw_ac": list(capacities),
            "dc_capacity_mw": [c * 1.275 for c in capacities],
            "tracking": ["single_axis"] * n,
            "tilt": [0.0] * n,
            "azimuth": [180.0] * n,
            "county": list(counties),
            "balancing_authority": [ba] * n,
            "operating_date": pd.to_datetime(["2020-01-01"] * n, utc=True),
        }
    )


def weather(run: str, leads=(1, 2, 3), plant_ids=(1, 2), **overrides) -> pd.DataFrame:
    run_time = pd.Timestamp(run, tz="UTC")
    rows = []
    for lead in leads:
        for pid in plant_ids:
            rows.append(
                {
                    "run_time": run_time,
                    "valid_time": run_time + pd.Timedelta(hours=lead),
                    "lead_hours": lead,
                    "plant_id": pid,
                    "dswrf": 500.0,
                    "dni": 700.0,
                    "dhi": 90.0,
                    "tcdc": 10.0,
                    "t2m": 300.0,
                    "w10m": 3.0,
                }
            )
    frame = pd.DataFrame(rows)
    for column, value in overrides.items():
        frame[column] = value
    return frame


# --- fleet ----------------------------------------------------------


def test_fleet_keeps_only_ciso() -> None:
    mixed = pd.concat(
        [registry(), registry(plant_ids=(3, 4), ba="LDWP")], ignore_index=True
    )
    kept = fleet(mixed)
    assert set(kept["plant_id"]) == {1, 2}


def test_an_unmapped_county_fails_loudly() -> None:
    """A silent unknown bucket would hide next year's plants."""
    with pytest.raises(ValueError, match="counties missing"):
        fleet(registry(counties=("Kern", "Atlantis")))


def test_every_plant_gets_a_zone() -> None:
    zoned = fleet(registry())
    assert set(zoned["zone"]) <= set(ZONES)
    assert zoned["zone"].notna().all()


# --- aggregate ------------------------------------------------------


def test_aggregate_collapses_plants_to_hours() -> None:
    plants = fleet(registry())
    out = aggregate(weather("2024-06-15 06:00"), plants)
    assert len(out) == 3, "three forecast hours, one row each"
    assert out["lead_hours"].tolist() == [1, 2, 3]


def test_aggregate_declares_every_zone_even_when_empty() -> None:
    """The table must not change width when a zone has no plants."""
    plants = fleet(registry(plant_ids=(1,), counties=("Kern",), capacities=(100.0,)))
    out = aggregate(weather("2024-06-15 06:00", plant_ids=(1,)), plants)
    for zone in ZONES:
        assert f"{zone}_dswrf" in out.columns
    assert out["coastal_dswrf"].isna().all(), "an absent zone has no temperature"
    assert out["kern_dswrf"].notna().all()


def test_capacity_weighting_favours_the_bigger_plant() -> None:
    plants = fleet(registry(capacities=(100.0, 900.0)))
    frame = weather("2024-06-15 06:00")
    frame.loc[frame["plant_id"] == 1, "dswrf"] = 0.0
    frame.loc[frame["plant_id"] == 2, "dswrf"] = 1000.0
    out = aggregate(frame, plants)
    assert out["fleet_dswrf"].iloc[0] == pytest.approx(900.0)


# --- physical -------------------------------------------------------


def test_physical_sums_zones_to_the_fleet() -> None:
    plants = fleet(registry())
    out = physical(weather("2024-06-15 06:00", leads=range(1, 20)), plants)
    zone_total = out[[f"{zone}_ac_mw" for zone in ZONES]].sum(axis=1)
    assert zone_total.to_numpy() == pytest.approx(out["fleet_ac_mw"].to_numpy())


def test_physical_ignores_plants_outside_ciso() -> None:
    """The weather store holds all 928 plants; only CISO is modelled."""
    plants = fleet(registry())
    both = pd.concat(
        [weather("2024-06-15 06:00"), weather("2024-06-15 06:00", plant_ids=(7, 8))],
        ignore_index=True,
    )
    out = physical(both, plants)
    only_mine = physical(weather("2024-06-15 06:00"), plants)
    assert out["fleet_ac_mw"].to_numpy() == pytest.approx(
        only_mine["fleet_ac_mw"].to_numpy()
    )


def test_an_empty_zone_is_zero_megawatts_not_null() -> None:
    plants = fleet(registry(plant_ids=(1,), counties=("Kern",), capacities=(100.0,)))
    out = physical(weather("2024-06-15 06:00", plant_ids=(1,)), plants)
    assert (out["coastal_ac_mw"] == 0.0).all(), "no plants really is no megawatts"


def test_physical_never_beats_the_nameplate() -> None:
    plants = fleet(registry())
    bright = weather("2024-06-15 06:00", leads=range(1, 25), dswrf=1100.0, dni=1000.0)
    out = physical(bright, plants)
    assert out["fleet_ac_mw"].max() <= plants["capacity_mw_ac"].sum() + 1e-9


# --- hourly ---------------------------------------------------------


def framed(run: str, values: list[float]) -> pd.DataFrame:
    run_time = pd.Timestamp(run, tz="UTC")
    return pd.DataFrame(
        {
            "run_time": run_time,
            "valid_time": [
                run_time + pd.Timedelta(hours=h) for h in range(1, len(values) + 1)
            ],
            "lead_hours": list(range(1, len(values) + 1)),
            "fleet_ac_mw": values,
        }
    )


def test_an_instant_becomes_the_mean_of_the_hour() -> None:
    out = hourly(framed("2024-06-15 06:00", [0.0, 100.0, 200.0]))
    assert out["fleet_ac_mw"].tolist() == [50.0, 150.0]


def test_the_last_forecast_hour_is_dropped() -> None:
    """It has no successor, so it cannot be turned into an hour mean."""
    out = hourly(framed("2024-06-15 06:00", [1.0, 2.0, 3.0, 4.0]))
    assert out["lead_hours"].tolist() == [1, 2, 3]


def test_hours_never_average_across_two_runs() -> None:
    """The bug this exists to catch is silent and ruins every boundary."""
    early = framed("2024-06-15 06:00", [10.0, 20.0])
    late = framed("2024-06-16 06:00", [1000.0, 2000.0])
    out = hourly(pd.concat([early, late], ignore_index=True))
    assert out["fleet_ac_mw"].tolist() == [15.0, 1500.0]
    assert len(out) == 2, "each run loses exactly its own last hour"


def test_the_index_columns_are_not_averaged() -> None:
    out = hourly(framed("2024-06-15 06:00", [1.0, 2.0, 3.0]))
    assert out["lead_hours"].tolist() == [1, 2]
    assert (out["run_time"] == pd.Timestamp("2024-06-15 06:00", tz="UTC")).all()
    starts = out["valid_time"].dt.hour.tolist()
    assert starts == [7, 8], "a row is still named by the hour it starts"


def test_a_hole_costs_the_hour_before_it() -> None:
    """Averaging 13:00 with 15:00 is not the mean of the 13:00 hour.

    Some forecast hours were never archived, so this is a real case,
    not a hypothetical one. The hour before the hole has no partner and
    must leave rather than be averaged across the gap.
    """
    frame = framed("2024-06-15 06:00", [1.0, 2.0, 3.0, 4.0])
    holed = frame[frame["lead_hours"] != 3]
    out = hourly(holed)
    assert out["lead_hours"].tolist() == [1], "lead 2 has no neighbour left"
    assert out["fleet_ac_mw"].tolist() == [1.5]


# --- calendar -------------------------------------------------------


def test_local_hour_is_local_not_utc() -> None:
    frame = pd.DataFrame({"valid_time": pd.to_datetime(["2024-06-15 20:00"], utc=True)})
    out = calendar(frame, TZ)
    assert out["local_hour"].iloc[0] == 13, "20:00 UTC is 13:00 PDT in June"


def test_local_hour_follows_daylight_saving() -> None:
    """The same UTC hour is a different local hour in January and June."""
    frame = pd.DataFrame(
        {"valid_time": pd.to_datetime(["2024-01-15 20:00", "2024-06-15 20:00"], utc=True)}
    )
    out = calendar(frame, TZ)
    assert out["local_hour"].tolist() == [12, 13]


def test_day_of_year_is_a_plain_number() -> None:
    frame = pd.DataFrame({"valid_time": pd.to_datetime(["2024-12-31 20:00"], utc=True)})
    out = calendar(frame, TZ)
    assert out["day_of_year"].iloc[0] == 366, "2024 is a leap year"
