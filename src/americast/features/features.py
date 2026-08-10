"""Feature engineering: plant-level weather -> one row per forecast hour.

The model predicts one number (statewide CAISO solar MW) per forecast
hour, but the weather store holds 788 plant rows for that same hour.
This module does the collapse -- and it collapses inside weather zones
rather than statewide, because fog in the Bay and clear sky in the
Mojave average to "half cloudy", which describes neither place.
"""

import pandas as pd

from americast.features.county import CISO_BA, COUNTY_ZONE

# The four HRRR fields, in schema order. Native GRIB units throughout —
# W/m², %, Kelvin, m/s — because a weighted mean is unit-agnostic.
WEATHER_VARS = ("dswrf", "tcdc", "t2m", "w10m")

# What identifies one forecast hour, and so one row of the model's table.
HOUR_KEYS = ["run_time", "valid_time"]


def fleet(plants: pd.DataFrame) -> pd.DataFrame:
    """The plants we model: CISO only, each labeled with its weather zone.

    plants is the PLANTS_CA registry frame; this reads balancing_authority
    and county, and returns the same columns plus `zone`.

    An unmapped county raises. A silent "unknown" bucket would hide next
    year's new plants instead of showing them -- capacity would drop out
    of every feature and nothing would look wrong.
    """
    ciso = plants[plants["balancing_authority"] == CISO_BA].copy()
    counties = ciso["county"].str.lower()
    unmapped = sorted(set(counties) - set(COUNTY_ZONE))
    if unmapped:
        raise ValueError(f"counties missing from ZONE_COUNTIES: {unmapped}")
    ciso["zone"] = counties.map(COUNTY_ZONE)
    return ciso.reset_index(drop=True)


def aggregate(weather: pd.DataFrame, plants: pd.DataFrame) -> pd.DataFrame:
    """Capacity-weighted zone means: 788 plant rows -> one row per hour.

    weather is the HRRR_WEATHER frame; plants is fleet()'s output. The
    inner join is where the CISO filter actually bites — the weather
    store holds all 928 registry plants, and the 140 outside CISO leave
    here.

    Returns one row per (run_time, valid_time), carrying lead_hours, a
    `{zone}_{var}` column for every zone and variable, and a
    `fleet_{var}` column weighted across the whole fleet. The fleet
    columns are not redundant: they are a fixed linear combination of
    the five zone means, and a tree can only approximate those with a
    stack of splits. Handing the number over is cheaper.
    """
    columns = ["plant_id", "zone", "capacity_mw_ac"]
    rows = weather.merge(plants[columns], on="plant_id", how="inner")
    absent = set(plants["plant_id"]) - set(rows["plant_id"])
    if absent:
        raise ValueError(f"{len(absent)} fleet plants have no weather rows")

    weighted = rows[[*HOUR_KEYS, "zone", "capacity_mw_ac"]].copy()
    for var in WEATHER_VARS:
        weighted[var] = rows[var] * rows["capacity_mw_ac"]

    by_zone = _means(weighted, [*HOUR_KEYS, "zone"])
    spread = by_zone.pivot(index=HOUR_KEYS, columns="zone", values=list(WEATHER_VARS))
    spread.columns = [f"{zone}_{var}" for var, zone in spread.columns]

    by_fleet = _means(weighted, HOUR_KEYS)
    renamed = {var: f"fleet_{var}" for var in WEATHER_VARS}

    out = spread.reset_index().merge(by_fleet.rename(columns=renamed), on=HOUR_KEYS)
    lead = out["valid_time"] - out["run_time"]
    out.insert(2, "lead_hours", (lead // pd.Timedelta(hours=1)).astype("int32"))
    return out.sort_values(HOUR_KEYS).reset_index(drop=True)


def _means(weighted: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Weighted mean per group: sum of (value x MW), over summed MW.

    Dividing by the summed capacity of whoever is present, rather than
    by a precomputed fleet total, keeps the mean honest when a plant is
    missing from an hour — the survivors renormalize instead of the
    average quietly sagging toward zero.
    """
    values = [*WEATHER_VARS, "capacity_mw_ac"]
    totals = weighted.groupby(keys, as_index=False)[values].sum()
    for var in WEATHER_VARS:
        totals[var] = totals[var] / totals["capacity_mw_ac"]
    return totals.drop(columns="capacity_mw_ac")
