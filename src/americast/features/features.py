"""Feature engineering: plant-level weather -> one row per forecast hour.

The model predicts one number (statewide CAISO solar MW) per forecast
hour, but the weather store holds 788 plant rows for that same hour.
This module does the collapse -- and it collapses inside weather zones
rather than statewide, because fog in the Bay and clear sky in the
Mojave average to "half cloudy", which describes neither place.

Two kinds of column come out. `aggregate` gives capacity-weighted
weather means, which is what the raw fields look like from a zone's
point of view. `physical` runs every plant through features/power.py
and sums the megawatts, which is what those fields mean. The model
gets both, because the physics is a strong prior and the raw fields
are where it can find what the physics missed.

`hourly` then does something small and easy to overlook: it turns
instants into hour means, so that a feature and its label describe the
same span of time.
"""

import pandas as pd

from americast.features.county import CISO_BA, COUNTY_ZONE, ZONES
from americast.features.power import estimate

# The four HRRR fields, in schema order. Native GRIB units throughout —
# W/m², %, Kelvin, m/s — because a weighted mean is unit-agnostic.
WEATHER_VARS = ("dswrf", "tcdc", "t2m", "w10m")

# What the physical model contributes per zone and per fleet.
POWER_VARS = ("ac_mw", "clear_mw")

# What identifies one forecast hour, and so one row of the model's table.
HOUR_KEYS = ["run_time", "valid_time"]

# Columns that name an hour rather than measure it, so `hourly` must
# leave them alone while it averages everything else.
INDEX_COLUMNS = (*HOUR_KEYS, "lead_hours")


def fleet(plants: pd.DataFrame) -> pd.DataFrame:
    """The plants we model: CISO only, each labeled with its weather zone.

    plants is the PLANTS_CISO registry frame; this reads balancing_authority
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
    spread = _spread(by_zone, WEATHER_VARS)

    by_fleet = _means(weighted, HOUR_KEYS)
    renamed = {var: f"fleet_{var}" for var in WEATHER_VARS}

    out = spread.reset_index().merge(by_fleet.rename(columns=renamed), on=HOUR_KEYS)
    lead = out["valid_time"] - out["run_time"]
    out.insert(2, "lead_hours", (lead // pd.Timedelta(hours=1)).astype("int32"))
    return out.sort_values(HOUR_KEYS).reset_index(drop=True)


def physical(weather: pd.DataFrame, plants: pd.DataFrame) -> pd.DataFrame:
    """Per-plant megawatts, summed to zone and fleet: the physics prior.

    weather is the HRRR_WEATHER frame; plants is fleet()'s output.
    Returns one row per (run_time, valid_time) carrying
    `{zone}_ac_mw`, `{zone}_clear_mw`, the two fleet totals, and
    `fleet_cos_zenith`.

    The weather store holds all 928 registry plants and this models the
    788 in CISO, so the filter happens here, before the physics runs.
    It is done by selection rather than by a join, because
    features/power.py raises on a plant it has no registry row for —
    that check is worth keeping sharp, and it can only stay sharp if
    the caller narrows the frame deliberately.

    Summing is exact and needs no weights: a megawatt at one plant is a
    megawatt at another. That is the whole advantage of modelling power
    per plant rather than averaging weather and modelling once. Zones
    that hold no plants yet still appear, as zeros, so the table keeps
    one shape across the years.

    `fleet_cos_zenith` is capacity-weighted, not summed. It carries the
    plain geometry of the day, which a tree would otherwise have to
    reconstruct from the calendar columns.
    """
    mine = weather[weather["plant_id"].isin(plants["plant_id"])]
    estimated = estimate(mine, plants)
    zoned = estimated.merge(
        plants[["plant_id", "zone", "capacity_mw_ac"]], on="plant_id", how="left"
    )

    by_zone = zoned.groupby([*HOUR_KEYS, "zone"], as_index=False)[list(POWER_VARS)].sum()
    spread = _spread(by_zone, POWER_VARS)

    totals = zoned.groupby(HOUR_KEYS, as_index=False)[list(POWER_VARS)].sum()
    renamed = {var: f"fleet_{var}" for var in POWER_VARS}

    zoned["weighted_cos"] = zoned["cos_zenith"] * zoned["capacity_mw_ac"]
    summed = zoned.groupby(HOUR_KEYS, as_index=False)[
        ["weighted_cos", "capacity_mw_ac"]
    ].sum()
    summed["fleet_cos_zenith"] = summed["weighted_cos"] / summed["capacity_mw_ac"]

    out = spread.reset_index().merge(totals.rename(columns=renamed), on=HOUR_KEYS)
    out = out.merge(summed[[*HOUR_KEYS, "fleet_cos_zenith"]], on=HOUR_KEYS)
    # A zone holding no plants yet is genuinely zero megawatts. The
    # same is not true of `aggregate`, where an absent zone has no
    # temperature and must stay null rather than be invented.
    return out.fillna(0.0).sort_values(HOUR_KEYS).reset_index(drop=True)


def hourly(frame: pd.DataFrame, within: tuple[str, ...] = ()) -> pd.DataFrame:
    """Instants at the hour mark -> means over the hour that follows.

    frame is one row per (run_time, valid_time), or per (run_time,
    valid_time, *within) when `within` names extra keys — the API
    passes `("plant_id",)` to align a run's 788 plants each on their
    own clock. Returns the same shape, minus the last forecast hour of
    every group, with every value column replaced by the average of
    itself and the next hour's value.

    **This is the alignment the whole table depends on.** HRRR's
    radiation fields are instantaneous readings at valid_time. CAISO's
    hourly label is the mean over the hour that starts at valid_time.
    Comparing one to the other is a like-for-like error that hides in
    plain sight: it looks correct at midday, when the curve is flat,
    and it is worst at sunrise and sunset, when the curve is steep and
    the two quantities differ most.

    Measured on 2024-06-15, statewide: the instant at 02:00 UTC reads
    2.6 times CAISO's mean for that hour, and the instant at 13:00
    reads a fifth of it. Averaging the two ends of each hour first cut
    the whole day's mean error from 1264 MW to 775 MW, a 39% drop, with
    no change to the physics at all.

    A trapezoid, not something cleverer, because two points is all a
    forecast hour gives us. It slightly overshoots a sunrise ramp,
    which curves.

    No future information enters. The next hour's value comes from the
    same forecast run — it is something the model already knew at
    run_time, not something the day revealed later.

    A row survives only if the very next hour of the same run is
    present. That drops the last forecast hour of every run, which has
    no successor, and it drops the hour before a hole. Holes are real:
    some forecast hours were never archived, and the backfill records
    those runs as `partial`. Averaging across a two-hour gap would
    quietly report the mean of 13:00 and 15:00 as the 13:00 hour, which
    is not an hour mean of anything. One hour in 48 is a cheap price
    for a column that means exactly one thing everywhere.
    """
    skip = {*INDEX_COLUMNS, *within}
    values = [c for c in frame.columns if c not in skip]
    ordered = frame.sort_values(["run_time", *within, "valid_time"])
    grouped = ordered.groupby(["run_time", *within])

    next_value = grouped[values].shift(-1)
    next_time = grouped["valid_time"].shift(-1)
    adjacent = next_time - ordered["valid_time"] == pd.Timedelta(hours=1)

    averaged = ordered.copy()
    averaged[values] = (ordered[values] + next_value) / 2.0
    return averaged[adjacent].reset_index(drop=True)


def calendar(frame: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """Local hour and day of year, the two things weather cannot say.

    frame carries valid_time in UTC; timezone is the region's IANA key.
    Returns frame with `local_hour` and `day_of_year` added.

    Storage stays UTC and only this conversion is local, because the
    things these columns stand for are local: when people are awake,
    and where the sun sits in the year. A model given UTC hours has to
    learn that the meaning of "hour 20" slides by one every March and
    November.

    Day of year is left as a plain number rather than split into a sine
    and cosine pair. A tree splits on thresholds, so it can carve the
    year into seasons on its own; the smooth encoding buys a linear
    model something a tree does not need.
    """
    local = frame["valid_time"].dt.tz_convert(timezone)
    out = frame.copy()
    out["local_hour"] = local.dt.hour.astype("int32")
    out["day_of_year"] = local.dt.dayofyear.astype("int32")
    return out


def _spread(long: pd.DataFrame, variables: tuple[str, ...]) -> pd.DataFrame:
    """Long zone rows -> one `{zone}_{var}` column each, shape guaranteed.

    Reindexed over ZONES rather than over whatever zones happened to
    appear. A pivot invents its columns from the data, so a zone with
    no plants would quietly vanish and the table would change width
    between one year and the next. The declared schema is what catches
    that, and it can only catch it if the column is present and null.
    """
    wide = long.pivot(index=HOUR_KEYS, columns="zone", values=list(variables))
    wide.columns = [f"{zone}_{var}" for var, zone in wide.columns]
    expected = [f"{zone}_{var}" for zone in ZONES for var in variables]
    return wide.reindex(columns=expected)


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
