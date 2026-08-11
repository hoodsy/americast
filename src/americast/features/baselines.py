"""The two forecasts the model has to beat, computed before it exists.

A baseline is not a formality. If a gradient-boosted model on 40
weather features cannot beat "yesterday, scaled", then the weather
features are not carrying information and the honest thing is to say
so. These are built first, and Gate 5 grades against them.

Both answer the same question in different ways: what did the sky do
recently, and will it do that again? Neither one reads a weather
forecast, which is exactly the point — the gap between them and the
model is the value of the forecast.

**Everything here is keyed on run_time, not valid_time.** A baseline
that used the day before the hour it predicts would be reading the
future for a lead of 30 hours. What the forecaster knew is what was
published before the run went out, and nothing later.
"""

import numpy as np
import pandas as pd

from americast.region import RegionConfig

# How many daylight hours a day must have before its clear-sky ratio is
# trusted. A day with two usable hours produces a ratio dominated by
# sunrise geometry, and that ratio would then scale a whole 48-hour
# forecast.
MIN_DAYLIGHT_HOURS = 6

# The window for smart persistence, in local days. Seven covers a full
# week, so it holds no day-of-week shape — solar has none — but it is
# long enough to survive one cloudy day and short enough to follow the
# season as it moves.
PERSISTENCE_DAYS = 7

# Below this the fleet ceiling is night or the edge of twilight, where
# the clearness ratio divides small numbers by smaller ones. In MW, at
# the scale of a 21.5 GW fleet.
DAYLIGHT_MW = 100.0


def attach(table: pd.DataFrame, region: RegionConfig) -> pd.DataFrame:
    """Add both baseline columns to the training table.

    table must already carry fleet_clear_mw and the joined solar_mw
    label. Returns it with `baseline_clear_sky_mw` and
    `baseline_smart_mw` added, both null wherever the run has no
    qualifying history behind it — which is most of the first week of
    the record, and any run following a gap.
    """
    out = table.copy()
    out["baseline_clear_sky_mw"] = clear_sky(table, region)
    out["baseline_smart_mw"] = smart(table, region)
    return out


def clear_sky(table: pd.DataFrame, region: RegionConfig) -> pd.Series:
    """Today's clear-sky curve, scaled by the last clear day's ratio.

    For every row: take the physical clear-sky ceiling for that hour,
    and multiply it by how much of its ceiling the fleet actually
    delivered on the most recent complete day before run_time.

    This is the baseline that knows the physics but not the forecast.
    It handles the two things naive persistence cannot: the seasonal
    march of day length, and the shape of a single day, both of which
    come free from the ceiling. What it cannot do is know that
    tomorrow is cloudier than today.

    The ratio is taken over a whole day's energy rather than hour by
    hour, so that one bad hour cannot swing the day. It is also why a
    ceiling that sits a few percent low does no damage: the same
    ceiling appears in the denominator here and the numerator in the
    multiplication, so a steady bias divides out.
    """
    daily = _daily_ratio(table, region)
    reference = _reference_day(table, daily)
    ratio = reference.map(daily["ratio"])
    return table["fleet_clear_mw"] * ratio.to_numpy()


def smart(table: pd.DataFrame, region: RegionConfig) -> pd.Series:
    """The same local hour, averaged over the last seven days.

    For every row: what the fleet produced at that hour of the day, on
    average, across the week before run_time.

    It knows nothing about the sun's geometry, so it lags every
    seasonal change by about half a week, and it cannot tell a clear
    day from a cloudy one. What it does have is the whole shape of a
    real day, including the parts the physics gets wrong — inverter
    startup, terrain shading, plants offline for maintenance. That is
    why it is a harder baseline than it looks, and why beating it at
    long leads is the real test.
    """
    by_hour = _hourly_history(table, region)
    rolling = by_hour.rolling(PERSISTENCE_DAYS, min_periods=PERSISTENCE_DAYS).mean()

    daily = _daily_ratio(table, region)
    reference = _reference_day(table, daily)
    local_hour = table["valid_time"].dt.tz_convert(region.timezone).dt.hour

    lookup = rolling.stack()
    keys = pd.MultiIndex.from_arrays([reference, local_hour])
    return pd.Series(lookup.reindex(keys).to_numpy(), index=table.index)


def _labelled_hours(table: pd.DataFrame, region: RegionConfig) -> pd.DataFrame:
    """One row per valid_time that carries a label: the shared history.

    The table holds the same valid_time under many run_times, so this
    collapses to one row each. The label is identical across those
    copies; the ceiling differs only by which run forecast the air
    temperature, so the mean of them is a stable curve.
    """
    labelled = table[table["solar_mw"].notna()]
    history = labelled.groupby("valid_time", as_index=False).agg(
        solar_mw=("solar_mw", "first"),
        fleet_clear_mw=("fleet_clear_mw", "mean"),
    )
    local = history["valid_time"].dt.tz_convert(region.timezone)
    history["local_date"] = local.dt.date
    history["local_hour"] = local.dt.hour
    return history


def _daily_ratio(table: pd.DataFrame, region: RegionConfig) -> pd.DataFrame:
    """Per local day: delivered energy over ceiling energy, and when it ended.

    `last_hour` is the latest daylight instant of that day. A run is
    only allowed to use a day whose last daylight hour is already past,
    which is what keeps a 48-hour forecast from quoting a day it has
    not seen the end of.
    """
    history = _labelled_hours(table, region)
    lit = history[history["fleet_clear_mw"] > DAYLIGHT_MW]
    grouped = lit.groupby("local_date")
    daily = grouped.agg(
        delivered=("solar_mw", "sum"),
        ceiling=("fleet_clear_mw", "sum"),
        hours=("solar_mw", "size"),
        last_hour=("valid_time", "max"),
    )
    usable = daily[daily["hours"] >= MIN_DAYLIGHT_HOURS].copy()
    usable["ratio"] = usable["delivered"] / usable["ceiling"]
    return usable.sort_index()


def _reference_day(table: pd.DataFrame, daily: pd.DataFrame) -> pd.Series:
    """The newest local day fully behind each row's run_time.

    searchsorted, not a join, because this is a "latest before" lookup
    against a sorted list — and it must compare against the day's last
    daylight hour rather than the date itself, or a run at 06z would
    claim a day that still had an afternoon left in it.
    """
    if daily.empty:
        return pd.Series(pd.NA, index=table.index, dtype="object")

    ends = daily["last_hour"].to_numpy()
    position = np.searchsorted(ends, table["run_time"].to_numpy(), side="left") - 1
    dates = daily.index.to_numpy()
    picked = np.where(position >= 0, dates[position.clip(min=0)], None)
    return pd.Series(picked, index=table.index, dtype="object")


def _hourly_history(table: pd.DataFrame, region: RegionConfig) -> pd.DataFrame:
    """Actual MW as a (local day x local hour) grid, gaps kept as gaps.

    Reindexed onto every calendar day in the span so that a missing day
    breaks the rolling window instead of silently shortening it — seven
    rows of a rolling mean must be seven consecutive days, not the last
    seven days that happened to have data.
    """
    history = _labelled_hours(table, region)
    grid = history.pivot_table(
        index="local_date", columns="local_hour", values="solar_mw", aggfunc="mean"
    )
    span = pd.date_range(min(grid.index), max(grid.index), freq="1D").date
    return grid.reindex(span)
