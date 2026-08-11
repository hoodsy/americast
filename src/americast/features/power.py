"""Per-plant PV power: irradiance at a coordinate -> AC megawatts.

Each plant's output is modelled from the weather at its own 3 km
gridpoint, and those estimates sum to county, zone and state. Only the
state total can be graded against CAISO, but the levels below it are
what a map needs, and they come free once the per-plant number exists.

The chain is: where is the sun (`position`), where the panels point
(`orient`), how much light reaches the panel plane (`poa`), how hot
the cells get (`heat`), what the panels make (`generate`), and what
the inverter lets through. Each step takes the step before it and adds
its columns.
"""

import numpy as np
import pandas as pd
import pvlib

# Below this elevation the sun is setting, refraction dominates, and
# 1/cos(zenith) runs away. HRRR's own numbers stop behaving here too:
# across June 2024 its only two negative dhi values both sit at a
# zenith near 89.4. Geometry past this line is not worth trusting, and
# cutting it off costs 0.0018% of the month's total irradiance.
HORIZON_ZENITH = 89.0

# Tracking types that follow the sun about one horizontal axis.
# "unknown" joins them because 18.2 of the fleet's 21.5 GW tracks, so
# an unlabelled plant is far likelier to track than not.
TRACKED = ("single_axis", "unknown")
KNOWN_TRACKING = ("fixed", "single_axis", "dual_axis", "unknown")

# Every tracker turns about a north-south axis, whatever azimuth the
# registry reports. The reason is a symmetry the data itself shows.
#
# An axis is a line, so either heading describes it equally well, and
# respondents who mean an axis should split between the two. For
# north-south they do: 6,821 MW says 0 and 8,959 MW says 180, a ratio
# of 1.31 to 1. For east-west they do not: 3,341 MW says 90 and 14 MW
# says 270, a ratio of 247 to 1. A line cannot have a preferred
# direction. That column is recording which way the panels face, and a
# respondent writing one direction naturally writes the morning one.
#
# It agrees with the plants themselves — the 90s include Daggett 3,
# California Valley Solar Ranch and Athos, all documented horizontal
# single-axis installations, which by construction turn about a
# north-south axis to follow the sun's daily arc.
AXIS_AZIMUTH = 180.0

# Where a reported tilt stops being an axis angle and starts being a
# rotation limit. EIA asks trackers for their axis tilt, and some
# answer with how far the tracker swings instead. The two do not
# overlap in practice: no utility-scale tracker stands its axis past
# about 30 degrees, and no rotation limit is set below it. Splitting
# there keeps both readings — 377 MW of genuinely tilted axes and
# 2.7 GW of real per-plant rotation limits.
AXIS_TILT_LIMIT = 30.0

# The rotation limit for the 17.1 GW that reports none. 60 is modal
# among those that do report one, at 59% of that capacity, and is the
# modern standard.
MAX_ROTATION = 60.0

# Ground coverage ratio: panel area over land area. Not reported by
# EIA at all, so this is a fleet constant. Utility-scale single-axis
# rows sit around a third, and the value only bites through
# backtracking, below.
GROUND_COVER = 0.33

# How the sky's diffuse light is spread across the dome. Isotropic
# treats it as uniform, which is simple and always underestimates a
# clear sky, because more diffuse light arrives from right around the
# sun than from the opposite horizon. Hay-Davies adds that circumsolar
# brightening. Perez adds a horizon band as well and is the most
# accurate published model, but it leans on empirical coefficients that
# misbehave at low sun — the hours a tracker is already working hardest.
# Hay-Davies is the steadier choice, and a steady bias is what the
# learned model downstream is best placed to absorb.
SKY_MODEL = "haydavies"

# Fraction of light the ground throws back up at the panels. Not
# reported per plant, and it only reaches a panel that is tilted, so it
# is a small term for a fleet that spends midday lying flat. 0.25 is
# pvlib's default and sits between desert sand and farmland.
GROUND_ALBEDO = 0.25

# How the panels are mounted, for the cell temperature model. Every
# utility-scale plant stands on an open rack, so that half is certain.
# The glass/glass against glass/polymer half is not: EIA reports no
# module construction, and the fleet holds both — glass and backsheet
# on the older plants, glass on both faces on the bifacial ones built
# since about 2021. Measured at 1000 W/m², the two sets run 2.9 to
# 3.6 degrees apart across 1 to 5 m/s of wind, which is near 1% of
# power. That is a steady bias for the learned model to absorb, not a
# number worth guessing per plant.
THERMAL_MOUNT = "open_rack_glass_glass"

# HRRR stores 2 m temperature in Kelvin. pvlib's thermal model wants
# Celsius, and nothing downstream can catch the confusion: a cell at
# 300 instead of 27 still multiplies cleanly, and only the power comes
# out wrong.
KELVIN_ZERO = 273.15

# The power a cell loses for each degree above 25 C. EIA reports no
# module type, so this is a fleet constant like the two above. Modern
# mono-PERC datasheets cluster at 0.34 to 0.37% per degree, and this
# fleet was built mostly after 2015. PVWatts' own "standard" default of
# 0.47% describes the older polycrystalline modules that California's
# utility-scale fleet largely is not.
GAMMA_PDC = -0.0035


def position(hours: pd.DataFrame, plants: pd.DataFrame) -> pd.DataFrame:
    """Solar zenith and azimuth for every (forecast hour, plant) row.

    hours carries valid_time and plant_id — the weather frame works
    directly. plants supplies latitude and longitude per plant_id.
    Returns hours with `zenith` and `solar_azimuth` added, both in
    degrees, plus `cos_zenith` floored at the horizon.

    Two things this deliberately does not do. It does not round-trip
    through local time: the sun's position depends on the instant and
    the coordinate, and valid_time is already an unambiguous UTC
    instant, so timezone conversion could only introduce a DST bug.
    And it does not compute one position for the whole fleet — the
    zone centroids span 4.6 degrees of longitude, which is 18 minutes
    of solar time, and that gap lands exactly at the shoulders of the
    day where the forecast is hardest.

    `apparent_zenith` is the refracted angle, the direction the sun
    appears to be rather than where it geometrically is. Refraction
    lifts the low sun by about half a degree, which matters only near
    sunrise and sunset — but that is precisely where a tracker is
    rotating hardest, so the apparent angle is the honest input.
    """
    missing = set(hours["plant_id"]) - set(plants["plant_id"])
    if missing:
        raise ValueError(f"{len(missing)} plant_ids absent from the registry")

    coords = plants[["plant_id", "latitude", "longitude"]]
    rows = hours.merge(coords, on="plant_id", how="left")

    # pvlib takes one (time, latitude, longitude) triple per row and
    # returns a frame in the same order, so this is a column-wise
    # calculation over the whole frame rather than a loop over plants.
    solar = pvlib.solarposition.get_solarposition(
        pd.DatetimeIndex(rows["valid_time"]),
        rows["latitude"].to_numpy(),
        rows["longitude"].to_numpy(),
    )
    out = rows.drop(columns=["latitude", "longitude"])
    out["zenith"] = solar["apparent_zenith"].to_numpy()
    out["solar_azimuth"] = solar["azimuth"].to_numpy()
    out["cos_zenith"] = _cos_zenith(out["zenith"])
    return out


def orient(positioned: pd.DataFrame, plants: pd.DataFrame) -> pd.DataFrame:
    """Where each plant's panels point, hour by hour.

    positioned is `position`'s output; plants supplies tracking, tilt
    and azimuth. Returns positioned with `surface_tilt` and
    `surface_azimuth` added — the plane the panels present to the sky,
    in the same degrees-from-horizontal and degrees-clockwise-from-north
    that pvlib's transposition step expects next.

    Three kinds of mounting, three answers:

    Fixed panels never move, so the registry's own angles are the
    answer. This is the one case where EIA's numbers are used as
    reported, and the one case where they vary meaningfully — 64% of
    fixed capacity faces due south, but a fifth of it faces southeast.

    Single-axis trackers chase the sun east to west about a north-south
    axis. Their reported tilt is read by magnitude: at or below
    AXIS_TILT_LIMIT it is how far the axis itself leans, above it is
    how far the tracker swings before it stops. Backtracking is on:
    packed close together, rows shade their neighbours near sunrise and
    sunset, so a real tracker gives up some of its angle to stay out of
    its own shadow. Turning it off would claim light that the row in
    front is standing in.

    Dual-axis mounts point straight at the sun, so the panel plane is
    simply the sun's own position.

    After dark every mount is parked flat. pvlib returns NaN for a
    tracker once the sun is down, and a NaN here would spread through
    the irradiance arithmetic and poison whole plant-days. There is no
    light to catch at that hour, so flat is both safe and true.
    """
    unknown = set(plants["tracking"]) - set(KNOWN_TRACKING)
    if unknown:
        raise ValueError(f"unrecognised tracking types: {sorted(unknown)}")

    columns = ["plant_id", "tracking", "tilt", "azimuth"]
    rows = positioned.merge(plants[columns], on="plant_id", how="left")
    tilt = pd.Series(np.nan, index=rows.index, dtype="float64")
    azimuth = pd.Series(np.nan, index=rows.index, dtype="float64")

    fixed = rows["tracking"] == "fixed"
    tilt[fixed] = rows.loc[fixed, "tilt"]
    azimuth[fixed] = rows.loc[fixed, "azimuth"]

    dual = rows["tracking"] == "dual_axis"
    tilt[dual] = rows.loc[dual, "zenith"]
    azimuth[dual] = rows.loc[dual, "solar_azimuth"]

    turning = rows[rows["tracking"].isin(TRACKED)]
    for (axis_tilt, limit), group in turning.groupby(
        [_axis_tilt(turning["tilt"]), _rotation_limit(turning["tilt"])]
    ):
        turned = _rotate(group, axis_tilt, limit)
        tilt[group.index] = turned["surface_tilt"].to_numpy()
        azimuth[group.index] = turned["surface_azimuth"].to_numpy()

    dark = rows["cos_zenith"] == 0.0
    tilt[dark] = 0.0
    azimuth[dark] = AXIS_AZIMUTH

    out = rows.drop(columns=["tracking", "tilt", "azimuth"])
    out["surface_tilt"] = tilt
    out["surface_azimuth"] = azimuth
    return out


def poa(oriented: pd.DataFrame) -> pd.DataFrame:
    """Irradiance landing on the panel face, in W/m².

    oriented is `orient`'s output, carrying both the sun's position and
    the plane the panels present to it, alongside HRRR's dni, dhi and
    dswrf. Returns it with `poa` added.

    Three separate contributions arrive at a panel, and pvlib sums
    them. The direct beam, dni scaled by how squarely it strikes the
    glass. Light scattered by the sky dome, from dhi, spread according
    to SKY_MODEL. And light bouncing off the ground, which reaches only
    a tilted panel and is small for a fleet that lies flat at midday.

    This is the step where a panel can beat flat ground. A tracker
    facing a low morning sun collects far more than a horizontal
    surface at the same moment, which is the whole reason 18.2 of the
    fleet's 21.5 GW tracks — and it is invisible in dswrf alone.

    Everything is forced to zero once the sun is at the horizon.
    Nothing else in this module can do it: the parked-flat panel from
    `orient` still presents a face to a sun 89.4 degrees up, and HRRR's
    dni there can read 3200 W/m², which would put 34 W/m² on the glass
    after dark. cos_zenith is already zero on those rows, so this is
    the same horizon rule applied where it would otherwise leak.
    """
    extra = pvlib.irradiance.get_extra_radiation(
        pd.DatetimeIndex(oriented["valid_time"])
    )
    sky = pvlib.irradiance.get_total_irradiance(
        surface_tilt=oriented["surface_tilt"].to_numpy(),
        surface_azimuth=oriented["surface_azimuth"].to_numpy(),
        solar_zenith=oriented["zenith"].to_numpy(),
        solar_azimuth=oriented["solar_azimuth"].to_numpy(),
        dni=oriented["dni"].clip(lower=0.0).to_numpy(),
        ghi=oriented["dswrf"].clip(lower=0.0).to_numpy(),
        dhi=oriented["dhi"].clip(lower=0.0).to_numpy(),
        dni_extra=np.asarray(extra),
        albedo=GROUND_ALBEDO,
        model=SKY_MODEL,
    )
    total = pd.Series(np.asarray(sky["poa_global"]), index=oriented.index)
    lit = oriented["cos_zenith"] > 0.0

    out = oriented.copy()
    out["poa"] = total.where(lit, 0.0).fillna(0.0)
    return out


def heat(irradiated: pd.DataFrame) -> pd.DataFrame:
    """How hot the cells run, in degrees Celsius.

    irradiated is `poa`'s output, carrying the light on the glass
    alongside HRRR's t2m and w10m. Returns it with `cell_temperature`
    added.

    A hot cell makes less power — roughly 0.35% less for every degree
    above 25 C — and a panel in full sun sits 25 to 35 degrees above the
    air around it. Air temperature alone would therefore overstate a
    desert afternoon by ten percent or more, and the afternoon is where
    the peak the grid cares about is set.

    The Sandia model, chosen for where it measures the wind. It heats
    the module by the light it absorbs, cools it by a factor that decays
    with wind speed, then adds a small step for the gap between the
    module back and the cell inside. pvlib's other thermal models
    (faiman, pvsyst_cell) want the wind at module height, near 2 m. This
    one wants it at 10 m, which is exactly what HRRR stores. The others
    would first need our wind pushed down a log profile, and that
    correction is a guess this does not have to make.

    No horizon rule is necessary. `poa` is already zero after dark, and
    with no light the model returns the air temperature, which is what
    an unlit panel really sits at. There is no division and no exponent
    that can run away, so a NaN cannot start here either.

    Thermal mass is ignored. A panel settles to a new temperature in 10
    to 20 minutes and these rows stand an hour apart, so the steady
    state answer is the right one at this resolution.
    """
    coefficients = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][THERMAL_MOUNT]
    air_celsius = irradiated["t2m"] - KELVIN_ZERO
    cell = pvlib.temperature.sapm_cell(
        poa_global=irradiated["poa"].to_numpy(),
        temp_air=air_celsius.to_numpy(),
        wind_speed=irradiated["w10m"].to_numpy(),
        **coefficients,
    )

    out = irradiated.copy()
    out["cell_temperature"] = cell
    return out


def generate(heated: pd.DataFrame, plants: pd.DataFrame) -> pd.DataFrame:
    """DC megawatts at the panels, before any inverter sees them.

    heated is `heat`'s output, carrying the light on the glass and the
    cell temperature. plants supplies dc_capacity_mw and
    operating_date. Returns heated with `dc_mw` added.

    The PVWatts model, which is three numbers multiplied: the plant's
    DC nameplate, the fraction of standard light that arrives, and a
    correction for a cell that is not at 25 C.

        dc = dc_capacity_mw * (poa / 1000) * (1 + GAMMA_PDC * (cell - 25))

    It is the right model because of what EIA gives us. A single-diode
    model is more accurate, but it wants cells in series, saturation
    currents and a module part number, and EIA reports none of them.
    What EIA does report is the DC nameplate, which is the plant's
    output under exactly the conditions this equation references —
    1000 W/m² on the panel and a 25 C cell. Feed those two numbers in
    and the answer is the nameplate itself, by construction.

    This deliberately returns more than the plant can deliver. The
    fleet holds 30.4 GW of panel behind 23.9 GW of inverter, a loading
    ratio of 1.275, so a clear midday drives DC well past what the grid
    ever sees. Cutting it back is the inverter's job and belongs to the
    next step, not this one — the size of that gap is the whole reason
    the registry's DC column was worth chasing.

    **A plant that does not exist yet makes nothing.** The registry is
    one 2025 snapshot, but the weather store reaches back to 2023, so
    every row asks about plants at 2025 sizes. Only 70.4% of today's
    capacity was running on 2023-01-01. Without the operating_date
    test, a 2023 January hour would collect power from panels that were
    still fields, and the model would learn that 2023 was sunnier than
    it was. This is the exact place to stop it, because it is the first
    step where capacity enters the arithmetic — and every later sum,
    including the clear-sky ceiling, inherits the correction for free.

    Two known omissions, both steady and both small. Light that
    reflects off the glass at a sharp angle is not removed here, and
    soiling, wiring and mismatch losses are not either. They belong
    with the inverter's own efficiency in the next step, where one
    loss factor can carry them together.
    """
    columns = ["plant_id", "dc_capacity_mw", "operating_date"]
    rows = heated.merge(plants[columns], on="plant_id", how="left")
    watts = pvlib.pvsystem.pvwatts_dc(
        effective_irradiance=rows["poa"].to_numpy(),
        temp_cell=rows["cell_temperature"].to_numpy(),
        pdc0=rows["dc_capacity_mw"].to_numpy(),
        gamma_pdc=GAMMA_PDC,
    )
    built = rows["valid_time"] >= rows["operating_date"]
    produced = pd.Series(watts, index=rows.index)

    out = rows.drop(columns=columns[1:])
    out["dc_mw"] = produced.where(built, 0.0)
    return out


def _axis_tilt(tilt: pd.Series) -> pd.Series:
    """The part of a reported tilt that can only be an axis angle."""
    return tilt.where(tilt <= AXIS_TILT_LIMIT, 0.0)


def _rotation_limit(tilt: pd.Series) -> pd.Series:
    """The part of a reported tilt that can only be a rotation limit."""
    return tilt.where(tilt > AXIS_TILT_LIMIT, MAX_ROTATION)


def _rotate(group: pd.DataFrame, axis_tilt: float, limit: float) -> pd.DataFrame:
    """Rotate one set of trackers that share an axis tilt and limit.

    pvlib takes these two as scalars, not per row, which is why the
    caller groups first. There are only about two dozen distinct pairs
    across the fleet, so this runs a couple of dozen times rather than
    once per plant.

    Wrapped because pvlib hands back a plain dict for array input and a
    DataFrame for Series input, and the caller should not have to care.
    """
    turned = pvlib.tracking.singleaxis(
        group["zenith"].to_numpy(),
        group["solar_azimuth"].to_numpy(),
        axis_tilt=axis_tilt,
        axis_azimuth=AXIS_AZIMUTH,
        max_angle=limit,
        backtrack=True,
        gcr=GROUND_COVER,
    )
    return pd.DataFrame(turned) if isinstance(turned, dict) else turned


def _cos_zenith(zenith: pd.Series) -> pd.Series:
    """cos(zenith), zeroed once the sun is at or below the horizon.

    Kept as its own step because every later stage divides or
    multiplies by it, and a stray negative here would quietly turn into
    negative power after dark. Zero is the right floor: no sun above
    the horizon means no beam on a horizontal surface.
    """
    below = zenith >= HORIZON_ZENITH
    cosine = np.cos(np.radians(zenith))
    return cosine.where(~below, 0.0)
