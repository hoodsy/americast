"""Per-plant PV power: irradiance at a coordinate -> AC megawatts.

Each plant's output is modelled from the weather at its own 3 km
gridpoint, and those estimates sum to county, zone and state. Only the
state total can be graded against CAISO, but the levels below it are
what a map needs, and they come free once the per-plant number exists.

The chain is: where is the sun (`position`), how much light reaches
the panel plane, how hot the cells get, and what the inverter lets
through. This module holds one step at a time; `position` is the
first.
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

# Every tracker is modelled on a flat north-south axis, whatever the
# registry says its azimuth is. A horizontal single-axis tracker rotates
# to follow the sun's daily east-to-west arc, and the axis it turns
# about must lie perpendicular to that travel — so north-south. 81.5%
# of tracked capacity is recorded that way (0 or 180, the same line
# written two ways). The other 15% reads 90, which would be an
# east-west axis following the sun's slow seasonal drift instead of
# its daily arc. Those are not exotic plants: they include Daggett 3,
# California Valley Solar Ranch and Athos, all documented horizontal
# single-axis installations. They are the panel sweep direction
# reported where the axis was asked for.
AXIS_AZIMUTH = 180.0

# How far a tracker can rotate before it stops following the sun.
# Recovered from the tilt column EIA cannot report consistently: among
# single-axis generators giving a non-zero "tilt", 84% of capacity
# reads 60, 52 or 45 degrees — rotation limits, not axis tilts. 60 is
# the modal value at 52% of that capacity and the modern standard;
# their capacity-weighted mean is 52.6, pulled down by a residue of
# genuine axis tilts near 25.
MAX_ROTATION = 60.0

# Ground coverage ratio: panel area over land area. Not reported by
# EIA at all, so this is a fleet constant. Utility-scale single-axis
# rows sit around a third, and the value only bites through
# backtracking, below.
GROUND_COVER = 0.33


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

    Single-axis trackers rotate about a flat north-south axis, chasing
    the sun east to west and stopping at MAX_ROTATION. Backtracking is
    on: packed close together, rows shade their neighbours near sunrise
    and sunset, so a real tracker gives up some of its angle to stay
    out of its own shadow. Turning it off would claim light that the
    row in front is standing in.

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

    tracked = rows["tracking"].isin(TRACKED)
    if tracked.any():
        turned = _rotate(rows.loc[tracked, "zenith"], rows.loc[tracked, "solar_azimuth"])
        tilt[tracked] = turned["surface_tilt"].to_numpy()
        azimuth[tracked] = turned["surface_azimuth"].to_numpy()

    dark = rows["cos_zenith"] == 0.0
    tilt[dark] = 0.0
    azimuth[dark] = AXIS_AZIMUTH

    out = rows.drop(columns=["tracking", "tilt", "azimuth"])
    out["surface_tilt"] = tilt
    out["surface_azimuth"] = azimuth
    return out


def _rotate(zenith: pd.Series, solar_azimuth: pd.Series) -> pd.DataFrame:
    """One tracker rotation, with the fleet's axis and limits applied.

    Split out because pvlib hands back a plain dict for array input and
    a DataFrame for Series input, and the caller should not have to
    care which.
    """
    turned = pvlib.tracking.singleaxis(
        zenith.to_numpy(),
        solar_azimuth.to_numpy(),
        axis_tilt=0.0,
        axis_azimuth=AXIS_AZIMUTH,
        max_angle=MAX_ROTATION,
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
