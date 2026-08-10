"""Per-plant PV power: irradiance at a coordinate -> AC megawatts.

Gate 4 is bottom-up. Each plant's output is modelled from the weather
at its own 3 km gridpoint, and those estimates sum to county, zone and
state. Only the state total can be graded against CAISO, but the
levels below it are what the map shows, and they come free once the
per-plant number exists.

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
# the two negative dhi values in the June 2024 pilot both sit at a
# zenith near 89.4. Geometry past this line is not worth trusting, and
# there is no meaningful power to lose by cutting it off.
HORIZON_ZENITH = 89.0


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
