"""Feature engineering: plant-level weather -> one row per forecast hour.

The model predicts one number (statewide CAISO solar MW) per forecast
hour, but the weather store holds 788 plant rows for that same hour.
This module does the collapse -- and it collapses inside weather zones
rather than statewide, because fog in the Bay and clear sky in the
Mojave average to "half cloudy", which describes neither place.
"""

import pandas as pd

# The balancing authority whose output the label measures. Plants in
# LDWP, IID, BANC, PACW, and WALC sit in California but not in CAISO's
# reported number, so their weather cannot explain it.
CISO_BA = "CISO"

# Zone -> the counties it covers. Written this way round because the
# review question is "which counties are in the Mojave?", not the
# reverse. Two known compromises: San Luis Obispo's 1.1 GW is on the
# Carrizo Plain, which is drier than the word "coastal" suggests; and
# the Sierra foothill counties (Placer, El Dorado, Amador, Tuolumne)
# plus Lassen are filed under central_valley for want of a better home.
# Both are under 25 MW except San Luis Obispo.
ZONE_COUNTIES = {
    "kern": ("Kern",),
    "mojave": ("Riverside", "San Bernardino", "Los Angeles", "Inyo"),
    "imperial": ("Imperial", "San Diego"),
    "central_valley": (
        "Fresno",
        "Kings",
        "Tulare",
        "Merced",
        "Stanislaus",
        "Madera",
        "San Benito",
        "San Joaquin",
        "Sacramento",
        "Yolo",
        "Solano",
        "Butte",
        "Tehama",
        "Shasta",
        "Placer",
        "El Dorado",
        "Amador",
        "Tuolumne",
        "Lassen",
    ),
    "coastal": (
        "San Luis Obispo",
        "Santa Barbara",
        "Ventura",
        "Monterey",
        "Santa Cruz",
        "Contra Costa",
        "Alameda",
        "Santa Clara",
        "San Mateo",
        "San Francisco",
        "Marin",
        "Napa",
        "Sonoma",
        "Humboldt",
        "Orange",
    ),
}

# Inverted once at import: lowercase county -> zone. Lowercase because
# EIA-860 spells one Kern row "KERN".
COUNTY_ZONE = {}
for _zone, _counties in ZONE_COUNTIES.items():
    for _county in _counties:
        COUNTY_ZONE[_county.lower()] = _zone


def fleet(plants: pd.DataFrame) -> pd.DataFrame:
    """The plants we model: CISO only, each labeled with its weather zone.

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
