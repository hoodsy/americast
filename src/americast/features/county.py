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

# Sorted, because a pandas pivot emits its columns sorted and the
# training table's declared schema has to predict that order exactly.
# One tuple decides both, so they cannot drift apart.
ZONES = tuple(sorted(ZONE_COUNTIES))
