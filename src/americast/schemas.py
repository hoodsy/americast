"""Table schemas: one declared pyarrow schema per stored dataset.

Writers must build their tables with `pa.Table.from_pandas(df, schema=...)`
so a wrong column, dtype, or unexpected null fails at write time, not
months later when something downstream reads it back.
"""

import pyarrow as pa

from americast.features.county import ZONES

# CAISO fuel-mix solar at the feed's native 5-minute resolution.
# utc_time is the interval start. solar_mw is average power over the
# interval; small negatives at night are real (station service draw).
CAISO_SOLAR_5MIN = pa.schema(
    [
        pa.field("utc_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("solar_mw", pa.float64(), nullable=False),
    ]
)

# CAISO's published wind-and-solar curtailment, solar only, hourly.
#
# utc_time is the interval start, matching the hourly label. Curtailed
# solar is sunlight the plants could have converted and were instructed
# not to — it never reaches the fuel mix, so it is absent from
# CAISO_SOLAR_5MIN by construction. `solar_mw + curtailed_mw` is
# therefore what the fleet would have produced if the grid had taken
# everything it offered, which is the quantity the physical model
# actually estimates.
#
# curtailed_mw comes from CAISO's MWh column over a one-hour interval,
# so it is a mean power like every other value in this project.
#
# CAISO's second column, an instantaneous peak reduction, is
# deliberately not stored. The legacy report leaves it blank on some
# System-reason rows while still reporting the energy, so summing
# categories gives an hour whose "peak" is below its own mean. A column
# that is trustworthy after mid-2025 and not before is a column holding
# two definitions, and the energy figure is the only one this project
# can add to a label.
CAISO_CURTAILMENT = pa.schema(
    [
        pa.field("utc_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("curtailed_mw", pa.float64(), nullable=False),
    ]
)

# The live forecast store: what the model published, before anyone
# knows whether it was right.
#
# One row per (run_time, valid_time), appended once a day and never
# rewritten. That is the whole point of the file — a forecast that can
# be edited after the fact is not a forecast, and a scoreboard built on
# one is not evidence. `grade_daily` reads these rows and writes its
# verdict elsewhere.
#
# lead_hours runs 1 to 47, not 48: features.hourly averages each
# instant with the next one to match the label's hour-mean convention,
# so the final forecast hour has no successor and is dropped.
LIVE_FORECASTS = pa.schema(
    [
        pa.field("run_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("valid_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("lead_hours", pa.int32(), nullable=False),
        pa.field("p10_mw", pa.float64(), nullable=False),
        pa.field("p50_mw", pa.float64(), nullable=False),
        pa.field("p90_mw", pa.float64(), nullable=False),
        pa.field("fleet_ac_mw", pa.float64(), nullable=False),
        pa.field("fleet_clear_mw", pa.float64(), nullable=False),
    ]
)

# The scoreboard: yesterday's forecasts joined to what happened.
#
# Separate from LIVE_FORECASTS so that grading can be re-run — a label
# arriving late, or a CAISO revision — without ever touching the
# forecast that was published. error_mw is signed, p50 minus actual, so
# a positive value is an over-prediction. inside_band answers the only
# question the confidence interval makes: did the truth land in it.
LIVE_SCORES = pa.schema(
    [
        pa.field("run_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("valid_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("lead_hours", pa.int32(), nullable=False),
        pa.field("p10_mw", pa.float64(), nullable=False),
        pa.field("p50_mw", pa.float64(), nullable=False),
        pa.field("p90_mw", pa.float64(), nullable=False),
        pa.field("solar_mw", pa.float64(), nullable=False),
        pa.field("error_mw", pa.float64(), nullable=False),
        pa.field("inside_band", pa.bool_(), nullable=False),
    ]
)

# EIA-923 monthly net generation, solar plants only.
#
# The only per-plant truth this project has. CAISO publishes one number
# for the whole state, which can say that the fleet out-produces the
# physical model but never which plants do. This can.
#
# month is the first instant of the reporting month, UTC. Net
# generation is metered at the plant's grid connection over the whole
# month, so it is energy, not power, and it is net of station service —
# the same convention as the CAISO label, one level down.
#
# Not every plant reports here. EIA collects monthly from larger
# generators and annually from the rest, so this covers about two
# thirds of CISO capacity. It is a sample, and a sample is enough to
# compare one vintage against another.
EIA923_SOLAR_MONTHLY = pa.schema(
    [
        pa.field("plant_id", pa.int64(), nullable=False),
        pa.field("month", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("net_generation_mwh", pa.float64(), nullable=False),
    ]
)

# HRRR forecast fields extracted at plant locations. One row per
# (run_time, forecast hour, plant). Native GRIB units stored — dswrf,
# dni and dhi W/m², tcdc %, t2m Kelvin, w10m m/s — unit conversion
# happens at feature time only, so a unit bug is a cheap feature-code
# fix, never a re-download.
# lead_hours = valid_time - run_time, always 1..48 for our runs.
#
# The three radiation fields are instantaneous values at valid_time, not
# means over the hour, and they are tied together by
#     dswrf = dni * cos(zenith) + dhi
# dni is measured normal to the beam; dswrf and dhi are on a horizontal
# surface. Keep that identity in mind before "fixing" an apparent
# mismatch between them.
HRRR_WEATHER = pa.schema(
    [
        pa.field("run_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("valid_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("lead_hours", pa.int32(), nullable=False),
        pa.field("plant_id", pa.int64(), nullable=False),
        pa.field("dswrf", pa.float64(), nullable=False),
        pa.field("dni", pa.float64(), nullable=False),
        pa.field("dhi", pa.float64(), nullable=False),
        pa.field("tcdc", pa.float64(), nullable=False),
        pa.field("t2m", pa.float64(), nullable=False),
        pa.field("w10m", pa.float64(), nullable=False),
    ]
)

# California utility-scale solar PV plant registry from EIA-860
# (operating plants, plant-level aggregation of the generator schedule).
# tracking is "single_axis" | "dual_axis" | "fixed" | "unknown" — the
# capacity-dominant type when a plant mixes them. county and
# balancing_authority (EIA code, e.g. CISO/LDWP/BANC/IID) support
# sub-state grouping; unknowns are filled with "UNKNOWN", never null.
#
# capacity_mw_ac is the grid-side limit; dc_capacity_mw is the panel
# side. Their ratio is the inverter loading ratio, and the gap between
# them is what makes a plant clip on a clear midday.
#
# tilt and azimuth are degrees, azimuth clockwise from true north. What
# they describe depends on tracking: for "fixed" they are the panel
# plane, for "single_axis" they are the tracker axis. A north-south
# axis is therefore recorded as 0 or 180 — the same line either way.
# Never average azimuth across plants; 0 and 180 average to an
# east-west axis that exists nowhere.
#
# tilt is stored exactly as EIA reported it, blanks aside. On a fixed
# mount it is the panel angle; on a tracker it is either the axis tilt
# or the rotation limit, depending on which question the respondent
# answered. features/power.py splits the two by magnitude.
#
# operating_date is the month the plant's first phase started
# generating, stored so historical aggregation can drop plants that did
# not exist yet.
PLANTS_CISO = pa.schema(
    [
        pa.field("plant_id", pa.int64(), nullable=False),
        pa.field("plant_name", pa.string(), nullable=False),
        pa.field("latitude", pa.float64(), nullable=False),
        pa.field("longitude", pa.float64(), nullable=False),
        pa.field("capacity_mw_ac", pa.float64(), nullable=False),
        pa.field("dc_capacity_mw", pa.float64(), nullable=False),
        pa.field("tracking", pa.string(), nullable=False),
        pa.field("tilt", pa.float64(), nullable=False),
        pa.field("azimuth", pa.float64(), nullable=False),
        pa.field("county", pa.string(), nullable=False),
        pa.field("balancing_authority", pa.string(), nullable=False),
        pa.field("operating_date", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

# The model's training table: one row per (run_time, valid_time).
#
# Every value column describes the HOUR THAT STARTS at valid_time, not
# the instant. HRRR reports instants, and the CAISO label is a mean
# over the hour, so features/features.py averages each instant with the
# next one before anything lands here. That single alignment cut the
# physical estimate's error by 39% on the pilot day; see
# docs/training_table.md.
#
# Weather columns are capacity-weighted means in native GRIB units, per
# zone and across the fleet. ac_mw and clear_mw are sums of the
# per-plant physical model — what the panels would make under HRRR's
# sky and under a clear one. Their ratio is the clearness index, and
# clear_mw is what the persistence baseline scales.
#
# solar_mw is the label: CAISO's reported utility-scale solar, averaged
# over the hour. It is nullable because the weather store reaches hours
# the label store has not yet covered, and n_intervals says how many
# 5-minute readings stand behind it (12 = complete).
#
# The two baseline columns are stored beside the label, not computed at
# evaluation time, so that the number the model is graded against is
# fixed in the same artifact as the model's inputs. Both are null where
# the run has no qualifying history behind it.
_ZONE_WEATHER = [
    pa.field(f"{zone}_{var}", pa.float64(), nullable=False)
    for zone in ZONES
    for var in ("dswrf", "tcdc", "t2m", "w10m")
]
_ZONE_POWER = [
    pa.field(f"{zone}_{var}", pa.float64(), nullable=False)
    for zone in ZONES
    for var in ("ac_mw", "clear_mw")
]
TRAIN_TABLE = pa.schema(
    [
        pa.field("run_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("valid_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("lead_hours", pa.int32(), nullable=False),
        *_ZONE_WEATHER,
        *[
            pa.field(f"fleet_{var}", pa.float64(), nullable=False)
            for var in ("dswrf", "tcdc", "t2m", "w10m")
        ],
        *_ZONE_POWER,
        pa.field("fleet_ac_mw", pa.float64(), nullable=False),
        pa.field("fleet_clear_mw", pa.float64(), nullable=False),
        pa.field("fleet_cos_zenith", pa.float64(), nullable=False),
        pa.field("local_hour", pa.int32(), nullable=False),
        pa.field("day_of_year", pa.int32(), nullable=False),
        pa.field("solar_mw", pa.float64(), nullable=True),
        pa.field("n_intervals", pa.int32(), nullable=True),
        pa.field("baseline_clear_sky_mw", pa.float64(), nullable=True),
        pa.field("baseline_smart_mw", pa.float64(), nullable=True),
    ]
)
