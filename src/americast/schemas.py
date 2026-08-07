"""Table schemas: one declared pyarrow schema per stored dataset.

Writers must build their tables with `pa.Table.from_pandas(df, schema=...)`
so a wrong column, dtype, or unexpected null fails at write time, not at
read time three gates later.
"""

import pyarrow as pa

# CAISO fuel-mix solar at the feed's native 5-minute resolution.
# utc_time is the interval start. solar_mw is average power over the
# interval; small negatives at night are real (station service draw).
CAISO_SOLAR_5MIN = pa.schema(
    [
        pa.field("utc_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("solar_mw", pa.float64(), nullable=False),
    ]
)

# HRRR forecast fields extracted at plant locations. One row per
# (run_time, forecast hour, plant). Native GRIB units stored — dswrf W/m²,
# tcdc %, t2m Kelvin, w10m m/s — unit conversion happens at feature time
# only, so a unit bug is a cheap feature-code fix, never a re-download.
# lead_hours = valid_time - run_time, always 1..48 for our runs.
HRRR_WEATHER = pa.schema(
    [
        pa.field("run_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("valid_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("lead_hours", pa.int32(), nullable=False),
        pa.field("plant_id", pa.int64(), nullable=False),
        pa.field("dswrf", pa.float64(), nullable=False),
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
PLANTS_CA = pa.schema(
    [
        pa.field("plant_id", pa.int64(), nullable=False),
        pa.field("plant_name", pa.string(), nullable=False),
        pa.field("latitude", pa.float64(), nullable=False),
        pa.field("longitude", pa.float64(), nullable=False),
        pa.field("capacity_mw_ac", pa.float64(), nullable=False),
        pa.field("tracking", pa.string(), nullable=False),
        pa.field("county", pa.string(), nullable=False),
        pa.field("balancing_authority", pa.string(), nullable=False),
    ]
)
