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
