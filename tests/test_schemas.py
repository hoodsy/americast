import pandas as pd
import pyarrow as pa
import pytest

from americast.schemas import CAISO_SOLAR_5MIN


def good_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "utc_time": pd.date_range("2024-06-15", periods=3, freq="5min", tz="UTC"),
            "solar_mw": [0.0, 12.5, 340.0],
        }
    )


def test_conforming_frame_passes() -> None:
    table = pa.Table.from_pandas(
        good_frame(), schema=CAISO_SOLAR_5MIN, preserve_index=False
    )
    assert table.num_rows == 3


def test_null_solar_mw_rejected() -> None:
    df = good_frame()
    df.loc[1, "solar_mw"] = None
    with pytest.raises(ValueError, match="non-nullable"):
        pa.Table.from_pandas(df, schema=CAISO_SOLAR_5MIN, preserve_index=False)


def test_missing_column_rejected() -> None:
    df = good_frame().drop(columns=["solar_mw"])
    with pytest.raises(KeyError):
        pa.Table.from_pandas(df, schema=CAISO_SOLAR_5MIN, preserve_index=False)


def test_naive_timestamp_assumed_utc() -> None:
    # Pinned pyarrow behavior: the schema cast does NOT reject naive
    # timestamps — it assumes they are UTC. Writers must guard tz-awareness
    # themselves (append_to_store does).
    df = good_frame()
    df["utc_time"] = df["utc_time"].dt.tz_localize(None)
    table = pa.Table.from_pandas(df, schema=CAISO_SOLAR_5MIN, preserve_index=False)
    assert table.num_rows == 3
