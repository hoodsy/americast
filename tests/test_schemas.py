import pandas as pd
import pyarrow as pa
import pytest

from americast.schemas import CAISO_SOLAR_5MIN, HRRR_WEATHER


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


def hrrr_frame() -> pd.DataFrame:
    run = pd.Timestamp("2024-06-01 06:00", tz="UTC")
    return pd.DataFrame(
        {
            "run_time": [run, run],
            "valid_time": [run + pd.Timedelta(hours=1), run + pd.Timedelta(hours=2)],
            "lead_hours": pd.array([1, 2], dtype="int32"),
            "plant_id": [56789, 56789],
            "dswrf": [0.0, 450.5],
            "tcdc": [100.0, 37.5],
            "t2m": [288.4, 290.1],
            "w10m": [3.2, 4.1],
        }
    )


def test_hrrr_conforming_frame_passes() -> None:
    table = pa.Table.from_pandas(
        hrrr_frame(), schema=HRRR_WEATHER, preserve_index=False
    )
    assert table.num_rows == 2


def test_hrrr_null_value_rejected() -> None:
    df = hrrr_frame()
    df.loc[0, "dswrf"] = None
    with pytest.raises(ValueError, match="non-nullable"):
        pa.Table.from_pandas(df, schema=HRRR_WEATHER, preserve_index=False)


def test_hrrr_int64_lead_hours_casts_to_int32() -> None:
    # Pinned: plain-int lead_hours arrive as int64; the schema cast narrows
    # them safely since values fit in int32.
    df = hrrr_frame()
    df["lead_hours"] = df["lead_hours"].astype("int64")
    table = pa.Table.from_pandas(df, schema=HRRR_WEATHER, preserve_index=False)
    assert table.schema.field("lead_hours").type == pa.int32()


def test_naive_timestamp_assumed_utc() -> None:
    # Pinned pyarrow behavior: the schema cast does NOT reject naive
    # timestamps — it assumes they are UTC. Writers must guard tz-awareness
    # themselves (append_to_store does).
    df = good_frame()
    df["utc_time"] = df["utc_time"].dt.tz_localize(None)
    table = pa.Table.from_pandas(df, schema=CAISO_SOLAR_5MIN, preserve_index=False)
    assert table.num_rows == 3
