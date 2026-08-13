"""EIA-923 parsing: wrapped headers, placeholder cells, repeated plants.

The workbook's headers carry embedded newlines and the wrapping has
moved between vintages, so every column here is found by flattened
text rather than by literal name. These tests pin that, and pin the
three ways a plant-month can be wrong while looking right.
"""

import pandas as pd
import pyarrow as pa
import pytest

from americast.ingest.eia923 import (
    CURRENT_YEAR,
    _columns,
    _empty,
    load_year,
    url_for,
    verify,
    write,
)
from americast.schemas import EIA923_SOLAR_MONTHLY

MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)


def workbook(rows: list[dict]) -> pd.DataFrame:
    """A sheet with EIA's wrapped headers and placeholder cells."""
    built = []
    for row in rows:
        record = {
            "Plant Id": row["plant"],
            "Reported\nPrime Mover": row.get("mover", "PV"),
            "Reported\nFuel Type Code": row.get("fuel", "SUN"),
        }
        for month in MONTHS:
            record[f"Netgen\n{month}"] = row.get(month, 0.0)
        built.append(record)
    return pd.DataFrame(built)


def sheet_to_frame(frame: pd.DataFrame, tmp_path, year: int = 2025) -> pd.DataFrame:
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path) as writer:
        # Five junk rows above the header, as EIA ships it.
        pd.DataFrame([[""]] * 5).to_excel(
            writer, sheet_name="Page 1 Generation and Fuel Data",
            index=False, header=False,
        )
        frame.to_excel(
            writer, sheet_name="Page 1 Generation and Fuel Data",
            index=False, startrow=5,
        )
    return load_year(path, year)


# --- the header wrapping ---------------------------------------------


def test_columns_are_found_through_the_newlines() -> None:
    """"Reported\\nFuel Type Code" must resolve without being named."""
    mapping = _columns(workbook([{"plant": 1}]))
    assert mapping["fuel"] == "Reported\nFuel Type Code"
    assert mapping["plant"] == "Plant Id"
    assert mapping["June"] == "Netgen\nJune"


def test_a_rewrapped_header_still_resolves() -> None:
    """EIA has moved the line breaks between vintages."""
    frame = workbook([{"plant": 1}])
    frame = frame.rename(columns={"Reported\nFuel Type Code": "Reported Fuel  Type\nCode"})
    assert _columns(frame)["fuel"] == "Reported Fuel  Type\nCode"


def test_a_missing_column_is_an_error_not_a_silent_zero() -> None:
    frame = workbook([{"plant": 1}]).drop(columns=["Netgen\nMarch"])
    with pytest.raises(KeyError, match="netgen"):
        _columns(frame)


# --- parsing ----------------------------------------------------------


def test_only_solar_rows_are_kept(tmp_path) -> None:
    out = sheet_to_frame(
        workbook(
            [
                {"plant": 1, "June": 100.0},
                {"plant": 2, "fuel": "NG", "June": 9999.0},
            ]
        ),
        tmp_path,
    )
    assert set(out["plant_id"]) == {1}


def test_a_plant_filing_twice_is_summed_not_stacked(tmp_path) -> None:
    """One site, two prime movers. It is still one plant-month."""
    out = sheet_to_frame(
        workbook(
            [
                {"plant": 7, "mover": "PV", "June": 100.0},
                {"plant": 7, "mover": "PVe", "June": 50.0},
            ]
        ),
        tmp_path,
    )
    june = out[out["month"] == pd.Timestamp("2025-06-01", tz="UTC")]
    assert len(june) == 1
    assert june["net_generation_mwh"].iloc[0] == pytest.approx(150.0)


def test_a_placeholder_month_is_dropped_not_zeroed(tmp_path) -> None:
    """EIA writes "." for a month a plant did not report.

    Zeroing it would claim the plant generated nothing, which is a
    different statement from the plant saying nothing.
    """
    out = sheet_to_frame(
        workbook([{"plant": 3, "June": 500.0, "July": "."}]), tmp_path
    )
    months = set(out["month"].dt.month)
    assert 6 in months
    assert 7 not in months


def test_months_become_the_first_instant_of_the_month(tmp_path) -> None:
    out = sheet_to_frame(workbook([{"plant": 1, "March": 42.0}]), tmp_path)
    march = out[out["net_generation_mwh"] == 42.0]["month"].iloc[0]
    assert march == pd.Timestamp("2025-03-01", tz="UTC")


def test_the_year_comes_from_the_caller_not_the_sheet(tmp_path) -> None:
    """The sheet names months but never the year."""
    out = sheet_to_frame(workbook([{"plant": 1, "May": 10.0}]), tmp_path, year=2023)
    assert out["month"].iloc[0].year == 2023


def test_a_sheet_with_no_solar_yields_the_declared_columns(tmp_path) -> None:
    out = sheet_to_frame(workbook([{"plant": 1, "fuel": "NG"}]), tmp_path)
    assert out.empty
    assert list(out.columns) == [f.name for f in EIA923_SOLAR_MONTHLY]


# --- urls -------------------------------------------------------------


def test_the_current_year_is_not_in_the_archive() -> None:
    assert "archive" not in url_for(CURRENT_YEAR)
    assert "archive" in url_for(CURRENT_YEAR - 1)


# --- the store --------------------------------------------------------


def test_the_store_conforms_to_the_schema(tmp_path) -> None:
    out = sheet_to_frame(workbook([{"plant": 1, "June": 100.0}]), tmp_path)
    path = tmp_path / "store.parquet"
    write(out, path)
    assert pa.parquet.read_schema(path).equals(EIA923_SOLAR_MONTHLY)


def test_verify_reports_coverage_against_the_registry(tmp_path) -> None:
    out = sheet_to_frame(
        workbook([{"plant": 1, "June": 100.0}, {"plant": 2, "June": 50.0}]), tmp_path
    )
    registry = pd.DataFrame(
        {"plant_id": [1, 2, 3], "capacity_mw_ac": [100.0, 100.0, 800.0]}
    )
    audit = verify(out, registry)
    assert audit["matched_plants"] == 2
    assert audit["matched_capacity_share"] == pytest.approx(0.2)


def test_verify_catches_unsummed_prime_mover_rows() -> None:
    """Two rows for one plant-month means the groupby was skipped."""
    doubled = pd.concat([_row(), _row()], ignore_index=True)
    assert verify(doubled)["duplicated"] == 1


def _row() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "plant_id": [1],
            "month": [pd.Timestamp("2025-06-01", tz="UTC")],
            "net_generation_mwh": [100.0],
        }
    )


def test_empty_has_the_declared_dtypes() -> None:
    frame = _empty()
    assert frame.empty
    assert frame["plant_id"].dtype == "int64"
