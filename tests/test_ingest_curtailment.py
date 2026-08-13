"""Curtailment parsing, on both of CAISO's report formats.

The real failure this guards against is not a missing day. It is
gridstatus handing back an object-dtype column and pandas
concatenating the digits, which produces a number that looks like
megawatts and is not.
"""

from datetime import date

import pandas as pd
import pyarrow as pa
import pytest

from americast.ingest.curtailment import (
    FLEET_CEILING_MW,
    _append,
    _energy_column,
    fetch_day,
    load,
    verify,
)
from americast.schemas import CAISO_CURTAILMENT


def legacy_report() -> pd.DataFrame:
    """The pre-2025 format: strings, blanks, and only non-zero rows."""
    return pd.DataFrame(
        [
            # 06:00 -- two solar categories at once, one with a blank MW
            {"Interval Start": "2024-03-15T13:00:00Z", "Fuel Type": "Solar",
             "Curtailment Type": "Economic", "Curtailment Reason": "Local",
             "Curtailment (MWh)": "17", "Curtailment (MW)": "58"},
            {"Interval Start": "2024-03-15T13:00:00Z", "Fuel Type": "Solar",
             "Curtailment Type": "Economic", "Curtailment Reason": "System",
             "Curtailment (MWh)": "8", "Curtailment (MW)": None},
            # wind in the same hour -- must not be counted
            {"Interval Start": "2024-03-15T13:00:00Z", "Fuel Type": "Wind",
             "Curtailment Type": "Economic", "Curtailment Reason": "Local",
             "Curtailment (MWh)": "900", "Curtailment (MW)": "950"},
            # 07:00
            {"Interval Start": "2024-03-15T14:00:00Z", "Fuel Type": "Solar",
             "Curtailment Type": "Economic", "Curtailment Reason": "Local",
             "Curtailment (MWh)": "366", "Curtailment (MW)": "735"},
        ]
    )


def current_report() -> pd.DataFrame:
    """The post-2025 format: no brackets, and explicit zeros."""
    return pd.DataFrame(
        [
            {"Interval Start": "2026-03-15T13:00:00Z", "Fuel Type": "Solar",
             "Curtailment Type": "Economic", "Curtailment Reason": "Local",
             "Curtailment MWH": 120.0, "Curtailment MW": 300.0},
            {"Interval Start": "2026-03-15T13:00:00Z", "Fuel Type": "Solar",
             "Curtailment Type": "Economic", "Curtailment Reason": "System",
             "Curtailment MWH": 80.0, "Curtailment MW": 210.0},
            {"Interval Start": "2026-03-15T13:00:00Z", "Fuel Type": "Wind",
             "Curtailment Type": "Economic", "Curtailment Reason": "Local",
             "Curtailment MWH": 500.0, "Curtailment MW": 600.0},
        ]
    )


def parse(raw: pd.DataFrame, monkeypatch) -> pd.DataFrame:
    monkeypatch.setattr(
        "americast.ingest.curtailment._raw_day", lambda day: raw
    )
    return fetch_day(date(2024, 3, 15))


# --- the parsing bug this module shipped with ------------------------


def test_string_columns_are_added_not_concatenated(monkeypatch) -> None:
    """"17" + "8" is 25 megawatts, never 178.

    gridstatus returns the legacy columns as objects. Summing them
    without coercion concatenates the digits, and the result is large
    enough to pass for a real reading.
    """
    out = parse(legacy_report(), monkeypatch)
    first = out.set_index("utc_time").loc[pd.Timestamp("2024-03-15T13:00:00Z")]
    assert first["curtailed_mw"] == pytest.approx(25.0)


def test_a_blank_reading_does_not_poison_the_hour(monkeypatch) -> None:
    """The System row has no MW value. Its energy still counts."""
    out = parse(legacy_report(), monkeypatch)
    assert out["curtailed_mw"].notna().all()
    assert (out["curtailed_mw"] > 0.0).all()


def test_wind_is_not_counted_as_solar(monkeypatch) -> None:
    """900 MWh of wind sits in the same hour and must be ignored."""
    out = parse(legacy_report(), monkeypatch)
    assert out["curtailed_mw"].max() < 900.0


def test_categories_are_summed_within_an_hour(monkeypatch) -> None:
    """Economic-Local and Economic-System curtail different plants at once."""
    out = parse(current_report(), monkeypatch)
    assert len(out) == 1
    assert out["curtailed_mw"].iloc[0] == pytest.approx(200.0)


# --- both report formats ---------------------------------------------


def test_the_energy_column_is_found_in_the_legacy_format() -> None:
    assert _energy_column(legacy_report()) == "Curtailment (MWh)"


def test_the_energy_column_is_found_in_the_current_format() -> None:
    assert _energy_column(current_report()) == "Curtailment MWH"


def test_the_energy_column_is_never_the_power_column() -> None:
    """"Curtailment MW" must not match a search for MWh."""
    for frame in (legacy_report(), current_report()):
        assert "MWH" in _energy_column(frame).upper().replace(" ", "")


# --- absent days ------------------------------------------------------


def test_a_missing_report_yields_no_rows_rather_than_raising(monkeypatch) -> None:
    """CAISO has gaps. One must not end a backfill of thirteen hundred."""
    out = parse(pd.DataFrame(), monkeypatch)
    assert out.empty
    assert list(out.columns) == [f.name for f in CAISO_CURTAILMENT]


def test_a_day_with_only_wind_yields_no_solar_rows(monkeypatch) -> None:
    wind_only = legacy_report()
    wind_only["Fuel Type"] = "Wind"
    assert parse(wind_only, monkeypatch).empty


# --- the store --------------------------------------------------------


def test_the_store_conforms_to_the_schema(tmp_path, monkeypatch) -> None:
    path = tmp_path / "curtailment.parquet"
    _append(parse(legacy_report(), monkeypatch), path)
    assert pa.parquet.read_schema(path).equals(CAISO_CURTAILMENT)


def test_appending_the_same_day_twice_changes_nothing(
    tmp_path, monkeypatch
) -> None:
    """Re-running a backfill must not double a day's curtailment."""
    path = tmp_path / "curtailment.parquet"
    day = parse(legacy_report(), monkeypatch)
    _append(day, path)
    first = load(path)
    _append(day, path)
    assert load(path).equals(first)


def test_the_store_stays_sorted(tmp_path, monkeypatch) -> None:
    path = tmp_path / "curtailment.parquet"
    later = parse(current_report(), monkeypatch)
    _append(later, path)
    _append(parse(legacy_report(), monkeypatch), path)
    assert load(path)["utc_time"].is_monotonic_increasing


# --- verify -----------------------------------------------------------


def test_verify_passes_a_clean_series(tmp_path, monkeypatch) -> None:
    audit = verify(parse(legacy_report(), monkeypatch))
    assert audit["negative"] == 0
    assert audit["implausible"] == 0
    assert audit["n_rows"] == 2


def test_verify_catches_the_concatenation_bug(monkeypatch) -> None:
    """The check has to be able to fail, or it is decoration."""
    broken = parse(legacy_report(), monkeypatch)
    broken.loc[0, "curtailed_mw"] = FLEET_CEILING_MW + 1.0
    assert verify(broken)["implausible"] == 1
