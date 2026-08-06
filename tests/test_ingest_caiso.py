from datetime import date

import numpy as np
import pandas as pd
import pytest

from americast.ingest.caiso import (
    _complete_days,
    _normalize_fuel_mix,
    append_to_store,
    qa_report,
    to_hourly,
)
from americast.region import CAISO_CA


def synthetic_raw(n: int = 288) -> pd.DataFrame:
    """A frame shaped like gridstatus's CAISO fuel-mix response."""
    times = pd.date_range(
        "2024-06-15 00:00", periods=n, freq="5min", tz="US/Pacific"
    )
    return pd.DataFrame(
        {
            "Time": times,
            "Interval Start": times,
            "Interval End": times + pd.Timedelta(minutes=5),
            "Solar": np.arange(n, dtype="int64") - 50,
            "Wind": np.full(n, 4800),
            "Natural Gas": np.full(n, 5700),
        }
    )


def test_normalize_keeps_only_solar_in_utc() -> None:
    out = _normalize_fuel_mix(synthetic_raw())
    assert list(out.columns) == ["utc_time", "solar_mw"]
    assert str(out["utc_time"].dt.tz) == "UTC"
    # Pacific midnight is 07:00 UTC in June (PDT)
    assert out["utc_time"].iloc[0] == pd.Timestamp("2024-06-15 07:00", tz="UTC")
    assert out["solar_mw"].dtype == "float64"
    assert out["solar_mw"].iloc[0] == -50.0


def test_normalize_sorts_by_time() -> None:
    shuffled = synthetic_raw().sample(frac=1, random_state=7)
    out = _normalize_fuel_mix(shuffled)
    assert out["utc_time"].is_monotonic_increasing


def test_to_hourly_means_and_counts() -> None:
    df = _normalize_fuel_mix(synthetic_raw())
    hourly = to_hourly(df)
    assert len(hourly) == 24
    assert (hourly["n_intervals"] == 12).all()
    # First hour covers values -50..-39 → mean -44.5
    assert hourly["solar_mw"].iloc[0] == -44.5


def test_to_hourly_exposes_gaps() -> None:
    df = _normalize_fuel_mix(synthetic_raw())
    # Remove hour 2 (Pacific) entirely and half of hour 3
    hole_start = pd.Timestamp("2024-06-15 09:00", tz="UTC")
    half_start = pd.Timestamp("2024-06-15 10:00", tz="UTC")
    keep = ~(
        ((df["utc_time"] >= hole_start) & (df["utc_time"] < half_start))
        | (
            (df["utc_time"] >= half_start)
            & (df["utc_time"] < half_start + pd.Timedelta(minutes=30))
        )
    )
    hourly = to_hourly(df[keep])
    by_time = hourly.set_index("utc_time")
    assert by_time.loc[hole_start, "n_intervals"] == 0
    assert pd.isna(by_time.loc[hole_start, "solar_mw"])
    assert by_time.loc[half_start, "n_intervals"] == 6


def day_frame(day: str, n: int = 288) -> pd.DataFrame:
    times = pd.date_range(f"{day} 00:00", periods=n, freq="5min", tz="US/Pacific")
    return pd.DataFrame(
        {
            "utc_time": times.tz_convert("UTC"),
            "solar_mw": np.linspace(0.0, 100.0, n),
        }
    )


def test_append_is_idempotent(tmp_path) -> None:
    path = tmp_path / "store.parquet"
    df = day_frame("2024-06-15")
    append_to_store(df, path)
    append_to_store(df, path)
    stored = pd.read_parquet(path)
    assert len(stored) == 288
    assert stored["utc_time"].is_monotonic_increasing


def test_append_keeps_newest_on_overlap(tmp_path) -> None:
    path = tmp_path / "store.parquet"
    append_to_store(day_frame("2024-06-15"), path)
    revised = day_frame("2024-06-15")
    revised["solar_mw"] += 1000.0
    append_to_store(revised, path)
    stored = pd.read_parquet(path)
    assert len(stored) == 288
    assert stored["solar_mw"].min() >= 1000.0


def test_append_rejects_naive_timestamps(tmp_path) -> None:
    df = day_frame("2024-06-15")
    df["utc_time"] = df["utc_time"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="tz-aware"):
        append_to_store(df, tmp_path / "store.parquet")


def test_complete_days_bookkeeping(tmp_path) -> None:
    path = tmp_path / "store.parquet"
    append_to_store(day_frame("2024-06-14"), path)  # complete
    append_to_store(day_frame("2024-03-10", n=276), path)  # spring-forward day
    append_to_store(day_frame("2024-06-13", n=200), path)  # interrupted day
    append_to_store(day_frame("2024-06-16"), path)  # complete but most recent
    done = _complete_days(path, CAISO_CA)
    assert date(2024, 6, 14) in done
    assert date(2024, 3, 10) in done, "DST spring day (276 rows) is complete"
    assert date(2024, 6, 13) not in done, "200-row day must be refetched"
    assert date(2024, 6, 16) not in done, "most recent day always refetched"


def test_qa_report_flags_planted_defects() -> None:
    df = day_frame("2024-06-15")
    df.loc[40, "solar_mw"] = -500.0  # implausible negative
    df.loc[100:115, "solar_mw"] = 777.0  # frozen for 16 intervals
    df = df.drop(index=range(200, 212)).reset_index(drop=True)  # 1-hour hole
    report = qa_report(df, CAISO_CA)
    assert len(report["missing_intervals"]) == 12
    assert len(report["large_negatives"]) == 1
    assert (report["frozen_runs"]["solar_mw"] == 777.0).all()
    assert len(report["frozen_runs"]) == 16
    assert not report["odd_days"].empty


def test_qa_report_clean_day_is_quiet() -> None:
    report = qa_report(day_frame("2024-06-15"), CAISO_CA)
    assert report["missing_intervals"].empty
    assert report["large_negatives"].empty
    assert report["frozen_runs"].empty
    assert report["odd_days"].empty
