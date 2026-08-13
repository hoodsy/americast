from typing import ClassVar

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import xarray as xr

from americast.ingest import hrrr
from americast.ingest.hrrr import build, extract, finalize, pilot, write
from americast.schemas import HRRR_WEATHER


def synthetic_grid(values: np.ndarray, name: str) -> xr.Dataset:
    """A 5x5 'projected' grid: 2D lat/lon coords over integer y/x dims,
    ~0.03 degrees (~3 km) apart, roughly over the Mojave."""
    y, x = np.arange(5), np.arange(5)
    lat = 35.0 + 0.03 * y[:, None] + 0.001 * x[None, :]
    lon = -118.0 + 0.03 * x[None, :] + 0.001 * y[:, None]
    return xr.Dataset(
        {name: (("y", "x"), values)},
        coords={
            "latitude": (("y", "x"), lat),
            "longitude": (("y", "x"), lon),
        },
    )


def plants_at_cells(cells: list[tuple[int, int]]) -> pd.DataFrame:
    ds = synthetic_grid(np.zeros((5, 5)), "dummy")
    return pd.DataFrame(
        {
            "plant_id": [100 + i for i in range(len(cells))],
            "latitude": [float(ds.latitude[y, x]) for y, x in cells],
            "longitude": [float(ds.longitude[y, x]) for y, x in cells],
        }
    )


def test_extracts_exact_cell_values() -> None:
    values = np.arange(25, dtype="float64").reshape(5, 5)
    ds = synthetic_grid(values, "sdswrf")
    plants = plants_at_cells([(0, 0), (2, 3), (4, 4)])
    out = extract([ds], plants)
    assert list(out["plant_id"]) == [100, 101, 102]
    assert list(out["sdswrf"]) == [0.0, 13.0, 24.0]


def test_merges_variables_across_datasets() -> None:
    ds_a = synthetic_grid(np.full((5, 5), 7.0), "t2m")
    ds_b = synthetic_grid(np.full((5, 5), 3.0), "tcc")
    plants = plants_at_cells([(1, 1), (3, 2)])
    out = extract([ds_a, ds_b], plants)
    assert set(out.columns) == {"plant_id", "t2m", "tcc"}
    assert (out["t2m"] == 7.0).all()
    assert (out["tcc"] == 3.0).all()


def test_far_off_grid_plant_fails_loudly() -> None:
    # Loud failure has two acceptable sources: Herbie's own max_distance
    # filter (drops the point, message below) or our fallback guards
    # (length mismatch / grid-distance check) for cases it lets through.
    ds = synthetic_grid(np.zeros((5, 5)), "sdswrf")
    plants = pd.DataFrame(
        {"plant_id": [1], "latitude": [45.0], "longitude": [-100.0]}
    )
    with pytest.raises(
        ValueError,
        match="max_distance|km from nearest gridpoint|points for",
    ):
        extract([ds], plants)


RUN = pd.Timestamp("2024-06-01 06:00", tz="UTC")


def tiny_ds(*names: str) -> xr.Dataset:
    return xr.Dataset({n: (("y", "x"), np.zeros((2, 2))) for n in names})


class FakeHerbie:
    """Returns a truncated decode until asked to overwrite the local file."""

    calls: ClassVar[list[bool]] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def xarray(
        self, search: str, remove_grib: bool = True, overwrite: bool = False
    ) -> list[xr.Dataset]:
        FakeHerbie.calls.append(overwrite)
        if overwrite:
            return [
                tiny_ds("sdswrf", "vbdsf", "vddsf"),
                tiny_ds("tcc"),
                tiny_ds("t2m", "u10", "v10"),
            ]
        return [tiny_ds("tcc"), tiny_ds("t2m")]


def test_fetch_retries_once_on_truncated_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeHerbie.calls = []
    monkeypatch.setattr(hrrr, "Herbie", FakeHerbie)
    dss = hrrr.fetch(RUN, 15)
    assert FakeHerbie.calls == [False, True], "one plain try, one forced refresh"
    assert hrrr._fields(dss) == hrrr.EXPECTED_VARS


def test_fetch_fails_loudly_when_refresh_is_still_short(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AlwaysShort(FakeHerbie):
        def xarray(self, *args: object, **kwargs: object) -> list[xr.Dataset]:
            return [tiny_ds("tcc")]

    monkeypatch.setattr(hrrr, "Herbie", AlwaysShort)
    with pytest.raises(ValueError, match="missing fields.*sdswrf"):
        hrrr.fetch(RUN, 15)


def raw_extract(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "plant_id": [100 + i for i in range(n)],
            "sdswrf": [800.0] * n,
            "vbdsf": [900.0] * n,
            "vddsf": [100.0] * n,
            "tcc": [25.0] * n,
            "t2m": [300.0] * n,
            "u10": [3.0] * n,
            "v10": [4.0] * n,
        }
    )


def test_finalize_shapes_and_maths() -> None:
    out = finalize(raw_extract(), RUN, 12)
    assert list(out.columns) == [f.name for f in HRRR_WEATHER]
    assert (out["w10m"] == 5.0).all(), "3-4-5 wind triangle"
    assert (out["dswrf"] == 800.0).all()
    assert (out["dni"] == 900.0).all(), "VBDSF lands in dni, not dswrf"
    assert (out["dhi"] == 100.0).all()
    assert (out["valid_time"] - out["run_time"] == pd.Timedelta(hours=12)).all()
    assert (out["lead_hours"] == 12).all()
    assert str(out["lead_hours"].dtype) == "int32"


def test_finalize_conforms_to_schema() -> None:
    out = finalize(raw_extract(), RUN, 1)
    table = pa.Table.from_pandas(out, schema=HRRR_WEATHER, preserve_index=False)
    assert table.num_rows == 3


def test_build_loops_all_48(monkeypatch: pytest.MonkeyPatch) -> None:
    fetched: list[int] = []

    def fake_fetch(
        run_time: pd.Timestamp, fhour: int, scratch: object = None
    ) -> list[str]:
        fetched.append(fhour)
        return ["sentinel"]

    def fake_extract(dss: list[str], plants: pd.DataFrame) -> pd.DataFrame:
        assert dss == ["sentinel"]
        return raw_extract(len(plants))

    monkeypatch.setattr(hrrr, "fetch", fake_fetch)
    monkeypatch.setattr(hrrr, "extract", fake_extract)

    plants = pd.DataFrame({"plant_id": [1, 2]})
    out = build(RUN, plants)
    assert fetched == list(range(1, 49))
    assert len(out) == 48 * 2
    assert sorted(out["lead_hours"].unique()) == list(range(1, 49))
    assert out["valid_time"].max() == RUN + pd.Timedelta(hours=48)


def test_write_roundtrip_and_idempotence(tmp_path) -> None:
    frame = finalize(raw_extract(), RUN, 12)
    path = write(frame, tmp_path)
    assert path.name == "hrrr_20240601_06z.parquet"
    write(frame, tmp_path)  # rewrite replaces, never appends
    stored = pd.read_parquet(path)
    assert len(stored) == 3
    assert list(stored["w10m"]) == [5.0, 5.0, 5.0]


def test_write_rejects_mixed_runs(tmp_path) -> None:
    a = finalize(raw_extract(), RUN, 1)
    b = finalize(raw_extract(), RUN + pd.Timedelta(hours=6), 1)
    with pytest.raises(ValueError, match="single run_time"):
        write(pd.concat([a, b], ignore_index=True), tmp_path)


def test_pilot_is_resumable(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_build(run_time: pd.Timestamp, plants: pd.DataFrame) -> pd.DataFrame:
        return finalize(raw_extract(), run_time, 1)

    monkeypatch.setattr(hrrr, "build", fake_build)
    plants = pd.DataFrame({"plant_id": [1]})

    first = pilot(tmp_path, plants)
    assert first == 30
    assert len(list(tmp_path.glob("hrrr_202406*_06z.parquet"))) == 30

    second = pilot(tmp_path, plants)
    assert second == 0, "existing files are the manifest; nothing refetched"


# --- backfill -------------------------------------------------------


def test_runs_lists_every_target_hour() -> None:
    out = hrrr.runs("2023-01-01", "2023-01-03", hours=(6, 18))
    assert len(out) == 6
    assert out[0] == pd.Timestamp("2023-01-01 06:00", tz="UTC")
    assert out[1] == pd.Timestamp("2023-01-01 18:00", tz="UTC")
    assert out[-1] == pd.Timestamp("2023-01-03 18:00", tz="UTC")


def test_pending_skips_stored_files_and_archive_holes(tmp_path) -> None:
    targets = hrrr.runs("2023-01-01", "2023-01-04", hours=(6,))
    # Target 0 is already stored on disk.
    hrrr.write(finalize(raw_extract(), targets[0], 1), tmp_path)
    # Targets 1-3 are recorded as missing, partial, failed.
    for run_time, status in zip(targets[1:], ["missing", "partial", "failed"]):
        hrrr.record(
            {
                "run_time": run_time.isoformat(),
                "status": status,
                "fhours": 0,
                "rows": 0,
                "seconds": 1.0,
            },
            tmp_path,
        )
    todo = hrrr.pending(targets, tmp_path)
    assert todo == targets[2:], "stored and missing drop out; partial/failed return"


def test_manifest_roundtrip(tmp_path) -> None:
    row = {
        "run_time": "2023-01-01T06:00:00+00:00",
        "status": "ok",
        "fhours": 48,
        "rows": 44544,
        "seconds": 401.2,
    }
    hrrr.record(row, tmp_path)
    hrrr.record({**row, "run_time": "2023-01-02T06:00:00+00:00"}, tmp_path)
    out = hrrr.read_manifest(tmp_path)
    assert len(out) == 2, "second line appends, header written once"
    assert list(out.columns) == hrrr.MANIFEST_COLUMNS
    assert out["run_time"].iloc[0] == pd.Timestamp("2023-01-01 06:00", tz="UTC")


def test_build_skips_a_hole_and_keeps_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_attempt(
        run_time: pd.Timestamp, fhour: int, scratch: object = None
    ) -> list[str] | None:
        return None if fhour == 13 else ["sentinel"]

    monkeypatch.setattr(hrrr, "attempt", fake_attempt)
    monkeypatch.setattr(hrrr, "extract", lambda dss, plants: raw_extract(len(plants)))

    out = build(RUN, pd.DataFrame({"plant_id": [1, 2]}))
    assert out["lead_hours"].nunique() == 47
    assert 13 not in set(out["lead_hours"])
    assert len(out) == 47 * 2, "one hole costs its own hour and nothing else"


def test_build_abandons_a_run_absent_from_the_archive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tried: list[int] = []

    def fake_attempt(
        run_time: pd.Timestamp, fhour: int, scratch: object = None
    ) -> None:
        tried.append(fhour)

    monkeypatch.setattr(hrrr, "attempt", fake_attempt)
    out = build(RUN, pd.DataFrame({"plant_id": [1]}))
    assert out.empty
    assert list(out.columns) == [f.name for f in HRRR_WEATHER]
    assert tried == [1, 2, 3], "stop after three dead hours, not 48"


def test_attempt_retries_then_gives_up(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def always_fails(
        run_time: pd.Timestamp, fhour: int, scratch: object = None
    ) -> list[str]:
        calls.append(fhour)
        raise OSError("connection reset")

    monkeypatch.setattr(hrrr, "fetch", always_fails)
    monkeypatch.setattr(hrrr.time, "sleep", lambda s: None)
    assert hrrr.attempt(RUN, 5) is None
    assert len(calls) == hrrr.TRIES


def test_attempt_returns_after_a_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fails_once(
        run_time: pd.Timestamp, fhour: int, scratch: object = None
    ) -> list[str]:
        calls.append(fhour)
        if len(calls) == 1:
            raise OSError("connection reset")
        return ["sentinel"]

    monkeypatch.setattr(hrrr, "fetch", fails_once)
    monkeypatch.setattr(hrrr.time, "sleep", lambda s: None)
    assert hrrr.attempt(RUN, 5) == ["sentinel"]
    assert len(calls) == 2


def test_one_stores_a_full_run(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_build(
        run_time: pd.Timestamp, plants: pd.DataFrame, scratch: object = None
    ) -> pd.DataFrame:
        return pd.concat(
            [finalize(raw_extract(), run_time, f) for f in range(1, 49)],
            ignore_index=True,
        )

    monkeypatch.setattr(hrrr, "build", fake_build)
    row = hrrr.one(RUN, tmp_path)
    assert row["status"] == "ok"
    assert row["fhours"] == 48
    assert row["rows"] == 48 * 3
    assert hrrr.run_path(RUN, tmp_path).exists()


def test_one_reports_a_missing_run_without_writing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = pd.DataFrame(columns=[f.name for f in HRRR_WEATHER])
    monkeypatch.setattr(hrrr, "build", lambda r, p, s=None: empty)
    row = hrrr.one(RUN, tmp_path)
    assert row["status"] == "missing"
    assert row["fhours"] == 0
    assert not hrrr.run_path(RUN, tmp_path).exists()


def test_one_converts_a_crash_into_a_status(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def explodes(
        run_time: pd.Timestamp, plants: pd.DataFrame, scratch: object = None
    ) -> pd.DataFrame:
        raise RuntimeError("cfgrib fell over")

    monkeypatch.setattr(hrrr, "build", explodes)
    row = hrrr.one(RUN, tmp_path)
    assert row["status"] == "failed", "a worker must never raise into the pool"
    assert not hrrr.run_path(RUN, tmp_path).exists()


def test_verify_reports_stored_shape(tmp_path) -> None:
    frame = pd.concat(
        [finalize(raw_extract(), RUN, f) for f in range(1, 49)], ignore_index=True
    )
    hrrr.write(frame, tmp_path)
    out = hrrr.verify(tmp_path)
    assert len(out) == 1
    assert out["fhours"].iloc[0] == 48
    assert out["plants"].iloc[0] == 3
    assert out["rows"].iloc[0] == 144


def test_uncovered_plants_finds_a_plant_the_store_never_sampled(tmp_path) -> None:
    """The check that has to run after every registry change.

    A plant added to the registry gets no weather from a rebuild. The
    grids were discarded, so only a refetch can sample a new coordinate,
    and `aggregate` inner-joins — a missing plant weighs nothing and
    raises nothing.
    """
    from test_features import registry as registry_fixture
    from test_features import weather

    from americast.ingest.hrrr import uncovered_plants

    stored = weather("2024-06-15 06:00", leads=range(1, 3), plant_ids=(1, 2))
    stored.to_parquet(tmp_path / "hrrr_20240615_06z.parquet")

    grown = registry_fixture(
        plant_ids=(1, 2, 3),
        counties=("Kern", "Imperial", "Yuma"),
        capacities=(100.0, 200.0, 150.0),
    )
    missing = uncovered_plants(grown, root=tmp_path)
    assert list(missing["plant_id"]) == [3]


def test_uncovered_plants_is_empty_when_the_store_matches(tmp_path) -> None:
    from test_features import registry as registry_fixture
    from test_features import weather

    from americast.ingest.hrrr import uncovered_plants

    weather("2024-06-15 06:00", leads=range(1, 3), plant_ids=(1, 2)).to_parquet(
        tmp_path / "hrrr_20240615_06z.parquet"
    )
    same = registry_fixture(
        plant_ids=(1, 2), counties=("Kern", "Imperial"), capacities=(100.0, 200.0)
    )
    assert uncovered_plants(same, root=tmp_path).empty


def test_pending_lists_the_store_once(tmp_path, monkeypatch) -> None:
    """One listing, not one probe per target.

    `exists` per run is free on a disk and a network round-trip against
    a bucket. At 1,590 targets that stalled a backfill for minutes
    before the first worker started, which looked like a hang.
    """
    from americast import storage
    from americast.ingest.hrrr import pending, runs

    probes = 0
    real_exists = storage.exists

    def counting_exists(location):
        nonlocal probes
        probes += 1
        return real_exists(location)

    monkeypatch.setattr(storage, "exists", counting_exists)
    targets = runs("2025-03-01", "2025-04-30", hours=(0, 6, 12, 18))
    assert len(targets) > 200

    todo = pending(targets, root=tmp_path)
    assert todo == targets, "an empty store means everything is pending"
    assert probes <= 1, f"probed {probes} times; the store should be listed once"


def test_pending_skips_runs_already_stored(tmp_path) -> None:
    from test_features import weather

    from americast.ingest.hrrr import pending, run_path, runs, write

    targets = runs("2025-03-01", "2025-03-03", hours=(6,))
    stored = weather(str(targets[0]), leads=range(1, 3), plant_ids=(1, 2))
    stored["run_time"] = targets[0]
    write(stored, root=tmp_path)
    assert run_path(targets[0], tmp_path).exists()

    todo = pending(targets, root=tmp_path)
    assert targets[0] not in todo
    assert len(todo) == len(targets) - 1
