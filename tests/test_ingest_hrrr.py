import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import xarray as xr

from americast.ingest import hrrr
from americast.ingest.hrrr import build, extract, finalize
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


def raw_extract(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "plant_id": [100 + i for i in range(n)],
            "sdswrf": [800.0] * n,
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
    assert (out["valid_time"] - out["run_time"] == pd.Timedelta(hours=12)).all()
    assert (out["lead_hours"] == 12).all()
    assert str(out["lead_hours"].dtype) == "int32"


def test_finalize_conforms_to_schema() -> None:
    out = finalize(raw_extract(), RUN, 1)
    table = pa.Table.from_pandas(out, schema=HRRR_WEATHER, preserve_index=False)
    assert table.num_rows == 3


def test_build_loops_all_48(monkeypatch: pytest.MonkeyPatch) -> None:
    fetched: list[int] = []

    def fake_fetch(run_time: pd.Timestamp, fhour: int) -> list[str]:
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
