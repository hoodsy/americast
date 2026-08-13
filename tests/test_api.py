import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from test_features import registry, weather

from americast.api import frames as build
from americast.api.app import app
from americast.api.models import LevelSeries, PlantFrames, PlantSeries
from americast.region import CAISO_CA
from americast.schemas import HRRR_WEATHER

ZONED = {
    "plant_ids": (1, 2, 3),
    "counties": ("Kern", "Imperial", "Fresno"),
    "capacities": (100.0, 200.0, 50.0),
}
RUN = pd.Timestamp("2024-06-15 06:00", tz="UTC")


def write_run(frame: pd.DataFrame, path) -> None:
    """Write like the ingest does, through the declared schema.

    `runs()` serves a file only if its schema matches HRRR_WEATHER
    exactly, so a fixture written by a bare `to_parquet` is correctly
    refused — int64 where the schema says int32. Writing it properly is
    also what makes this store a faithful stand-in for a real one.
    """
    table = pa.Table.from_pandas(frame, schema=HRRR_WEATHER, preserve_index=False)
    pq.write_table(table, path)


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> tuple:
    """A two-run weather store and a matching registry."""
    root = tmp_path_factory.mktemp("api")
    hrrr = root / "hrrr"
    hrrr.mkdir()
    for day in ("20240615", "20240616"):
        frame = weather(
            f"2024-06-{day[-2:]} 06:00",
            leads=range(1, 25),
            plant_ids=ZONED["plant_ids"],
        )
        write_run(frame, hrrr / f"hrrr_{day}_06z.parquet")

    registry_path = root / "plants.parquet"
    registry(**ZONED).to_parquet(registry_path)
    region = CAISO_CA.__class__(
        id="test",
        name="TEST",
        kind="iso",
        timezone=CAISO_CA.timezone,
        iso="TESTISO",
        plant_registry_path=registry_path,
    )
    return region, hrrr, root


@pytest.fixture
def client(store, monkeypatch) -> TestClient:
    """The real app, pointed at the synthetic store."""
    region, hrrr, _ = store
    monkeypatch.setattr("americast.api.app.HRRR_DIR", hrrr)
    monkeypatch.setattr("americast.api.app.CAISO_CA", region)
    with TestClient(app) as running:
        yield running


# --- the contract enforces itself -----------------------------------


def test_a_ragged_plant_series_is_refused() -> None:
    """The invariant a client relies on to index arrays together."""
    with pytest.raises(ValidationError, match="different length"):
        PlantFrames(
            run_time=RUN,
            valid_times=[RUN, RUN + pd.Timedelta(hours=1)],
            plants=[PlantSeries(plant_id=1, mw=[1.0], clearness=[None])],
        )


def test_a_county_cannot_claim_to_be_validated() -> None:
    """Only the statewide number is graded against CAISO."""
    with pytest.raises(ValidationError, match="only state is graded"):
        LevelSeries(
            level="county", name="Kern", validated=True, mw=[1.0], clear_mw=[2.0]
        )


def test_the_state_must_claim_to_be_validated() -> None:
    with pytest.raises(ValidationError, match="only state is graded"):
        LevelSeries(
            level="state", name="CISO", validated=False, mw=[1.0], clear_mw=[2.0]
        )


# --- building payloads ----------------------------------------------


def test_runs_are_listed_newest_first(store) -> None:
    _, hrrr, _ = store
    listed = build.runs(hrrr).runs
    assert len(listed) == 2
    assert listed[0] > listed[1]


def test_a_file_of_the_wrong_schema_is_not_served(store) -> None:
    """A store mid-refetch should shrink, never lie."""
    _, hrrr, _ = store
    bad = pd.DataFrame({"run_time": [RUN], "nonsense": [1.0]})
    bad.to_parquet(hrrr / "hrrr_20991231_06z.parquet")  # bare write: wrong schema
    try:
        assert len(build.runs(hrrr).runs) == 2, "the malformed run was skipped"
    finally:
        (hrrr / "hrrr_20991231_06z.parquet").unlink()


def test_plants_match_the_registry(store) -> None:
    region, _, _ = store
    listed = build.plants(region).plants
    assert len(listed) == len(ZONED["plant_ids"])
    assert {p.zone for p in listed} == {"kern", "imperial", "central_valley"}


def test_every_series_matches_the_clock(store) -> None:
    region, hrrr, _ = store
    payload = build.frames(RUN, hrrr, region)
    assert len(payload.valid_times) == 23, "24 forecast hours in, 23 hour means out"
    for series in payload.plants:
        assert len(series.mw) == len(payload.valid_times)
        assert len(series.clearness) == len(payload.valid_times)


def test_an_unstored_run_raises_for_the_route_to_catch(store) -> None:
    region, hrrr, _ = store
    with pytest.raises(FileNotFoundError, match="no stored run"):
        build.frames(pd.Timestamp("1999-01-01 06:00", tz="UTC"), hrrr, region)


def test_clearness_is_null_when_the_sun_is_too_low(store) -> None:
    """Null is not zero: the question does not apply, rather than a value."""
    region, hrrr, _ = store
    payload = build.frames(RUN, hrrr, region)
    every = [c for series in payload.plants for c in series.clearness]
    assert any(c is None for c in every), "a 24-hour run contains night"
    assert any(c is not None for c in every), "and contains day"


def test_zones_and_counties_sum_to_the_state(store) -> None:
    region, hrrr, _ = store
    payload = build.totals(RUN, hrrr, region)
    state = next(level for level in payload.levels if level.level == "state")
    for kind in ("zone", "county"):
        stacked = [level.mw for level in payload.levels if level.level == kind]
        summed = [sum(hour) for hour in zip(*stacked)]
        assert summed == pytest.approx(state.mw, abs=0.5), f"{kind} does not add up"


def test_only_the_state_is_marked_validated(store) -> None:
    region, hrrr, _ = store
    payload = build.totals(RUN, hrrr, region)
    graded = [level for level in payload.levels if level.validated]
    assert [level.level for level in graded] == ["state"]


def test_a_run_is_computed_once_and_then_reused(store) -> None:
    """The cache is what makes on-demand computation comfortable."""
    region, hrrr, _ = store
    build._aligned.cache_clear()
    build.frames(RUN, hrrr, region)
    build.totals(RUN, hrrr, region)
    assert build._aligned.cache_info().misses == 1
    assert build._aligned.cache_info().hits >= 1


# --- routes ---------------------------------------------------------


def test_the_routes_answer(client: TestClient) -> None:
    assert client.get("/runs").status_code == 200
    assert client.get("/plants").status_code == 200
    assert client.get(f"/runs/{RUN.isoformat()}/plants").status_code == 200
    assert client.get(f"/runs/{RUN.isoformat()}/totals").status_code == 200


def test_latest_resolves_to_the_newest_run(client: TestClient) -> None:
    listed = client.get("/runs").json()["runs"]
    latest = client.get("/runs/latest/totals").json()
    assert latest["run_time"] == listed[0]


def test_an_unknown_run_is_a_404(client: TestClient) -> None:
    assert client.get("/runs/1999-01-01T06:00:00Z/plants").status_code == 404
    assert client.get("/runs/1999-01-01T06:00:00Z/totals").status_code == 404


def test_a_malformed_timestamp_is_a_422(client: TestClient) -> None:
    assert client.get("/runs/not-a-time/plants").status_code == 422


def test_the_dev_frontend_is_allowed_through_cors(client: TestClient) -> None:
    reply = client.get("/runs", headers={"Origin": "http://localhost:5173"})
    assert reply.headers["access-control-allow-origin"] == "http://localhost:5173"
