"""The daily loop: run selection, forecasting, grading, idempotence.

No network. The HRRR fetch is replaced by a synthetic run so the whole
path from weather to published JSON is exercised without leaving the
machine.
"""

import json

import pandas as pd
import pytest
from test_features import registry, weather
from test_features_table import ALL_ZONES
from test_model_split import table as synthetic_table

from americast.daily import grade_daily, run_daily
from americast.ingest import hrrr
from americast.model import model as boosters
from americast.model.split import split
from americast.region import CAISO_CA
from americast.schemas import LIVE_FORECASTS, LIVE_SCORES

FAST = {"learning_rate": 0.3, "num_leaves": 7}
RUN = pd.Timestamp("2024-06-15 06:00", tz="UTC")


@pytest.fixture(scope="module")
def models() -> dict:
    fitted, _ = boosters.train(split(synthetic_table()), params=FAST)
    return fitted


@pytest.fixture(scope="module")
def region(tmp_path_factory):
    root = tmp_path_factory.mktemp("daily")
    path = root / "plants.parquet"
    registry(**ALL_ZONES).to_parquet(path)
    return CAISO_CA.__class__(
        id="test",
        name="TEST",
        kind="iso",
        timezone=CAISO_CA.timezone,
        iso=CAISO_CA.iso,
        plant_registry_path=path,
    )


@pytest.fixture(scope="module")
def published(models, region, monkeypatch_module) -> pd.DataFrame:
    frame = weather(str(RUN), leads=range(1, 49), plant_ids=ALL_ZONES["plant_ids"])
    monkeypatch_module.setattr(
        "americast.daily.run_daily.hrrr.build", lambda run, plants: frame
    )
    return run_daily.forecast(RUN, models, region)


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch

    patch = MonkeyPatch()
    yield patch
    patch.undo()


# --- which run the loop picks ----------------------------------------


def test_the_loop_waits_for_the_archive_to_finish() -> None:
    """06z is not readable at 06z. Asking early loses forecast hours."""
    just_after = pd.Timestamp("2026-06-15 06:30", tz="UTC")
    assert run_daily.latest(just_after) == pd.Timestamp("2026-06-14 06:00", tz="UTC")


def test_the_loop_takes_today_once_the_archive_has_landed() -> None:
    settled = pd.Timestamp("2026-06-15 09:00", tz="UTC")
    assert run_daily.latest(settled) == pd.Timestamp("2026-06-15 06:00", tz="UTC")


def test_the_loop_always_picks_an_06z_run() -> None:
    """The run hour the model was trained on; see the module docstring."""
    for hour in range(0, 24, 3):
        stamp = pd.Timestamp(f"2026-06-15 {hour:02d}:00", tz="UTC")
        assert run_daily.latest(stamp).hour == run_daily.RUN_HOUR


# --- the forecast -----------------------------------------------------


def test_the_forecast_carries_every_declared_column(published) -> None:
    assert list(published.columns) == [field.name for field in LIVE_FORECASTS]


def test_the_forecast_covers_forty_seven_hours(published) -> None:
    """48 forecast hours in, 47 hour-means out: the last has no successor."""
    assert len(published) == 47
    assert published["lead_hours"].max() == 47


def test_the_forecast_never_predicts_the_past(published) -> None:
    assert (published["valid_time"] > published["run_time"]).all()


def test_the_band_is_ordered_and_positive(published) -> None:
    audit = run_daily.verify(published)
    assert audit["band_inverted"] == 0
    assert audit["negative"] == 0
    assert audit["night_not_zero"] == 0
    assert audit["predicts_the_past"] == 0


def test_an_empty_archive_is_an_error_not_an_empty_file(models, region, monkeypatch) -> None:
    """A morning with no forecast is a failure to report, not a blank page."""
    monkeypatch.setattr(
        "americast.daily.run_daily.hrrr.build",
        lambda run, plants: pd.DataFrame(columns=["run_time"]),
    )
    with pytest.raises(RuntimeError, match="archive has nothing"):
        run_daily.forecast(RUN, models, region)


# --- idempotence, the gate's own criterion ---------------------------


def test_re_running_a_day_changes_nothing(published, tmp_path) -> None:
    """Gate 6's definition of done, asserted on bytes."""
    path = tmp_path / "forecasts.parquet"
    first = run_daily.append(published, path)
    before = path.read_bytes()

    again = run_daily.append(published, path)
    assert first == len(published)
    assert again == 0, "a re-run must add no rows"
    assert path.read_bytes() == before, "and must not rewrite the file"


def test_a_re_run_upgrades_a_partial_day(published, tmp_path) -> None:
    """A day fetched during an archive hole must improve, not duplicate."""
    path = tmp_path / "forecasts.parquet"
    run_daily.append(published.head(20), path)
    run_daily.append(published, path)

    stored = run_daily.load(path)
    assert len(stored) == len(published)
    assert not stored.duplicated(["run_time", "valid_time"]).any()


def test_two_runs_live_side_by_side(published, tmp_path) -> None:
    path = tmp_path / "forecasts.parquet"
    run_daily.append(published, path)
    tomorrow = published.copy()
    tomorrow["run_time"] = tomorrow["run_time"] + pd.Timedelta(days=1)
    tomorrow["valid_time"] = tomorrow["valid_time"] + pd.Timedelta(days=1)
    run_daily.append(tomorrow, path)

    assert run_daily.load(path)["run_time"].nunique() == 2


# --- the JSON contract ------------------------------------------------


def test_every_series_is_the_same_length(published) -> None:
    """A client indexes these together; a short array is a silent bug."""
    payload = run_daily.to_json(published)
    n = len(payload["valid_times"])
    for key in ("p10_mw", "p50_mw", "p90_mw", "physical_mw", "clear_sky_mw", "lead_hours"):
        assert len(payload[key]) == n, key


def test_the_json_publishes_only_the_newest_run(published, tmp_path) -> None:
    older = published.copy()
    older["run_time"] = older["run_time"] - pd.Timedelta(days=1)
    both = pd.concat([older, published], ignore_index=True)

    payload = run_daily.to_json(both)
    assert payload["run_time"] == published["run_time"].max().isoformat()
    assert len(payload["valid_times"]) == len(published)


def test_the_json_round_trips(published, tmp_path) -> None:
    path = tmp_path / "forecast.json"
    run_daily.publish(published, path)
    payload = json.loads(path.read_text())
    assert payload["units"] == "MW"
    assert payload["validated"] is True


# --- grading ----------------------------------------------------------


def labels(frame: pd.DataFrame, intervals: int = 12) -> pd.DataFrame:
    """A label store covering a forecast's hours."""
    return pd.DataFrame(
        {
            "utc_time": frame["valid_time"],
            "solar_mw": frame["p50_mw"] + 250.0,
            "n_intervals": intervals,
        }
    )


def test_grading_scores_every_hour_the_label_reached(published) -> None:
    scored = grade_daily.grade(published, labels(published))
    assert len(scored) == len(published)
    assert list(scored.columns) == [field.name for field in LIVE_SCORES]


def test_an_ungraded_hour_is_absent_not_zero(published) -> None:
    """Tomorrow has not happened. It must not enter the file as a miss."""
    partial = labels(published).head(10)
    assert len(grade_daily.grade(published, partial)) == 10


def test_a_short_hour_is_not_graded(published) -> None:
    """Three 5-minute readings is a different measurement, not a worse one."""
    thin = labels(published, intervals=3)
    assert grade_daily.grade(published, thin).empty


def test_error_keeps_its_sign(published) -> None:
    """Actual is 250 MW above p50 everywhere, so the model under-predicts."""
    scored = grade_daily.grade(published, labels(published))
    assert scored["error_mw"].max() == pytest.approx(-250.0)


def test_the_band_verdict_is_recorded(published) -> None:
    scored = grade_daily.grade(published, labels(published))
    assert scored["inside_band"].dtype == bool
    inside = (scored["solar_mw"] >= scored["p10_mw"]) & (
        scored["solar_mw"] <= scored["p90_mw"]
    )
    assert (scored["inside_band"] == inside).all()


def test_re_grading_a_day_changes_nothing(published, tmp_path) -> None:
    path = tmp_path / "scores.parquet"
    scored = grade_daily.grade(published, labels(published))
    grade_daily.append(scored, path)
    before = path.read_bytes()

    assert grade_daily.append(scored, path) == 0
    assert path.read_bytes() == before


def test_a_revised_label_replaces_its_verdict(published, tmp_path) -> None:
    """A CAISO revision must update the score, not store both."""
    path = tmp_path / "scores.parquet"
    grade_daily.append(grade_daily.grade(published, labels(published)), path)

    revised = labels(published)
    revised["solar_mw"] = revised["solar_mw"] + 1000.0
    grade_daily.append(grade_daily.grade(published, revised), path)

    stored = grade_daily.load(path)
    assert len(stored) == len(published)
    assert not stored.duplicated(["run_time", "valid_time"]).any()


# --- the rolling summary ----------------------------------------------


def test_the_summary_scores_daylight_only(published) -> None:
    scored = grade_daily.grade(published, labels(published))
    summary = grade_daily.rolling(scored)
    assert summary["n"] < len(scored), "night must have been cut"
    assert summary["mae_mw"] > 0.0


def test_the_summary_survives_an_empty_scoreboard() -> None:
    """Day one of the loop, before anything has been graded."""
    empty = pd.DataFrame({field.name: [] for field in LIVE_SCORES})
    assert grade_daily.rolling(empty)["n"] == 0


def test_the_scoreboard_json_lines_up(published) -> None:
    scored = grade_daily.grade(published, labels(published))
    payload = grade_daily.to_json(scored)
    n = len(payload["days"])
    for key in ("daily_mae_mw", "daily_coverage", "daily_hours"):
        assert len(payload[key]) == n, key


def test_verify_catches_a_forecast_graded_against_itself(published) -> None:
    """Zero error everywhere means the join found the forecast, not the truth."""
    self_graded = grade_daily.grade(
        published,
        pd.DataFrame(
            {
                "utc_time": published["valid_time"],
                "solar_mw": published["p50_mw"],
                "n_intervals": 12,
            }
        ),
    )
    assert grade_daily.verify(self_graded)["perfect"] == len(self_graded)


# --- the multi-region contract ---------------------------------------


def test_the_contract_is_versioned(published) -> None:
    """A frozen public contract needs a version before it needs to change."""
    assert run_daily.to_json(published)["schema_version"] == run_daily.SCHEMA_VERSION


def test_the_region_identifies_itself(published, region) -> None:
    """A national map needs more than a name string per region."""
    block = run_daily.to_json(published, region=region)["region"]
    assert block == {
        "id": "test",
        "name": "TEST",
        "kind": "iso",
        "timezone": region.timezone,
        "graded": True,
    }


def test_an_ungraded_region_says_so(published, region) -> None:
    """HRRR covers the whole country; actuals feeds do not.

    A region we can forecast but cannot score is a different product,
    and a consumer must be able to tell without asking.
    """
    ungraded = CAISO_CA.__class__(
        id="nogrid", name="No Grid", kind="balancing_authority",
        timezone=region.timezone, iso=region.iso,
        plant_registry_path=region.plant_registry_path, graded=False,
    )
    payload = run_daily.to_json(published, region=ungraded)
    assert payload["region"]["graded"] is False
    assert payload["validated"] is False


def test_the_peak_matches_the_series(published) -> None:
    payload = run_daily.to_json(published)
    assert payload["peak"]["p50_mw"] == pytest.approx(max(payload["p50_mw"]))
    index = payload["p50_mw"].index(max(payload["p50_mw"]))
    assert payload["peak"]["valid_time"] == payload["valid_times"][index]


def test_generated_at_is_reported(published) -> None:
    """Distinguishes a stale run from a pipeline that never fired."""
    stamped = pd.Timestamp("2026-08-13T09:04:12Z")
    payload = run_daily.to_json(published, generated_at=stamped)
    assert payload["generated_at"] == stamped.isoformat()


def test_accuracy_is_absent_rather_than_faked_on_day_one(published) -> None:
    """Before the first grading there is no error to report."""
    assert run_daily.to_json(published)["accuracy"] is None


def test_accuracy_travels_with_the_forecast(published) -> None:
    scored = grade_daily.grade(published, labels(published))
    summary = grade_daily.rolling(scored)
    payload = run_daily.to_json(
        published,
        accuracy={
            "window_days": summary["days"], "mae_mw": round(summary["mae_mw"], 1),
            "bias_mw": round(summary["bias_mw"], 1),
            "coverage": round(summary["coverage"], 3), "graded_hours": summary["n"],
        },
    )
    assert payload["accuracy"]["mae_mw"] > 0.0
    assert payload["accuracy"]["window_days"] == grade_daily.ROLLING_DAYS


def test_the_index_lists_where_to_fetch_each_region() -> None:
    payload = run_daily.index()
    assert payload["schema_version"] == run_daily.SCHEMA_VERSION
    entry = payload["regions"][0]
    assert entry["id"] == CAISO_CA.id
    assert entry["forecast"] == f"{CAISO_CA.id}/forecast.json"
    assert entry["scoreboard"] == f"{CAISO_CA.id}/scoreboard.json"


def test_the_index_grows_with_regions(region) -> None:
    """One entry today; the shape is what has to be right."""
    payload = run_daily.index(regions=[CAISO_CA, region])
    assert [entry["id"] for entry in payload["regions"]] == ["caiso", "test"]


def test_published_objects_are_filed_under_their_region() -> None:
    """Public URLs. Moving one later breaks every client that saved it."""
    assert str(run_daily.JSON_PATH).endswith(f"{CAISO_CA.id}/forecast.json")
    assert str(grade_daily.JSON_PATH).endswith(f"{CAISO_CA.id}/scoreboard.json")


def test_the_daily_run_stores_the_weather_it_fetched(monkeypatch, tmp_path, region) -> None:
    """The archive has to keep growing or the next retrain has a hole in it."""
    frame = weather(str(RUN), leads=range(1, 49), plant_ids=ALL_ZONES["plant_ids"])
    monkeypatch.setattr("americast.daily.run_daily.hrrr.build", lambda run, plants: frame)
    monkeypatch.setenv("AMERICAST_DATA_ROOT", str(tmp_path))

    fetched = run_daily.fetch(RUN, region, root=tmp_path / "hrrr")
    assert len(fetched) == len(frame)
    assert hrrr.run_path(RUN, tmp_path / "hrrr").exists()


def test_the_index_points_at_the_run_archive() -> None:
    entry = run_daily.index()["regions"][0]
    assert entry["runs"] == "caiso/runs.json"
    assert entry["plants"] == "caiso/plants.json.gz"
