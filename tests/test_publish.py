"""The publisher: does a run project into the object a browser reads?

No network. Every store is a synthetic frame, so the whole path from
parquet to published dict is exercised without leaving the machine.
"""

import gzip
import json

import pandas as pd
import pytest

from americast import storage
from americast.daily import publish
from americast.schemas import LIVE_SCORES

RUN = pd.Timestamp("2026-08-17 06:00", tz="UTC")
NOW = pd.Timestamp("2026-08-17 09:45", tz="UTC")


def forecasts(run_time: pd.Timestamp = RUN, hours: int = publish.RUN_HOURS):
    """One run's published curve, shaped like the real store."""
    valid = [run_time + pd.Timedelta(hours=lead) for lead in range(1, hours + 1)]
    frame = pd.DataFrame(
        {
            "run_time": [run_time] * hours,
            "valid_time": valid,
            "lead_hours": list(range(1, hours + 1)),
            "p10_mw": [800.0] * hours,
            "p50_mw": [1000.0] * hours,
            "p90_mw": [1200.0] * hours,
            "fleet_ac_mw": [900.0] * hours,
            "fleet_clear_mw": [1500.0] * hours,
        }
    )
    return frame.astype({"lead_hours": "int32"})


def scores(run_time: pd.Timestamp = RUN, hours: int = 3):
    """The first `hours` of that run, graded."""
    issued = forecasts(run_time).head(hours).copy()
    issued["solar_mw"] = 950.0
    issued["error_mw"] = issued["p50_mw"] - issued["solar_mw"]
    issued["inside_band"] = True
    return issued[[field.name for field in LIVE_SCORES]]


def empty_scores():
    return LIVE_SCORES.empty_table().to_pandas()


# --- the curve --------------------------------------------------------


def test_the_curve_carries_the_run_as_issued() -> None:
    built = publish.curve(RUN, forecasts(), empty_scores(), updated_at=NOW)
    assert built["run_time"] == RUN.isoformat()
    assert len(built["valid_times"]) == publish.RUN_HOURS
    assert len(built["p50_mw"]) == publish.RUN_HOURS


def test_an_ungraded_hour_is_none_not_zero() -> None:
    """Zero says the fleet made nothing. None says nobody has checked."""
    built = publish.curve(RUN, forecasts(), scores(hours=3), updated_at=NOW)
    assert built["observed_mw"][:3] == [950.0, 950.0, 950.0]
    assert built["observed_mw"][3] is None


def test_observed_runs_parallel_to_valid_times() -> None:
    built = publish.curve(RUN, forecasts(), scores(hours=3), updated_at=NOW)
    assert len(built["observed_mw"]) == len(built["valid_times"])


def test_error_is_absent_before_anything_is_graded() -> None:
    built = publish.curve(RUN, forecasts(), empty_scores(), updated_at=NOW)
    assert built["error"] is None


def test_error_reports_this_run_not_the_rolling_window() -> None:
    built = publish.curve(RUN, forecasts(), scores(hours=3), updated_at=NOW)
    assert built["error"]["graded_hours"] == 3
    assert built["error"]["mae_mw"] == 50.0
    assert built["error"]["bias_mw"] == 50.0
    assert built["error"]["coverage"] == 1.0


def test_an_unknown_run_is_an_error_not_an_empty_object() -> None:
    with pytest.raises(ValueError, match="no stored forecast"):
        publish.curve(
            pd.Timestamp("2020-01-01 06:00", tz="UTC"),
            forecasts(),
            empty_scores(),
            updated_at=NOW,
        )


def test_generated_at_survives_a_rewrite_and_updated_at_moves() -> None:
    """generated_at describes the forecast; updated_at describes the object."""
    issued = pd.Timestamp("2026-08-17 09:45", tz="UTC")
    later = pd.Timestamp("2026-08-19 09:45", tz="UTC")
    built = publish.curve(
        RUN, forecasts(), scores(), generated_at=issued, updated_at=later
    )
    assert built["generated_at"] == issued.isoformat()
    assert built["updated_at"] == later.isoformat()


# --- sealing ----------------------------------------------------------


def test_a_fresh_partly_graded_run_is_open() -> None:
    assert not publish.sealed(RUN, scores(hours=3), now=NOW)


def test_a_fully_graded_run_seals() -> None:
    assert publish.sealed(RUN, scores(hours=publish.RUN_HOURS), now=NOW)


def test_an_old_run_seals_even_with_hours_that_never_graded() -> None:
    """CAISO does not re-send telemetry, so some hours never become gradeable.

    Without the age backstop those runs are rewritten every morning forever.
    """
    old = NOW + pd.Timedelta(days=publish.SEAL_AFTER_DAYS)
    assert publish.sealed(RUN, scores(hours=3), now=old)


def test_the_header_follows_the_seal() -> None:
    assert publish.caching(sealed=True) == publish.IMMUTABLE
    assert publish.caching(sealed=False) == publish.BRIEF


def test_the_run_key_spells_the_weather_file() -> None:
    assert publish.run_key(RUN) == "20260817T06z"


# --- writing objects --------------------------------------------------


@pytest.fixture
def bucket(tmp_path, monkeypatch):
    """A local stand-in for the bucket, holding only what we write to it.

    The live stores are handed to `write` rather than seeded here. Every
    store path in this project is a module constant resolved at import,
    so a test that moves the root cannot reach them by writing files.
    """
    monkeypatch.setenv(storage.ENV_VAR, str(tmp_path))
    return tmp_path


def stores() -> dict:
    """The two live stores as `write` and `refresh` want them."""
    return {"forecasts": forecasts(), "scores": scores(hours=3)}


def test_the_forecast_object_lands_under_its_run(bucket, monkeypatch) -> None:
    monkeypatch.setattr(publish, "_map_objects", lambda *a, **k: None)
    written = publish.write(RUN, now=NOW, **stores())
    stored = json.loads(storage.read_text(written["forecast"]))
    assert stored["run_time"] == RUN.isoformat()
    assert stored["sealed"] is False


def test_a_rewrite_keeps_generated_at_and_moves_updated_at(bucket, monkeypatch) -> None:
    monkeypatch.setattr(publish, "_map_objects", lambda *a, **k: None)
    first = publish.write(RUN, now=NOW, **stores())
    issued = json.loads(storage.read_text(first["forecast"]))["generated_at"]

    later = NOW + pd.Timedelta(days=1)
    publish.write(RUN, now=later, **stores())
    stored = json.loads(storage.read_text(first["forecast"]))
    assert stored["generated_at"] == issued
    assert stored["updated_at"] == later.isoformat()


def test_an_immutable_object_is_never_rebuilt(bucket, monkeypatch) -> None:
    """Rewriting an object already sent with `immutable` is invisible to
    every reader that cached it, so the publisher must not do it.

    The map halves are also the expensive ones, so the guard saves a
    400 KB rebuild every morning as well as the correctness problem.
    """
    prefix = publish.run_prefix(RUN)
    storage.write_text(storage.child(prefix, "totals.json"), "{}")
    storage.write_gzip(storage.child(prefix, "plants.json.gz"), "{}")

    def refuse(*args, **kwargs):
        raise AssertionError("rebuilt an object that was already immutable")

    monkeypatch.setattr(publish.frames, "totals", refuse)
    monkeypatch.setattr(publish.frames, "frames", refuse)

    publish.write(RUN, now=NOW + pd.Timedelta(days=1), **stores())
    assert storage.read_text(storage.child(prefix, "totals.json")) == "{}"


def test_a_gzipped_object_decompresses_to_the_same_json(tmp_path) -> None:
    path = tmp_path / "plants.json.gz"
    storage.write_gzip(path, '{"plants": []}', cache_control=publish.IMMUTABLE)
    assert json.loads(gzip.decompress(path.read_bytes())) == {"plants": []}


# --- the index --------------------------------------------------------


def two_runs():
    """Yesterday's run and today's, in one frame."""
    older = forecasts(RUN - pd.Timedelta(days=1))
    return pd.concat([older, forecasts(RUN)], ignore_index=True)


def test_the_index_is_newest_first() -> None:
    listing = publish.catalogue(forecasts=two_runs(), scores=empty_scores(), now=NOW)
    times = [entry["run_time"] for entry in listing["runs"]]
    assert times == sorted(times, reverse=True)


def test_every_entry_carries_its_own_path() -> None:
    """A client must never build a key. That rule is what lets a second
    region, or a second run hour a day, appear without a frontend deploy."""
    listing = publish.catalogue(forecasts=forecasts(), scores=empty_scores(), now=NOW)
    assert listing["runs"][0]["path"] == "caiso/runs/20260817T06z/"


def test_an_ungraded_run_reports_no_error_rather_than_zero() -> None:
    listing = publish.catalogue(forecasts=forecasts(), scores=empty_scores(), now=NOW)
    assert listing["runs"][0]["mae_mw"] is None


def test_a_graded_run_carries_its_error() -> None:
    listing = publish.catalogue(forecasts=forecasts(), scores=scores(hours=3), now=NOW)
    assert listing["runs"][0]["mae_mw"] == 50.0


def test_the_index_carries_the_peak_so_a_picker_needs_no_run_objects() -> None:
    listing = publish.catalogue(forecasts=forecasts(), scores=empty_scores(), now=NOW)
    assert listing["runs"][0]["peak_mw"] == 1000.0


def test_refresh_returns_only_the_open_runs(bucket, monkeypatch) -> None:
    monkeypatch.setattr(publish, "_map_objects", lambda *a, **k: None)
    assert publish.refresh(now=NOW, **stores()) == [RUN]


def test_refresh_leaves_a_sealed_run_alone(bucket, monkeypatch) -> None:
    monkeypatch.setattr(publish, "_map_objects", lambda *a, **k: None)
    old = NOW + pd.Timedelta(days=publish.SEAL_AFTER_DAYS)
    assert publish.refresh(now=old, **stores()) == []


def test_refresh_writes_the_index(bucket, monkeypatch) -> None:
    monkeypatch.setattr(publish, "_map_objects", lambda *a, **k: None)
    publish.refresh(now=NOW, **stores())
    listing = json.loads(storage.read_text(publish.index_path()))
    assert listing["region"] == "caiso"
    assert len(listing["runs"]) == 1


# --- verify -----------------------------------------------------------


def test_verify_reports_a_missing_object_rather_than_raising(
    bucket, monkeypatch
) -> None:
    """The house rule: verify reports, and decides nothing."""
    monkeypatch.setattr(publish, "_map_objects", lambda *a, **k: None)
    publish.refresh(now=NOW, **stores())

    audit = publish.verify(now=NOW, **stores())
    key = publish.run_key(RUN)
    assert audit["runs"] == 1
    assert audit["open"] == 1
    assert audit["missing_objects"] == [
        f"caiso/runs/{key}/totals.json",
        f"caiso/runs/{key}/plants.json.gz",
    ]


def test_verify_counts_a_whole_run(bucket, monkeypatch) -> None:
    monkeypatch.setattr(publish, "_map_objects", lambda *a, **k: None)
    publish.refresh(now=NOW, **stores())
    assert publish.verify(now=NOW, **stores())["short_runs"] == []


def test_verify_notices_a_run_with_holes_in_it(bucket, monkeypatch) -> None:
    """A short run means the weather archive had gaps, and the page will too."""
    monkeypatch.setattr(publish, "_map_objects", lambda *a, **k: None)
    partial = forecasts(hours=12)
    publish.refresh(now=NOW, forecasts=partial, scores=empty_scores())

    audit = publish.verify(now=NOW, forecasts=partial, scores=empty_scores())
    assert audit["short_runs"] == [RUN.isoformat()]


def test_a_run_with_no_stored_weather_still_publishes_its_curve(bucket) -> None:
    """Every run issued before the job began storing its weather is in
    this state. Publishing nothing for those mornings would be worse
    than publishing half, and verify reports the gap either way."""
    written = publish.write(RUN, hrrr_dir=bucket / "nowhere", now=NOW, **stores())
    assert "forecast" in written
    assert "totals" not in written
    assert json.loads(storage.read_text(written["forecast"]))["run_time"] == (
        RUN.isoformat()
    )
