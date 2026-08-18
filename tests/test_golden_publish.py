"""Golden-answer tests over the real weather store.

The unit tests prove the publisher assembles what it says it assembles.
These prove the objects it writes describe California, round-trip through
gzip, and agree with each other. Skipped where no conforming run is
stored.

The map halves come from real weather. The statewide curve does not: no
live forecast store exists on a development machine, so a stand-in frame
covers that half and the assertions here stay off it.
"""

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest
from test_golden_api import a_stored_run

from americast import storage
from americast.daily import publish
from americast.ingest.hrrr import HRRR_DIR
from americast.region import CAISO_CA
from americast.schemas import LIVE_SCORES

RUN = a_stored_run()
WEATHER = HRRR_DIR
NOW = pd.Timestamp("2026-08-18 09:45", tz="UTC")

pytestmark = pytest.mark.skipif(
    RUN is None, reason="no stored HRRR run matches the current schema"
)


def a_forecast(run_time: pd.Timestamp) -> pd.DataFrame:
    """A stand-in curve, so `write` can be exercised whole."""
    hours = publish.RUN_HOURS
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


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    """One real run published into a throwaway root.

    The weather store stays where it is and is handed in, because the
    point of these tests is real weather through the real payload models.
    """
    root = tmp_path_factory.mktemp("archive")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(storage.ENV_VAR, str(root))
        yield publish.write(
            RUN,
            CAISO_CA,
            hrrr_dir=WEATHER,
            now=NOW,
            forecasts=a_forecast(RUN),
            scores=LIVE_SCORES.empty_table().to_pandas(),
        )


def test_the_run_publishes_forty_seven_hours(written) -> None:
    levels = json.loads(storage.read_text(written["totals"]))
    assert len(levels["valid_times"]) == publish.RUN_HOURS


def test_every_series_runs_parallel_to_valid_times(written) -> None:
    levels = json.loads(storage.read_text(written["totals"]))
    hours = len(levels["valid_times"])
    for series in levels["levels"]:
        assert len(series["mw"]) == hours, series["name"]
        assert len(series["clear_mw"]) == hours, series["name"]


def test_only_the_state_total_claims_to_be_graded(written) -> None:
    """Counties and zones are estimates that sum to the graded state number."""
    levels = json.loads(storage.read_text(written["totals"]))
    graded = [s["level"] for s in levels["levels"] if s["validated"]]
    assert graded == ["state"]


def test_the_compressed_object_round_trips(written) -> None:
    """Read the bytes back the way a browser would, not the way we wrote them."""
    payload = json.loads(gzip.decompress(Path(str(written["plants"])).read_bytes()))
    assert payload["plants"]
    assert len(payload["plants"][0]["mw"]) == len(payload["valid_times"])


def test_the_map_and_the_curve_describe_the_same_hours(written) -> None:
    """Compared as instants, not as strings — see the test below."""
    curve = json.loads(storage.read_text(written["forecast"]))
    levels = json.loads(storage.read_text(written["totals"]))
    assert [pd.Timestamp(stamp) for stamp in curve["valid_times"]] == [
        pd.Timestamp(stamp) for stamp in levels["valid_times"]
    ]


def test_the_two_objects_spell_utc_differently_on_purpose(written) -> None:
    """`+00:00` from pandas, `Z` from pydantic. Both are ISO-8601 UTC.

    This is pinned rather than fixed. The contract everywhere in this
    project is that arrays are parallel and a reader indexes them
    together — `docs/web_handoff.md` says never to match on a timestamp —
    so the spelling is not load-bearing. Normalising it would mean either
    changing `forecast.json`, which is live and documented, or changing
    the payload models, which the local API also serves.

    If this test ever fails because the spellings converged, that is
    fine: delete it. It exists so that nobody compares these as strings
    and concludes the hours disagree.
    """
    curve = json.loads(storage.read_text(written["forecast"]))
    levels = json.loads(storage.read_text(written["totals"]))
    assert curve["valid_times"][0].endswith("+00:00")
    assert levels["valid_times"][0].endswith("Z")
    assert curve["valid_times"] != levels["valid_times"]


def test_the_fleet_never_exceeds_what_is_installed(written) -> None:
    """Above the installed capacity is a bug, not a sunny day."""
    levels = json.loads(storage.read_text(written["totals"]))
    state = next(s for s in levels["levels"] if s["level"] == "state")
    assert max(state["mw"]) <= 21_520.0


def test_publishing_twice_writes_identical_bytes(written) -> None:
    """Matches the idempotence run_daily.append already promises."""
    before = Path(str(written["plants"])).read_bytes()
    publish.write(
        RUN,
        CAISO_CA,
        hrrr_dir=WEATHER,
        now=NOW + pd.Timedelta(days=1),
        forecasts=a_forecast(RUN),
        scores=LIVE_SCORES.empty_table().to_pandas(),
    )
    assert Path(str(written["plants"])).read_bytes() == before
