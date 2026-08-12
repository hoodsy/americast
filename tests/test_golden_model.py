"""Golden-answer tests over the real model and the real training table.

The unit tests prove each function does what it says. These prove the
whole assembly landed on a real place in a real year, and that the
numbers in `docs/model.md` and on the Gate 5 page are the numbers the
code produces. Skipped where no conforming table or stored model
exists.
"""

import pandas as pd
import pytest
from test_golden_train import stored_table

from americast.model import eval as scoring
from americast.model import model as boosters
from americast.model.model import MODEL_DIR
from americast.model.split import graded, split

# The frozen slice: fit on 2023, early-stop on 2024 H1, score the first
# week of 2024 July. Recomputed on every run from the stored table, so
# this number moves only if the features, the split, the parameters or
# the seed move — which is exactly when it should move, and when
# docs/model.md needs re-reading.
FROZEN_MAE = 501.113

# How far above the highest generation CAISO actually reported in the
# test period a prediction is allowed to reach. Deliberately not the
# registry's installed capacity: CAISO's own peak already exceeds the
# registry's CISO nameplate, because the snapshot is missing plants
# that are running. Bounding predictions by a stale nameplate would
# fail a model for correctly following the truth.
HEADROOM = 1.10


def stored_model():
    """The saved boosters if they exist, else None."""
    if not (MODEL_DIR / "meta.json").exists():
        return None
    try:
        return boosters.load(MODEL_DIR)
    except (OSError, ValueError):
        return None


pytestmark = pytest.mark.skipif(
    stored_table() is None or stored_model() is None,
    reason="no stored training table and model",
)


@pytest.fixture(scope="module")
def parts() -> dict:
    return split(stored_table())


@pytest.fixture(scope="module")
def models() -> dict:
    fitted, _ = stored_model()
    return fitted


@pytest.fixture(scope="module")
def test_rows(parts: dict, models: dict) -> pd.DataFrame:
    return boosters.attach(models, graded(parts["test"]))


# --- the split is honest ---------------------------------------------


def test_the_periods_do_not_share_a_single_hour(parts: dict) -> None:
    """The one failure that would invalidate every number below."""
    from americast.model.split import verify

    assert verify(parts)["overlap"] == 0


def test_the_test_period_is_large_enough_to_mean_something(parts: dict) -> None:
    assert len(graded(parts["test"])) > 5_000


# --- the predictions are physical ------------------------------------


def test_the_model_stays_inside_what_california_has_ever_produced(
    test_rows: pd.DataFrame,
) -> None:
    """An upper bound taken from the label, not from the stale registry."""
    observed = test_rows["solar_mw"].max()
    assert test_rows["p90_mw"].max() <= observed * HEADROOM


def test_caiso_already_exceeds_the_registry_it_is_modelled_from(
    test_rows: pd.DataFrame,
) -> None:
    """Why the bound above cannot be the nameplate — the same finding again.

    The registry's CISO fleet tops out below what CAISO reports, so the
    ceiling every ratio is measured against is too small in the test
    period. This is the drift, seen from the other end.
    """
    assert test_rows["solar_mw"].max() > test_rows["fleet_clear_mw"].max()


def test_the_band_is_ordered_and_positive(models: dict, parts: dict) -> None:
    audit = boosters.verify(models, graded(parts["test"]))
    assert audit["band_inverted"] == 0
    assert audit["negative"] == 0
    assert audit["night_not_zero"] == 0


def test_the_model_tracks_caiso_through_the_day(test_rows: pd.DataFrame) -> None:
    """The check that would catch a whole-table time shift."""
    assert test_rows["p50_mw"].corr(test_rows["solar_mw"]) > 0.95


# --- the gate's own criterion ----------------------------------------


def test_the_model_beats_every_reference(test_rows: pd.DataFrame) -> None:
    scored = scoring.score(test_rows).set_index("column")
    model_mae = scored.loc["p50_mw", "mae"]
    for column in ("fleet_ac_mw", "baseline_clear_sky_mw", "baseline_smart_mw"):
        assert model_mae < scored.loc[column, "mae"], f"lost to {column}"


def test_the_exit_criterion_is_met(test_rows: pd.DataFrame) -> None:
    """Gate 5's definition of done, asserted rather than admired."""
    verdict = scoring.criterion(test_rows)
    assert verdict["passed"]
    assert verdict["skill"] > 0.15
    assert not verdict["buckets_lost"]


def test_the_model_wins_at_long_leads_not_only_short_ones(
    test_rows: pd.DataFrame,
) -> None:
    """A model that only wins at 1-6h is nowcasting, not forecasting."""
    long_lead = test_rows[test_rows["lead_hours"] >= 24]
    model_error = (long_lead["p50_mw"] - long_lead["solar_mw"]).abs().mean()
    base_error = (
        long_lead["baseline_clear_sky_mw"] - long_lead["solar_mw"]
    ).abs().mean()
    assert model_error < base_error


def test_beating_the_physics_is_what_the_training_bought(
    test_rows: pd.DataFrame,
) -> None:
    """The unfitted physics is the honest bar, and it is a hard one."""
    scored = scoring.score(test_rows).set_index("column")
    assert scored.loc["p50_mw", "mae"] < scored.loc["fleet_ac_mw", "mae"]


# --- the documented weaknesses are still the documented weaknesses ---


def test_the_fleet_drifted_out_from_under_the_model(parts: dict) -> None:
    """The finding that explains the bias. Pinned so it cannot go unnoticed.

    If this ever fails, the registry has been refreshed and both
    docs/model.md and the Gate 5 page need rewriting — which is the
    point of asserting it.
    """
    drifted = scoring.drift(parts).set_index("period")
    assert drifted.loc["train", "residual_median"] < 1.0
    assert drifted.loc["test", "residual_median"] > 1.0
    assert (
        drifted.loc["train", "residual_median"]
        < drifted.loc["validate", "residual_median"]
        < drifted.loc["test", "residual_median"]
    )


def test_the_band_still_sits_too_low(test_rows: pd.DataFrame) -> None:
    """The known miscalibration, asserted in the direction it fails.

    More hours fall above p90 than below p10. That asymmetry is the
    drift above, not a band that is merely narrow, and the distinction
    decides which fix is worth trying.
    """
    band = scoring.coverage(test_rows)
    assert band["coverage"] < band["nominal"]
    assert band["above_p90"] > band["below_p10"]


def test_lead_time_and_time_of_day_are_still_one_axis(
    test_rows: pd.DataFrame,
) -> None:
    """Pass 3 of the backfill is what makes this fail. Until then it holds."""
    audit = scoring.confounded(test_rows)
    assert audit["run_hours"] == [6]
    assert audit["confounded"]


# --- reproducibility --------------------------------------------------


def test_the_frozen_slice_gives_the_frozen_number() -> None:
    """A fixed seed on a fixed slice, to four decimal places.

    This is the test that makes every other number in the gate
    trustworthy: it says the pipeline is deterministic, so a metric that
    moved did so because something changed, not because LightGBM felt
    differently today.
    """
    whole = pd.concat(split(stored_table()).values(), ignore_index=True)
    stamps = whole["valid_time"]
    frozen = {
        "train": whole[stamps.dt.year == 2023],
        "validate": whole[(stamps >= "2024-01-01") & (stamps < "2024-07-01")],
    }
    scored = graded(whole[(stamps >= "2024-07-01") & (stamps < "2024-07-08")])

    fitted, _ = boosters.train(frozen)
    predicted = boosters.predict(fitted, scored)
    mae = (predicted["p50_mw"] - scored["solar_mw"]).abs().mean()

    assert float(mae) == pytest.approx(FROZEN_MAE, abs=1e-3)


def test_the_stored_model_records_its_own_provenance() -> None:
    _, meta = stored_model()
    assert meta["target"] == "ratio"
    assert meta["seed"]
    assert len(meta["features"]) > 30
    assert "solar_mw" not in meta["features"]


# --- the report renders from the real numbers ------------------------


def test_the_report_renders(test_rows: pd.DataFrame, parts: dict) -> None:
    from americast.model import report

    page = report.render(test_rows, parts)
    assert page.startswith("<!doctype html>")
    assert page.count("plotly-graph-div") == 5, "five figures on the page"
    assert "MET" in page


def test_the_report_carries_every_number_it_charts(
    test_rows: pd.DataFrame, parts: dict
) -> None:
    """The contrast relief: aqua and yellow need the table to be legible."""
    from americast.model import report

    page = report.render(test_rows, parts)
    for name in scoring.PREDICTORS.values():
        assert name in page
