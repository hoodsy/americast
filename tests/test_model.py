"""The boosters: do they fit, predict physically, and reproduce?

Trained on the synthetic table from test_model_split, so these check
the machinery rather than the meteorology. The golden tests read the
real model and ask whether it beat anything.
"""

import numpy as np
import pandas as pd
import pytest
from test_model_split import table

from americast.model.model import (
    QUANTILES,
    attach,
    importance,
    load,
    predict,
    save,
    train,
    verify,
)
from americast.model.split import design, graded, split

# Enough rounds to learn the synthetic shape, few enough to keep the
# suite quick. The real model uses the module defaults.
FAST = {"learning_rate": 0.2, "num_leaves": 15}


@pytest.fixture(scope="module")
def parts() -> dict:
    return split(table())


@pytest.fixture(scope="module")
def fitted(parts: dict) -> tuple:
    return train(parts, params=FAST)


@pytest.fixture(scope="module")
def scored(parts: dict, fitted: tuple) -> pd.DataFrame:
    models, _ = fitted
    return attach(models, graded(parts["test"]))


# --- shape and provenance -------------------------------------------


def test_one_booster_per_quantile(fitted: tuple) -> None:
    models, _ = fitted
    assert set(models) == set(QUANTILES)


def test_the_metadata_records_how_the_fit_was_made(fitted: tuple) -> None:
    """Provenance is the point: a model nobody can reconstruct is a guess."""
    _, meta = fitted
    assert meta["seed"]
    assert meta["features"]
    assert meta["target"] == "ratio"
    assert set(meta["best_iteration"]) == set(QUANTILES)
    assert meta["n_train_rows"] > 0


def test_predict_returns_the_three_bands(scored: pd.DataFrame) -> None:
    for name in QUANTILES:
        assert f"{name}_mw" in scored.columns


def test_predict_does_not_change_the_row_count(parts: dict, fitted: tuple) -> None:
    models, _ = fitted
    rows = graded(parts["test"])
    assert len(predict(models, rows)) == len(rows)


# --- the predictions are physical ------------------------------------


def test_the_band_is_ordered(scored: pd.DataFrame) -> None:
    """Three independent fits can cross; `predict` sorts them."""
    assert (scored["p10_mw"] <= scored["p50_mw"]).all()
    assert (scored["p50_mw"] <= scored["p90_mw"]).all()


def test_the_fleet_never_generates_backwards(scored: pd.DataFrame) -> None:
    assert (scored["p10_mw"] >= 0.0).all()


def test_the_night_is_predicted_as_zero(parts: dict, fitted: tuple) -> None:
    """No model is needed to know the sun is down."""
    models, _ = fitted
    everything = design(table())
    dark = everything[everything["fleet_clear_mw"] == 0.0]
    assert len(dark) > 0
    predicted = predict(models, dark)
    assert (predicted == 0.0).to_numpy().all()


def test_the_model_did_not_learn_one_number(scored: pd.DataFrame) -> None:
    """A flat prediction posts a mediocre score rather than an obvious failure."""
    assert scored["p50_mw"].std() > 1000.0


def test_the_band_has_width(scored: pd.DataFrame) -> None:
    assert (scored["p90_mw"] - scored["p10_mw"]).mean() > 0.0


def test_verify_reports_a_clean_fit(parts: dict, fitted: tuple) -> None:
    models, _ = fitted
    audit = verify(models, graded(parts["test"]))
    assert audit["band_inverted"] == 0
    assert audit["negative"] == 0
    assert audit["night_not_zero"] == 0
    assert audit["flat"] > 0.0


# --- reproducibility --------------------------------------------------


def test_the_same_seed_gives_the_same_model(parts: dict, fitted: tuple) -> None:
    """The claim the golden metrics rest on."""
    first, _ = fitted
    again, _ = train(parts, params=FAST)
    rows = graded(parts["test"]).head(500)
    assert predict(first, rows).to_numpy() == pytest.approx(
        predict(again, rows).to_numpy()
    )


def test_a_different_seed_gives_a_different_model(parts: dict, fitted: tuple) -> None:
    """Otherwise the seed test above proves nothing."""
    first, _ = fitted
    other, _ = train(parts, params={**FAST, "seed": 7, "bagging_seed": 7})
    rows = graded(parts["test"]).head(500)
    assert not np.allclose(
        predict(first, rows).to_numpy(), predict(other, rows).to_numpy()
    )


def test_a_saved_model_predicts_what_it_predicted(
    parts: dict, fitted: tuple, tmp_path
) -> None:
    models, meta = fitted
    save(models, meta, tmp_path)
    restored, restored_meta = load(tmp_path)

    rows = graded(parts["test"]).head(500)
    assert predict(restored, rows).to_numpy() == pytest.approx(
        predict(models, rows).to_numpy()
    )
    assert restored_meta["features"] == meta["features"]


# --- what the model leaned on ----------------------------------------


def test_importance_ranks_every_feature(fitted: tuple) -> None:
    models, meta = fitted
    gains = importance(models)
    assert len(gains) == len(meta["features"])
    assert gains.is_monotonic_decreasing
