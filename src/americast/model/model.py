"""Three LightGBM boosters: a median forecast and a confidence band.

## What gradient boosting is doing here

A boosted tree model is a long sum of small decision trees. The first
tree makes a crude guess, the second is fitted to what the first got
wrong, the third to what the pair still gets wrong, and so on for
hundreds of rounds. Each tree is shallow — ours are capped at 31 leaves
— so no single one can memorise a day; the fit comes from the sum.
`learning_rate` shrinks each tree's contribution before it is added, so
a low rate with many trees explores more carefully than a high rate with
few. That is the whole algorithm, and everything below is bookkeeping
around it.

LightGBM wants its data in an `lgb.Dataset`, which pre-bins every
feature into at most 255 buckets. Binning is why it is fast: a split
search over 255 bucket edges is cheap where a search over 18,000
distinct float values is not. It also means the model is insensitive to
monotone rescaling of any feature, so nothing here is normalised.

## Why three models and not one

A point forecast that is right on average tells a grid operator
nothing about its own reliability. The `quantile` objective replaces
squared error with **pinball loss**, which charges a different price for
being too high than for being too low: at alpha 0.1 an over-prediction
costs nine times an under-prediction, so the fit settles where 10% of
the truth falls below it. Three separate fits at alpha 0.1, 0.5 and 0.9
therefore trace a band that should contain the truth 80% of the time,
and `eval.coverage` checks whether it does. At alpha 0.5 pinball loss is
mean absolute error, so the median model is also the MAE-optimal point
forecast — one code path serves both jobs.

## The weight, which is not a detail

The boosters are fitted on a **ratio** to the clear-sky ceiling, while
the project is graded in **megawatts**. The two are related by
`MW error = clear_mw x ratio error`, so passing `fleet_clear_mw` as the
sample weight makes the loss being minimised exactly the loss being
reported. Without it the fit would spend its capacity on the twilight
band, where the ratio is large and noisy and the megawatts are nearly
nothing.

## Reproducibility

`deterministic` and `force_row_wise` together make LightGBM's histogram
construction order-independent, and `num_threads` is pinned because
thread count alone can otherwise move the last decimal. With those and
a fixed seed, a frozen slice gives the same numbers on every machine,
which is what the golden tests hold it to.
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from americast import storage
from americast.model.split import (
    FEATURES,
    SCALE,
    TARGET,
    TRAIN_END,
    VAL_END,
    graded,
)

MODEL_DIR = storage.key("model")

# The band, and the names its columns carry everywhere downstream.
QUANTILES = {"p10": 0.1, "p50": 0.5, "p90": 0.9}

SEED = 20260813

# Small data — about 18,000 graded training hours — so the guards
# against memorising it matter more than capacity. `min_data_in_leaf`
# is well above the default 20 because neighbouring hours of one
# afternoon are near-duplicates of each other, and a leaf holding 20
# rows can easily be holding one cloudy Tuesday.
PARAMS = {
    "objective": "quantile",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 4,
    "verbosity": -1,
}

MAX_ROUNDS = 2000
EARLY_STOPPING = 100


def train(
    parts: dict[str, pd.DataFrame], params: dict | None = None
) -> tuple[dict[str, lgb.Booster], dict]:
    """Fit the three boosters, stopping each on the validation period.

    Takes the output of `split.split`. Returns the fitted models keyed
    by band name, and a metadata dict recording exactly how they were
    made.

    Every model gets its own early stop. The three quantiles do not
    converge at the same rate — the tails are harder than the median and
    generally want more rounds — so a shared round count would either
    under-fit the band or over-fit the middle.

    Validation is used for one thing only: deciding when to stop. The
    models are not refitted on train plus validate afterwards. That
    would buy six months of extra data at the cost of a round count
    chosen on rows the final model had then seen, and the point of this
    gate is a number nobody can argue with.
    """
    fit_rows = graded(parts["train"])
    stop_rows = graded(parts["validate"])

    fit_set = _dataset(fit_rows)
    stop_set = _dataset(stop_rows, reference=fit_set)

    models = {}
    rounds = {}
    for name, alpha in QUANTILES.items():
        settings = {**PARAMS, **(params or {}), "alpha": alpha}
        booster = lgb.train(
            settings,
            fit_set,
            num_boost_round=MAX_ROUNDS,
            valid_sets=[stop_set],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
        )
        models[name] = booster
        rounds[name] = booster.best_iteration

    meta = {
        "seed": SEED,
        "params": {**PARAMS, **(params or {})},
        "features": list(FEATURES),
        "target": TARGET,
        "scale": SCALE,
        "quantiles": QUANTILES,
        "best_iteration": rounds,
        "train_end": str(TRAIN_END),
        "validate_end": str(VAL_END),
        "n_train_rows": len(fit_rows),
        "n_validate_rows": len(stop_rows),
        "train_span": [str(fit_rows["valid_time"].min()), str(fit_rows["valid_time"].max())],
    }
    return models, meta


def predict(models: dict[str, lgb.Booster], frame: pd.DataFrame) -> pd.DataFrame:
    """Predict p10, p50 and p90 in megawatts, for any slice of the table.

    The frame must already carry the design columns, which `split.split`
    adds; `split.design` alone is enough for a slice that was never
    split.

    Three corrections happen here, in order, and each one is a claim
    about physics rather than about statistics:

    1. **Ratios become megawatts** by multiplying through the clear-sky
       ceiling. Where the ceiling is zero the answer is zero — the sun
       is down, and no model is needed to say so.
    2. **The band is sorted.** Nothing ties the three fits together, so
       an occasional row comes back with p10 above p50. Sorting the
       triple is the standard repair and is honest: it changes which
       model supplied a number, never the set of numbers.
    3. **Negative power is clipped to zero.** The fleet cannot generate
       backwards in daylight. The small negatives CAISO reports at night
       are station-service draw, and those rows are already zero by
       step 1.
    """
    ceiling = frame[SCALE].to_numpy()
    features = frame[list(FEATURES)]

    ratios = np.column_stack([models[name].predict(features) for name in QUANTILES])
    megawatts = ratios * ceiling[:, None]
    megawatts = np.sort(megawatts, axis=1)
    megawatts = np.clip(megawatts, 0.0, None)
    megawatts[ceiling <= 0.0, :] = 0.0

    return pd.DataFrame(
        megawatts, columns=[f"{name}_mw" for name in QUANTILES], index=frame.index
    )


def attach(models: dict[str, lgb.Booster], frame: pd.DataFrame) -> pd.DataFrame:
    """The frame with the three prediction columns added."""
    return frame.join(predict(models, frame))


def save(models: dict[str, lgb.Booster], meta: dict, directory: Path = MODEL_DIR) -> None:
    """Write the boosters and their provenance.

    LightGBM's own text format, not pickle. A pickled booster is tied to
    the version of the library that made it; the text format is a
    readable list of trees that loads years later, and can be diffed.

    Written through `model_to_string` rather than `save_model`, because
    LightGBM writes to a filename and cannot reach object storage. The
    string is the same bytes either way.
    """
    for name, booster in models.items():
        text = booster.model_to_string(num_iteration=booster.best_iteration)
        storage.write_text(_member(directory, f"{name}.txt"), text)
    storage.write_text(
        _member(directory, "meta.json"), json.dumps(meta, indent=2, default=str)
    )


def load(directory: Path = MODEL_DIR) -> tuple[dict[str, lgb.Booster], dict]:
    """Read back what `save` wrote."""
    models = {
        name: lgb.Booster(model_str=storage.read_text(_member(directory, f"{name}.txt")))
        for name in QUANTILES
    }
    meta = json.loads(storage.read_text(_member(directory, "meta.json")))
    return models, meta


def _member(directory: Path | str, name: str) -> Path | str:
    """One file inside the model directory, local or remote.

    Object storage has no directories, so joining with `/` is the only
    operation that means the same thing on both sides.
    """
    return Path(directory) / name if isinstance(directory, Path) else f"{directory}/{name}"


def importance(models: dict[str, lgb.Booster], name: str = "p50") -> pd.Series:
    """Which features the median model actually split on, by total gain.

    Gain, not split count. Split count rewards a feature that gets used
    for hundreds of tiny late-round corrections; gain measures how much
    error each feature's splits actually removed, which is the question
    being asked.
    """
    booster = models[name]
    gains = booster.feature_importance(importance_type="gain")
    series = pd.Series(gains, index=booster.feature_name())
    return series.sort_values(ascending=False)


def verify(models: dict[str, lgb.Booster], frame: pd.DataFrame) -> dict:
    """Checks on the predictions themselves, independent of any score.

    A model can post a fine MAE and still be broken in ways a mean
    hides. These are the ways:

    - `band_inverted`: rows where p10 > p50 > p90 survived sorting. Any
      is a bug in `predict`.
    - `negative`: predicted megawatts below zero. Same.
    - `above_nameplate`: predictions beyond the fleet's installed AC
      capacity, which is physically unreachable.
    - `night_not_zero`: a prediction where the ceiling is zero.
    - `flat`: the standard deviation of p50. A model that has learned
      nothing predicts one number forever, and that failure otherwise
      shows up only as a mediocre score.
    - `band_width`: mean p90 - p10, in MW. A band that collapses to
      nothing is as wrong as one that spans the whole fleet.
    """
    predicted = predict(models, frame)
    low = predicted["p10_mw"]
    mid = predicted["p50_mw"]
    high = predicted["p90_mw"]
    ceiling = frame[SCALE]

    return {
        "n_rows": len(frame),
        "band_inverted": int(((low > mid) | (mid > high)).sum()),
        "negative": int((low < 0.0).sum()),
        "night_not_zero": int(((ceiling <= 0.0) & (mid > 0.0)).sum()),
        "flat": float(mid.std()),
        "band_width": float((high - low).mean()),
        "p50_max": float(mid.max()),
        "p50_mean": float(mid.mean()),
    }


def _dataset(rows: pd.DataFrame, reference: lgb.Dataset | None = None) -> lgb.Dataset:
    """One graded slice as an lgb.Dataset, weighted by the ceiling.

    Weights are divided by their own mean so they average to 1. The
    absolute scale of a weight does not change where the loss is
    minimised, but it does change `min_data_in_leaf`'s hessian-based
    sibling and the printed loss values, and a weight of 18,000 makes
    both meaningless.

    The validation set is built with `reference` pointing at the
    training set so that both use identical bin edges. Without it
    LightGBM bins each set on its own quantiles, and the same megawatt
    value would land in different buckets in the two — a quiet, small
    corruption of every early-stopping decision.
    """
    weight = rows[SCALE].to_numpy()
    return lgb.Dataset(
        rows[list(FEATURES)],
        label=rows[TARGET].to_numpy(),
        weight=weight / weight.mean(),
        reference=reference,
        free_raw_data=False,
    )


if __name__ == "__main__":
    from americast.features.table import load
    from americast.model.split import split
    from americast.model.split import verify as verify_split

    parts = split(load())
    layout = verify_split(parts)
    print("  rows:", layout["graded_rows"], f"overlap={layout['overlap']}")
    for name in ("train", "validate", "test"):
        start, end = layout["spans"][name]
        print(f"  {name:9s} {start:%Y-%m-%d} -> {end:%Y-%m-%d}")

    fitted, provenance = train(parts)
    save(fitted, provenance)
    print(f"trained {len(fitted)} boosters -> {MODEL_DIR}")
    print(f"  rounds {provenance['best_iteration']}")
    print(f"  fitted on {provenance['n_train_rows']:,} graded hours")
    print("\ntop features by gain:")
    print(importance(fitted).head(10).round(0).to_string())
