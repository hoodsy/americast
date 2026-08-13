# The model

What Gate 5 built, what it beat, and the three measurements that decide
how much to trust it.

## What it holds

`data/model/` — three LightGBM boosters in LightGBM's own text format,
plus `meta.json` recording the seed, the parameters, the feature list
and the round count each model stopped at.

```sh
uv run python -m americast.model.model    # fit and save
uv run python -m americast.model.eval     # score the test period
uv run python -m americast.model.report   # write data/reports/gate5.html
```

Audit what was built:

```python
from americast.model import eval as scoring, model
from americast.model.split import split, graded, verify
from americast.features.table import load

parts = split(load())
verify(parts)                                  # can anything cross the cut?
models, meta = model.load()
scoring.verify(model.attach(models, graded(parts["test"])))
```

No parquet store and no pyarrow schema, because this gate stores a
model rather than a dataset. Gate 6 writes the first forecast table,
and that one gets a declared schema like every other.

## The target is a ratio, and that was forced

The model predicts `solar_mw / fleet_clear_mw` — the share of its
clear-sky ceiling the fleet delivers — and multiplies back through the
ceiling to get megawatts.

This is not a preference. **10.1% of test hours sit above the highest
label in training.** The fleet grew from a 19,379 MW peak in 2023-24 to
23,208 MW in the test period, and a boosted tree is a sum of piecewise
constants: it cannot return a value outside the range of what it was
fitted on. A megawatt-target model is therefore structurally unable to
reach a tenth of its own test period, and would post a bad score for
arithmetic reasons that look like modelling reasons.

Predicting the ratio makes fleet growth, day length and season come
free from the ceiling, which is rebuilt from the registry and already
knows when each plant switched on.

**Sample weight is `fleet_clear_mw`.** Error in megawatts is
`clear_mw × ratio error`, so weighting the ratio loss by the ceiling
makes the loss being minimised the same quantity as the loss being
reported. Without it the fit spends its capacity on the twilight band,
where the ratio is large and noisy and the megawatts are nearly
nothing.

## The split, and the rule that keeps it honest

| Period | Span | Graded daylight hours |
|---|---|---|
| train | 2023-01-08 → 2024-12-31 | 18,284 |
| validate | 2025-01-01 → 2025-06-30 | 4,674 |
| test | 2025-07-01 → 2026-08-06 | 10,242 |

**A row joins a period only when `run_time` and `valid_time` are both
inside it.** Assigning by `valid_time` alone would let the last
training run — which forecasts 47 hours ahead — carry validation labels
into the fit. Assigning by `run_time` alone has the mirror fault. The
straddle rule costs 119 rows of 61,899, and every alternative buys them
back with a leak.

Validation decides one thing only: when to stop boosting. The models
are **not** refitted on train plus validate afterwards. That would buy
six months of data at the cost of a round count chosen on rows the
final model had then seen.

The two baseline columns are legal features — both are keyed on
`run_time` and read no future — and are still held out. A model handed
smart persistence would beat smart persistence and prove nothing about
whether the weather forecast carries information.

## What it achieves

Test period, 10,242 graded daylight hours, every predictor on identical
rows:

| Predictor | MAE | RMSE | Bias | MAE skill |
|---|---|---|---|---|
| **Model (p50)** | **1236 MW** | 1659 MW | −405 MW | **+0.283** |
| Physical model | 1525 MW | 1981 MW | −89 MW | +0.115 |
| Clear-sky persistence | 1723 MW | 2594 MW | −1 MW | 0.000 |
| Smart persistence | 1665 MW | 2490 MW | −10 MW | +0.033 |
| Naive zero | 12,944 MW | 14,598 MW | −12,944 MW | −6.513 |

Skill is against clear-sky persistence. Against the unfitted physics —
the harder bar — the model scores **+0.190 on MAE** and +0.163 on RMSE.

**The exit criterion is met.** At leads of 4 hours and beyond the model
beats clear-sky persistence in all eight lead buckets, with 28.3%
skill. `eval.criterion` computes this rather than leaving it to the
eye, and `test_the_exit_criterion_is_met` asserts it.

Gate 4 predicted this outcome and the reason held: both large terms in
the physics residual — spring curtailment and missing capacity — are
learnable from features the table already carries, while cloud error is
the part no model removes. The two features the median model leans on
hardest are `day_of_year` and `fleet_clearness`, which is curtailment
season and cloud, in that order.

## The lead-time axis is not a lead-time axis

While only the 06z run is stored, each lead hour reaches just **2.00**
distinct local hours, and those two are one hour apart, chosen by the
daylight-saving calendar rather than by anything the forecast knows.
Lead time and time of day are the same axis. `eval.confounded` measures
this instead of asserting it.

MAE by lead bucket makes the problem obvious:

| Lead | Model | Clear-sky persistence |
|---|---|---|
| 1-6h | 55 | 155 |
| 7-12h | 1144 | 1486 |
| 13-18h | 1479 | 1788 |
| 19-24h | 376 | 802 |
| 25-30h | 56 | 155 |
| 31-36h | 1189 | 1856 |
| 37-42h | 1501 | 2169 |
| 43-48h | 406 | 844 |

The 1-6h and 25-30h buckets are not the model at its best. They are the
hours when California is generating almost nothing, 24 hours apart,
which is the same clock reading twice. Nothing here says how error
grows with lead time, and the report carries that warning on the page.
Pass 3 of the backfill is what separates the two axes.

## The band does not cover what it claims

The p10–p90 band should hold the truth 80% of the time. It holds it
**58.6%**, and it fails asymmetrically:

| | measured | nominal |
|---|---|---|
| coverage | 58.6% | 80% |
| below p10 | 13.5% | 10% |
| above p90 | 27.9% | 10% |
| mean width | 2899 MW | — |

Nearly three times as many hours fall above p90 as the band allows,
against a near-correct miss rate below p10. **The band is not narrow.
It sits too low.** That distinction decides which fix is worth trying,
and a wider band is not it.

The p90 model also stopped after 59 boosting rounds against 269 for the
median and 875 for p10. Early stopping was doing its job: the upper
quantile's validation loss stopped improving almost immediately,
because the validation period had already drifted away from the
training period in the direction p90 cares about most.

## Why the model is biased and the physics is not

This is the finding of the gate, and it is not a modelling error.

Divide the weather out by comparing CAISO against the physical model's
own answer rather than against the ceiling. HRRR's clouds are then
inside the denominator, so what is left moves only if the fleet itself
changed:

| Period | CAISO / physics (median) | Hours |
|---|---|---|
| train | 0.9666 | 17,736 |
| validate | 0.9847 | 4,495 |
| test | **1.0230** | 9,971 |

CAISO delivered 0.967× the physics during training and 1.023× during
the test period — a 5.8% climb the weather features cannot see. **The
model learned the first number and was graded against the second.**
That is the whole of its −405 MW bias, and it is why the unfitted
physics, having learned nothing, is nearly unbiased instead at −89 MW.

The cause is in the registry. Its newest plant is dated **2025-12**,
so every plant commissioned during the test period generates real
megawatts and contributes no ceiling. CAISO's own peak already exceeds
the registry's CISO nameplate, which is why golden tests bound
predictions by the observed label rather than by installed capacity.

This is a stale input, not a retuning problem. Refitting a constant
against the test period would remove the bias and would also be the
exact leak this gate exists to prevent.

`test_the_fleet_drifted_out_from_under_the_model` pins the three
numbers. When the registry is refreshed, that test fails, and this
section needs rewriting — which is the point of asserting it.

## Reproducibility

`deterministic`, `force_row_wise` and a pinned `num_threads` together
make the fit independent of thread scheduling; the seed does the rest.
A frozen slice — fit on 2023, stop on 2024 H1, score the first week of
2024 July — returns **501.113 MW** MAE, and
`test_the_frozen_slice_gives_the_frozen_number` holds it to three
decimals across processes and machines.

That test is what makes every other number here trustworthy. A metric
that moves has moved because something changed, not because LightGBM
felt differently today.

## What is not here

- **No refit on train plus validate.** See the split section.
- **No calibration of the band.** Conformal widening on the validation
  period would fix coverage without touching test data, and is the
  obvious next thing to try. It is not in the build plan, so it is an
  open question rather than a decision.
- **No 00z, 12z or 18z runs.** Pass 3 of the backfill. Until it lands,
  no statement about lead time means what it appears to mean.
- **No stored predictions.** Gate 6 writes `data/live/forecasts.parquet`
  with a declared schema. This gate keeps the model and recomputes.
