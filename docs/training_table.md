# The training table

What Gate 4 built, why the columns mean what they mean, and the two
measurements that shaped it.

## What it holds

`data/train/table.parquet` — one row per (run_time, valid_time), the
only thing the model in Gate 5 reads.

| Group | Columns | What it is |
|---|---|---|
| Keys | run_time, valid_time, lead_hours | when the forecast was made, and for when |
| Zone weather | `{zone}_dswrf|tcdc|t2m|w10m` | capacity-weighted means, native GRIB units |
| Fleet weather | `fleet_*` | the same across all of CISO |
| Zone power | `{zone}_ac_mw`, `{zone}_clear_mw` | the physical model, summed |
| Fleet power | `fleet_ac_mw`, `fleet_clear_mw` | the same across all of CISO |
| Geometry | fleet_cos_zenith | capacity-weighted, the plain shape of the day |
| Calendar | local_hour, day_of_year | Pacific, the only local values in the project |
| Label | solar_mw, n_intervals | CAISO's hourly mean, and how many 5-minute rows back it |
| Baselines | baseline_clear_sky_mw, baseline_smart_mw | what the model has to beat |

Five zones: kern, mojave, imperial, central_valley, coastal.

Rebuild it with `uv run python -m americast.features.table`. The build
is a fold over run files and takes about twenty minutes for 1100 runs.
Nothing is incremental. Features change far more often than the weather
store grows, and an append-only table is one that quietly holds two
definitions of the same column.

Audit what was built:

```python
from americast.features.table import load, verify
verify(load())   # short runs, missing days, unlabelled rows, physical bias
```

## Every value column describes an hour, not an instant

This is the single most important thing in the table, and it is
invisible if you do not go looking for it.

HRRR's radiation fields are **instantaneous** readings at valid_time.
CAISO's hourly label is the **mean over the hour that starts** at
valid_time. They are not the same quantity. Comparing them looks
correct at midday, when the curve is flat, and is worst at sunrise and
sunset, when the curve is steep and the two differ most.

Measured statewide on 2024-06-15:

| Hour (UTC) | Instant | Hour mean | CAISO |
|---|---|---|---|
| 13:00 | 0.70 GW | 5.33 GW | 3.61 GW |
| 19:00 | 17.94 GW | 17.93 GW | 17.79 GW |
| 02:00 | 6.99 GW | 3.62 GW | 2.66 GW |

The instant reads a fifth of the truth at dawn and 2.6 times the truth
at dusk, while midday is fine either way. `features.hourly()` averages
each instant with the next one before anything reaches the table. That
one change cut the day's mean error from **1264 MW to 775 MW**, a 39%
drop, with no change to the physics at all.

Two consequences worth knowing:

- **The last forecast hour of every run is dropped.** It has no
  successor to average with. Lead hours therefore run 1 to 47, not 48.
- **A hole costs the hour before it.** Some forecast hours were never
  archived. Averaging 13:00 with 15:00 would report the mean of two
  hours as one, so the orphaned row leaves instead.

No future information enters. The next hour comes from the same
forecast run, which the model already had at run_time.

## The physical model, and what it already achieves

Every CISO plant goes through `features/power.py` — sun position, panel
orientation, transposition, cell temperature, DC power, inverter — and
the megawatts are summed. The chain runs twice, once on HRRR's sky and
once on a clear one, so every level carries both a forecast and a
ceiling.

Scored on daylight hours where all three predictors exist, over
2023-01 → 2026-01:

| Predictor | MAE | Bias |
|---|---|---|
| Physical model | 1215 MW | +179 MW |
| Smart persistence | 1335 MW | −13 MW |
| Clear-sky persistence | 1418 MW | −6 MW |
| Naive zero | 10,785 MW | −10,785 MW |

The physics beats both baselines with nothing fitted to anything. That
is the bar Gate 5 has to clear, and it is a demanding one.

**The +179 MW bias is a seasonal cancellation, not a flat offset.** See
the decomposition below: it is +33% in March and −9% in September, so
`SYSTEM_LOSSES` cannot be tuned to remove it. Fitting that constant
against the whole record would also put test-period information into a
training-time value, which is the leak Gate 5 exists to avoid.

## The ceiling is calibrated onto HRRR

`clear_mw` is a **reference** clear sky, not a hard bound, and it needed
a correction before anything could be built on the clearness ratio.

Uncalibrated, the physical estimate exceeded its own ceiling on **71.3%
of daylight rows**, median 1.059, reaching 1.88 at dawn. The cause is
not the atmosphere: on hours where HRRR itself reports under 5% cloud,
its GHI sits **8.9% above Ineichen's** at the same place and instant.
HRRR's shortwave scheme runs high in clear skies. Which model is right
does not matter — clearness is a ratio of an HRRR-driven numerator to
this denominator, and a ratio between two models means nothing unless
they agree about a cloudless sky.

**Fixed 2026-08-12 by `CLEAR_SKY_CALIBRATION = 1.089`**, scaling
Ineichen's output onto HRRR, fitted across 13 runs of 2023-24 and
491,712 plant-hours. No label enters the fit, so nothing can leak into
Gate 5's test period.

The output is scaled rather than the turbidity, having measured both:

| | clear hour reads | p99 | across zenith 0→75 |
|---|---|---|---|
| turbidity ×0.597 | 1.000 | 1.062 | 0.995 → 0.942 |
| **output ×1.089** | 1.000 | 1.165 | 0.974 → 1.004 |

Fitting turbidity instead needs a Linke of 1.29 — below a pure Rayleigh
atmosphere, so no longer "turbidity" in any physical sense — and it
over-corrects at low sun, making a clear late afternoon read 0.942. All
three components are scaled together, so `ghi = dni·cos(z) + dhi` still
closes.

**Altitude was never the cause,** and was re-tested at the hours where
it should have mattered most. At 800 m the dawn ratio gets *worse*,
2.54 → 2.93.

After calibration, a statewide cloudless hour reads **1.012**. The ratio
is still not capped at 1: broken cloud genuinely pushes panels above
their clear-sky value, and hiding that would be a worse lie than
reporting it.

**Near the horizon the ratio stays meaningless whatever the
calibration.** `power.clearness()` therefore refuses to report it above
a zenith of 75° — a plant there has not started for the day, which is a
different claim from "0% clear". That costs 3.79% of the fleet's
horizontal irradiance. On the reported rows of a real June run: median
1.004, p99 1.198, and 0.28% above 1.3.

## Why the error is 1215 MW, and what it is made of

Measured over 2023-01 → 2026-01, 27,684 graded daylight hours: 11.3% of
mean daylight generation, 5.6% of installed capacity. Almost none of it
is fixable by calibration — removing the flat bias takes it to 1204 MW,
and a single scale factor makes it *worse* at 1242 MW.

The reason is that the residual is two large effects of opposite sign
plus cloud noise. Restricted to clear midday hours, where the physics
should be at its best, the bias by month:

| Month | Jan | Feb | **Mar** | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bias | +1% | +7% | **+33%** | +13% | +5% | −3% | −6% | −7% | −9% | −7% | −6% | −2% |

- **Spring over-prediction is curtailment.** March reads +33%. High
  output, mild demand and hydro running is exactly when CAISO curtails
  solar hardest. Curtailed generation is real sunlight that never
  reaches the fuel mix, and no irradiance model can see an economic
  decision.
- **Autumn under-prediction is missing capacity.** June to December run
  −2 to −9%. The registry is state-filtered, so the ~2.5 GW of Arizona
  and Nevada solar inside CAISO's balancing authority is absent (see
  `plant_registry.md`). With curtailment low then, that gap shows.
- **Clouds are the smallest term.** Bucketing midday hours by HRRR's own
  cloud cover: clear 8.0% error, few 10%, scattered 12%, broken 14%,
  overcast 16%. Going from a clear sky to a broken one adds ~270 MW.

The pattern is stable — it was first measured on 22 months and held
when the table grew to three years.

This is good news for Gate 5. Both large terms are **learnable from
features the table already carries** — curtailment from month, hour and
output level, missing capacity from a scale the model can fit — while
cloud error is the part no model can remove. A tree should beat 1141 MW
by a wide margin, and doing so is mostly a matter of learning
California's grid economics rather than its weather.

## What is not in the table, and why

- **Per-plant rows.** They are computed on the fly and never stored.
  They would double the weather store to hold numbers that rebuild in
  1.4 seconds per run.
- **Rooftop solar.** Out of scope; the label does not contain it.
- **Plants outside CISO.** 140 of the 928 registry plants sit in LDWP,
  IID, BANC, PACW or WALC. Their weather cannot explain CAISO's number.
  `fleet()` drops them and `aggregate()` is where the cut bites.
- **Plants that did not exist yet.** `generate()` zeroes any plant
  before its operating_date. Ignoring it would have inflated the
  2023-06-15 fleet peak from 19.39 GW to 27.28 GW, a 40.7% leak.

## Reading the lead-time chart while pass 3 is outstanding

`data/reports/gate4.html` plots error against lead time. Until the
backfill's pass 3 lands, only 06z runs are stored, so **every lead hour
falls at a fixed time of day**. The low error at 19–30 hours is not
skill; those are the hours when California is barely generating. The
00z, 12z and 18z runs are what separate lead time from time of day, and
the chart carries this warning on the page itself.
