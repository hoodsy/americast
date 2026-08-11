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
is a fold over run files and takes about ten minutes for 653 runs.
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

Scored on daylight hours where all three predictors exist:

| Predictor | MAE | Bias |
|---|---|---|
| Physical model | 1141 MW | +423 MW |
| Smart persistence | 1262 MW | −54 MW |
| Clear-sky persistence | 1312 MW | −15 MW |
| Naive zero | 10,092 MW | −10,092 MW |

The physics beats both baselines with nothing fitted to anything. That
is the bar Gate 5 has to clear, and it is a demanding one.

**The +423 MW bias is a real signal, not noise.** It is about 4%, and
the chain has exactly one constant that could produce a flat
multiplicative offset: `SYSTEM_LOSSES`, currently 14%. It has been left
alone deliberately. Fitting it against the whole record would put test
period information into a training-time constant, which is the leak
Gate 5 exists to avoid. Fit it on 2023–2024 only, if at all.

## The ceiling sits under the real sky, and by how much

`clear_mw` is a **reference** clear sky, not a hard bound — and the gap
is bigger than a footnote. The physical estimate exceeds its own
ceiling on **71.3% of daylight rows**, median ratio 1.059.

| Local hour | 5 | 7 | 10-15 | 17 | 19 |
|---|---|---|---|---|---|
| median estimate / ceiling | 1.88 | 1.13 | ~1.05 | 1.11 | 1.42 |

Two different faults wear one number:

- **Midday, a steady +4.6 to +6.5%.** On a clear June day HRRR's GHI
  runs ~10% above Ineichen's at every hour from 09:00 to 16:00.
- **The shoulders, where the ratio breaks.** At a zenith near 86°,
  Ineichen attenuates the beam far harder than HRRR: 75 W/m² of DNI
  against HRRR's 102 at dawn, 66 against 223 at dusk. The megawatts
  there are small — hours 5 and 19 carry 0.1% and 1.9% of total error
  — so this is a broken ratio, not a broken forecast.

**Altitude is not the cause.** The first test measured it at the noon
peak, which is the one hour where airmass matters least. Re-tested
across the day: at 800 m the dawn ratio gets *worse*, 2.54 → 2.93. The
suspect is the Linke turbidity climatology, which reads high over clean
dry air.

What this costs: the clearness index is unusable as a diagnostic (it
exceeds 1 most of the time), and `clear_mw`'s shape is distorted at
dawn and dusk. What it does not cost: the persistence baseline, which
divides by the ceiling and multiplies by it again, so a steady bias
cancels. No cap is applied, because a cap would not have that property.

## Why the error is 1141 MW, and what it is made of

11.3% of mean daylight generation, 5.3% of installed capacity. Almost
none of it is fixable by calibration — removing the flat bias takes it
to 1103 MW, and a single scale factor makes it *worse* at 1175 MW.

The reason is that the residual is two large effects of opposite sign
plus cloud noise. Restricted to clear midday hours, where the physics
should be at its best, the bias by month is:

| Month | Jan | Feb | **Mar** | Apr | May | Jun | Jul | Aug | Sep | Oct |
|---|---|---|---|---|---|---|---|---|---|---|
| bias | +8% | +6% | **+32%** | +13% | +7% | −2% | −4% | −6% | −8% | −6% |

- **Spring over-prediction is curtailment.** March reads +32% — 3762 MW
  on clear midday hours. High output, mild demand and hydro running is
  exactly when CAISO curtails solar hardest. Curtailed generation is
  real sunlight that never reaches the fuel mix, and no irradiance
  model can see an economic decision.
- **Summer under-prediction is missing capacity.** June to December run
  −2 to −8%. The registry is state-filtered, so the ~2.5 GW of Arizona
  and Nevada solar inside CAISO's balancing authority is absent (see
  `plant_registry.md`). With curtailment low in summer, that gap shows.
- **Clouds are the smallest term.** Bucketing midday hours by HRRR's own
  cloud cover: clear 8.0% error, few 10%, scattered 12%, broken 14%,
  overcast 16%. Going from a clear sky to a broken one adds ~270 MW.

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
