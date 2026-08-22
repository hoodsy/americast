# Americast

Americast forecasts California's utility-scale solar generation 48 hours
ahead, and grades every forecast against what the grid actually reported.

![The Americast web app: a map of California's solar plants coloured by
clearness, above a 48-hour statewide generation curve](docs/images/webapp.png)

Utility-scale solar only. Rooftop solar is out of scope, because the target
is CAISO's reported utility-scale number, hourly, in MW.

## How it works

Every plant is modelled from the weather at its own location: where the sun
is, which way the panels point, how much light reaches the panel face, how
hot the cells get, and what the inverter lets through. Those megawatts sum to
a state total. The same chain runs a second time on a clear sky, and the
ratio of the two is the clearness index. A LightGBM model then corrects the
sum. It predicts the share of the clear-sky ceiling the fleet delivers, not
megawatts directly, because the fleet keeps growing past its own training
range.

A GitHub Actions job runs this once a day and writes static JSON to S3. The
web app reads those files directly, so there is no server in the read path.

```sh
curl -s https://americast-data.s3.us-west-2.amazonaws.com/americast/public/caiso/forecast.json | head
```

## Data

| Source | What it gives |
|---|---|
| [NOAA HRRR](https://rapidrefresh.noaa.gov/hrrr/), read with [Herbie](https://github.com/blaylockbk/Herbie) | 3 km weather forecasts, hourly, 1 to 48 hours ahead |
| [EIA-860 and 860M](https://www.eia.gov/electricity/data/eia860/) | the fleet: location, AC and DC capacity, tilt, azimuth, mount type |
| CAISO fuel mix, read with [gridstatus](https://github.com/gridstatus/gridstatus) | 5-minute solar actuals — the number the forecast is graded against |
| CAISO curtailment | the hours when output was cut on purpose |

The fleet is 833 operating plants and 24.23 GW AC. The filter is the
balancing authority, not the state: CAISO reaches into Arizona and Nevada,
and some Californian plants belong to other operators. Getting this wrong put
CAISO's peak above the whole modelled fleet, which is impossible —
[`docs/plant_registry.md`](docs/plant_registry.md).

Weather grids are never stored. Each HRRR run is sampled at every plant and
saved as one small table — [`docs/hrrr_backfill.md`](docs/hrrr_backfill.md).

## Results

Trained on 2023–2024, early-stopped on 2025 H1, graded on 2025-07 onward.

| Predictor | MAE | Skill |
|---|---|---|
| **Americast** | **1236 MW** | **+0.283** |
| Physical model, unfitted | 1525 MW | +0.115 |
| Clear-sky persistence | 1723 MW | 0.000 |

It wins every lead bucket at 4 hours and beyond.

One weakness, measured rather than hidden: the p10–p90 band covers 58.6% of
the hours where it claims 80%, and it sits too low rather than being too
narrow. Plants built during the test period make real megawatts but add no
ceiling. Why refitting that constant would be a leak —
[`docs/model.md`](docs/model.md).

Only the state total is graded. County and zone figures are estimates that
sum to it. No hourly public truth exists to check them against.

## The API

A read-only FastAPI service over the same data. The web app does not need
it — the browser reads static JSON from S3 — but it is the fastest way to
explore a run locally.

```sh
uv run python -m americast.api.app    # http://localhost:8000, docs at /docs
```

`/runs` lists stored runs. `/runs/{run_time}/totals` gives the state, zone
and county curves. `/runs/{run_time}/plants` gives per-plant megawatts and
clearness. `/runs/latest/...` is an alias for the newest run. Every value
array is the same length as `valid_times`, and every level carries
`validated`, true only for the state total.

## Related work

- [CAISO's own day-ahead renewable forecast](https://oasis.caiso.com/mrioasis/logon.do)
  — the operator publishes a solar forecast for the same hours (OASIS,
  `SLD_REN_FCST`, by trading hub). The obvious next benchmark. Americast is
  not graded against it yet.
- [Open Source Quartz Solar Forecast](https://github.com/openclimatefix/open-source-quartz-solar-forecast)
  — the closest open-source relative. Also boosted trees on numerical weather,
  also 0 to 48 hours, but one site at a time and trained on UK data.
- [Sheffield Solar PV_Live](https://www.solar.sheffield.ac.uk/pvlive/) — the
  UK counterpart on the actuals side. It includes rooftop, which CAISO's fuel
  mix does not.
- [Solar Forecast Arbiter](https://forecastarbiter.epri.com/) (EPRI) — an open
  framework for grading solar forecasts on equal terms.

## Run it

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13+.

```sh
uv sync
uv run pytest
```

```sh
uv run python -m americast.features.table    # rebuild the training table
uv run python -m americast.model.model       # fit p10, p50 and p90
uv run python -m americast.model.eval        # score the test period
uv run python -m americast.daily.run_daily   # one day: forecast, grade, publish
```

## Docs

- [`plant_registry.md`](docs/plant_registry.md) — the fleet, and the balancing-authority trap
- [`hrrr_backfill.md`](docs/hrrr_backfill.md) — HRRR sampling, and how to restart a backfill
- [`training_table.md`](docs/training_table.md) — the training table, column by column
- [`caiso_data_quality.md`](docs/caiso_data_quality.md) — what is wrong with the actuals
- [`model.md`](docs/model.md) — features, split, scores, and the calibration problem
- [`publish.md`](docs/publish.md) — the public objects, and their cache headers
- [`operations.md`](docs/operations.md) — what runs where, and what it costs
- [`web_handoff.md`](docs/web_handoff.md) — wiring a frontend to live data
