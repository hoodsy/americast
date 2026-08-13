# americast

Statewide day-ahead solar generation forecast for the California grid (CAISO),
built to extend to other US regions later. A tree model on weather-model
features, graded daily against published actuals.

**Scope:** utility-scale solar generation only. Rooftop (behind-the-meter)
solar is explicitly out of scope — the target is CAISO's reported
utility-scale solar output, hourly, in MW.

## Data

- `data/caiso/solar_5min.parquet` — CAISO fuel-mix solar actuals at native
  5-minute resolution, 2023-01-01 → present. Quality notes:
  `docs/caiso_data_quality.md`.
- `data/registry/plants_ciso.parquet` — the CAISO utility-scale solar PV
  fleet: 833 operating plants, 24.23 GW AC, with county and balancing
  authority for sub-state grouping. Built from EIA-860M (monthly
  inventory, June 2026) for who is running, and the EIA-860 annual Solar
  schedule (2025 Early Release) for array geometry.

- `data/hrrr/hrrr_<YYYYMMDD>_<HH>z.parquet` — HRRR forecasts sampled at
  every plant, one file per model run, f01–f48. Weather grids are never
  stored. Design and restart notes: `docs/hrrr_backfill.md`.
- `data/train/table.parquet` — the model's training table: one row per
  (run_time, valid_time), with zone weather, the physical model's
  megawatts, calendar columns, the CAISO label and two baselines.
  Details: `docs/training_table.md`.

**The filter is the balancing authority, not the state.** CAISO is a
balancing authority whose territory reaches into Arizona and Nevada, so a
`state == CA` filter was wrong in both directions at once: it admitted 140
Californian plants in LDWP, IID, BANC, PacifiCorp and WALC whose output
never reaches CAISO's number, and it excluded 2,478 MW of Arizona and
Nevada solar whose output does. The second error made the modelled
clear-sky ceiling smaller than the fleet it was meant to bound — CAISO's
23.21 GW peak sat above the whole modelled fleet, which is impossible and
was the tell. At 24.23 GW the peak/installed ratio is a physically
sensible 0.96. Details: `docs/plant_registry.md`.

## The physical model

Before any learning, every plant is modelled from its own 3 km weather:
sun position, panel orientation (fixed, single-axis with backtracking,
dual-axis), transposition onto the panel face, cell temperature, DC
power, then inverter losses and clipping. Those megawatts sum to county,
zone and state. The same chain runs a second time on a clear sky, and
the ratio of the two is the clearness index.

On daylight hours across 2023-01 → 2024-10, the unfitted physics reaches
**1141 MW mean absolute error** against CAISO, beating both persistence
baselines (1262 and 1312 MW) and a naive zero (10,092 MW). That is the
bar the model has to clear, and it is a demanding one.

Only the statewide number is graded. County and zone figures are
physically-derived estimates that sum to it, and no hourly public truth
exists to check them against.

```sh
uv run python -m americast.features.table    # rebuild the training table
uv run python -m americast.features.report   # write data/reports/gate4.html
```

## The model

Three LightGBM boosters — p10, p50 and p90 — fitted on 2023-2024,
early-stopped on 2025 H1, and graded on 2025-07 onward. They predict the
share of the clear-sky ceiling the fleet delivers, not megawatts
directly, because the fleet outgrew its own training range: a tenth of
the test period sits above the highest label the model ever saw.

On the test period the model reaches **1236 MW mean absolute error**
against 1723 MW for clear-sky persistence and 1525 MW for the unfitted
physics — 28.3% skill against the baseline the build plan names, and
19.0% against the physics. It wins every lead bucket at 4 hours and
beyond.

Two things it does not do well, both measured rather than hidden. The
p10–p90 band covers 58.6% of hours where it claims 80%, and it sits too
low rather than being too narrow. The cause is that CAISO delivered
0.967× the physics during training and 1.023× during the test period:
the registry's newest plant is dated 2025-12, so plants commissioned
during the test period generate real megawatts and add no ceiling. The
model learned the first number and was graded against the second.
Details, and why refitting that constant would be a leak:
`docs/model.md`.

```sh
uv run python -m americast.model.model       # fit and save to data/model/
uv run python -m americast.model.eval        # score the test period
uv run python -m americast.model.report      # write data/reports/gate5.html
```

## The API

A read-only FastAPI service over the same data, for a separate React map
frontend. Local only for now; hosting is its own decision.

```sh
uv run python -m americast.api.app           # http://localhost:8000, docs at /docs
```

| Endpoint | Returns |
|---|---|
| `GET /runs` | stored model runs, newest first |
| `GET /plants` | 788 plants: id, name, lat, lon, capacity, county, zone |
| `GET /runs/{run_time}/plants` | per-plant `mw` and `clearness`, 47 hours |
| `GET /runs/{run_time}/totals` | state, zone and county curves |
| `GET /runs/latest/...` | alias for the newest run |

Two things the contract enforces rather than documents. Every value
array is exactly as long as `valid_times`, so a client can index them
together. And every aggregation level carries `validated`, true only for
the state total — county and zone are estimates that sum to the graded
number, and no consumer can present them as forecasts by accident.

Per-plant values are computed on demand (~1.5 s per run) and cached, so
nothing new is stored. Design notes:
`docs/superpowers/specs/2026-08-12-map-api-design.md`.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13+.

```sh
uv sync
uv run pytest
```
