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
- `data/registry/plants_ca.parquet` — California utility-scale solar PV
  registry from EIA-860 (2025 Early Release): 928 operating plants,
  23.88 GW AC, with county and balancing authority for sub-state grouping.

- `data/hrrr/hrrr_<YYYYMMDD>_<HH>z.parquet` — HRRR forecasts sampled at
  every plant, one file per model run, f01–f48. Weather grids are never
  stored. Design and restart notes: `docs/hrrr_backfill.md`.
- `data/train/table.parquet` — the model's training table: one row per
  (run_time, valid_time), with zone weather, the physical model's
  megawatts, calendar columns, the CAISO label and two baselines.
  Details: `docs/training_table.md`.

**Registry sanity check:** the CAISO-BA slice of the registry is 21.52 GW
across 788 plants. Observed fuel-mix peak (Aug 2026) is 23.35 GW — higher,
because (1) CAISO's balancing authority includes ~2.5 GW of solar in
Arizona and Nevada that a state-filtered registry excludes, and (2) the
2025 filing cannot see plants energized in 2026. Including out-of-state
CISO plants, installed CISO capacity is 24.0 GW and the peak/installed
ratio is a physically sensible 0.97. Details: `docs/plant_registry.md`.

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
bar the Gate 5 model has to clear.

Only the statewide number is graded. County and zone figures are
physically-derived estimates that sum to it, and no hourly public truth
exists to check them against.

```sh
uv run python -m americast.features.table    # rebuild the training table
uv run python -m americast.features.report   # write data/reports/gate4.html
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
