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

**Registry sanity check:** the CAISO-BA slice of the registry is 21.52 GW
across 788 plants. Observed fuel-mix peak (Aug 2026) is 23.35 GW — higher,
because (1) CAISO's balancing authority includes ~2.5 GW of solar in
Arizona and Nevada that a state-filtered registry excludes, and (2) the
2025 filing cannot see plants energized in 2026. Including out-of-state
CISO plants, installed CISO capacity is 24.0 GW and the peak/installed
ratio is a physically sensible 0.97. Details: `docs/plant_registry.md`.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13+.

```sh
uv sync
uv run pytest
```
